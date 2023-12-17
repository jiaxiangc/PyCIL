# Diffusion model fine-tune in imagenet, cifar10, cifar100
# TODO: Check paper, weights, fid, is

import os
import json
import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from diffusers import UNet2DModel, DDIMScheduler, DDIMPipeline

from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy


init_epoch = 100
init_lr = 4e-4
init_milestones = [70]
init_lr_decay = 1e-2
init_weight_decay = 0.0005
batch_size = 64
num_workers = 8
beta_1, beta_2 = 0.9, 0.999
adam_epsilon = 1e-8


# epochs = 20
# lrate = 0.0001
# milestones = [10, 15]
# lrate_decay = 0.1
# weight_decay = 2e-4

class DMFinetune(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        # self._network = IncrementalNet(args, False)
        self.args = args
        unet_config = json.load(open(os.path.join(args["pretrained_model_path"], "config.json"), "r"))
        self._network = UNet2DModel(**unet_config)
        self._noise_scheduler = DDIMScheduler.from_pretrained(args["pretrained_model_path"])

        logging.info("Successfully create diffusion model")

    def after_task(self):
        self._known_classes = self._total_classes
        # self._log_validation()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        # self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        optimizer = optim.AdamW(
            self._network.parameters(),
            lr=init_lr,
            weight_decay=init_weight_decay,
            betas=(beta_1, beta_2),
            eps=adam_epsilon,
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
        )
        self._train_loop(train_loader, test_loader, optimizer, scheduler)

    def _train_loop(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            for i, (_, images, labels) in enumerate(train_loader):
                images = images.to(self._device)
                labels = labels.to(self._device)

                noise = torch.randn_like(images)
                bsz = images.shape[0]
                
                timesteps = torch.randint(0, self._noise_scheduler.config.num_train_timesteps, (bsz,), device=self._device)
                timesteps = timesteps.long()

                noisy_images = self._noise_scheduler.add_noise(images, noise, timesteps)

                if self._noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif self._noise_scheduler.config.prediction_type == "v_prediction":
                    target = self._noise_scheduler.get_velocity(images, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {self._noise_scheduler.config.prediction_type}")
                
                model_pred = self._network(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                )
                self._log_validation(epoch)
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                )

            prog_bar.set_description(info)

        logging.info(info)

    @torch.no_grad()
    def _log_validation(self, epoch):
        if hasattr(self._network, "module"):
            unet = self._network.module
        # Load pipeline
        pipeline = DDIMPipeline.from_pretrained(
            self.args["pretrained_model_path"],
            unet=unet,
        )
        pipeline.to(self._device)
        # Inference
        images = pipeline(batch_size=16, num_inference_steps=50, output_type="pt").images
        
        # Save images
        images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
        grid_images = make_grid(images, nrow=4, normalize=True, value_range=(0, 1))
        if not os.path.exists(self.args["output_dir"]):
            os.makedirs(self.args["output_dir"])
            logging.info(f"Create {self.args['output_dir']}")

        save_image(grid_images, os.path.join(self.args["output_dir"], f"task_{self._cur_task:02d}_epoch_{epoch:03d}.jpg"))