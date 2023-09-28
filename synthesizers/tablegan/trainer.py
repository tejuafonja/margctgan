import os
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import binary_cross_entropy_with_logits

from synthesizers.tablegan.model import (
    determine_layers,
    Generator,
    Discriminator,
    Classifier,
    weights_init,
)
from metrics import *
from utils.sampler import DataSampler

### TableGAN model adapted from https://github.com/sdv-dev/SDGym/blob/master/sdgym/synthesizers/tablegan.py
class TableGANSynthesizer(object):
    """TableGAN Synthesizer."""

    def __init__(self, args):
        self.random_dim = args.random_dim
        self.num_channels = args.num_channels
        self.side = args.side
        self.l2scale = args.lr

        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.resume = args.resume

        self.use_sampler = args.use_sampler
        self.discrete_columns = args.discrete_columns
        self.metadata = args.metadata

        self._device = args.device
        self._log_dir = args.log_dir
        self.args = args

        self.data_transformer = args.data_transformer
        self.tablegan_transformer = args.tablegan_transformer

    def fit(self, train_data, validation_data=None):
        """Fit the TableGAN Synthesizer model to the training data.

        Args:
            train_data (pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.

            validation_data (pandas.DataFrame):
                Validation Data. It must be a 2-dimensional numpy array or a pandas.DataFrame. Defaults to `None`
        """

        self.data_transformer.fit(train_data, discrete_columns=self.discrete_columns)
        self.tablegan_transformer.fit(self.metadata)

        data = self.data_transformer.transform(train_data)

        ## setup models
        if self.resume:
            self.load()
        else:
            layers_D, layers_G, layers_C = determine_layers(
                self.side, self.random_dim, self.num_channels
            )

            self.generator = Generator(self.side, layers_G).to(self._device)
            self.discriminator = Discriminator(self.side, layers_D).to(self._device)
            self.classifier = Classifier(
                self.side, layers_C, self.data_transformer, self._device
            ).to(self._device)

            self.generator.apply(weights_init)
            self.discriminator.apply(weights_init)
            self.classifier.apply(weights_init)

        ## optimizers
        optimizer_params = dict(
            lr=2e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale
        )
        optimizerG = Adam(self.generator.parameters(), **optimizer_params)
        optimizerD = Adam(self.discriminator.parameters(), **optimizer_params)
        optimizerC = Adam(self.classifier.parameters(), **optimizer_params)

        # fmt: off
        if self.use_sampler:
            data_sampler = DataSampler(data, self.data_transformer.output_info_list, log_frequency=True)
            steps_per_epoch = max(len(data) // self.batch_size, 1)
        else:
            data = self.tablegan_transformer.transform(data)
            data = torch.from_numpy(data.astype("float32"))
            dataset = TensorDataset(data)

            if self.batch_size < len(data):
                loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            else:
                loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
            steps_per_epoch = len(loader)
        # fmt: on

        for epoch in range(self.epochs):
            # fmt: off
            for id_ in range(steps_per_epoch):

                if self.use_sampler:
                    real = data_sampler.sample_data(self.batch_size, None, None)
                    real = self.tablegan_transformer.transform(real)
                    real = torch.from_numpy(real.astype("float32")).to(self._device)
                else:
                    real = next(iter(loader))[0].to(self._device)

                ## discriminator
                optimizerD.zero_grad()
                noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self._device)
                fake = self.generator(noise)
                
                y_real = self.discriminator(real)
                y_fake = self.discriminator(fake)
                
                loss_d_real = -(torch.log(y_real + 1e-4).mean())
                loss_d_fake = -(torch.log(1.0 - y_fake + 1e-4).mean())
                loss_d = loss_d_real + loss_d_fake
                
                loss_d.backward()
                optimizerD.step()

                ## generator
                optimizerG.zero_grad()
                noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self._device)
                
                fake = self.generator(noise)
                y_fake = self.discriminator(fake)
                
                loss_g = -(torch.log(y_fake + 1e-4).mean())
                loss_g.backward(retain_graph=True)

                ## information loss
                loss_mean = torch.norm(torch.mean(fake, dim=0) - torch.mean(real, dim=0), p=1)
                loss_std = torch.norm(torch.std(fake, dim=0) - torch.std(real, dim=0), p=1)
                
                loss_info = loss_mean + loss_std
                loss_info.backward()
                optimizerG.step()

                ## classifier
                if self.classifier.valid:
                    noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self._device)
                    fake = self.generator(noise)
                    
                    real_prediction, real_label = self.classifier(real)
                    fake_prediction, fake_label = self.classifier(fake)

                    loss_c_real = binary_cross_entropy_with_logits(real_prediction, real_label)
                    loss_c_fake = binary_cross_entropy_with_logits(fake_prediction, fake_label)

                    ## update generator
                    optimizerG.zero_grad()
                    loss_c_fake.backward()
                    optimizerG.step()

                    ## update classifier
                    optimizerC.zero_grad()
                    loss_c_real.backward()
                    optimizerC.step()

                    loss_c_real = loss_c_real.item()
                    loss_c_fake = loss_c_fake.item()

                else:
                    loss_c_real = 0
                    loss_c_fake = 0

            ## saved model
            if epoch + 1 % self.args.save_after == 0:
                self.save()

        ## Save final model
        self.save()

    def sample(self, n):
        self.generator.eval()

        steps = n // self.batch_size + 1
        data = []

        for i in range(steps):
            noise = torch.randn(
                self.batch_size, self.random_dim, 1, 1, device=self._device
            )
            fake = self.generator(noise)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)[:n]
        data = self.tablegan_transformer.inverse_transform(data)
        data = self.data_transformer.inverse_transform(data)

        self.generator.train()
        return data

    def save(self):
        print("Saving model...")
        torch.save(self.generator, os.path.join(self._log_dir, "generator.pk"))
        torch.save(self.discriminator, os.path.join(self._log_dir, "discriminator.pk"))
        torch.save(self.classifier, os.path.join(self._log_dir, "classifier.pk"))
        print(f"Model successfully saved to {self._log_dir}")

    def load(self):
        print(f"Loading saved model from {self._log_dir}...")
        self.generator = torch.load(os.path.join(self._log_dir, "generator.pk"))
        self.discriminator = torch.load(os.path.join(self._log_dir, "discriminator.pk"))
        self.classifier = torch.load(os.path.join(self._log_dir, "classifier.pk"))
        print("Saved model successfully loaded.")
