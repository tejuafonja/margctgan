import warnings

import pandas as pd
import torch
from torch import pca_lowrank

warnings.filterwarnings("ignore")

from ctgan.synthesizers.base import BaseSynthesizer as Base, random_state


class PCA(Base):
    def __init__(self, n_components=None, device="cpu"):
        self.n_components = n_components
        self._device = device

    @random_state
    def fit(self, X):
        if type(X) != torch.Tensor:
            X = self.to_torch(X)

        X = X.to(self._device)

        n_samples, n_features = X.shape

        if self.n_components is None or self.n_components > n_samples:
            self.n_components = min(n_samples, n_features)
            
        self.U, self.S, self.V = pca_lowrank(X, q=self.n_components, center=False)

        explained_variance_ = (self.S**2) / (n_samples - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var

        self.explained_variance_ = explained_variance_[: self.n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[: self.n_components]

        self.n_samples = n_samples
        self.n_features = n_features

        return self

    def set_device(self, device):
        self._device = device

    def transform(self, X):
        if type(X) == pd.DataFrame:
            self.columns = X.columns
        else:
            self.columns = None

        if type(X) != torch.Tensor:
            X = self.to_torch(X)

        X = X.to(self._device)
        self.dtype = X.dtype

        X_bar = torch.matmul(X, self.V[:, : self.n_components])
        return X_bar

    def inverse_transform(self, X_bar):
        X_prime = torch.matmul(X_bar, self.V[:, : self.n_components].T)

        X_prime = X_prime.to(self._device)
        X_prime = X_prime.to(self.dtype).numpy()

        if self.columns is not None:
            X_prime = pd.DataFrame(X_prime, columns=self.columns)

        return X_prime

    def to_torch(self, X):
        if type(X) == pd.DataFrame:
            X = X.values

        X = torch.from_numpy(X)
        return X

