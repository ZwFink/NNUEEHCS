from torch import nn
from deltauq import deltaUQ_MLP, deltaUQ_CNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pytorch_lightning as L
import copy

training_defaults = {
    'learning_rate': 1e-3,
    'batch_size': 32,
    'num_workers': 1,
    'num_epochs': 10,
    'loss': 'l1_loss',
}


class WrappedModelBase(pl.LightningModule):
    def __init__(self, train_config=None,
                 validation_config=None
                 ):
        super(WrappedModelBase, self).__init__()
        self.train_config = copy.deepcopy(training_defaults)
        self.validation_config = copy.deepcopy(training_defaults)
        self.set_train_config(train_config)
        self.set_validation_config(validation_config)

    def set_train_config(self, train_config):
        if train_config is None:
            # We have a default train config,
            # we need to process it instead of
            # any user-defined overrides
            self.set_train_config(self.train_config)
            return

        self.train_config.update(train_config)
        self.loss = self.get_loss_fn(train_config['loss'])

    def set_validation_config(self, validation_config):
        if validation_config is None:
            # Yes, we default to the train config
            self.set_validation_config(self.train_config)
            return

        self.validation_config.update(validation_config)
        self.val_loss = self.get_loss_fn(validation_config['loss'])

    def get_loss_fn(self, name):
        try:
            return getattr(F, name)
        except AttributeError:
            raise ValueError(f"Unknown loss function: {name}")

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def on_train_start(self):
        all_params = {'train_config': self.train_config,
                      'validation_config': self.validation_config
                      }
        self.logger.log_hyperparams(all_params)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class EnsembleModel(WrappedModelBase):
    def __init__(self, models, **kwargs):
        super(EnsembleModel, self).__init__(**kwargs)
        self.models = nn.ModuleList(models)

    def forward(self, x, return_ue=False):
        outputs = torch.stack([model(x) for model in self.models])
        if return_ue:
            std = outputs.std(0)
            return outputs.mean(0), std
        return outputs.mean(0)


class MLPModel(WrappedModelBase):
    def __init__(self, model, **kwargs):
        super(MLPModel, self).__init__(**kwargs)
        self.model = model

    def forward(self, x):
        return self.model(x)


class KDEMLPModel(MLPModel):
    pass


class DeltaUQMLP(deltaUQ_MLP, WrappedModelBase):
    def __init__(self, base_model, estimator, **kwargs):
        deltaUQ_MLP.__init__(self, base_model, estimator)
        # somehow, the constructor of WrappedModelBase
        # removes our 'net' member. We need to re-add it
        # after the initialization
        net = self.net
        WrappedModelBase.__init__(self, **kwargs)
        self.net = net

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, torch.cat((y, y), dim=0))
        self.log('train_loss', loss)
        return loss


class PAGERMLP(DeltaUQMLP):
    pass