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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_config['learning_rate'],
                                      weight_decay=self.train_config.get('weight_decay', 0))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def get_callbacks(self):
        return []


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


class MCDropoutModel(WrappedModelBase):
    def __init__(self, model, num_samples=100, dropout_percent = 0.5, **kwargs):
        super(MCDropoutModel, self).__init__(**kwargs)
        self.model = model
        self.num_samples = num_samples
        self.dropout_perent = dropout_percent


    def forward(self, x, return_ue=False):
        preds = torch.stack([self.model(x) for _ in range(self.num_samples)])
        if return_ue:
            return preds.mean(0), preds.std(0)
        return preds.mean(0)


    def eval(self):
        super().eval()
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

class MLPModel(WrappedModelBase):
    def __init__(self, model, **kwargs):
        super(MLPModel, self).__init__(**kwargs)
        self.model = model

    def forward(self, x):
        return self.model(x)


class KDEMLPModel(MLPModel):
    def __init__(self, base_model, bandwidth='scott', rtol=0.1, train_fit_prop=1.0, **kwargs):
        super(KDEMLPModel, self).__init__(base_model, **kwargs)
        self.bandwidth = bandwidth
        self.rtol = rtol/10000
        self.kde = None
        self.train_fit_prop = train_fit_prop

    def fit_kde(self, data):
        from sklearn.neighbors import KernelDensity
        kde = KernelDensity(bandwidth=self.bandwidth, rtol=self.rtol)
        # randomly select 'train_fit_prop' of the data
        train_idxes = torch.randperm(len(data))[:int(self.train_fit_prop * len(data))]
        train_data = data[train_idxes].detach().cpu().numpy()
        kde.fit(train_data)
        self.kde = kde


    def forward(self, x, return_ue=False):
        if return_ue and self.kde is None:
            raise ValueError("KDE not fitted yet")
        pred = super().forward(x)
        if return_ue:
            import time
            test = time.time()
            log_dens = self.kde.score_samples(x.detach().cpu().numpy())
            dens = torch.exp(torch.tensor(log_dens))
            tend = time.time()
            print(tend-test)
            return pred, dens
        return pred

    class KDEFitCallback(L.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self._train_data_to_fit = []
            self._epochs = 0

        def on_train_epoch_end(self, trainer, pl_module):
            if self._epochs == 0:
                trn_data = torch.cat(self._train_data_to_fit)
                pl_module.fit_kde(trn_data)
            self._epochs += 1

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if self._epochs == 0:
                self._train_data_to_fit.append(batch[0])

    def get_callbacks(self):
        return [KDEMLPModel.KDEFitCallback()]


class DeltaUQMLP(deltaUQ_MLP, WrappedModelBase):
    def __init__(self, base_model, estimator='std', num_anchors=5, **kwargs):
        deltaUQ_MLP.__init__(self, base_model, estimator)
        # somehow, the constructor of WrappedModelBase
        # removes our 'net' member. We need to re-add it
        # after the initialization
        net = self.net
        WrappedModelBase.__init__(self, **kwargs)
        self.net = net
        self.num_anchors = num_anchors

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, torch.cat((y, y), dim=0))
        self.log('train_loss', loss)
        return loss

    def forward(self, x, return_ue=False):
        if self.training:
            return deltaUQ_MLP.forward(self, x)
        else:
            if not hasattr(self, 'anchors'):
                return deltaUQ_MLP.forward(self, x)
            return deltaUQ_MLP.forward(self, x, anchors=self.anchors, n_anchors=self.num_anchors, return_std=return_ue)

    @property
    def anchors(self):
        return self._anchors
    
    @anchors.setter
    def anchors(self, value):
        if not hasattr(self, '_anchors'):
            self.register_buffer('_anchors', value)
        else:
            self._anchors = value.detach().clone()

    class DeltaUQGetAnchorsCallback(L.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self._train_data_to_fit = []
            self._epochs = 0

        def on_train_epoch_end(self, trainer, pl_module):
            if self._epochs == 0:
                trn_data = torch.cat(self._train_data_to_fit)
                pl_module.anchors = trn_data[0:pl_module.num_anchors].detach().clone()
            self._epochs += 1

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            bs = batch[0].shape[0]
            if self._epochs == 0 and bs*len(self._train_data_to_fit) < pl_module.num_anchors:
                self._train_data_to_fit.append(batch[0].detach())


    def get_callbacks(self):
        return [DeltaUQMLP.DeltaUQGetAnchorsCallback()]


class PAGERMLP(DeltaUQMLP, WrappedModelBase):
    def forward(self, x, return_ue=False):
        res = super().forward(x, return_ue)
        if not return_ue:
            return res

        pred, uncertainty = res
        conformal_scores = self._score_samples(x, self.anchors,
                                               self.anchors_Y)

        return pred, (uncertainty, conformal_scores)

    def _anchored_predictions(self, x, anchors):
        p_matrix = list()
        for sample in x:
            if len(sample.shape) == 1:
                sample = sample.unsqueeze(0)
            p = deltaUQ_MLP.forward(self,
                                    anchors,
                                    anchors=sample,
                                    n_anchors=len(sample),
                                    return_pred_matrix=True
                                    )
            p_matrix.append(p)
        return torch.concat(p_matrix).squeeze(-1)


    def _score_samples(self, x, anchors_X, anchors_Y):
        p_matrix = self._anchored_predictions(x, anchors_X)
        score = torch.max(torch.abs(p_matrix - anchors_Y.T), dim=1)[0]
        return score


    def get_callbacks(self):
        return [PAGERMLP.PAGERGetAnchorsCallback()]

    @property
    def anchors_Y(self):
        return self._anchors_Y
    
    @anchors_Y.setter
    def anchors_Y(self, value):
        if not hasattr(self, '_anchors_Y'):
            self.register_buffer('_anchors_Y', value)
        else:
            self._anchors_Y = value.detach().clone()

    class PAGERGetAnchorsCallback(L.callbacks.Callback):
        # Like DeltaUQGetAnchorsCallback, but we need
        # the input and outputs of the anchors
        def __init__(self):
            super().__init__()
            self._anchor_X = []
            self._anchor_Y = []
            self._epochs = 0

        def on_train_epoch_end(self, trainer, pl_module):
            if self._epochs == 0:
                nanchors = pl_module.num_anchors
                anchor_X = torch.cat(self._anchor_X)
                anchor_Y = torch.cat(self._anchor_Y)
                pl_module.anchors = anchor_X[0:nanchors].detach().clone()
                pl_module.anchors_Y = anchor_Y[0:nanchors].detach().clone()
            self._epochs += 1

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            bs = batch[0].shape[0]
            if self._epochs == 0 and bs*len(self._anchor_X) < pl_module.num_anchors:
                self._anchor_X.append(batch[0].detach())
                self._anchor_Y.append(batch[1].detach())


