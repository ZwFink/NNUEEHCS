from torch import nn
from deltauq import deltaUQ_MLP, deltaUQ_CNN
import kde

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pytorch_lightning as L
import copy
from torch.func import vmap, stack_module_state

training_defaults = {
    'learning_rate': 1e-3,
    'batch_size': 32,
    'num_workers': 1,
    'num_epochs': 10,
    'loss': 'l1_loss',
}

# Custom loss functions registry
CUSTOM_LOSS_FUNCTIONS = {}

def register_loss_function(name, loss_fn):
    """Register a custom loss function"""
    CUSTOM_LOSS_FUNCTIONS[name] = loss_fn

def evidential_loss(outputs, targets, lam=1.0, epsilon=1e-4):
    """
    Evidential loss function for Deep Evidential Regression
    
    Args:
        outputs: Model outputs [gamma, nu, alpha, beta] where gamma is prediction
        targets: Ground truth targets
        lam: Regularization coefficient
        epsilon: Small constant for numerical stability
    
    Returns:
        Total loss combining likelihood and regularization terms
    """
    gamma, nu, alpha, beta = torch.chunk(outputs, 4, dim=-1)
    
    # Ensure positive values for variance parameters
    nu = F.softplus(nu) + epsilon
    alpha = F.softplus(alpha) + 1 + epsilon  # alpha > 1
    beta = F.softplus(beta) + epsilon
    
    # Compute the negative log likelihood
    diff = (targets - gamma) ** 2
    likelihood_loss = 0.5 * torch.log(torch.pi / nu) + alpha * torch.log(2 * beta) + \
                     (alpha + 0.5) * torch.log(nu * diff + 2 * beta) - \
                     torch.lgamma(alpha + 0.5) + torch.lgamma(alpha)
    
    # Regularization term (encourages higher uncertainty for wrong predictions)
    error = torch.abs(targets - gamma)
    reg = error * (2 * nu + alpha)
    
    return likelihood_loss.mean() + lam * reg.mean()

# Register the evidential loss
register_loss_function('evidential_loss', evidential_loss)


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
        # Only set loss function if 'loss' is specified in config
        # Some models (like DeepEvidentialModel) use custom loss functions
        if 'loss' in self.train_config:
            self.loss = self.get_loss_fn(self.train_config['loss'])
        else:
            # Use a default loss function as fallback
            self.loss = self.get_loss_fn('mse_loss')

    def set_validation_config(self, validation_config):
        if validation_config is None:
            # Yes, we default to the train config
            self.set_validation_config(self.train_config)
            return

        self.validation_config.update(validation_config)
        # Only set loss function if 'loss' is specified in config
        if 'loss' in self.validation_config:
            self.val_loss = self.get_loss_fn(self.validation_config['loss'])
        else:
            # Use a default loss function as fallback
            self.val_loss = self.get_loss_fn('mse_loss')

    def get_loss_fn(self, name):
        """Get loss function by name, supporting both built-in and custom functions"""
        # First check if it's a custom loss function
        if name in CUSTOM_LOSS_FUNCTIONS:
            return CUSTOM_LOSS_FUNCTIONS[name]
        
        # Then check if it's a built-in function
        try:
            return getattr(F, name)
        except AttributeError:
            raise ValueError(f"Unknown loss function: {name}")

    def get_custom_loss_fn(self):
        """Override this method to provide a custom loss function for specific models"""
        return None

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Use custom loss if available, otherwise use configured loss
        custom_loss = self.get_custom_loss_fn()
        if custom_loss is not None:
            loss = custom_loss(y_hat, y)
        else:
            loss = self.loss(y_hat, y)
        
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Use custom loss if available, otherwise use configured loss
        custom_loss = self.get_custom_loss_fn()
        if custom_loss is not None:
            loss = custom_loss(y_hat, y)
        else:
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
        
    def get_non_torch_memory_usage_mb(self):
        return 0


class EnsembleModel(WrappedModelBase):
    def __init__(self, models, vectorize=False, **kwargs):
        super(EnsembleModel, self).__init__(**kwargs)
        self.models = nn.ModuleList(models)
        self.vectorize = vectorize
        if vectorize:
            self.params, self.buffers = torch.func.stack_module_state(self.models)

    def call_single_model(self, params, buffers, x):
        return torch.func.functional_call(self.models[0], (params, buffers), (x,))

    def forward(self, x, return_ue=False):
        if self.vectorize:
            outputs = torch.vmap(self.call_single_model, (0, 0, None))(self.params, self.buffers, x)
        else:
            outputs = torch.stack([model(x) for model in self.models])
        
        if return_ue:
            std = outputs.std(0)
            return outputs.mean(0), std
        return outputs.mean(0)
        
    def to(self, device):
        super().to(device)
        for model in self.models:
            model.to(device)
            
        if self.vectorize:
            for param_name in self.params:
                self.params[param_name] = self.params[param_name].to(device)
                
            for buffer_name in self.buffers:
                self.buffers[buffer_name] = self.buffers[buffer_name].to(device)
        return self


class MCDropoutModel(WrappedModelBase):
    def __init__(self, model, num_samples=100, dropout_percent=0.5, vectorize=False, **kwargs):
        super(MCDropoutModel, self).__init__(**kwargs)
        self.model = model
        self.num_samples = num_samples
        self.dropout_percent = dropout_percent
        self.vectorize = vectorize
        
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_percent
                
        self.stacked = False
        
    def _ensure_stacked(self):
        if not self.stacked and self.vectorize:
            params, buffers = stack_module_state([self.model] * self.num_samples)
            self.params, self.buffers = params, buffers
            self.stacked = True

    def call_single_forward(self, params, buffers, x):
        return torch.func.functional_call(self.model, (params, buffers), (x,))

    def forward(self, x, return_ue=False):
        if self.training:
            return self.model(x)
        else:
            if self.vectorize:
                self._ensure_stacked()
                preds = torch.vmap(self.call_single_forward, (0, 0, None), 
                                  randomness='different')(self.params, self.buffers, x)
            else:
                preds = []
                for _ in range(self.num_samples):
                    preds.append(self.model(x))
                preds = torch.stack(preds)
            
            if return_ue:
                return preds.mean(0), preds.std(0)
            return preds.mean(0)

    def eval(self):
        super().eval()
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
                
    def to(self, device):
        super().to(device)
        self.model.to(device)
        if self.stacked:
            for param_name in self.params:
                self.params[param_name] = self.params[param_name].to(device)
            for buffer_name in self.buffers:
                self.buffers[buffer_name] = self.buffers[buffer_name].to(device)
        return self


class MLPModel(WrappedModelBase):
    def __init__(self, model, **kwargs):
        super(MLPModel, self).__init__(**kwargs)
        self.model = model

    def forward(self, x):
        return self.model(x)


class BaselineModel(WrappedModelBase):
    """Baseline/dummy UE method that trains a normal network but returns dummy uncertainty values.
    
    This serves as a baseline for comparison with actual UE methods. It trains a standard neural
    network and returns dummy uncertainty estimates when requested, allowing it to be used with
    the same evaluation infrastructure as real UE methods.
    """
    def __init__(self, model, uncertainty_value=1.0, **kwargs):
        """
        Args:
            model: The base neural network
            uncertainty_value: Constant value to return as uncertainty estimate (default: 1.0)
        """
        super(BaselineModel, self).__init__(**kwargs)
        self.model = model
        self.uncertainty_value = uncertainty_value

    def forward(self, x, return_ue=False):
        """Forward pass through the model.
        
        Args:
            x: Input tensor
            return_ue: If True, return (predictions, dummy_uncertainty), else just predictions
            
        Returns:
            If return_ue=False: predictions tensor
            If return_ue=True: (predictions, dummy_uncertainty_tensor)
        """
        predictions = self.model(x)
        
        if return_ue:
            # Return dummy uncertainty values - same shape as predictions but constant value
            uncertainty = torch.full_like(predictions, self.uncertainty_value)
            return predictions, uncertainty
        
        return predictions

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
            # negate so ID scores associated with higher density
            # receive lower uncertainty scores.
            dens = -torch.exp(torch.tensor(log_dens))
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
            print(f"Fitting KDE on {len(self._train_data_to_fit)} samples")
            if self._epochs == 0:
                trn_data = torch.cat(self._train_data_to_fit)
                pl_module.fit_kde(trn_data)
            self._epochs += 1

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if self._epochs == 0:
                self._train_data_to_fit.append(batch[0])

    def get_callbacks(self):
        return [KDEMLPModel.KDEFitCallback()]


class KNNKDEMLPModel(MLPModel):
    def __init__(self, base_model, bandwidth='scott', k=10, train_fit_prop=1.0, **kwargs):
        super(KNNKDEMLPModel, self).__init__(base_model, **kwargs)
        self.bandwidth = bandwidth
        self.k = k
        self.train_fit_prop = train_fit_prop
        self._kde = kde.KNNKDE(k=self.k, bandwidth=self.bandwidth)

    def fit_kde(self, data):
        self._kde.fit(data)

    def forward(self, x, return_ue=False):
        if return_ue and self._kde is None:
            raise ValueError("KDE not fitted yet")
        pred = super().forward(x)
        if return_ue:
            # negate so ID scores associated with higher density
            # receive lower uncertainty scores.
            return pred, -self._kde.kernel_density(x)
        return pred

    class KNNKDEFitCallback(L.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self._train_data_to_fit = []
            self._epochs = 0

        def on_validation_epoch_start(self, trainer, pl_module):
            if self._epochs == 0:
                trn_data = torch.cat(self._train_data_to_fit)
                pl_module.fit_kde(trn_data)
            self._epochs += 1

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if self._epochs == 0:
                self._train_data_to_fit.append(batch[0])

    def get_callbacks(self):
        callbacks = [self.KNNKDEFitCallback()]
        return callbacks

    def get_non_torch_memory_usage_mb(self):
        return self._kde.get_gpu_memory_use_mb()


class DeltaUQMLP(deltaUQ_MLP, WrappedModelBase):
    def __init__(self, base_model, estimator='std', num_anchors=5, anchored_batch_size=None, **kwargs):
        deltaUQ_MLP.__init__(self, base_model, estimator)
        # somehow, the constructor of WrappedModelBase
        # removes our 'net' member. We need to re-add it
        # after the initialization
        net = self.net
        WrappedModelBase.__init__(self, **kwargs)
        self.net = net
        self.num_anchors = num_anchors
        if anchored_batch_size is None:
            import sys
            self.batch_size = sys.maxsize
        else:
            self.batch_size = anchored_batch_size
        # Initialize anchors as None to prevent errors
        self.register_buffer('_anchors', None)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, torch.cat((y, y), dim=0))
        self.log('train_loss', loss)
        return loss

    def forward(self, x, return_ue=False):
        batch_size = self.batch_size
        
        if self.training:
            forward_pass_fn = lambda inputs: deltaUQ_MLP.forward(self, inputs)
        else:
            if not hasattr(self, 'anchors') or self._anchors is None:
                if return_ue:
                    print("WARNING: Returning UE without anchors")
                forward_pass_fn = lambda inputs: deltaUQ_MLP.forward(self, inputs)
            else:
                forward_pass_fn = lambda inputs: deltaUQ_MLP.forward(
                    self, inputs, anchors=self.anchors, n_anchors=self.num_anchors, return_std=return_ue
                )
        
        # not even a full batch
        if len(x) <= batch_size:
            return forward_pass_fn(x)

        outputs = []
        for i in range(0, len(x), batch_size):
            batch = x[i:i+batch_size]
            batch_output = forward_pass_fn(batch)
            outputs.append(batch_output)
        
        if isinstance(outputs[0], tuple):
            return tuple(torch.cat([o[i] for o in outputs], dim=0) for i in range(len(outputs[0])))
        else:
            return torch.cat(outputs, dim=0)

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

        def on_validation_epoch_start(self, trainer, pl_module):
            if self._epochs == 0 and len(self._train_data_to_fit) > 0:
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
    def __init__(self, base_model, estimator='std', anchored_batch_size=None, num_anchors=5, vectorize=False, **kwargs):
        DeltaUQMLP.__init__(self, base_model, estimator, anchored_batch_size)
        # somehow, the constructor of WrappedModelBase
        # removes our 'net' member. We need to re-add it
        # after the initialization
        net = self.net
        WrappedModelBase.__init__(self, **kwargs)
        self.net = net
        self.num_anchors = num_anchors
        self.vectorize = vectorize
        # Initialize anchors as None to prevent errors
        self.register_buffer('_anchors', None)

    def forward(self, x, return_ue=False):
        res = DeltaUQMLP.forward(self, x, return_ue)
        if not return_ue:
            return res

        pred, uncertainty = res
        conformal_scores = self._score_samples(x, self.anchors,
                                               self.anchors_Y)

        # uncertainty score should be the max of uncertainty, conformal_scores
        uncertainty_score = torch.maximum(uncertainty, conformal_scores)
        
        return pred, uncertainty_score

    def _anchored_predictions(self, x, anchors):
        if self.vectorize:
            batch_size = x.shape[0]
            
            all_samples = x
            
            p_matrix = deltaUQ_MLP.forward(self,
                                          anchors,
                                          anchors=all_samples,
                                          n_anchors=batch_size,
                                          return_pred_matrix=True)
        else:
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
            p_matrix = torch.concat(p_matrix)
        
        return p_matrix.squeeze(-1)

    def _score_samples(self, x, anchors_X, anchors_Y):
        p_matrix = self._anchored_predictions(x, anchors_X)
        score = torch.max(torch.abs(p_matrix - anchors_Y.T), dim=1)[0].unsqueeze(-1)
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

        def on_validation_epoch_start(self, trainer, pl_module):
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


class DeepEvidentialModel(WrappedModelBase):
    """Deep Evidential Regression model that outputs evidential parameters and computes uncertainty."""
    
    def __init__(self, model, lam=1.0, **kwargs):
        """
        Args:
            model: Base neural network that outputs 4 values [gamma, nu, alpha, beta]
            lam: Regularization coefficient for evidential loss
        """
        super(DeepEvidentialModel, self).__init__(**kwargs)
        self.model = model
        self.lam = lam
        self.epsilon = 1e-4  # Fixed numerical stability constant

    def get_raw_outputs(self, x):
        """Get raw 4-parameter outputs for loss computation, regardless of training/eval mode."""
        return self.model(x)

    def forward(self, x, return_ue=False):
        """Forward pass through the evidential model.
        
        Args:
            x: Input tensor
            return_ue: If True, return (predictions, uncertainty), else just predictions
            
        Returns:
            If training: raw outputs (4 parameters for loss function)
            If return_ue=False (inference): predictions (gamma values)
            If return_ue=True (inference): (predictions, epistemic_uncertainty)
        """
        outputs = self.model(x)
        
        # During training, return raw outputs for the loss function
        if self.training:
            return outputs
        
        # During inference, process the outputs
        gamma, nu, alpha, beta = torch.chunk(outputs, 4, dim=-1)
        
        # Ensure positive values for variance parameters
        nu = F.softplus(nu) + self.epsilon
        alpha = F.softplus(alpha) + 1 + self.epsilon  # alpha > 1
        beta = F.softplus(beta) + self.epsilon
        
        predictions = gamma  # The mean prediction
        
        if return_ue:
            # Epistemic uncertainty
            epistemic_uncertainty = beta / (alpha - 1)
            return predictions, -epistemic_uncertainty
        
        return predictions

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.get_raw_outputs(x)  # Always get raw outputs for loss
        
        # Use custom loss function
        custom_loss = self.get_custom_loss_fn()
        if custom_loss is not None:
            loss = custom_loss(y_hat, y)
        else:
            loss = self.loss(y_hat, y)
        
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.get_raw_outputs(x)  # Always get raw outputs for loss
        
        # Use custom loss function
        custom_loss = self.get_custom_loss_fn()
        if custom_loss is not None:
            loss = custom_loss(y_hat, y)
        else:
            loss = self.loss(y_hat, y)
        
        self.log('val_loss', loss)
        return loss

    def get_custom_loss_fn(self):
        """Override to use evidential loss function"""
        def loss_fn(outputs, targets):
            return evidential_loss(outputs, targets, lam=self.lam, epsilon=self.epsilon)
        return loss_fn

    def get_evidential_parameters(self, x):
        """Get all evidential parameters for analysis.
        
        Returns:
            Dict with keys: gamma (mean), nu (precision), alpha, beta (variance params)
        """
        outputs = self.model(x)
        gamma, nu, alpha, beta = torch.chunk(outputs, 4, dim=-1)
        
        # Ensure positive values for variance parameters
        nu = F.softplus(nu) + self.epsilon
        alpha = F.softplus(alpha) + 1 + self.epsilon
        beta = F.softplus(beta) + self.epsilon
        
        return {
            'gamma': gamma,
            'nu': nu, 
            'alpha': alpha,
            'beta': beta
        }


