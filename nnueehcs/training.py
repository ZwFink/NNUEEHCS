import pytorch_lightning as L
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def _inst_init_if_not_none(inst, attr, val, default):
    if val is not None:
        setattr(inst, attr, val)
    else:
        setattr(inst, attr, default)


class Trainer(L.Trainer):
    def __init__(self, name, trainer_config, logger=None, callbacks=None):
        self.name = name
        _inst_init_if_not_none(self, 'callbacks', 
                                callbacks, [EarlyStopping(monitor='val_loss')]
                                )
        _inst_init_if_not_none(self, 'logger', logger, 
                                L.loggers.CSVLogger('logs', name=name)
                                )

        super().__init__(callbacks=self.callbacks, logger=self.logger, 
                         **trainer_config
                         )
        self.logger.log_hyperparams(trainer_config)

    def get_logger(self):
        return self.logger

    def get_callbacks(self):
        return self.callbacks