import pytest
import torch
from nnueehcs.model_builder import EnsembleModelBuilder, KDEModelBuilder, DeltaUQMLPModelBuilder
import pytorch_lightning as L
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd


def is_within_tolerance(number, target, tolerance):
    upper_bound = target * (1 + tolerance)
    return number <= upper_bound


@pytest.fixture()
def trainer_config():
    return {
            'accelerator': 'cpu',
            'max_epochs': 5000,
            'overfit_batches': 1,
            'log_every_n_steps': 5,
            'num_sanity_val_steps': 0,
            'gradient_clip_val': 5}


@pytest.fixture()
def training_config():
    return {'loss': 'l1_loss'}


@pytest.fixture()
def network_descr():
    return [
        {'Linear': {'args': [3, 256]}},
        {'ReLU': {}},
        {'Linear': {'args': [256, 1]}}
    ]


@pytest.fixture()
def train_dataset():
    x = torch.randn(32, 3)
    return torch.utils.data.TensorDataset(x, x.sum(1, keepdim=True))


@pytest.fixture()
def train_dataloader(train_dataset):
    return torch.utils.data.DataLoader(train_dataset, batch_size=32)


@pytest.fixture(autouse=True)
def cleanup_files():
    yield
    import shutil
    shutil.rmtree('logs', ignore_errors=True)

    import os
    if os.path.exists('model.pth'):
        os.remove('model.pth')


def get_trainer(trainer_config, name):
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=200, verbose=False, mode="min")
    logger = L.loggers.CSVLogger("logs", name)
    return L.Trainer(callbacks=[early_stop_callback], **trainer_config, logger=logger), logger


def model_accuracy_assertions(log_dir):
    logger_path = f'{log_dir}/metrics.csv'
    val_loss = pd.read_csv(logger_path)['val_loss']
    min_loss = val_loss.min()

    assert is_within_tolerance(min_loss, 0.0018693739548325, 0.99)
    assert min_loss < 0.01
    assert val_loss.idxmin() > val_loss.idxmax()
    assert val_loss.min()*100 < val_loss.max()


def prediction_assertions(model):
    torch.save(model, 'model.pth')
    model = torch.load('model.pth')
    x = torch.randn(1, 3)
    y = model(x)
    assert torch.allclose(y, model(x))


def test_ensembles(trainer_config, training_config, network_descr, train_dataloader):
    trainer, logger = get_trainer(trainer_config, 'ensembles')

    ensemble_descr = {'num_models': 3}
    ensembles = EnsembleModelBuilder(network_descr, ensemble_descr, train_config=training_config).build()
    trainer.fit(ensembles, train_dataloader, train_dataloader)

    model_accuracy_assertions(logger.log_dir)
    prediction_assertions(ensembles)


def test_kde(trainer_config, training_config, network_descr, train_dataloader):
    trainer, logger = get_trainer(trainer_config, 'kde')

    kde = KDEModelBuilder(network_descr, {}, train_config=training_config).build()
    trainer.fit(kde, train_dataloader, train_dataloader)

    model_accuracy_assertions(logger.log_dir)
    prediction_assertions(kde)


def test_duq(trainer_config, training_config, network_descr, train_dataloader):
    trainer, logger = get_trainer(trainer_config, 'kde')

    duq = DeltaUQMLPModelBuilder(network_descr, {'estimator': 'std'}, train_config=training_config).build()
    print(duq)
    trainer.fit(duq, train_dataloader, train_dataloader)

    model_accuracy_assertions(logger.log_dir)
    prediction_assertions(duq)
