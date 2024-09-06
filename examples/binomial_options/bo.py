import torch
from torch import nn
from torch.utils.data import DataLoader
from nnueehcs.model_builder import EnsembleModelBuilder, KDEModelBuilder, DeltaUQMLPModelBuilder
from nnueehcs.training import Trainer
from nnueehcs.data_utils import get_dataset_from_config
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import yaml
import click
import matplotlib.pyplot as plt

def get_trainer(trainer_config, name, model):
    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.00, patience=10, verbose=False, mode='min')]
    extra_cbs = model.get_callbacks()
    if extra_cbs:
        callbacks.extend(extra_cbs)
    return Trainer(name, trainer_config, callbacks=callbacks)


@click.command()
@click.option('--eval_only', is_flag=True)
def main(eval_only):
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
        trainer_cfg = config['trainer']
        training_cfg = config['training']
        model_cfg = config['model']
        dataset_cfg = config['datasets']


    if eval_only is False:
        name = 'binomial_options'

        # builder = EnsembleModelBuilder(model_cfg['architecture'], model_cfg['ensemble_model'])

        dset = get_dataset_from_config(dataset_cfg, 'bo_ds')
        builder = DeltaUQMLPModelBuilder(model_cfg['architecture'], model_cfg['delta_uq'])
        model = builder.build().to(dset.dtype)
        trainer = get_trainer(trainer_cfg, name, model)


        ipt = dset.input
        opt = dset.output
        # do min-max scaling to get it to 0-1
        opt = (opt - opt.min()) / (opt.max() - opt.min())
        dset.output = opt
        ipt = (ipt - ipt.min()) / (ipt.max() - ipt.min())
        dset.input = ipt



        split_data = dset.train_test_split(test_proportion=training_cfg['validation_split'])
        train, test = split_data
        train_dl = DataLoader(train, batch_size=training_cfg['batch_size'], shuffle=True)
        test_dl = DataLoader(test, batch_size=training_cfg['batch_size'], shuffle=False)
        trainer.fit(model, train_dl, test_dl)
        torch.save(model, 'model.pth')

    model = torch.load('model.pth')

    model.eval()
    with torch.no_grad():
        dset_id = get_dataset_from_config(dataset_cfg, 'bo_ds')
        dset_ood = get_dataset_from_config(dataset_cfg, 'bo_ood')

        id_ipt = dset_id.input.to(model.device)
        id_opt = dset_id.output.to(model.device)

        id_ipt = (id_ipt - id_ipt.min()) / (id_ipt.max() - id_ipt.min())
        id_opt = (id_opt - id_opt.min()) / (id_opt.max() - id_opt.min())

        ood_ipt = dset_ood.input.to(model.device)
        ood_opt = dset_ood.output.to(model.device)

        ood_ipt = (ood_ipt - id_ipt.min()) / (id_ipt.max() - id_ipt.min())
        ood_opt = (ood_opt - id_opt.min()) / (id_opt.max() - id_opt.min())
        id_ipt = id_ipt[0:20000]
        id_opt = id_opt[0:20000]
        ood_ipt = ood_ipt[0:20000]
        ood_opt = ood_opt[0:20000]
        id_preds, id_ue = model(id_ipt, return_ue=True)
        ood_preds, ood_ue = model(ood_ipt, return_ue=True)
        
        id_loss = nn.L1Loss()(id_preds, id_opt)
        ood_loss = nn.L1Loss()(ood_preds, ood_opt)
        print(f'ID Loss: {id_loss}')
        print(f'OOD Loss: {ood_loss}')

        print(f'Average ID uncertainty: {id_ue.mean()}')
        print(f'Average OOD uncertainty: {ood_ue.mean()}')
        from scipy.stats import wasserstein_distance
        print(wasserstein_distance(id_ue.flatten(), ood_ue.flatten()))

        fig, ax = plt.subplots()
        ax.ecdf(id_ue.flatten(), label='ID')
        ax.ecdf(ood_ue.flatten(), label='OOD')
        ax.legend()
        # save it to png file
        plt.savefig('uncertainty.png')




if __name__ == '__main__':
    main()


