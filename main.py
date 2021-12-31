import os
import yaml
import wandb
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from experiments import select_experiment
from pathlib import Path
from data.datamodule import StaticDataModule
from torchvision import transforms as tt

def set_default_config(dict):
    dict.setdefault('networks', {})
    dict.setdefault('optimization', {})
    dict.setdefault('dataset', 'ToyDataset')
    dict.setdefault('experiment_id', 'no_name')
    dict['networks'].setdefault('variational', False)
    dict['optimization'].setdefault('input_frames', 4)
    return dict


def _read_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    if config is None:
        config = {}
    return config

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--exp', default='standard')
    parser.add_argument('--onnx', default=0)
    parser.add_argument('--eval', default=False)
    parser.add_argument('--paramfile', default=None)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--test', default=False)
    parser.add_argument('--name', default=None)
    parser.add_argument('--modelfile', default=None)



    args = parser.parse_args()

    DEFAULT_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    CONFIG_ROOT = 'configs'
    wandb.login()

    if args.eval:
        params = _read_config(f'{args.paramfile}')
        wandb_logger = WandbLogger()
        if args.test:
            params['optimization']['precision'] = 32
            params['optimization']['batch_size'] = 2
            params['dataset']['sequence_length'] = 30
            params['dataset']['data_path'] = params['dataset']['data_path_local']
            wandb.init(project='hgn-poke', entity='stdhd', config=params, dir=os.path.join("runs"))
            if wandb.run.name == None:
                wandb.run.name = "offline"
            logging_root = os.path.join("runs", wandb.run.name)
        else:
            wandb.init(project='hgn-poke', entity='stdhd', config=params,
                       dir=os.path.join("runs"))
            logging_root = os.path.join("runs", wandb.run.name,
                                        "callback_checkpoints")
        model = select_experiment(params, resume=False).load_from_checkpoint(args.modelfile, params=params, resume=False, strict=False)
        model.save_test_dir = params['logging'].get('save_test_dir', '')
        model.init_test_setting()
        trainer = pl.Trainer(gpus=args.gpus, max_epochs=params["optimization"]["epochs"],
                             precision=params["optimization"]["precision"],
                             move_metrics_to_cpu=True,
                             check_val_every_n_epoch=1,
                             resume_from_checkpoint=args.resume,
                             default_root_dir=logging_root,
                             logger=wandb_logger,)

        datamod = StaticDataModule(params, datakeys=params['dataset']['datakeys'], custom_transforms=[
            tt.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )] if params['dataset'].get('normalize', False) else None)
        datamod.setup()
        trainer.test(model, datamodule=datamod)
    else:
        params = _read_config(f'{os.path.join(DEFAULT_PROJECT_ROOT, CONFIG_ROOT, args.exp)}.yml')
        params = set_default_config(params)
        wandb_logger = WandbLogger()

        if args.test:
            print(params['dataset'])
            params['optimization']['precision'] = 32
            params['optimization']['batch_size'] = 2
            params['dataset']['data_path'] = params['dataset']['data_path_local']
            wandb.init(project='hgn-poke', entity='stdhd', config=params, dir=os.path.join("runs"))
            wandb.init(project='hgn-poke', entity='stdhd', config=params, dir=os.path.join("runs"))
            if wandb.run.name == None:
                wandb.run.name = "offline"
            logging_root = os.path.join("runs", wandb.run.name)
        else:
            wandb.init(project='hgn-poke', entity='stdhd', config=params, dir=os.path.join("runs"))
            logging_root = os.path.join("runs", wandb.run.name,
                                        "callback_checkpoints")

        wandb.define_metric("val/fvd_sampling", summary="min")
        wandb.define_metric("val/fvd_reconstruction", summary="min")
        wandb.define_metric("val/kld", summary="min")
        wandb.define_metric("train/kld", summary="min")
        wandb.define_metric("train/loss", summary="min")
        nt = select_experiment(params)(params=params)
        if wandb.run.name == None:
            wandb.run.name = "offline"
        if args.name:
            wandb.run.name = f"{args.name}_{wandb.run.name}"

        Path(logging_root).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(logging_root, 'config.yml'), 'w') as file:
            _ = yaml.dump(params, file)

        checkpoint_callback_monitor = nt.get_checkpoint_monitoring(logging_root)

        trainer = pl.Trainer(gpus=args.gpus, max_epochs=params["optimization"]["epochs"],
                             precision=params["optimization"]["precision"],
                             move_metrics_to_cpu=True,
                             check_val_every_n_epoch=1,
                             resume_from_checkpoint=args.resume,
                             default_root_dir=logging_root,
                             logger=wandb_logger,
                             callbacks=[checkpoint_callback_monitor])

        print('batch size is', params['optimization']['batch_size'])

        datamod = StaticDataModule(params, datakeys=params['dataset']['datakeys'], custom_transforms=[
            tt.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )] if params['dataset'].get('normalize', False) else None)
        datamod.setup()
        trainer.fit(nt, datamodule=datamod)
        wandb.finish()

