# adapted from
# Author: Yuan Gong

import os
import sys
import torch.utils.data
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
from src import dataloader
from src.models import create_audio_model
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint

torch.cuda.empty_cache()

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"  # only on Windows

pl.seed_everything(42, workers=True)

config = yaml.safe_load(open('params.yaml'))

if __name__ == '__main__':
    parser = ArgumentParser(description='Extract BS features.')
    parser.add_argument("--gpu", help="Specify how many gpus to use.")
    parser.add_argument("--nodes", help="Specify how many nodes to use.")
    parser.add_argument("--exp_dir", help='Specify where to store the results.')
    parser.add_argument("--fold", help='Specify fold of cross-validation.')
    parser.add_argument("--cpu", help='Specify how many cpu use to dataloader.')
    parser.add_argument("--job_name", help='SLURM job name.')
    parser.add_argument("--resume_ckpt", help='Resume training from checkpoint.')
    args = parser.parse_args()

    device_num = int(args.gpu)
    nodes = int(args.nodes)
    exp_dir = args.exp_dir
    fold = args.fold
    num_workers = int(args.cpu)
    job_name = args.job_name
    ckpt = args.resume_ckpt

    audioset_data = dataloader.data_prep(device_num=device_num, batch_size=config.get('batch_size'), num_workers=num_workers,
                                         fold=fold, train_audio_conf=config.get('train_audio_conf'), val_audio_conf=config.get('val_audio_conf'),
                                         label_csv=config.get('label_csv'), path=config.get('path'), bal=config.get('bal'))

    if config.get('loss') == 'FL':
        if not config.get('bal'):
            weight = audioset_data.calculate_weights()
            print('using weighted loss')
        else:
            weight = None
    else:
        weight = None

    print("\nCreating experiment directory: %s" % exp_dir)
    if os.path.exists("%s/checkpoint" % exp_dir) == False:
        os.makedirs("%s/checkpoint" % exp_dir, exist_ok=True)

    audio_model = create_audio_model(config, weight, exp_dir)

    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(dirpath=f'{exp_dir}/checkpoint/', filename="model-{epoch:02d}", save_on_train_epoch_end=True)
    if ckpt is not None:
        trainer = pl.Trainer(deterministic=True, accelerator='gpu', devices=device_num, max_epochs=config.get('n_epochs'),
                         check_val_every_n_epoch=1, enable_checkpointing=True, strategy='ddp', # DDPStrategy(find_unused_parameters=False,static_graph=True, gradient_as_bucket_view=True),
                         num_nodes=nodes, log_every_n_steps=1000, flush_logs_every_n_steps=1000,
                         logger=MLFlowLogger(), num_sanity_val_steps=0, replace_sampler_ddp=False, default_root_dir=f'{exp_dir}/checkpoint/',
                         precision=16, callbacks=[checkpoint_callback],
                         resume_from_checkpoint=ckpt
                         )
    else:
        trainer = pl.Trainer(deterministic=True, accelerator='gpu', devices=device_num,
                             max_epochs=config.get('n_epochs'),
                             check_val_every_n_epoch=1, enable_checkpointing=True, strategy='ddp',
                             num_nodes=nodes, log_every_n_steps=1000, flush_logs_every_n_steps=1000,
                             logger=MLFlowLogger(), num_sanity_val_steps=0, replace_sampler_ddp=False,
                             default_root_dir=f'{exp_dir}/checkpoint/',
                             precision=16, callbacks=[checkpoint_callback]
                             )

    trainer.logger.log_hyperparams({"fold": fold})
    for param in config.keys():
        if type(config.get(param)) != dict:
            trainer.logger.log_hyperparams({param: config.get(param)})
        else:
            if param == 'model':
                trainer.logger.log_hyperparams({'model': config.get('model')['name']})
            else:
                for subparam in config.get(param).keys():
                    trainer.logger.log_hyperparams({f'{param}_{subparam}': config.get(param)[subparam]})

    trainer.logger.log_hyperparams({'num_workers': num_workers})
    trainer.logger.log_hyperparams({'gpus': device_num})
    trainer.logger.log_hyperparams({'exp_dir': exp_dir})
    trainer.logger.log_hyperparams({'nodes': nodes})

    trainer.fit(audio_model, audioset_data)
