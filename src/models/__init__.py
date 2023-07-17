from .cnn_models import *


def create_audio_model(config, weight, exp_dir):
    if config.get('model')['name'] == 'CNN':
        audio_model = CNN(label_dim=config.get('n_class'),
                                scheduler=config.get('scheduler'), loss=config.get('loss'),
                                exp_dir=exp_dir,
                                weight=weight, warmup=config.get('warmup'))
    else:
        raise ValueError('Missing configuration for model {}'.format(config.get('model')['name']))

    return audio_model
