# modified from:
# Author: Yuan Gong
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch

import csv
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import pandas as pd
from scipy.signal import butter, filtfilt
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import math
import torch.distributed as dist
import glob
from pathlib import Path
from catalyst.data.sampler import DistributedSamplerWrapper
from torch_audiomentations import Compose, AddBackgroundNoise


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    images = images['labels'].to_numpy()
    count[0] = len(np.where(images=='nonbs')[0])
    count[1] = len(np.where(images=='bs')[0])
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = np.array([0] * len(images))
    weight[np.where(images=='nonbs')[0]] = weight_per_class[0]
    weight[np.where(images=='bs')[0]] = weight_per_class[1]
    return weight


def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup


def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup


def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list


def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:]-coeff*signal[:-1])


def find_shift(window_size, duration_lbl, overlap, note, sr):
    if note == 'long event':
        left_shift = - int(window_size/2)
        right_shift = int(window_size/2)
    elif note == 'start':
        left_shift = - int((1-overlap) * window_size)
        right_shift = int(overlap * window_size)
    elif note == 'end':
        left_shift = - int(overlap * window_size)
        right_shift = int((1-overlap) * window_size)
    elif note == 'short event':
        left_shift = - int(duration_lbl/4)
        right_shift = int(duration_lbl/4)
    else:
        left_shift = - int(0.01 * sr)
        right_shift = int(0.01 * sr)
    return left_shift, right_shift


class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, audio_data, sr, audio_conf, label_csv=None, path='', eval=False):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """

        if type(dataset_json_file) == str:
            self.datapath = dataset_json_file
            self.data = pd.read_json(dataset_json_file, 'records')  # TODO test
        else:
            self.data = dataset_json_file

        self.audio_data = audio_data
        self.sr = sr

        self.audio_conf = audio_conf
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('now use noise augmentation')

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print('number of classes is {:d}'.format(self.label_num))
        self.shift = self.audio_conf.get('shift')
        if self.shift:
            print('now use time shift augmentation')
        self.path = path
        self.pitchshift = self.audio_conf.get('pitch shift') if self.audio_conf.get('pitch shift') is not None else False
        if self.pitchshift:
            print('now use pitch shift augmentation')
        self.background_noise = self.audio_conf.get('background noise') if self.audio_conf.get('background noise') is not None else False

        if self.background_noise:
            self.bg_noise = Compose([AddBackgroundNoise(background_paths=self.audio_conf.get('path noises'), mode='per_example', p=0.5, sample_rate=sr)])

    def _wav2fbank(self, filename, start, end, labels, filename2=None, duration_lbl=None, overlap=None, note=None):
        start_memory = start
        end_lookahead = end

        waveform = self.audio_data[Path(filename)][:, start:end]

        if (self.shift or self.pitchshift or self.background_noise):
            wf = waveform
            if self.shift:
                left_shift, right_shift = find_shift(end-start, duration_lbl, overlap, note, self.sr)
                shift_size = np.random.randint(left_shift, right_shift)
                if shift_size > 0:
                    extra_signal = self.audio_data[Path(filename)][:, min(start_memory, start) - shift_size: min(start_memory, start)]
                    if extra_signal.shape[1] < shift_size:
                        extra_signal = torch.cat([torch.zeros((1, shift_size - extra_signal.shape[1])), extra_signal], dim=1)
                    wf = torch.cat([extra_signal, wf[:, :-shift_size]], dim=1)
                elif shift_size < 0:
                    extra_signal = self.audio_data[Path(filename)][:, max(end, end_lookahead): max(end, end_lookahead) + shift_size]
                    if extra_signal.shape[1] < abs(shift_size):
                        extra_signal = torch.cat([extra_signal, torch.zeros((1, abs(shift_size) - extra_signal.shape[1]))], dim=1)
                    wf = torch.cat([wf[:, abs(shift_size):], extra_signal], dim=1)
            if self.pitchshift:
                shift = torchaudio.transforms.PitchShift(sample_rate=self.sr, n_steps= np.random.randint(-4, 4),
                                                         n_fft=512)
                wf = shift(wf)
            if self.background_noise:
                wf = self.bg_noise(wf[None, :, :])
                wf = wf[0, :, :]
            waveform = wf
            assert len(waveform) > 0, f"Something went wrong. {filename}, {start}, {end}"

        mfcc = torchaudio.transforms.MFCC(sample_rate=self.sr, n_mfcc=self.audio_conf.get('num_mel_bins'),
                                          melkwargs=dict(n_fft=int(self.sr * 0.1), hop_length=int(self.sr * 0.1),
                                                         center=False))
        fbank = mfcc(waveform)

        return fbank

    def __getitem__(self, index):
        # do mix-up for this sample (controlled by the given mixup rate)
        label_indices = np.zeros(self.label_num)
        if 'duration_lbl' in self.data.columns:
            duration_lbl, overlap, note = self.data.loc[index, 'duration_lbl'], self.data.loc[index, 'overlap'], self.data.loc[index, 'note']
        else:
            duration_lbl, overlap, note = None, None, None
        fbank = self._wav2fbank(filename=self.path + self.data.loc[index, 'wav'],
                                                start=self.data.loc[index, 'start'],
                                                end=self.data.loc[index, 'end'], labels=self.data.loc[index, 'labels'],
                                            duration_lbl=duration_lbl, overlap=overlap, note=note)
        for label_str in self.data.loc[index, 'labels'].split(','):
            label_indices[int(self.index_dict[label_str])] = 1.0

        label_indices = torch.FloatTensor(label_indices)

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        meta_data = torch.FloatTensor([int(self.data.loc[index, 'wav'][9:12]), int(self.data.loc[index, 'wav'][13:16]),
                                       int(self.data.loc[index, 'wav'][-5:-4]), self.data.loc[index, 'start'], self.data.loc[index, 'end'], -1, -1, -1])

        return fbank, label_indices, meta_data

    def __len__(self):
         return len(self.data)


class data_prep(pl.LightningDataModule):
    def __init__(self, device_num, batch_size, num_workers, fold, train_audio_conf, val_audio_conf, label_csv=None, path='', bal=True):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        data_train = pd.read_json(f'bs/data/{fold}/data_train.json')
        data_val = pd.read_json(f'bs/data/{fold}/data_val.json')
        id_train = np.load(f'bs/data/{fold}/id_train.npy')
        self.targets = np.zeros((data_train.shape[0], 2))
        self.targets[data_train['labels']=='nonbs', 0] = 1
        self.targets[data_train['labels']=='bs', 1] = 1
        self.targets = torch.LongTensor(self.targets)
        self.targets = torch.argmax(self.targets, dim=1)
        self.bal = bal

        audio_data_train = {}
        audio_data_val = {}
        sr = 16000

        for filename in glob.glob(f'{path}/data/*/*'):
            index = filename.find('data')
            part = filename[index + len('data/'): index + len('data/') + 3]

            wfile, _ = torchaudio.load(filename)

            wfile = torchaudio.functional.highpass_biquad(wfile, sr, 60)

            if int(part) in id_train:
                audio_data_train[Path(filename)] = wfile
            else:
                audio_data_val[Path(filename)] = wfile

            del wfile

        self.train_dataset = AudiosetDataset(dataset_json_file=data_train, audio_data=audio_data_train, sr=sr, audio_conf=train_audio_conf, label_csv=label_csv, path=path, eval=False)  # this is an audioset dataset
        self.eval_dataset = AudiosetDataset(dataset_json_file=data_val, audio_data=audio_data_val, sr=sr, audio_conf=val_audio_conf, label_csv=label_csv, path=path, eval=True)
        self.device_num = device_num

        print('Now train with {:d} training samples, evaluate with {:d} samples'.format(len(self.train_dataset),
                                                                                                  len(self.eval_dataset)))

    def calculate_weights(self):
        class_sample_count = torch.tensor(
            [(self.targets == t).sum() for t in torch.unique(self.targets, sorted=True)])
        weight = 1. / class_sample_count.double()
        return weight

    def train_dataloader(self):
        if not self.bal:
            train_sampler = DistributedSampler(self.train_dataset, shuffle=True) if self.device_num > 1 else None
        else:
            # train_sampler = DistributedWeightedSampler(self.train_dataset, self.targets) if self.device_num > 1 else None
            class_sample_count = torch.tensor(
                [(self.targets == t).sum() for t in torch.unique(self.targets, sorted=True)])
            weight = 1. / class_sample_count.double()
            samples_weight = torch.tensor([weight[t] for t in self.targets])
            # train_sampler = DistributedSamplerWrapper(WeightedRandomSampler(samples_weight, len(samples_weight))) if self.device_num > 1 else WeightedRandomSampler(samples_weight, len(samples_weight))
            train_sampler = DistributedWeightedSampler(self.train_dataset, self.targets) if self.device_num > 1 else WeightedRandomSampler(
                samples_weight, len(samples_weight))
        if train_sampler is not None:
            shuffle = False
        else:
            shuffle = True
        train_loader = DataLoader(
            dataset=self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size // self.device_num,
            shuffle=shuffle,
            sampler=train_sampler,
            pin_memory=True
        )
        return train_loader

    def val_dataloader(self):
        eval_sampler = DistributedSampler(self.eval_dataset, shuffle=False) if self.device_num > 1 else None
        eval_loader = DataLoader(
            dataset=self.eval_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size // self.device_num,
            shuffle=False,
            sampler=eval_sampler,
            pin_memory=False
        )
        return eval_loader

    def test_dataloader(self):
        test_sampler = DistributedSampler(self.eval_dataset, shuffle=False) if self.device_num > 1 else None
        test_loader = DataLoader(
            dataset=self.eval_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size // self.device_num,
            shuffle=False,
            sampler=test_sampler,
            pin_memory=False
        )
        return test_loader


class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset, targets, num_replicas=None, rank=None, replacement=True, shuffle=True, weights=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.targets = targets
        self.shuffle = shuffle
        self.weights = weights

    def calculate_weights(self, targets):
        assert len(targets) == self.num_samples
        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return samples_weight

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # get targets (you can alternatively pass them in __init__, if this op is expensive)
        # select only the wanted targets for this subsample
        targets = self.targets[indices]
        assert len(targets) == self.num_samples
        # randomly sample this subset, producing balanced classes
        weights = self.calculate_weights(targets)
        subsample_balanced_indicies = torch.multinomial(weights, self.num_samples, self.replacement)
        # now map these target indicies back to the original dataset index...
        dataset_indices = torch.tensor(indices)[subsample_balanced_indicies]

        return iter(dataset_indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
