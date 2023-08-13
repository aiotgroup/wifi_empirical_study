import os
import logging
import torch
import numpy as np
from torch.utils.data.dataset import Dataset

from .dataset_config import DatasetConfig
from .util import load_mat

logger = logging.getLogger(__name__)


class HTHIDatasetConfig(DatasetConfig):
    def __init__(self, datasource_path: os.path, index: int, use_pha: bool, segment_length_rate: float,
                 overlap_length_rate: float):
        super(HTHIDatasetConfig, self).__init__('hthi', datasource_path)
        self.sample_rate = 210
        self.index = index
        self.use_pha = use_pha
        self.segment_length_rate = segment_length_rate
        self.overlap_length_rate = overlap_length_rate


def load_hthi_data(config: HTHIDatasetConfig):
    train_mat = load_mat(os.path.join(config.datasource_path,
                                      'train_dataset_%d_%s_%.2f_%.2f.h5' % (
                                          config.index, 'pha' if config.use_pha else 'none',
                                          config.segment_length_rate,
                                          config.overlap_length_rate)))
    test_mat = load_mat(os.path.join(config.datasource_path,
                                     'test_dataset_%d_%s_%.2f_%.2f.h5' % (
                                         config.index, 'pha' if config.use_pha else 'none',
                                         config.segment_length_rate,
                                         config.overlap_length_rate)))

    train_data = {
        'data': train_mat['data'],
        'freq_data': train_mat['freq_data'],
        'label': train_mat['label'],
    }
    test_data = {
        'data': test_mat['data'],
        'freq_data': test_mat['freq_data'],
        'label': test_mat['label'],
    }
    return train_data, test_data


class HTHIDataset(Dataset):
    def __init__(self, mat_data):
        super(HTHIDataset, self).__init__()

        logger.info('加载HTHI数据集')

        self.data = torch.from_numpy(mat_data['data']).float()
        self.freq_data = torch.from_numpy(mat_data['freq_data']).float()
        self.label = torch.from_numpy(mat_data['label']).squeeze().long()

        self.label_n_class = len(np.bincount(np.squeeze(mat_data['label'])))

        self.num_sample, self.n_channel, self.seq_len = self.data.size()
        _, self.freq_n_channel, self.freq_seq_len = self.freq_data.size()

        assert self.num_sample == self.label.size(0)
        assert self.num_sample == self.freq_data.size(0)

    def __getitem__(self, index):
        return {
            'data': self.data[index],
            'freq_data': self.freq_data[index],
            'label': self.label[index],
        }

    def __len__(self):
        return self.num_sample

    def get_n_channels(self):
        return {
            'data': self.n_channel,
            'freq_data': self.freq_n_channel,
        }

    def get_seq_lens(self):
        return {
            'data': self.seq_len,
            'freq_data': self.freq_seq_len,
        }

    def get_n_classes(self):
        return {
            'label': self.label_n_class,
        }


if __name__ == '__main__':
    datasource_path = os.path.join("")
    train_data, test_data = load_hthi_data(HTHIDatasetConfig(datasource_path, 0, False, 2, 0.75))
    train_dataset = HTHIDataset(train_data)
    test_dataset = HTHIDataset(test_data)
    print(len(train_dataset))
    print(len(test_dataset))
    print(train_dataset.get_n_classes(), test_dataset.get_n_classes())
    print(train_dataset.get_n_channels(), test_dataset.get_n_channels())
    print(train_dataset.get_seq_lens(), test_dataset.get_seq_lens())
