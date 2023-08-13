import os
import logging
import torch
import numpy as np
from torch.utils.data.dataset import Dataset

from .dataset_config import DatasetConfig
from .util import load_mat

logger = logging.getLogger(__name__)


class ARILDatasetConfig(DatasetConfig):
    def __init__(self, datasource_path: os.path, use_pha: bool, segment_length_rate: float, overlap_length_rate: float):
        super(ARILDatasetConfig, self).__init__('aril', datasource_path)
        self.sample_rate = 60
        self.use_pha = use_pha
        self.segment_length_rate = segment_length_rate
        self.overlap_length_rate = overlap_length_rate


def load_aril_data(config: ARILDatasetConfig):
    train_mat = load_mat(os.path.join(config.datasource_path,
                                      'train_dataset_%s_%.2f_%.2f.h5' % ('pha' if config.use_pha else 'none',
                                                                         config.segment_length_rate,
                                                                         config.overlap_length_rate)))
    test_mat = load_mat(os.path.join(config.datasource_path,
                                     'test_dataset_%s_%.2f_%.2f.h5' % ('pha' if config.use_pha else 'none',
                                                                       config.segment_length_rate,
                                                                       config.overlap_length_rate)))

    train_data = {
        'data': train_mat['data'],
        'freq_data': train_mat['freq_data'],
        'activity_label': train_mat['activity_label'],
        'location_label': train_mat['location_label'],
    }
    test_data = {
        'data': test_mat['data'],
        'freq_data': test_mat['freq_data'],
        'activity_label': test_mat['activity_label'],
        'location_label': test_mat['location_label'],
    }
    return train_data, test_data


class ARILDataset(Dataset):
    def __init__(self, mat_data):
        super(ARILDataset, self).__init__()

        logger.info('加载ARIL数据集')

        self.data = torch.from_numpy(mat_data['data']).float()
        self.freq_data = torch.from_numpy(mat_data['freq_data']).float()
        self.activity_label = torch.from_numpy(mat_data['activity_label']).squeeze().long()
        self.location_label = torch.from_numpy(mat_data['location_label']).squeeze().long()

        self.activity_label_n_class = len(np.bincount(np.squeeze(mat_data['activity_label'])))
        self.location_label_n_class = len(np.bincount(np.squeeze(mat_data['location_label'])))

        self.num_sample, self.n_channel, self.seq_len = self.data.size()
        _, self.freq_n_channel, self.freq_seq_len = self.freq_data.size()

        assert self.num_sample == self.freq_data.size(0)
        assert self.num_sample == self.activity_label.size(0)
        assert self.num_sample == self.location_label.size(0)

    def __getitem__(self, index):
        return {
            'data': self.data[index],
            'freq_data': self.freq_data[index],
            'activity': self.activity_label[index],
            # 'location': self.location_label[index],
        }

    def __len__(self):
        return self.num_sample

    def get_n_classes(self):
        return {
            'activity': self.activity_label_n_class,
            # 'location': self.location_label_n_class,
        }

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


if __name__ == '__main__':
    datasource_path = os.path.join("")
    train_data, test_data = load_aril_data(ARILDatasetConfig(datasource_path, False, 2, 0.75))
    train_dataset, test_dataset = ARILDataset(train_data), ARILDataset(test_data)
    print(len(train_dataset))
    print(len(test_dataset))
    print(train_dataset.get_n_classes(), test_dataset.get_n_classes())
    print(train_dataset.get_n_channels(), test_dataset.get_n_channels())
    print(train_dataset.get_seq_lens(), test_dataset.get_seq_lens())
