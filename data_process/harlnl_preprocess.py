import os
import math
import random
import time
import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.signal import stft

from data_process.util import load_mat, save_mat


def load_harlnl_data(datasource_path: os.path, label_index: int):
    # csidata(seq_len, 90)
    def deal_with_inf_data(csi_data):
        seq_len = csi_data.shape[2]
        median = np.median(csi_data, axis=-1, keepdims=True)
        median = median.repeat(seq_len, axis=-1)
        where = np.isinf(csi_data)
        csi_data[where] = median[where]
        return csi_data

    datasource_train = {
        'data': list(),
        'label': list(),
    }
    datasource_test = {
        'data': list(),
        'label': list(),
    }
    dir_name = str(label_index)
    file_list = os.listdir(os.path.join(datasource_path, 'TrainWhole', dir_name))
    for j in tqdm(range(len(file_list))):
        mat_file = scio.loadmat(os.path.join(datasource_path, 'TrainWhole', dir_name, file_list[j]))
        csi_data = np.expand_dims(mat_file['csidata'], 0)
        csi_data = csi_data.transpose((0, 2, 1))
        csi_data = deal_with_inf_data(csi_data)
        datasource_train['data'].append(csi_data)
        datasource_train['label'].append(label_index)
    file_list = os.listdir(os.path.join(datasource_path, 'TestWhole', dir_name))
    for j in tqdm(range(len(file_list))):
        mat_file = scio.loadmat(os.path.join(datasource_path, 'TestWhole', dir_name, file_list[j]))
        csi_data = np.expand_dims(mat_file['csidata'], 0)
        csi_data = csi_data.transpose((0, 2, 1))
        csi_data = deal_with_inf_data(csi_data)
        datasource_test['data'].append(csi_data)
        datasource_test['label'].append(label_index)
    return datasource_train, datasource_test


def split_train_test(datasource_path: os.path):
    train_mat = {
        'data': [],
        'label': [],
    }
    test_mat = {
        'data': [],
        'label': [],
    }

    for i in range(n_class):
        datasource_train, datasource_test = load_harlnl_data(datasource_path, i)

        n_sample = len(datasource_train['label'])
        assert n_sample == len(datasource_train['data'])
        train_mat['data'].extend(datasource_train['data'])
        train_mat['label'].extend(datasource_train['label'])

        n_sample = len(datasource_test['label'])
        assert n_sample == len(datasource_test['data'])
        test_mat['data'].extend(datasource_test['data'])
        test_mat['label'].extend(datasource_test['label'])
    return train_mat, test_mat


def _normalize(train_mat, test_mat):
    def get_mean_std(data):
        # data shape: list( example(1, channel, variable_seq_len) )
        mean = np.mean(data[0], axis=-1, keepdims=True)
        for example in data[1:]:
            mean = np.vstack((mean, np.mean(example, axis=-1, keepdims=True)))
        mean = np.mean(mean, axis=0, keepdims=True)
        std = np.mean((data[0] - mean) ** 2, axis=-1, keepdims=True)
        for example in data[1:]:
            std = np.vstack((std, np.mean((example - mean) ** 2, axis=-1, keepdims=True)))
        std = np.sqrt(np.mean(std, axis=0, keepdims=True))
        return mean, std

    def normalize(data, mean, std):
        for i in range(len(data)):
            data[i] = (data[i] - mean) / std

    # shape: 1, channel, 1
    data_mean, data_std = get_mean_std(train_mat['data'])
    normalize(train_mat['data'], data_mean, data_std)
    normalize(test_mat['data'], data_mean, data_std)


def normalize_train_test(train_mat, test_mat):
    _normalize(train_mat, test_mat)
    return train_mat, test_mat


def get_mean_seq_len(train_mat, test_mat):
    seq_lens = []
    for sample in train_mat['data']:
        seq_lens.append(sample.shape[2])
    for sample in test_mat['data']:
        seq_lens.append(sample.shape[2])
    return np.mean(seq_lens)


def stft_train_test(train_mat, test_mat, sample_rate, segment_length_rate: float, overlap_length_rate: float):
    def _stft_data(data, sample_rate, segment_length_rate: float, overlap_length_rate: float):
        # channel: original_channel * (nperseg // 2 + 1)
        # time: ceil(original_time / (nperseg - noverlap)) + 1
        # 采样率固定，那么测试采样率左右的 segment 长度 和 hop size，一个是在整段看到更多东西，一个是精细的转换数据
        # shape: 1, channel, seq_len
        nperseg = int(sample_rate * segment_length_rate)
        noverlap = int(nperseg * overlap_length_rate)
        data = stft(data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)[2]
        data = np.abs(data)
        data = data.reshape(data.shape[0], data.shape[1] * data.shape[2], data.shape[3])
        return data

    train_mat['freq_data'] = list()
    for sample in train_mat['data']:
        train_mat['freq_data'].append(_stft_data(np.expand_dims(sample, axis=0),
                                                 sample_rate, segment_length_rate, overlap_length_rate).squeeze(0))
    train_mat['freq_data'] = np.array(train_mat['freq_data'])

    test_mat['freq_data'] = list()
    for sample in test_mat['data']:
        test_mat['freq_data'].append(_stft_data(np.expand_dims(sample, axis=0),
                                                sample_rate, segment_length_rate, overlap_length_rate).squeeze(0))
    test_mat['freq_data'] = np.array(test_mat['freq_data'])

    print(train_mat['data'].shape, train_mat['label'].shape, train_mat['freq_data'].shape)
    print(test_mat['data'].shape, test_mat['label'].shape, test_mat['freq_data'].shape)
    return train_mat, test_mat


def unify_seq_len(train_mat, test_mat, target_seq_len: int):
    def _unify_seq_len(data, target_seq_len: int):
        # shape: 1, channel, seq_len
        data = torch.Tensor(data)
        data = F.interpolate(data, target_seq_len, mode='linear', align_corners=True)
        data = data.numpy()
        return data

    size = len(train_mat['data'])
    assert size == len(train_mat['label'])
    train_result = {
        'data': np.zeros((size, train_mat['data'][0].shape[1], target_seq_len), dtype=np.float32),
        'label': np.zeros((size, 1), dtype=np.uint8),
    }
    for i in range(size):
        train_result['data'][i, :, :] = _unify_seq_len(train_mat['data'][i], target_seq_len)[0, :, :]
        train_result['label'][i, :] = train_mat['label'][i]

    size = len(test_mat['data'])
    assert size == len(test_mat['label'])
    test_result = {
        'data': np.zeros((size, test_mat['data'][0].shape[1], target_seq_len), dtype=np.float32),
        'label': np.zeros((size, 1), dtype=np.uint8),
    }
    for i in range(size):
        test_result['data'][i, :, :] = _unify_seq_len(test_mat['data'][i], target_seq_len)[0, :, :]
        test_result['label'][i, :] = test_mat['label'][i]
    del train_mat, test_mat
    return train_result, test_result


if __name__ == '__main__':
    datasource_path = os.path.join('')
    random_times = 1

    n_class = 12
    target_seq_len = 640
    sample_rate = 320
    segment_length_rates = [0.5, 1, 2]
    overlap_length_rates = [0, 0.25, 0.5, 0.75]

    for i in range(random_times):
        """
            1. 首先将整个数据集分为训练集和测试集
        """
        before_time = time.time()
        train_mat, test_mat = split_train_test(datasource_path)
        after_time = time.time()
        print("Train-Test Splitting Time Consuming: %d." % (after_time - before_time))
        """
            2. 数据归一化：计算出训练集的均值和标准差，以此归一化训练集和测试集
        """
        before_time = time.time()
        train_mat, test_mat = normalize_train_test(train_mat, test_mat)
        after_time = time.time()
        print("Normalization Time Consuming: %d." % (after_time - before_time))

        """
            获取平均长度
        """
        mean_seq_len = get_mean_seq_len(train_mat, test_mat)
        mean_seq_len = int(mean_seq_len)
        print("Mean Sequence length = %d." % mean_seq_len)

        """
            3. 时间维度下采样，统一时间长度
        """
        before_time = time.time()
        train_mat, test_mat = unify_seq_len(train_mat, test_mat, target_seq_len)
        after_time = time.time()
        print("数据下采样 Time Consuming: %d." % (after_time - before_time))

        new_sample_rate = int((target_seq_len / mean_seq_len) * sample_rate)
        print("After unify sequence length, sample rate = %d." % new_sample_rate)

        for segment_length_rate in segment_length_rates:
            for overlap_length_rate in overlap_length_rates:
                """
                    4. 频域变换： 在原始mat文件中添加上经过STFT转换后的频域数据
                """
                before_time = time.time()
                train_result, test_result = stft_train_test(train_mat, test_mat,
                                                            new_sample_rate, segment_length_rate, overlap_length_rate)
                after_time = time.time()
                print("STFT频域提取 Time Consuming: %d." % (after_time - before_time))

                """
                    5. 数据保存
                """
                save_mat(os.path.join(datasource_path,
                                      'train_dataset_%d_%.2f_%.2f.h5' % (i, segment_length_rate,
                                                                         overlap_length_rate)),
                         train_result)
                save_mat(os.path.join(datasource_path,
                                      'test_dataset_%d_%.2f_%.2f.h5' % (i, segment_length_rate,
                                                                        overlap_length_rate)),
                         test_result)

                del train_result, test_result
