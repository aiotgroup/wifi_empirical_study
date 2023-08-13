import os
import time
import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F
from scipy.signal import stft
from sklearn.model_selection import train_test_split

from data_process.util import load_mat, save_mat


def split_train_test(datasource_path: os.path, index: int, test_ratio=0.1):
    datasource = scio.loadmat(os.path.join(datasource_path, 'csi_amp_all.mat'))
    datasource['csi_amp'] = np.transpose(datasource['csi_amp'], (0, 2, 1))

    n_classes = len(np.bincount(np.squeeze(datasource['label'])))
    n_samples = datasource['csi_amp'].shape[0]
    assert n_samples == datasource['label'].shape[1]

    data = [[] for _ in range(n_classes)]
    label = [[] for _ in range(n_classes)]

    for i in range(n_samples):
        data[datasource['label'][0][i]].append(datasource['csi_amp'][i])
        label[datasource['label'][0][i]].append(datasource['label'][0][i])

    train_data = [[] for _ in range(n_classes)]
    train_label = [[] for _ in range(n_classes)]

    test_data = [[] for _ in range(n_classes)]
    test_label = [[] for _ in range(n_classes)]

    for i in range(n_classes):
        train_data[i], test_data[i], train_label[i], test_label[i] = train_test_split(data[i], label[i],
                                                                                      test_size=test_ratio,
                                                                                      shuffle=True)

    train_mat = {
        'data': [],
        'label': [],
    }
    test_mat = {
        'data': [],
        'label': [],
    }

    for i in range(n_classes):
        train_mat['data'].extend(train_data[i])
        train_mat['label'].extend(train_label[i])
        test_mat['data'].extend(test_data[i])
        test_mat['label'].extend(test_label[i])

    train_mat['data'] = np.array(train_mat['data'])
    train_mat['label'] = np.array(train_mat['label'])
    test_mat['data'] = np.array(test_mat['data'])
    test_mat['label'] = np.array(test_mat['label'])

    # scio.savemat(os.path.join(datasource_path, 'train_dataset_%d.mat' % index), train_mat)
    # scio.savemat(os.path.join(datasource_path, 'test_dataset_%d.mat' % index), test_mat)
    save_mat(os.path.join(datasource_path, 'train_dataset_%d.h5' % index), train_mat)
    save_mat(os.path.join(datasource_path, 'test_dataset_%d.h5' % index), test_mat)


def _normalize(train_data, test_data):
    def get_mean_std(data):
        channel = data.shape[1]
        mean = np.mean(data.transpose(1, 0, 2).reshape(channel, -1), axis=-1)
        mean = np.expand_dims(mean, 0)
        mean = np.expand_dims(mean, 2)
        std = np.std(data.transpose(1, 0, 2).reshape(channel, -1), axis=-1)
        std = np.expand_dims(std, 0)
        std = np.expand_dims(std, 2)
        return mean, std

    def normalize(data, mean, std):
        return (data - mean) / std

    amp_mean, amp_std = get_mean_std(train_data['data'])
    train_data['data'] = normalize(train_data['data'], amp_mean, amp_std)
    test_data['data'] = normalize(test_data['data'], amp_mean, amp_std)


def normalize_train_test(datasource_path: os.path, index: int):
    # train_data = scio.loadmat(os.path.join(datasource_path, 'train_dataset_%d.mat' % index))
    # test_data = scio.loadmat(os.path.join(datasource_path, 'test_dataset_%d.mat' % index))
    train_data = load_mat(os.path.join(datasource_path, 'train_dataset_%d.h5' % index))
    test_data = load_mat(os.path.join(datasource_path, 'test_dataset_%d.h5' % index))

    _normalize(train_data, test_data)

    # scio.savemat(os.path.join(datasource_path, 'train_dataset_%d.mat' % index), train_data)
    # scio.savemat(os.path.join(datasource_path, 'test_dataset_%d.mat' % index), test_data)
    save_mat(os.path.join(datasource_path, 'train_dataset_%d.h5' % index), train_data)
    save_mat(os.path.join(datasource_path, 'test_dataset_%d.h5' % index), test_data)


def downsample_train_test(datasource_path: os.path, index: int, downsample_factor: int = 10):
    # train_data = scio.loadmat(os.path.join(datasource_path, 'train_dataset_%d.mat' % index))
    # test_data = scio.loadmat(os.path.join(datasource_path, 'test_dataset_%d.mat' % index))
    train_data = load_mat(os.path.join(datasource_path, 'train_dataset_%d.h5' % index))
    test_data = load_mat(os.path.join(datasource_path, 'test_dataset_%d.h5' % index))

    seq_len = train_data['data'].shape[2]
    assert seq_len == test_data['data'].shape[2]

    train_data['data'] = torch.Tensor(train_data['data'])
    train_data['data'] = F.interpolate(train_data['data'], seq_len // downsample_factor, mode='linear',
                                       align_corners=True)
    train_data['data'] = train_data['data'].numpy()

    test_data['data'] = torch.Tensor(test_data['data'])
    test_data['data'] = F.interpolate(test_data['data'], seq_len // downsample_factor, mode='linear',
                                      align_corners=True)
    test_data['data'] = test_data['data'].numpy()

    # scio.savemat(os.path.join(datasource_path, 'train_dataset_%d.mat' % index), train_data)
    # scio.savemat(os.path.join(datasource_path, 'test_dataset_%d.mat' % index), test_data)
    save_mat(os.path.join(datasource_path, 'train_dataset_%d.h5' % index), train_data)
    save_mat(os.path.join(datasource_path, 'test_dataset_%d.h5' % index), test_data)


def stft_train_test(datasource_path: os.path, index: int, segment_length_rate: float, overlap_length_rate: float):
    # channel: original_channel * (nperseg + 1)
    # time: ceil(original_time / (nperseg - noverlap)) + 1
    sample_rate = 100
    nperseg = int(sample_rate * segment_length_rate)
    noverlap = int(nperseg * overlap_length_rate)
    # train_data = scio.loadmat(os.path.join(datasource_path, 'train_dataset_%d.mat' % index))
    # test_data = scio.loadmat(os.path.join(datasource_path, 'test_dataset_%d.mat' % index))
    train_data = load_mat(os.path.join(datasource_path, 'train_dataset_%d.h5' % index))
    test_data = load_mat(os.path.join(datasource_path, 'test_dataset_%d.h5' % index))

    train_data['freq_data'] = stft(train_data['data'], fs=sample_rate, nperseg=nperseg, noverlap=noverlap)[2]
    train_data['freq_data'] = np.abs(train_data['freq_data'])
    train_data['freq_data'] = train_data['freq_data'].reshape(train_data['freq_data'].shape[0],
                                                              train_data['freq_data'].shape[1] *
                                                              train_data['freq_data'].shape[2],
                                                              train_data['freq_data'].shape[3])
    print(train_data['freq_data'].shape)
    test_data['freq_data'] = stft(test_data['data'], fs=sample_rate, nperseg=nperseg, noverlap=noverlap)[2]
    test_data['freq_data'] = np.abs(test_data['freq_data'])
    test_data['freq_data'] = test_data['freq_data'].reshape(test_data['freq_data'].shape[0],
                                                            test_data['freq_data'].shape[1] *
                                                            test_data['freq_data'].shape[2],
                                                            test_data['freq_data'].shape[3])
    # scio.savemat(os.path.join(datasource_path, 'train_dataset_%d_%.2f_%.2f.mat' % (index, segment_length_rate,
    #                                                                                overlap_length_rate)), train_data)
    # scio.savemat(os.path.join(datasource_path, 'test_dataset_%d_%.2f_%.2f.mat' % (index, segment_length_rate,
    #                                                                               overlap_length_rate)), test_data)
    save_mat(os.path.join(datasource_path, 'train_dataset_%d_%.2f_%.2f.h5' % (index, segment_length_rate,
                                                                               overlap_length_rate)), train_data)
    save_mat(os.path.join(datasource_path, 'test_dataset_%d_%.2f_%.2f.h5' % (index, segment_length_rate,
                                                                              overlap_length_rate)), test_data)


if __name__ == '__main__':
    datasource_path = os.path.join()
    random_times = 5
    do_split = False
    do_normalize = False
    do_downsample = False
    do_stft = True

    segment_length_rates = [0.5, 1, 2]
    overlap_length_rates = [0, 0.25, 0.5, 0.75]

    """
        1. 首先将整个数据集分为训练集和测试集
    """
    if do_split:
        for i in range(random_times):
            before_time = time.time()

            split_train_test(datasource_path, i)

            after_time = time.time()
            print("Train-Test Splitting Time Consuming: %d." % (after_time - before_time))
    """
        2. 数据归一化：计算出训练集的均值和标准差，以此归一化训练集和测试集
    """
    if do_normalize:
        for i in range(random_times):
            before_time = time.time()

            normalize_train_test(datasource_path, i)

            after_time = time.time()
            print("Normalization Time Consuming: %d." % (after_time - before_time))
    """
        3. 下采样：时间维度过长，通过下采样精简数据
    """
    if do_downsample:
        for i in range(random_times):
            before_time = time.time()

            downsample_train_test(datasource_path, i)

            after_time = time.time()
            print("Down Sampling Time Consuming: %d." % (after_time - before_time))

    """
        4. 频域变换： 在原始mat文件中添加上经过STFT转换后的频域数据
    """
    if do_stft:
        for i in range(random_times):
            for segment_length_rate in segment_length_rates:
                for overlap_length_rate in overlap_length_rates:
                    before_time = time.time()
                    stft_train_test(datasource_path, i, segment_length_rate, overlap_length_rate)
                    after_time = time.time()
                    print("STFT Time Consuming: %d." % (after_time - before_time))
