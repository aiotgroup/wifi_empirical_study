import os
import numpy as np
import scipy.io as scio
from scipy.signal import stft

from data_process.util import load_mat, save_mat


def data_format_transform(datasource_path: os.path, use_pha: bool):
    train_data_split_amp = scio.loadmat(os.path.join(datasource_path, 'train_data_split_amp.mat'))
    train_data_split_pha = scio.loadmat(os.path.join(datasource_path, 'train_data_split_pha.mat'))
    test_data_split_amp = scio.loadmat(os.path.join(datasource_path, 'test_data_split_amp.mat'))
    test_data_split_pha = scio.loadmat(os.path.join(datasource_path, 'test_data_split_pha.mat'))

    if use_pha:
        train_data = np.concatenate((train_data_split_amp['train_data'], train_data_split_pha['train_data']), axis=1)
        test_data = np.concatenate((test_data_split_amp['test_data'], test_data_split_pha['test_data']), axis=1)
    else:
        train_data = train_data_split_amp['train_data']
        test_data = test_data_split_amp['test_data']

    train_mat = {
        'data': train_data,
        'activity_label': train_data_split_amp['train_activity_label'],
        'location_label': train_data_split_amp['train_location_label'],
    }
    test_mat = {
        'data': test_data,
        'activity_label': test_data_split_amp['test_activity_label'],
        'location_label': test_data_split_amp['test_location_label'],
    }
    # scio.savemat(os.path.join(datasource_path, 'train_dataset_%s.mat' % ('pha' if use_pha else 'none')), train_mat)
    # scio.savemat(os.path.join(datasource_path, 'test_dataset_%s.mat' % ('pha' if use_pha else 'none')), test_mat)
    save_mat(os.path.join(datasource_path, 'train_dataset_%s.h5' % ('pha' if use_pha else 'none')), train_mat)
    save_mat(os.path.join(datasource_path, 'test_dataset_%s.h5' % ('pha' if use_pha else 'none')), test_mat)


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


def normalize_train_test(datasource_path: os.path, use_pha: bool):
    # train_data = scio.loadmat(os.path.join(datasource_path, 'train_dataset_%s.mat' % ('pha' if use_pha else 'none')))
    # test_data = scio.loadmat(os.path.join(datasource_path, 'test_dataset_%s.mat' % ('pha' if use_pha else 'none')))
    train_data = load_mat(os.path.join(datasource_path, 'train_dataset_%s.h5' % ('pha' if use_pha else 'none')))
    test_data = load_mat(os.path.join(datasource_path, 'test_dataset_%s.h5' % ('pha' if use_pha else 'none')))

    _normalize(train_data, test_data)

    # scio.savemat(os.path.join(datasource_path, 'train_dataset_%s.mat' % ('pha' if use_pha else 'none')), train_data)
    # scio.savemat(os.path.join(datasource_path, 'test_dataset_%s.mat' % ('pha' if use_pha else 'none')), test_data)
    save_mat(os.path.join(datasource_path, 'train_dataset_%s.h5' % ('pha' if use_pha else 'none')), train_data)
    save_mat(os.path.join(datasource_path, 'test_dataset_%s.h5' % ('pha' if use_pha else 'none')), test_data)


def stft_train_test(datasource_path: os.path, use_pha: bool, segment_length_rate: float, overlap_length_rate: float):
    # 采样率固定，那么测试采样率左右的 segment 长度 和 hop size，一个是在整段看到更多东西，一个是精细的转换数据
    sample_rate = 60
    nperseg = int(sample_rate * segment_length_rate)
    noverlap = int(nperseg * overlap_length_rate)
    # train_data = scio.loadmat(os.path.join(datasource_path, 'train_dataset_%s.mat' % ('pha' if use_pha else 'none')))
    # test_data = scio.loadmat(os.path.join(datasource_path, 'test_dataset_%s.mat' % ('pha' if use_pha else 'none')))
    train_data = load_mat(os.path.join(datasource_path, 'train_dataset_%s.h5' % ('pha' if use_pha else 'none')))
    test_data = load_mat(os.path.join(datasource_path, 'test_dataset_%s.h5' % ('pha' if use_pha else 'none')))

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
    # scio.savemat(os.path.join(datasource_path, 'train_dataset_%s_%.2f_%.2f.mat' % ('pha' if use_pha else 'none',
    #                                                                                segment_length_rate,
    #                                                                                overlap_length_rate)), train_data)
    # scio.savemat(os.path.join(datasource_path, 'test_dataset_%s_%.2f_%.2f.mat' % ('pha' if use_pha else 'none',
    #                                                                               segment_length_rate,
    #                                                                               overlap_length_rate)), test_data)
    save_mat(os.path.join(datasource_path, 'train_dataset_%s_%.2f_%.2f.h5' % ('pha' if use_pha else 'none',
                                                                               segment_length_rate,
                                                                               overlap_length_rate)), train_data)
    save_mat(os.path.join(datasource_path, 'test_dataset_%s_%.2f_%.2f.h5' % ('pha' if use_pha else 'none',
                                                                              segment_length_rate,
                                                                              overlap_length_rate)), test_data)


if __name__ == '__main__':
    datasource_path = os.path.join("")
    use_pha = False
    do_normalize = True
    do_stft = True

    segment_length_rates = [0.5, 1, 2]
    overlap_length_rates = [0, 0.25, 0.5, 0.75]

    """
        0. 数据格式变换
    """
    data_format_transform(datasource_path, use_pha)
    """
        1. 数据归一化：计算出训练集的均值和标准差，以此归一化训练集和测试集
    """
    if do_normalize:
        normalize_train_test(datasource_path, use_pha)
    """
        2. 频域变换： 在原始mat文件中添加上经过STFT转换后的频域数据
    """
    if do_stft:
        for segment_length_rate in segment_length_rates:
            for overlap_length_rate in overlap_length_rates:
                stft_train_test(datasource_path, use_pha, segment_length_rate, overlap_length_rate)
