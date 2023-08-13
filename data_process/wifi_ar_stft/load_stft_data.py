import scipy.signal
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np


def plot_stft(t, f, Sxx, data):
    plt.figure(figsize=(20, 5))

    ax1 = plt.subplot(2, 1, 1)
    plt.plot(data)
    ax1.set_title('antenna {} Amplitude, label: {}')

    ax1 = plt.subplot(2, 1, 2)
    plt.pcolormesh(t, f, Sxx)
    plt.show()


def scipy_spec(data, nperseg=128, noverlap=75, nfft=128, cutoff_freq=120):
    '''

    :param data:
    :param nperseg: 每个段的长度，默认是None
    :param noverlap: 窗口重叠，默认是None
    :param nfft: 控制时频图频率的长度，默认是None
    :param cutoff_freq: 时频图的频率范围，最大值是采样率的一半，500
    :return:
    '''

    f, t, Sxx = scipy.signal.spectrogram(data, fs=1000,
                                         window=('tukey', 0.25),
                                         nperseg=nperseg,  # 每个段的长度
                                         noverlap=noverlap,  # 窗口重叠
                                         nfft=nfft,
                                         detrend='constant',
                                         return_onesided=True,
                                         scaling='spectrum',
                                         axis=- 1,
                                         mode='psd')

    a = 0
    for i in range(len(f)):
        if f[i] > cutoff_freq:
            a = i
            break

    return t, f[0:a], Sxx[0:a, :]

    # return t, f, Sxx


def test(csi_data, nperseg=128, noverlap=75, nfft=128, cutoff_freq=120):
    idx = 200
    print(csi_data.shape)

    t, f, Sxx = scipy_spec(csi_data[idx, :, 0], nperseg=nperseg, noverlap=noverlap,
                           nfft=nfft, cutoff_freq=cutoff_freq)

    plot_stft(t, f, Sxx, csi_data[idx, :, 0])
    print(Sxx.shape)


def stft_data(csi_data):
    stft_list = []
    for idx in range(csi_data.shape[0]):
        # for idx in range(20):

        temp_list = []
        for channel in range(90):
            _, _, Sxx = scipy_spec(csi_data[idx, :, channel], nperseg=128, noverlap=75,
                                   nfft=128, cutoff_freq=120)

            temp_list.append(Sxx.transpose())

        stft_map = np.concatenate(temp_list, axis=1)
        t, f = stft_map.shape
        stft_list.append(stft_map.reshape((1, t, f)))

        print(idx)

    stft_train = np.concatenate(stft_list, axis=0)
    print(stft_train.shape)

    scio.savemat('stft_all.mat', mdict={'stft_map': stft_train})


if __name__ == '__main__':
    data = scio.loadmat('csi_amp_all.mat')

    # -----测试-----
    #
    # test(data['csi_amp'], nperseg = 128, noverlap = 75, nfft = 128, cutoff_freq=120)
    # test(data['csi_amp'], nperseg=None, noverlap=None, nfft=None, cutoff_freq=150)
    # test(data['csi_amp'], nperseg=256, noverlap=175, nfft=256, cutoff_freq=120)

    stft_data(data['csi_amp'])

    # draw_csi(data['csi_amp'][idx,:,:],
    #          data['label'][0][idx],
    #          data['salient_label'][idx,:],
    #          data['segment_label'][idx,:])
