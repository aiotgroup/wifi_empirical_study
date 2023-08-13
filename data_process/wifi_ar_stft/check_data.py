import glob
import re

import pandas as pd
import numpy as np
import scipy.io as scio

import matplotlib.pyplot as plt

def draw_csi(data, label, salient_label, segment_label):

    plt.figure(figsize=(20, 25))

    for i in range(3):
        ax1 = plt.subplot(5, 1, i + 1)
        plt.plot(data[:, i * 30: i * 30 + 30])
        ax1.set_title('antenna {} Amplitude, label: {}'.format(i + 1, label))

    ax1 = plt.subplot(5, 1, 4)
    plt.plot(salient_label)
    ax1.set_title('salient_label')

    ax1 = plt.subplot(5, 1, 5)
    plt.plot(segment_label)
    ax1.set_title('segment_label')

    plt.show()


def plot_stft(data,csi_data, i, idx):
    f_length = int(data.shape[1] / 90)
    t_length = data.shape[0]

    plt.figure(figsize=(20, 5))
    t = np.arange(0, t_length * 10, 10)
    f = np.arange(0, 120, 120 / f_length)
    plt.subplot(2, 1, 1)
    plt.pcolormesh(t,
                   f,
                   data[:, i*f_length: i*f_length + f_length].transpose()
                   )
    plt.title('stft: idx: {}'.format(idx))

    plt.subplot(2, 1, 2)
    plt.plot(csi_data[:, i])
    plt.xlim(0, csi_data.shape[0])
    plt.show()

if __name__ == '__main__':

    idx = 550

    data = scio.loadmat('csi_amp_all.mat')
    # (557, 15800, 90)
    print(data['csi_amp'].shape,
          data['label'].shape,
          data['salient_label'].shape,
          data['segment_label'].shape)

    # draw_csi(data['csi_amp'][idx,:,:],
    #          data['label'][0][idx],
    #          data['salient_label'][idx,:],
    #          data['segment_label'][idx,:])

    stft_data = scio.loadmat('stft_all.mat')

    plot_stft(stft_data['stft_map'][idx, :, :], data['csi_amp'][idx,:,:],50,idx)
