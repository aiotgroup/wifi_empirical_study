import os

from utils import *

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    cuda = 3

    config = DatasetDefaultConfig()

    model_list = [
        ('resnet1d_101', 'resnet1d_span_cls_raw_time', 128),
        ('resnet1d_101', 'resnet1d_span_cls_raw_channel', 128),
        ('resnet1d_101', 'resnet1d_span_cls_freq_time', 128),
        ('resnet1d_101', 'resnet1d_span_cls_freq_channel', 64),

        ('resnet1d_50', 'resnet1d_span_cls_raw_time', 128),
        ('resnet1d_50', 'resnet1d_span_cls_raw_channel', 128),
        ('resnet1d_50', 'resnet1d_span_cls_freq_time', 128),
        ('resnet1d_50', 'resnet1d_span_cls_freq_channel', 64),

        ('resnet1d_34', 'resnet1d_span_cls_raw_time', 128),
        ('resnet1d_34', 'resnet1d_span_cls_raw_channel', 128),
        ('resnet1d_34', 'resnet1d_span_cls_freq_time', 128),
        ('resnet1d_34', 'resnet1d_span_cls_freq_channel', 64),

        ('resnet1d_18', 'resnet1d_span_cls_raw_time', 128),
        ('resnet1d_18', 'resnet1d_span_cls_raw_channel', 128),
        ('resnet1d_18', 'resnet1d_span_cls_freq_time', 128),
        ('resnet1d_18', 'resnet1d_span_cls_freq_channel', 64),
    ]

    for dataset_name in config.dataset_list:
        for module in model_list:
            backbone_name = module[0]
            head_name = dataset_name_to_head_name_mapping(dataset_name)
            strategy_name = module[1]
            batch_size = module[2]
            os.system(
                './script_run.sh %d %s %s %s %s %d' %
                (cuda, dataset_name, backbone_name, head_name, strategy_name, batch_size)
            )