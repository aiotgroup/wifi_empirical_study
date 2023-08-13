import os

from utils import *

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    cuda = 0

    config = DatasetDefaultConfig()

    model_list = [
        ('vit_b_2', 'vit_span_cls_freq', 32),
        ('vit_b_4', 'vit_span_cls_freq', 32),
        ('vit_b_8', 'vit_span_cls_freq', 32),
        ('vit_b_16', 'vit_span_cls_raw', 64),
        ('vit_b_32', 'vit_span_cls_raw', 64),
        ('vit_b_64', 'vit_span_cls_raw', 64),

        ('vit_s_2', 'vit_span_cls_freq', 64),
        ('vit_s_4', 'vit_span_cls_freq', 64),
        ('vit_s_8', 'vit_span_cls_freq', 64),
        ('vit_s_16', 'vit_span_cls_raw', 64),
        ('vit_s_32', 'vit_span_cls_raw', 64),
        ('vit_s_64', 'vit_span_cls_raw', 64),

        ('vit_ms_2', 'vit_span_cls_freq', 64),
        ('vit_ms_4', 'vit_span_cls_freq', 64),
        ('vit_ms_8', 'vit_span_cls_freq', 64),
        ('vit_ms_16', 'vit_span_cls_raw', 64),
        ('vit_ms_32', 'vit_span_cls_raw', 64),
        ('vit_ms_64', 'vit_span_cls_raw', 64),

        ('vit_es_2', 'vit_span_cls_freq', 64),
        ('vit_es_4', 'vit_span_cls_freq', 64),
        ('vit_es_8', 'vit_span_cls_freq', 64),
        ('vit_es_16', 'vit_span_cls_raw', 64),
        ('vit_es_32', 'vit_span_cls_raw', 64),
        ('vit_es_64', 'vit_span_cls_raw', 64),
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
