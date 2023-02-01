# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from typing import Type

from PIL import Image
import numpy as np
import torch
from mmengine.model import revert_sync_batchnorm
import mmcv

from mmseg.apis import inference_model, init_model
from mmseg.utils import register_all_modules
from mmseg.visualization import SegLocalVisualizer
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData


import torch.nn as nn


class Recorder:
    """record the forward output feature map and save to data_buffer
    """

    def __init__(self) -> None:
        self.data_buffer = list()

    def __enter__(self, ):
        self._data_buffer = list()

    def record_data_hook(self, model: nn.Module, input: Type, output: Type):
        self.data_buffer.append(output)

    def __exit__(self, *args, **kwargs):
        pass


def main():
    parser = ArgumentParser(
        description="To draw the feature_map durning inference")
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # show all named module in the model
    # for name, module in model.named_modules():
    #     print(name)

    source = ['decode_head.psp_modules.3.1.activate',
              'decode_head.lateral_convs.2.activat',
              'decode_head.fpn_bottleneck.activate']
    source = dict.fromkeys(source)

    count = 0
    recorder = Recorder()
    # registry the forward hook
    for name, module in model.named_modules():
        if name in source:
            count += 1
            module.register_forward_hook(recorder.record_data_hook)
            if count == len(source):
                break

    with recorder:
        # test a single image, and record feature map to data_buffer
        result = inference_model(model, args.img)

    seg_visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='WandbVisBackend')],
        save_dir='temp_dir',
        alpha=0.5)
    seg_visualizer.dataset_meta = dict(
        classes=model.dataset_meta['classes'],
        palette=model.dataset_meta['palette'])

    original_img = mmcv.imread(args.img, channel_order='rgb')
    # fuse the model output datasample result and original img
    seg_visualizer.add_datasample(
        name='gt_mask',
        image=original_img,
        data_sample=result,
        draw_gt=False,
        draw_pred=True,
        wait_time=0,
        out_file=None,
        show=False)

    # original test don't have mask, make a pseudo sample contain gt_mask
    gt_sample = SegDataSample()
    gt_mask = Image.open("/root/jxw/mmsegmentation/demo/IMG_01822-10.png")
    gt_mask = torch.from_numpy(np.array(gt_mask))
    gt_sample.set_data({
        'gt_sem_seg':
        PixelData(**{'data': gt_mask})
    })
    seg_visualizer.add_datasample(
        name='predict',
        image=original_img,
        data_sample=gt_sample,
        draw_gt=True,
        draw_pred=False,
        wait_time=0,
        out_file=None,
        show=False)

    seg_visualizer.add_image('original_img', original_img)

    # add feature map to wandb visualizer
    for i in range(len(recorder.data_buffer)):
        feature = recorder.data_buffer[i][0]  # remove the batch
        drawn_img = seg_visualizer.draw_featmap(
            feature, original_img, channel_reduction='select_max')
        seg_visualizer.add_image(f'feature_map{i}', drawn_img)


if __name__ == '__main__':
    main()
