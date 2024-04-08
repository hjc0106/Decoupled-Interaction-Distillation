# Untitled

# DID based on MMRotate

## 1: Install mmrotate

Install mmrotate based [instruction](https://github.com/open-mmlab/mmdetection3d/blob/1.0/docs/en/getting_started.md).

- torch==1.13.0+cu117
- torchvision==0.14.0+cu117
- mmcv-full==1.6.0
- mmdet==2.26.0
- mmrotate==0.3.4
- yapf==0.40.1

## 2: Add KD training to mmrotate

- Replace their codes with our codes, including mmcv/runner/epoch_based_runner.py, mmrotate/apis, mmrotate/models
- Add our configs/distillation to configs from mmrotate
- Add our tools to tools from mmrotate

## 3: Teacher Preparation

Download the official teacher models from Openmmlab or train the teacher models.

```python
mkdir teacher_checkpoints
# Download Retinanet-R50
cd teacher_checkpoints
wget https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90/rotated_retinanet_obb_r50_fpn_1x_dota_le90-c0097bc4.pth
# Train Retinanet-R101
python tools/train.py configs/distillation/rotated_retinanet_obb_r101_fpn_1x_dota_le90.py
```

Put it in teacher_checkpoints. You need to create this folder by yourself.

# DID based on MMDetection3d

## 1: Install mmdetection3d

Install mmdetection3d based [instruction](https://github.com/open-mmlab/mmdetection3d/blob/1.0/docs/en/getting_started.md).

- torch==1.13.0+cu117
- torchvision==0.14.0+cu117
- mmcv-full==1.6.0
- mmdet==2.26.0
- mmsegmentation==0.29.1
- mmdet3d==1.00rc3
- numpy==1.23.5
- yapf==0.40.1

## 2: Add KD training to mmdetection3d

- Replace their codes with our codes, including mmcv/runner/epoch_based_runner.py, mmdet3d/apis, mmdet3d/models
- Add our configs/distillation to configs from mmdetection3d
- Add our tools to tools from mmdetection3d

## 3: Teacher Preparation

Download the official teacher models from Openmmlab or train the teacher models.

```python
# Download PointPillars
wget https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth
# Train Retinanet-R101
python tools/train.py configs/distillation/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py
```

Put it in teacher_checkpoints. You need to create this folder by yourself.
