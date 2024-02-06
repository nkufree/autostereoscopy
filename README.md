# 基于遮挡视频实例分割的裸眼3D实现

## 项目简介

本项目基于[GenVIS模型](https://github.com/miranheo/GenVIS)实现了裸眼3D的效果，使用模型为OVIS训练集上训练的模型。

在原项目的基础上，修改部分代码使其在Python3.10版本中运行。

关于裸眼3D的部分，主要代码分为以下部分：

1. `demo/autostereoscopy.py`文件中实现视频的输入和输出。
2. `demo/predictor.py`文件中增加裸眼3D的适配。
3. `demo/visualizer.py`文件中实现方法`draw_autostereoscopy`，用于绘制裸眼3D的必要边框。

项目效果展示[点击此处](https://www.bilibili.com/video/BV1xZ421U7M1/)。

## 环境配置

原项目环境配置可参考：[installation instructions](https://github.com/sukjunhwang/VITA/blob/main/INSTALL.md).

我使用的环境为docker，配置方式如下：

```shell
docker pull pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

docker run --name pytorch2 --gpus all --privileged -v $PWD/conda:/home/condashare -dt -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all --shm-size 8G pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel
```

在这里我是将Windows上的一个目录挂载到容器中，后续需要在这个文件夹中克隆仓库，这样方便查看输出。

之后按照原项目的配置方式即可，注意python版本的区别，在后续需要克隆detectron2仓库，该项目的`setup.cfg`文件中使用的版本为3.7，需要修改为3.10。

## 运行方式

我使用的视频分辨率为1920x1080。

使用R50为backbone的模型，运行方式如下：

```shell
CUDA_VISIBLE_DEVICES=0 python demo/autostereoscopy.py --config-file configs/genvis/ovis/genvis_R50_bs8_online.yaml --video-input /path/to/video --output /path/to/output --opts MODEL.WEIGHTS /path/to/checkpoint_file
```

使用Swin-L为backbone的模型，运行方式如下(其实改一下config和模型路径就行)：

```shell
CUDA_VISIBLE_DEVICES=0 python demo/autostereoscopy.py --config-file configs/genvis/ovis/genvis_SWIN_bs8_online.yaml --video-input /path/to/video --output /path/to/output--save-frames true --opts MODEL.WEIGHTS /path/to/checkpoint_file
```

这里需要给出单个路径，视频路径、输出文件夹路径和模型路径。

测试了这两个模型，使用Swin-L的效果更好。

以下为原项目的部分README。

## Getting Started

We provide a script `train_net_genvis.py`, that is made to train all the configs provided in GenVIS.

To train a model with "train_net_genvis.py" on VIS, first
setup the corresponding datasets following
[Preparing Datasets](https://github.com/sukjunhwang/VITA/blob/main/datasets/README.md).

Then run with pretrained weights on target VIS dataset in [VITA's Model Zoo](https://github.com/sukjunhwang/VITA#model-zoo):
```
python train_net_genvis.py --num-gpus 4 \
  --config-file configs/genvis/ovis/genvis_R50_bs8_online.yaml \
  MODEL.WEIGHTS vita_r50_ovis.pth
```

To evaluate a model's performance, use
```
python train_net_genvis.py --num-gpus 4 \
  --config-file configs/genvis/ovis/genvis_R50_bs8_online.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

## <a name="ModelZoo"></a>Model Zoo
**Additional weights will be updated soon!**
### YouTubeVIS-2019
| Backbone | Method | AP | AP50 | AP75| AR1 | AR10 | Download |
| :---: | :---: | :--: | :---: | :---: | :---: | :---: | :---: |
| R-50 | online | 50.0 | 71.5 | 54.6 | 49.5 | 59.7 | [model](https://drive.google.com/file/d/1WdDsE4EGAuYQ1hqLB4XtZoYO0iSehnZo/view?usp=share_link) |
| R-50 | semi-online | 51.3 | 72.0 | 57.8 | 49.5 | 60.0 | [model](https://drive.google.com/file/d/1yQVzuFFrHsRDd96ywMsGLTDwVqKShFZt/view?usp=share_link) |
| Swin-L | online | 64.0 | 84.9 | 68.3 | 56.1 | 69.4 | [model](https://drive.google.com/file/d/1TZvH5qlhTnZ6WXk1oNmCmYz_cq1m5AuO/view?usp=share_link) |
| Swin-L | semi-online | 63.8 | 85.7 | 68.5 | 56.3 | 68.4 | [model](https://drive.google.com/file/d/1PTtkH-Angrw92D7P7-BXvtAQZ8nWmJ6Q/view?usp=share_link) |

### YouTubeVIS-2021
| Backbone | Method | AP | AP50 | AP75| AR1 | AR10 | Download |
| :---: | :---: | :--: | :---: | :---: | :---: | :---: | :---: |
| R-50 | online | 47.1 | 67.5 | 51.5 | 41.6 | 54.7 | [model](https://drive.google.com/file/d/1-WcWxoBRBIAyxhH0-1X2ywe1bquOWjkO/view?usp=share_link) |
| R-50 | semi-online | 46.3 | 67.0 | 50.2 | 40.6 | 53.2 | [model](https://drive.google.com/file/d/1AMqKe9OX-wsr39RUxggTwPY25cvABoub/view?usp=share_link) |
| Swin-L | online | 59.6 | 80.9 | 65.8 | 48.7 | 65.0 | [model](https://drive.google.com/file/d/1cHEfYb6QLGllR1i2xvL-AZnrthKx3wbV/view?usp=share_link) |
| Swin-L | semi-online | 60.1 | 80.9 | 66.5 | 49.1 | 64.7 | [model](https://drive.google.com/file/d/1Nl8bE5JXFdLSoABrvNax_rrnLrt0ZSNc/view?usp=share_link) |

### OVIS
| Backbone | Method | AP | AP50 | AP75| AR1 | AR10 | Download |
| :---: | :---: | :--: | :---: | :---: | :---: | :---: | :---: |
| R-50 | online | 35.8 | 60.8 | 36.2 | 16.3 | 39.6 | [model](https://drive.google.com/file/d/15Iitl2sSmAxFXT-PJCYfY37vcc7_iEO7/view?usp=share_link) |
| R-50 | semi-online | 34.5 | 59.4 | 35.0 | 16.6 | 38.3 | [model](https://drive.google.com/file/d/1Y8d0ETmW3XoD-zGxvZNRVvlz1jTsXY5a/view?usp=share_link) |
| Swin-L | online | 45.2 | 69.1 | 48.4 | 19.1 | 48.6 | [model](https://drive.google.com/file/d/11aqfoqDoyEIDcDmYqcWDEX3FK7ChIRks/view?usp=share_link) |
| Swin-L | semi-online | 45.4 | 69.2 | 47.8 | 18.9 | 49.0 | [model](https://drive.google.com/file/d/17uErrcAZ6-5ewdzUy9CxDK6tjOe5Xp93/view?usp=share_link) |

## License
The majority of GenVIS is licensed under a
[Apache-2.0 License](LICENSE).
However portions of the project are available under separate license terms: Detectron2([Apache-2.0 License](https://github.com/facebookresearch/detectron2/blob/main/LICENSE)), IFC([Apache-2.0 License](https://github.com/sukjunhwang/IFC/blob/master/LICENSE)), Mask2Former([MIT License](https://github.com/facebookresearch/Mask2Former/blob/main/LICENSE)), Deformable-DETR([Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE)), and VITA([Apache-2.0 License](https://github.com/sukjunhwang/VITA/blob/main/LICENSE)).

