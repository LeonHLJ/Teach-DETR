## Teach-DETR

> **Teach-DETR: Better Training DETR with Teachers**<br> 
> Linjiang Huang (CUHK), Kaixin Lu (Shanghai University), Guanglu Song (Sensetime Research), Liang Wang (CASIA),
> Si Liu (Beihang University), Yu Liu (Sensetime Research), Hongsheng Li (CUHK)


### Coming soon.

- [ ] Release auxiliary boxes of MSCOCO 2017
- [x] Release code for H-Deformable DETR
- [ ] Release code for DINO
- [ ] Release models


## Introduction

In this paper, we present a novel training scheme, namely Teach-DETR, to learn better DETR-based detectors from versatile teacher detectors. We show that the predicted boxes from teacher detectors are effective medium to transfer knowledge of teacher detectors, which could be either RCNN-based or DETR-based detectors, to train a more accurate and robust DETR model. This new training scheme can easily incorporate the predicted boxes from multiple teacher detectors, each of which provides parallel supervisions to the student DETR. Our strategy introduces no additional parameters and adds negligible computational cost to the original detector during training. During inference, Teach-DETR brings zero additional overhead and maintains the merit of requiring no non-maximum suppression. Extensive experiments show that our method leads to consistent improvement for various DETR-based detectors. Specifically, we improve the state-of-the-art detector DINO with Swin-Large backbone, 4 scales of feature maps and 36-epoch training schedule, from 57.8\% to 58.9\% in terms of mean average precision on MSCOCO 2017 validation set.

<div align=center>
<img src=figures/pipeline.png width=70%>
</div>

## Results of DETR-based detectors on COCO

| Model  | Backbone | Epochs | Queries | AP |
| ------ | -------- | ------ | ------- | -- |
| Conditional-DETR-DC5 | R101 | 50 | 300 | 45.0 |
| Conditional-DETR-DC5 + Aux | R101 | 50 | 300 | 46.7 $\color{green}{(+1.7)}$ |
| DAB-DETR-DC5 | R101 | 50 | 300 | 45.8 |
| DAB-DETR-DC5 + Aux | R101 | 50 | 300 | 48.5 $\color{green}{(+2.7)}$ |
| DN-DETR-DC5 | R101 | 50 | 300 | 47.3 |
| DN-DETR-DC5 + Aux | R101 | 50 | 300 | 49.9 $\color{green}{(+2.6)}$|

## Results of two atypical DETR-based detectors on COCO

| Model  | Backbone | Epochs | Queries | AP |
| ------ | -------- | ------ | ------- | -- |
| YOLOS | DeiT-S | 150 | 100 | 35.6 |
| YOLOS + Aux | DeiT-S | 150 | 100 | 38.0 $\color{green}{(+2.4)}$|
| ViDT | Swin-S | 50 | 100 | 47.2 |
| ViDT + Aux | Swin-S | 50 | 100 | 49.0 $\color{green}{(+1.8)}$|

## Results of Deformable-DETR-based detectors on COCO

| Model  | Backbone | Epochs | Queries | AP |
| ------ | -------- | ------ | ------- | -- |
| Deformable-DETR | Swin-S | 36 | 300 | 50.7 |
| Deformable-DETR + Aux | Swin-S | 36 | 300 | 53.2 $\color{green}{(+2.5)}$ |
| Deformable-DETR + tricks $\dagger$ | Swin-S | 36 | 300 | 53.8 |
| Deformable-DETR + tricks $\dagger$ + Aux | Swin-S | 36 | 300 | 55.5 $\color{green}{(+1.7)}$ |
| H-Deformable-DETR | R50 | 36 | 300 | 50.0 |
| H-Deformable-DETR + Aux | R50 | 36 | 300 | 51.9 $\color{green}{(+1.9)}$ |
| H-Deformable-DETR | Swin-S | 36 | 300 | 54.2 |
| H-Deformable-DETR + Aux | Swin-S | 36 | 300 | 55.8 $\color{green}{(+1.6)}$ |
| H-Deformable-DETR | Swin-L (IN-22K) | 36 | 300 | 57.1 |
| H-Deformable-DETR + Aux | Swin-L (IN-22K) | 36 | 300 | 58.0 $\color{green}{(+0.9)}$ |
| H-Deformable-DETR $\ddagger$ | Swin-L (IN-22K) | 36 | 900 | 57.6 |
| H-Deformable-DETR $\ddagger$ + Aux | Swin-L (IN-22K) | 36 | 900 | 58.5 $\color{green}{(+0.9)}$ |
| DINO $\ddagger$ | Swin-L (IN-22K, 384) | 36 | 900 | 57.8 |
| DINO $\ddagger$ + Aux | Swin-L (IN-22K, 384) | 36 | 900 | 58.9 $\color{green}{(+1.1)}$ |

Note: all deformable-DETR-based detectors are in the two-stage manner.

$\dagger$ tricks denote dropout rate 0 within transformer, mixed query selection and look forward twice.

$\ddagger$ using top 300 predictions for evaluation.

## Installation
We test our models under ```python=3.7.10,pytorch=1.10.1,cuda=10.2```. Other versions might be available as well.

1. Clone this repo
```sh
git https://github.com/HDETR/H-Deformable-DETR.git
cd H-Deformable-DETR
```

2. Install Pytorch and torchvision

Follow the instruction on https://pytorch.org/get-started/locally/.
```sh
# an example:
conda install -c pytorch pytorch torchvision
```

3. Install other needed packages
```sh
pip install -r requirements.txt
pip install openmim
mim install mmcv-full
pip install mmdet
```

4. Compiling CUDA operators
```sh
cd models/ops
python setup.py build install
# unit test (should see all checking is True)
python test.py
cd ../..
```

## Data

Please download [COCO 2017](https://cocodataset.org/) dataset and organize them as following:
```
coco_path/
  ├── train2017/
  ├── val2017/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json
```
## Run
### To train a model using 8 cards

```Bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 <config path> \
    --coco_path <coco path>
```

To train/eval a model with the swin transformer backbone, you need to download the backbone from the [offical repo](https://github.com/microsoft/Swin-Transformer#main-results-on-imagenet-with-pretrained-models) frist and specify argument`--pretrained_backbone_path` like [our configs](./configs/two_stage/deformable-detr-hybrid-branch/36eps/swin).

### To eval a model using 8 cards

```Bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 <config path> \
    --coco_path <coco path> --eval --resume <checkpoint path>
```

### Distributed Run

You can refer to [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) to enable training on multiple nodes.


## License

This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
```bibtex
@misc{huang2023teachdetr,
      title={Teach-DETR: Better Training DETR with Teachers}, 
      author={Linjiang Huang and Kaixin Lu and Guanglu Song and Liang Wang and Si Liu and Yu Liu and Hongsheng Li},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      year={2023},
      publisher={IEEE}
}
```
