## Teach-DETR

> **Teach-DETR: Better Training DETR with Teachers**<br> 
> Linjiang Huang (CUHK), Kaixin Lu (Shanghai University), Guanglu Song (Sensetime Research), Liang Wang (CASIA),
> Si Liu (Beihang University), Yu Liu (Sensetime Research), Hongsheng Li (CUHK)


### Coming soon.

- [ ] Release auxiliary boxes of MSCOCO 2017
- [ ] Release code for DINO
- [ ] Release code for H-Deformable DETR
- [ ] Release models


## Introduction

In this paper, we present a novel training scheme, namely Teach-DETR, to learn better DETR-based detectors from versatile teacher detectors. We show that the predicted boxes from teacher detectors are effective medium to transfer knowledge of teacher detectors, which could be either RCNN-based or DETR-based detectors, to train a more accurate and robust DETR model. This new training scheme can easily incorporate the predicted boxes from multiple teacher detectors, each of which provides parallel supervisions to the student DETR. Our strategy introduces no additional parameters and adds negligible computational cost to the original detector during training. During inference, Teach-DETR brings zero additional overhead and maintains the merit of requiring no non-maximum suppression. Extensive experiments show that our method leads to consistent improvement for various DETR-based detectors. Specifically, we improve the state-of-the-art detector DINO with Swin-Large backbone, 4 scales of feature maps and 36-epoch training schedule, from 57.8\% to 58.9\% in terms of mean average precision on MSCOCO 2017 validation set.


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

$\dagger$ denote dropout rate 0 within transformer, mixed query selection and look forward twice.

$\ddagger$ using top 300 predictions for evaluation.

## License

This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
```bibtex
@misc{huang2022teachdetr,
      title={Teach-DETR: Better Training DETR with Teachers}, 
      author={Linjiang Huang and Kaixin Lu and Guanglu Song and Liang Wang and Si Liu and Yu Liu and Hongsheng Li},
      year={2022},
      eprint={2211.11953},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
