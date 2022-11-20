# Pytorch Medical Segmentation

<i>Read Chinese Introduction：<a href='https://github.com/MontaEllis/Pytorch-Medical-Segmentation/blob/master/README-zh.md'>Here！</a></i><br />

## Notes

We are planning a major update to the code in the near future, so if you have any suggestions, please feel free to email [me](elliszkn@163.com) or mention them in the issue.

## Recent Updates

* 2021.1.8 The train and test codes are released.
* 2021.2.6 A bug in dice was fixed with the help of [Shanshan Li](https://github.com/ssli23).
* 2021.2.24 A video tutorial was released(https://www.bilibili.com/video/BV1gp4y1H7kq/).
* 2021.5.16 A bug in Unet3D implement was fixed.
* 2021.5.16 The metric code is released.
* 2021.6.24 All parameters can be adjusted in hparam.py.
* 2021.7.7 Now you can refer medical classification in [Pytorch-Medical-Classification](https://github.com/MontaEllis/Pytorch-Medical-Classification)
* 2022.5.15 Now you can refer semi-supervised learning on medical segmentation in [SSL-For-Medical-Segmentation](https://github.com/MontaEllis/SSL-For-Medical-Segmentation)
* 2022.5.17 We update the training and inference code and fix some bugs.

## Requirements

* pytorch1.7
* torchio<=0.18.20
* python>=3.6

## Notice

* You can modify **hparam.py** to determine whether 2D or 3D segmentation and whether multicategorization is possible.
* We provide algorithms for almost all 2D and 3D segmentation.
* This repository is compatible with almost all medical data formats(e.g. nii.gz, nii, mhd, nrrd, ...), by modifying **fold_arch** in **hparam.py** of the config. **I would like you to convert both the source and label images to the same type before using them, where labels are marked with 1, not 255.**
* If you want to use a **multi-category** program, please modify the corresponding codes by yourself. I cannot identify your specific categories.
* Whether in 2D or 3D, this project is processed using **patch**. Therefore, images do not have to be strictly the same size. In 2D, however, you should set the patch large enough.

## Models have Completed in Project

| Publication Date | Model Name | The First and Last Authors |  Title | Reference|
| :---: | :---: | :---: | :---: | :---: |
| 2016-10 |  3D U-Net  | Özgün Çiçek and Ronneberger, Olaf | 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation | [MICCAI2016](https://link.springer.com/chapter/10.1007/978-3-319-46723-8_49)|
| 2016-10 | 3D V-Net | Fausto Milletari and Seyed-Ahmad Ahmadi | V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation | [3DV2016](https://ieeexplore.ieee.org/abstract/document/7785132) |
| 2017-09 | 3D DenseVoxelNet  | Lequan Yu and Jing Qin & Pheng-Ann Heng | Automatic 3D Cardiovascular MR Segmentation with Densely-Connected Volumetric ConvNets | [MICCAI2017](https://link.springer.com/chapter/10.1007/978-3-319-66185-8_33) |
| 2017-09 | 3D DenseNet | Toan Duc Bui and Taesup Moon | 3D Densely Convolutional Networks for Volumetric Segmentation | [arxiv](https://arxiv.org/abs/1709.03199) |
| 2017-05 | 3D HighResNet | Wenqi Li and M. Jorge Cardoso & Tom Vercauteren | On the Compactness, Efficiency, and Representation of 3D Convolutional Networks: Brain Parcellation as a Pretext Task | [IPMI2017](https://link.springer.com/chapter/10.1007/978-3-319-59050-9_28) |
| 2017-05 | 3D Residual U-Net | Kisuk Lee and H. Sebastian Seung | Superhuman Accuracy on the SNEMI3D Connectomics Challenge | [arxiv](https://arxiv.org/abs/1706.00120) |
| 2021-10 |  CSR-Net   | Cheng Chen and Ruoxiu Xiao | CSR-Net: Cross-Scale Residual Network for multi-objective scaphoid fracture segmentation | [CIBM2021](https://www.sciencedirect.com/science/article/pii/S0010482521005709) |
| 2022 | UNETR | Ali Hatamizadeh and Daguang Xu | UNETR: Transformers for 3D Medical Image Segmentation | [CVPR2022](https://openaccess.thecvf.com/content/WACV2022/html/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.html) |

## Prepare Your Dataset

### Example1

if your source dataset is :

```
source_dataset
├── source_1.mhd
├── source_1.zraw
├── source_2.mhd
├── source_2.zraw
├── source_3.mhd
├── source_3.zraw
├── source_4.mhd
├── source_4.zraw
└── ...
```

and your label dataset is :

```
label_dataset
├── label_1.mhd
├── label_1.zraw
├── label_2.mhd
├── label_2.zraw
├── label_3.mhd
├── label_3.zraw
├── label_4.mhd
├── label_4.zraw
└── ...
```

then your should modify **fold_arch** as **\*.mhd**, **source_train_dir** as **source_dataset** and **label_train_dir** as **label_dataset** in **hparam.py**

### Example2

if your source dataset is :

```
source_dataset
├── 1
    ├── source_1.mhd
    ├── source_1.zraw
├── 2
    ├── source_2.mhd
    ├── source_2.zraw
├── 3
    ├── source_3.mhd
    ├── source_3.zraw
├── 4
    ├── source_4.mhd
    ├── source_4.zraw
└── ...
```

and your label dataset is :

```
label_dataset
├── 1
    ├── label_1.mhd
    ├── label_1.zraw
├── 2
    ├── label_2.mhd
    ├── label_2.zraw
├── 3
    ├── label_3.mhd
    ├── label_3.zraw
├── 4
    ├── label_4.mhd
    ├── label_4.zraw
└── ...
```

then your should modify **fold_arch** as **\*/\*.mhd**, **source_train_dir** as **source_dataset** and **label_train_dir** as **label_dataset** in **hparam.py**

## Training

* without pretrained-model

```
set hparam.train_or_test to 'train'
python main.py
```

* with pretrained-model

```
set hparam.train_or_test to 'train'
python main.py -k True
```

## Inference

* testing

```
set hparam.train_or_test to 'test'
python main.py
```

## Examples

![](https://ellis.oss-cn-beijing.aliyuncs.com/img/20210108185333.png)
![](https://ellis.oss-cn-beijing.aliyuncs.com/img/2021-02-06%2022-40-07%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

## Tutorials

* https://www.bilibili.com/video/BV1gp4y1H7kq/

## Done

### Network

* 2D

* [x] unet
* [x] unet++
* [x] miniseg
* [x] segnet
* [x] pspnet
* [x] highresnet(copy from https://github.com/fepegar/highresnet, Thank you to [fepegar](https://github.com/fepegar) for your generosity!)
* [x] deeplab
* [x] fcn

* 3D

* [x] unet3d
* [x] residual-unet3d
* [x] densevoxelnet3d
* [x] fcn3d
* [x] vnet3d
* [x] highresnert(copy from https://github.com/fepegar/highresnet, Thank you to [fepegar](https://github.com/fepegar) for your generosity!)
* [x] densenet3d
* [x] unetr (copy from https://github.com/tamasino52/UNETR)

### Metric

* [x] metrics.py to evaluate your results

## TODO

* [ ] dataset
* [ ] benchmark
* [ ] nnunet

## By The Way

This project is not perfect and there are still many problems. If you are using this project and would like to give the author some feedbacks, you can send [Me](elliszkn@163.com) an email.

## Acknowledgements

This repository is an unoffical PyTorch implementation of Medical segmentation in 3D and 2D and highly based on [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch) and [torchio](https://github.com/fepegar/torchio). Thank you for the above repo. The project is done with the supervisions of [Prof. Ruoxiu Xiao](http://enscce.ustb.edu.cn/Teach/TeacherList/2020-10-16/114.html) and [Dr. Cheng Chen](b20170310@xs.ustb.edu.cn). Thank you to [Youming Zhang](zhangym0820@csu.edu.cn), [Daiheng Gao](https://github.com/tomguluson92), [Jie Zhang](jpeter.zhang@connect.polyu.hk), [Xing Tao](kakatao@foxmail.com), [Weili Jiang](1379252229@qq.com) and [Shanshan Li](https://github.com/ssli23) for all the help I received.

## Related works

If this code is helpful for you, you can cite these for us. Thank you.

```
[1] Chen C, Zhou K. An Effective Deep Neural Network for Lung Lesions Segmentation from COVID-19 CT Images[J]. IEEE Transactions on Industrial Informatics, 2021.
[2] Chen C, Zhang T, et al. Pathological lung segmentation in chest CT images based on improved random walker[J]. Computer Methods and Programs in Biomedicine, 2021, 200: 105864.
```
