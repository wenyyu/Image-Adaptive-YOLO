# Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions
####  Accepted by AAAI 2022 [[arxiv]](https://arxiv.org/abs/2112.08088) 
Wenyu Liu, Gaofeng Ren, Runsheng Yu, Shi Guo, [Jianke Zhu](https://person.zju.edu.cn/jkzhu/645901.html), [Lei Zhang](https://web.comp.polyu.edu.hk/cslzhang/)
      
![image](https://user-images.githubusercontent.com/24246792/146731560-fa69fe86-fbf8-4a96-8bd8-a500490ec41d.png)
# Update
The image-adaptive filtering techniques used in the segmentation task can be found in our preprint paper.
#### "Improving Nighttime Driving-Scene Segmentation via Dual Image-adaptive Learnable Filters". [[arxiv]](https://arxiv.org/abs/2207.01331)
# Installation
```bash
$ git clone https://github.com/wenyyu/Image-Adaptive-YOLO.git  
$ cd Image-Adaptive-YOLO  
# Require python3 and tensorflow
$ pip install -r ./docs/requirements.txt
```

# Datasets and Models
[PSCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) [RTTS](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2) [ExDark](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/tree/master/Dataset)  
Voc_foggy_test & Voc_dark_test & Models: [Google Drive](https://drive.google.com/drive/folders/1P0leuiGHH69kVxyNVFuiCdCYXyYquPqM), [Baidu Netdisk](https://pan.baidu.com/s/1GQE_80rEzs0uCrzauHxwdw) (key: iayl)  
# Quick test
```bash  
# put checkpoint model in the corresponding directory 
# change the data and model paths in core/config.py
$ python evaluate.py 
```

![image](https://user-images.githubusercontent.com/24246792/146735760-4fcf7be9-fdd2-4694-8d91-d254144c52eb.png)

# Train and Evaluate on the datasets
1. Download VOC PASCAL trainval and test data
```bashrc
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```
Extract all of these tars into one directory and rename them, which should have the following basic structure.
```bashrc

VOC           # path:  /home/lwy/work/code/tensorflow-yolov3/data/VOC
├── test
|    └──VOCdevkit
|        └──VOC2007 (from VOCtest_06-Nov-2007.tar)
└── train
     └──VOCdevkit
         └──VOC2007 (from VOCtrainval_06-Nov-2007.tar)
         └──VOC2012 (from VOCtrainval_11-May-2012.tar)
                     
$ python scripts/voc_annotation.py
```
2. Generate Voc_foggy_train and Voc_foggy_val dataset offline
```bash  
# generate ten levels' foggy training images and val images, respectively
$ python ./core/data_make.py 
```

3. Edit core/config.py to configure  
```bashrc
--vocfog_traindata_dir'  = '/data/vdd/liuwenyu/data_vocfog/train/JPEGImages/'
--vocfog_valdata_dir'    = '/data/vdd/liuwenyu/data_vocfog/val/JPEGImages/'
--train_path             = './data/dataset_fog/voc_norm_train.txt'
--test_path              = './data/dataset_fog/voc_norm_test.txt'
--class_name             = './data/classes/vocfog.names'
```
4. Train and Evaluate
```bash  
$ python train.py # we trained our model from scratch.  
$ python evaluate.py   
$ cd ./experiments/.../mAP & python main.py 
``` 
5. More details of Preparing dataset or Train with your own dataset  
   reference the implementation [tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3).
   
# Train and Evaluate on low_light images
The overall process is the same as above, run the *_lowlight.py to train or evaluate.

# Acknowledgments

The code is based on [tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3), [exposure](https://github.com/yuanming-hu/exposure).

# Citation

```shell
@inproceedings{liu2022imageadaptive,
  title={Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions},
  author={Liu, Wenyu and Ren, Gaofeng and Yu, Runsheng and Guo, Shi and Zhu, Jianke and Zhang, Lei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2022}
}

@article{liu2022improving,
  title={Improving Nighttime Driving-Scene Segmentation via Dual Image-adaptive Learnable Filters},
  author={Liu, Wenyu and Li, Wentong and Zhu, Jianke and Cui, Miaomiao and Xie, Xuansong and Zhang, Lei},
  journal={arXiv e-prints},
  pages={arXiv--2207},
  year={2022}
}
```
