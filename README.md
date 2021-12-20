# Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions

AAAI2022 [[arxiv]](https://arxiv.org/abs/2112.08088) 

![image](https://user-images.githubusercontent.com/24246792/146731560-fa69fe86-fbf8-4a96-8bd8-a500490ec41d.png)

# Installation

$ git clone https://github.com/wenyyu/Image-Adaptive-YOLO.git  
$ cd Image-Adaptive-YOLO  
$ pip install -r ./docs/requirements.txt

# Datasets
[[PSCAL VOC]](http://host.robots.ox.ac.uk/pascal/VOC/) [[RTTS]](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2) [[ExDark]](https://drive.google.com/file/d/1GZqHFzTLDI-1rcOctHdf-c16VgagWocd/view)  

# Quick test

1. Download the model and test data [[BaiduClound]](https://pan.baidu.com/s/1GQE_80rEzs0uCrzauHxwdw) (Password: iayl)  
2. Run the test script  
$ python evaluate.py 

      ![image](https://user-images.githubusercontent.com/24246792/146735760-4fcf7be9-fdd2-4694-8d91-d254144c52eb.png)

# Train and Evaluate on the datasets

1. Prepare the training and testing datasets, edit "./core/config.py" to configure  
$ python train.py # we trained our model from scratch.  
$ python evaluate.py   
$ cd mAP & python main.py 

2. Train with your own dataset  
   reference the Implementations [[tensorflow-yolov3]](https://github.com/YunYang1994/tensorflow-yolov3) to prepare the files.

# Acknowledgments

The code is based on [[tensorflow-yolov3]](https://github.com/YunYang1994/tensorflow-yolov3), [[exposure]](https://github.com/yuanming-hu/exposure).

# Citation

```shell
@inproceedings{liu2022imageadaptive,
  title={Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions},
  author={Liu, Wenyu and Ren, Gaofeng and Yu, Runsheng and Guo, Shi and Zhu, Jianke and Zhang, Lei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2022}
}
```
