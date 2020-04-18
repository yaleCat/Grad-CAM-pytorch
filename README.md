# Unofficial implementation for Grad-CAM in Pytorch with Multi Network Structures #

### What makes the network think the image label is 'dog' and 'cat':
![Dog](https://github.com/yaleCat/Grad-CAM-pytorch/blob/master/assets/demo_raw.png?raw=true)

### Combining Grad-CAM with Guided Backpropagation for the 'dog' class:
![Gb_dog](https://github.com/yaleCat/Grad-CAM-pytorch/blob/master/assets/demo_gb_dog.jpg?raw=true)

Gradient class activation maps are a visualization technique for deep learning networks.

See the paper: https://arxiv.org/pdf/1610.02391v1.pdf


----------
## In this Repo ##

### Grad-CAM, Guided-Backpropagation with Grad-CAM ###

#### VGG19 ####
----------
![Grad_cam_dog36](https://github.com/yaleCat/Grad-CAM-pytorch/blob/master/assets/cam36.jpg?raw=true)![GB_dog36](https://github.com/yaleCat/Grad-CAM-pytorch/blob/master/assets/cam_gb36.jpg?raw=true)

#### VGG 19 Layer1, Layer20 and Layer36 ####
----------
![vgg4](https://github.com/yaleCat/Grad-CAM-pytorch/blob/master/assets/cam4.jpg?raw=true)![vgg20](https://github.com/yaleCat/Grad-CAM-pytorch/blob/master/assets/cam20.jpg?raw=true)![vgg36](https://github.com/yaleCat/Grad-CAM-pytorch/blob/master/assets/cam36.jpg?raw=true)


#### EfficientNet-b0 ####
----------
![Grad_cam_dog15](https://github.com/yaleCat/Grad-CAM-pytorch/blob/master/assets/cam15.jpg?raw=true)![GB_dog15](https://github.com/yaleCat/Grad-CAM-pytorch/blob/master/assets/cam_gb15.jpg?raw=true)

#### EfficientNet-b0 Layer1, Layer10 and Layer15 ####
----------
![eff1](https://github.com/yaleCat/Grad-CAM-pytorch/blob/master/assets/cam1.jpg?raw=true)![eff10](https://github.com/yaleCat/Grad-CAM-pytorch/blob/master/assets/cam10.jpg?raw=true)![eff15](https://github.com/yaleCat/Grad-CAM-pytorch/blob/master/assets/cam15.jpg?raw=true)

----------
### What exactly is the difference among this repository and the others? ###
A: For example, these two are the most popular efficientdet-pytorch,

- Add EfficientNet for Visualization (Can use for both torchvision.models and EfficientNet from [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch))
- Add multi-layer visualization for comparison
- Switch GuidedBackPropagationReLU to GuidedBackPropagationSwish for EfficientNet

----------


Usage: `python grad-cam.py --image-path <path_to_image>`

To use with CUDA:
`python grad-cam.py --image-path <path_to_image> --use-cuda`


Reference:
Appreciate the great work from the following repositories:

- [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
- [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam.git)

