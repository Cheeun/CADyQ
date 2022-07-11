# CADyQ : Contents-Aware Dynamic Quantization for Image Super Resolution

This resposity is the official implementation of our ECCV2022 [paper]().

![The framework of our paper.](https://github.com/Cheeun/CADyQ-pytorch/visualization/method-overview.png)

The overview of the proposed quantization framework CADyQ for SR network, which we illustrate with a residual block based backbone.
For each given patch and each layer, our CADyQ module introduces a light-weight bit selector that dynamically selects the bit-width and its corresponding quantization function $Q_{b^{k}}$ among the candidate quantization functions with distinct bit-widths.
The bit selector is conditioned on the estimated quantization sensitivity (the average gradient magnitude ${|\nabla{}|}$ of the given patch and the standard deviation $\sigma $ of the layer feature).
Qconv denotes the convolution layer of the quantized features and weights. 

Our implementation is based on [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and [PAMS(PyTorch)](https://github.com/colorjam/PAMS).



### Conda Environment setting
```
conda env create -f environment.yml --name CADyQ
conda activate CADyQ
```

### Dependencies
* Python 3.6
* PyTorch == 1.1.0
* coloredlogs >= 14.0
* scikit-image


### Datasets
* For training, we use [DIV2K datasets](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).

* For testing, we use [benchmark datasets](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) and [Test2K,4K.8K](https://drive.google.com/drive/folders/18b3QKaDJdrd9y0KwtrWU2Vp9nHxvfTZH?usp=sharing).
Test8K contains the images (index 1401-1500) from [DIV8K](https://competitions.codalab.org/competitions/22217#participate). Test2K/4K contain the images (index 1201-1300/1301-1400) from DIV8K which are downsampled to 2K and 4K resolution.

```
  # for training
  DIV2K 

  # for testing
  benchmark
  Test2K
  Test4K
  Test8K
```




### How to train CADyQ
```
sh train_carn_cadyq.sh
sh train_idn_cadyq.sh
sh train_edsrbaseline_cadyq.sh
sh train_srresnet_cadyq.sh
```
Pretrained model for stduent and teacher model to start training from can be accessed from [Google Drive]().


### How to test CADyQ
```
sh test_cadyq_patch.sh # for patch-wise inference
sh test_cadyq_image.sh # for image-wise inference
```
Our pretrained model can be accessed from [Google Drive]().



### Citation
```

```

### Contact
Email : cheeun914@snu.ac.kr 
