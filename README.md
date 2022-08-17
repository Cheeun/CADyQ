# CADyQ : Content-Aware Dynamic Quantization for Image Super Resolution

This respository is the official implementation of our ECCV2022 paper.
<!-- [paper](). -->

![The framework of our paper.](https://github.com/Cheeun/CADyQ/blob/main/visualization/method-overview.png)
<!-- (https://raw.githubusercontent.com/Cheeun/CADyQ/visualization/method-overview.png) -->



The overview of the proposed quantization framework CADyQ for SR network, which we illustrate with a residual block based backbone.
For each given patch and each layer, our CADyQ module introduces a light-weight bit selector that dynamically selects the bit-width and its corresponding quantization function $Q_{b^{k}}$ among the candidate quantization functions with distinct bit-widths.
The bit selector is conditioned on the estimated quantization sensitivity (the average gradient magnitude ${|\nabla{}|}$ of the given patch 
and the standard deviation ${\sigma}$ of the layer feature).
Qconv denotes the convolution layer of the quantized features and weights. 

Our implementation is based on [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and [PAMS(PyTorch)](https://github.com/colorjam/PAMS).



### Conda Environment setting
```
conda env create -f environment.yml --name CADyQ
conda activate CADyQ
```

### Dependencies
* kornia (pip install kornia)
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
Model weights for stduent and teacher model to start training from can be accessed from [Google Drive](https://drive.google.com/drive/folders/1nPVdcqLqcaq-fp3Kg04WCUKav0PFER6x?usp=sharing).


### How to test CADyQ
```
sh test_cadyq_patch.sh # for patch-wise inference
sh test_cadyq_image.sh # for image-wise inference
```
- One example of the inference command
```
CUDA_VISIBLE_DEVICES=0 python3 main.py \
--test_only --cadyq --search_space 4+6+8 --scale 4 --k_bits 8 \
--model CARN --n_feats 64 --n_resblocks 9 --group 1 --multi_scale \
--student_weights dir/for/our/pretrained_model \
--data_test Urban100 --dir_data dir/for/datasets \
```
- Our pretrained model can be accessed from [Google Drive](https://drive.google.com/drive/folders/1pkbG4bQG6CoTxJKrUnBAEOxeQVUzvAsp?usp=sharing).



### Citation
If you found our implementation useful, please consider citing our paper:
```
@article{hong2022cadyq,
  title={CADyQ: Content-Aware Dynamic Quantization for Image Super-Resolution},
  author={Hong, Cheeun and Baik, Sungyong and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
  journal={arXiv preprint arXiv:2207.10345},
  year={2022}
}
```

### Contact
Email : cheeun914@snu.ac.kr 
