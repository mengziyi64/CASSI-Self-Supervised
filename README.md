# CASSI-Self-Supervised
This repository contains the codes for paper **Self-supervised Neural Networks for Spectral Snapshot Compressive Imaging** (***ICCV (2021)***) by [Ziyi Meng](https://github.com/mengziyi64), Zhenming Yu, KunXu, Xin Yuan.
[[pdf]](https://arxiv.org/pdf/2108.12654.pdf)

## Overviewer
This repository uses a self-supervised neural networks to solve the reconstruction problem of snapshot compressive imaging (SCI), which uses a two-dimensional (2D) detector to capture a high-dimensional (usually 3D) data-cube in a compressed manner. This source code provides the reconstruction of 10 synthetic data originally used in [TSA-Net](https://github.com/mengziyi64/TSA-Net). So far this version of code only includes the PnP-DIP for the synthetic data.

## Results
<p align="center">
<img src="Data/Images/Fig1.png" width="800">
</p>
Fig. 1 Reconstructed synthetic data (sRGB) by 8 algorithms. We show the reconstructed spectral curves on selected regions to compare the spectral accuracy of different algorithms.

## Usage
### Download the SMEM repository and model file
0. Requirements are Python 3 and Pytorch 1.6 
1. Download this repository via git
2. Run **main.py** or **main.ipynb** to do reconstruction of one scene.

## Citation
```
@article{meng2021self,
  title={Self-supervised Neural Networks for Spectral Snapshot Compressive Imaging},
  author={Meng, Ziyi and Yu, Zhenming and Xu, Kun and Yuan, Xin},
  journal={arXiv preprint arXiv:2108.12654},
  year={2021}
}
```
## Contact
Ziyi Meng, Email: mengziyi64@163.com

Xin Yuan, Westlake University, Email: xyuan@westlake.edu.cn
