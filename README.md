## An Ensemble of Epoch-wise Empirical Bayes for Few-shot Learning

[![LICENSE](https://img.shields.io/github/license/yaoyao-liu/E3BM)](https://github.com/yaoyao-liu/E3BM/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.2.0-%237732a8)

### Paper: <https://arxiv.org/pdf/1904.08479> (last revised 16 Mar 2020)


## Installation

In order to run this repository, we advise you to install python 3.6 and PyTorch 1.2.0 with Anaconda.
You may download Anaconda and read the installation instruction on their official website:
<https://www.anaconda.com/download/>

Create a new environment and install PyTorch and torchvision on it:
```bash
conda create --name e3bm-pytorch python=3.6
conda activate e3bm-pytorch
conda install pytorch=1.2.0 
conda install torchvision -c pytorch
```

Install other requirements:
```bash
pip install -r requirements.txt
```

## Running Experiments

Run meta-training with default settings (data and pre-trained model will be downloaded automatically):
```bash
python main.py --phase_sib=meta_train
```

Run meta-test with our checkpoint (data and the checkpoint will be downloaded automatically):
```bash
python main.py --phase_sib=meta_eval
```

Run meta-test with other checkpoints:
```bash
python main.py --phase_sib=meta_eval --meta_eval_load_path=<your_ckpt_dir>
```

## Citation

Please cite our paper if it is helpful to your work:

```
@article{Liu2020E3BM
  author    = {Yaoyao Liu and
               Bernt Schiele and
               Qianru Sun},
  title     = {{LCC:} Learning to Customize and Combine Neural Networks for Few-Shot
               Learning},
  journal   = {aeXiv},
  volume    = {1904.08479},
  year      = {2019},
  url       = {http://arxiv.org/abs/1904.08479}
}
```

## Acknowledgements

Our implementations use the source code from the following repositories and users:

* [Learning Embedding Adaptation for Few-Shot Learning](https://github.com/Sha-Lab/FEAT)

* [Empirical Bayes Transductive Meta-Learning with Synthetic Gradients](https://github.com/hushell/sib_meta_learn)
