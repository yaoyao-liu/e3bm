## An Ensemble of Epoch-wise Empirical Bayes for Few-shot Learning

[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.2.0-%237732a8)

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

Run the experiment with default settings:
```bash
python main.py
```
