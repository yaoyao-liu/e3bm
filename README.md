# Transductive Setting

### Running Experiments

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

### Citation

Please cite our paper if it is helpful to your work:

```bibtex
@article{Liu2020E3BM,
  author    = {Yaoyao Liu and
               Bernt Schiele and
               Qianru Sun},
  title     = {An Ensemble of Epoch-wise Empirical Bayes for Few-shot Learning},
  journal   = {arXiv},
  year      = {2020}
}
```

### Acknowledgements

Our implementation uses the source code from the following repositories:

* [Learning Embedding Adaptation for Few-Shot Learning](https://github.com/Sha-Lab/FEAT)

* [Empirical Bayes Transductive Meta-Learning with Synthetic Gradients](https://github.com/hushell/sib_meta_learn)
