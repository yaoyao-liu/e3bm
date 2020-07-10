# Transductive Setting Experiments

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
