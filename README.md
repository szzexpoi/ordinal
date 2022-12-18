# Ordinal Relation prediction

This code implements a set of simple deep neural networks for predicting the ordinal relationship between two selected points. In the current stage, the experiments are restricted to LSP dataset.

### Training Command
```
python main_lsp.py --checkpoint_dir ./log_baseline_sigma_08_pos --lr 5e-4 --batch_size 32 --bbox_size 128 --mask_size 32 --mask_sigma 0.8 --epoch 240
```

### Evaluation Command
```
python main_lsp.py --weights ./weights/model_best.pth --batch_size 32 --bbox_size 128 --mask_size 32 --mask_sigma 0.8 --mode eval
```
