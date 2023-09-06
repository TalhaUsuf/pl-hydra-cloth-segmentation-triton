
# Purpose 

This code uses pl-lightning + hydra to train a cloth segmentation unet model using DDP multi-gpu strategy by using the 11k training samples from VTON_HD dataset.

All configurations are controlled via cfg files located inside [configs](configs) folder.

```bash
configs/
├── callbacks
│   ├── default.yaml
│   ├── early_stopping.yaml
│   ├── model_checkpoint.yaml
│   ├── model_summary.yaml
│   ├── none.yaml
│   └── rich_progress_bar.yaml
├── data
│   ├── mnist.yaml
│   └── viton_dataset.yaml
├── debug
│   ├── default.yaml
│   ├── fdr.yaml
│   ├── limit.yaml
│   ├── overfit.yaml
│   └── profiler.yaml
├── eval.yaml
├── experiment
│   └── example.yaml
├── extras
│   └── default.yaml
├── hparams_search
│   └── mnist_optuna.yaml
├── hydra
│   └── default.yaml
├── __init__.py
├── local
├── logger
│   ├── aim.yaml
│   ├── comet.yaml
│   ├── csv.yaml
│   ├── many_loggers.yaml
│   ├── mlflow.yaml
│   ├── neptune.yaml
│   ├── tensorboard.yaml
│   └── wandb.yaml
├── model
│   ├── mnist.yaml
│   └── viton_segment_model.yaml
├── paths
│   └── default.yaml
├── trainer
│   ├── cpu.yaml
│   ├── ddp_sim.yaml
│   ├── ddp.yaml
│   ├── default.yaml
│   ├── gpu.yaml
│   └── mps.yaml
└── train.yaml

```


`configs\train.yaml` will be used by the training script. It has default config to use for training. 


# Setup env.

```bash
conda env create -f environment.yaml
conda activate segment_vton

```




# Dataset

Download dataset from here `https://www.dropbox.com/s/10bfat0kg4si1bu/zalando-hd-resized.zip?dl=0`

unzip it in the repo. root dir., the unzipped data dir. structure should be following:

```bash
viton_hd/
├── test
│   ├── agnostic-v3.2
│   ├── cloth
│   ├── cloth-mask
│   ├── image
│   ├── image-densepose
│   ├── image-parse-agnostic-v3.2
│   ├── image-parse-v3
│   ├── openpose_img
│   └── openpose_json
├── test_pairs.txt
├── train
│   ├── agnostic-v3.2
│   ├── cloth
│   ├── cloth-mask
│   ├── image
│   ├── image-densepose
│   ├── image-parse-agnostic-v3.2
│   ├── image-parse-v3
│   ├── openpose_img
│   └── openpose_json
└── train_pairs.txt

```





# Train 

you can change the dataset related settings in `configs\data\viton_dataset.yaml`


```bash
python src/train.py

```


# Configure everything from CLI

⚡insted of changing yaml files you can change everything from CLI like:


You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.bs=64  data.workers=8 data.paired_unpaired='unpaired'
```

**To change the model encoder building blocks:**

```bash
python src/train.py trainer.max_epochs=20 model.encoder_name='resnet50'
```
# Configure backbone encoders

You can provide following encoder blocks names:

```markdown

 - resnet18
 - resnet34
 - resnet50
 - resnet101
 - resnet152
 - resnext50_32x4d
 - resnext101_32x4d
 - resnext101_32x8d
 - resnext101_32x16d
 - resnext101_32x32d
 - resnext101_32x48d

 - efficientnet-b0
 - efficientnet-b1
 - efficientnet-b2
 - efficientnet-b3
 - efficientnet-b4
 - efficientnet-b5
 - efficientnet-b6
 - efficientnet-b7
```

# TO DO
I am working on following features:

 - [ ] Add more backbone encoders
 - [ ] **Make model transformers api compatible**
 - [ ] Add model prediction code and results
 - [ ] Add pretrained model weights links
 - [ ] Add onnx model export code + Nvidia Triton inference server + Docker Compose









