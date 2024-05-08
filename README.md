## Quantizing Supernetworks For Medical Image Segmentation

## Requirements

+ Ubuntu
+ python 3.8
+ torch >= 1.0
+ torchvision >= 0.2.1
+ tqdm
+ numpy
+ pydicom (for chao dataset)
+ SimpleITK (for promise12 dataset)
+ Pillow
+ scipy
+ scikit-image
+ adabound
+ PyYAML
+ opencv-python
+ wandb
+ matplotlib
+ pydensecrf
+ pygraphviz
+ graphviz

## Usage

### Install dependencies

Note that depending on your system environment, you might need to handle package versioning manually.

```bash
pip3 install -r requirements.txt
```

### 1. Search the architecture

```bash
cd experiment
# Search at full precision on pascal voc2012
python search_cell.py --config ../configs/nas_unet/nas_unet_voc.yml
```

### 2. Update Genotype

Update Genotype at geno_searched.py, and add variable name to respective config file.

### 3. Train on downstream task
```bash
cd experiment
# Train on promise12 using optimized cell architecture
python train.py --config ../configs/nas_unet/nas_unet_promise12.yml
```

### Options

#### Cell Search

```bash
cd experiment
# Get a list of possible commands
python search_cell.py --help
```

```
usage: search_cell.py [-h] [--config [CONFIG]] [--quantize] [--mixed_precision] [--low_prec_optim]

config

optional arguments:
  -h, --help         show this help message and exit
  --config [CONFIG]  Configuration file to use (default is nas_unet_voc.yml)
  --quantize         Whether to quantize or not (default is False)
  --mixed_precision  Whether to use mixed precision or not (default is False)
  --low_prec_optim   Whether to use a low precision optimizer or not (default is False)
```
NOTE: Running multiple low precision options at the same time is untested behavior; please use only one such tag at a time.

#### Downstream Training

```bash
cd experiment
# Get a list of possible commands
python train.py --help
```

```
usage: train.py [-h] [--config [CONFIG]] [--model [MODEL]] [--ft] [--warm [WARM]]

config

optional arguments:
  -h, --help         show this help message and exit
  --config [CONFIG]  Configuration file to use
  --model [MODEL]    Model to train and evaluation
  --ft               finetuning on a different dataset
  --warm [WARM]      warm up from pre epoch
```
## Datasets

Datasets are all assumed to be at location ```/train_tiny_data/imgseg/```.
