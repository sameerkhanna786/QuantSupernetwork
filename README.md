## NAS-Unet: Neural Architecture Search for Medical Image Segmentation

## Requirement

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

```bash
pip3 install -r requirements.txt
```

### Search the architecture

1. Cell search
```bash
cd experiment
# Search at full precision on pascal voc2012
python search_cell.py --config ../configs/nas_unet/nas_unet_voc.yml
```

2. Update Genotype

Update Genotype at geno_searched.py, and add variable name to respective config file.

3. Train on downstream task
```bash
cd experiment
# Search at full precision on promise12
python train.py --config ../configs/nas_unet/nas_unet_promise12.yml
```

#### Options

```bash
cd experiment
# Get a list of possible commands
python search_cell.py --help
```

NOTE: Running multiple low precision options at the same time is untested behavior. 
Please use only one such tag at a time.

```bash
cd experiment
# Get a list of possible commands
python train.py --help
```

## Datasets

Datasets are all assumed to be at location ```/train_tiny_data/imgseg/```.