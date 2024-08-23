## Environment Set Up

### My Environment

NVIDIA GeForce RTX 4090
CUDA 12.1
Ubuntu 22.01

### Install Dependencies

```bash
$ conda --version
conda 24.7.1

$ protoc --version
libprotoc 3.21.12

```

### Create the environment, enter the directory and install some needed packages

```bash
conda create -n AMR python=3.8.5 && conda activate AMR
cd train

make install
```

if encounter this error 
```bash
option --use-feature: invalid choice: '2020-resolver' (choose from 'fast-deps', 'truststore', 'no-binary-enable-wheel-cache')
make: *** [Makefile:12: api] Error 2
```
then open the Makefile, change line 15 from 

`python -m pip install --use-feature=2020-resolver .`

to 

`python -m pip install .` 

and try again


if it installs successfully, then it would show

```bash
....................
----------------------------------------------------------------------
Ran 24 tests in 8.664s

OK (skipped=1)
```

### Install other packages

`pip install -r requirements.txt`

## Running

Put your testing images at [here](images/test)

`python test_images.py` or if you prefer using [Jupter Notebook](test_image.ipynb)

And the result will be stored at [here](images/test_annotated)

## Training

First of all, you have to create the working directory and enter it

```bash
make workspace-mask SAVE_DIR=workspace NAME=test-mask

cd workspace/test-mask 
```

Put your training images at [here](train/workspace/test-mask/images/train), and your validation images at [here](train/workspace/test-mask/images/val)

**Note:** Your annotation files must follow COCO format.

`make gen-tfrecord` to generate `label_map.pbtxt`

### Download the Pre-trained Model

`make dl-model` (The default selection here is **mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8**)

And the structure will be

```bash
└── test-mask
    └── pre-trained-models
        └── mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8
            ├── checkpoint
            ├── pipeline.config
            └── saved_model
```

### Configure Training Pipeline

Create a corresponding model folder in the models directory, such as: 

**mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8**, and copy 

**pre-trained-models/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8/pipeline config**

```bash
└── test-mask
    ├── models
    │   └── mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8
    │       └── pipeline.config
    └── pre-trained-models
```

After setting, type `make train` to start training.

## Model Export

After training, you should export the trained model by using `make export`

## Reference

[Training reference](https://github.com/TannerGilbert/Tensorflow-Object-Detection-API-train-custom-Mask-R-CNN-model) 
