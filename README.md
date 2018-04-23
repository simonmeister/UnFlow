# *UnFlow*: Unsupervised Learning of Optical Flow with a Bidirectional Census Loss

This repository contains the TensorFlow implementation of the paper

[UnFlow: Unsupervised Learning of Optical Flow with a Bidirectional Census Loss](https://arxiv.org/abs/1711.07837) (AAAI 2018)

[Simon Meister](http://simonmeister.org),
[Junhwa Hur](http://www.visinf.tu-darmstadt.de), and
[Stefan Roth](http://www.visinf.tu-darmstadt.de).

### Citation

If you find UnFlow useful in your research, please consider citing:

    @inproceedings{Meister:2018:UUL,
      title  = {{UnFlow}: Unsupervised Learning of Optical Flow
                with a Bidirectional Census Loss},
      author = {Simon Meister and Junhwa Hur and Stefan Roth},
      address = {New Orleans, Louisiana},
      booktitle = {AAAI},
      month = feb,
      year = {2018}
    }

### License

UnFlow is released under the MIT License (refer to the LICENSE file for details).


## Contents
1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Replicating our Models](#replicating-our-models)
4. [Navigating the Code](#navigating-the-code)

## Introduction

Our paper describes a method to train end-to-end deep networks for dense optical flow
without the need for ground truth optical flow.

This implementation supports all training and evaluation styles described
in the paper. This includes
- unsupervised training with our proxy loss(es) on
[SYNTHIA](http://synthia-dataset.net/),
[KITTI raw](http://www.cvlibs.net/datasets/kitti/raw_data.php),
[Cityscapes](https://www.cityscapes-dataset.com/),
and [FlyingChairs](https://arxiv.org/abs/1504.06852),
- supervised fine-tuning on
[KITTI 2012 flow](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow)
and [KITTI 2015 flow](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow),
- evaluation on
[KITTI 2012 flow](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow),
[KITTI 2015 flow](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow),
[Middlebury flow](http://vision.middlebury.edu/flow/),
and [MPI Sintel flow](http://sintel.is.tue.mpg.de/).

The supported network architectures are
[FlowNetS](https://arxiv.org/abs/1504.06852),
[FlowNetC](https://arxiv.org/abs/1504.06852), as well as stacked variants of these networks
as introduced in [FlowNet 2.0](https://arxiv.org/abs/1612.01925).

All datasets required for training and evaluation are downloaded on-demand
by the code (except Cityscapes, which has to be downloaded manually if needed).
Please ensure that the `data` directory specified in the configuration file
has enough space to hold at least 150 GB (when using SYNTHIA and KITTI only).

## Usage

Please refer to the configuration file template (`config_template/config.ini`) for a detailed description
of the different operating modes.

### Hardware requirements
- at least one NVIDIA GPU (multi-GPU training is supported)

### Software requirements
- python 3
- gcc4
- RAR backend tool for `rarfile` (see https://pypi.python.org/pypi/rarfile/)
- python packages: `matplotlib pypng rarfile pillow` and `tensorflow-gpu` (at least version 1.7)
- CUDA and CuDNN. You should use the installer downloaded from the NVIDIA website and install to /usr/local/cuda. Please make sure that the versions you install are compatible with the version of `tensorflow-gpu` you are using (see e.g. https://github.com/tensorflow/tensorflow/releases).

### Prepare environment
- copy `config_template/config.ini` to `./` and modify settings in the `[dir]`, `[run]`
and `[compile]` sections for your environment (see comments in the file).

### Run & evaluate experiments
- adapt settings in `./config.ini` for your experiment
- `cd src`
- train with `python run.py --ex my_experiment`. Evaluation is run during training as specified
in `config.ini`, but no visualizations are saved.
- evaluate (multiple) experiments with `python eval_gui.py --ex experiment_name_1[, experiment_name_2 [...]]`
and visually compare results side-by-side.

You can use the --help flag with run.py or eval_gui.py to view all available flags.

### View tensorboard logs
- view logs for all experiments with `tensorboard --logdir=<log_dir>/ex`

### Pre-trained models
We provide checkpoints for the 
`C_SYNTHIA`, `CS_SYNTHIA`, `CSS_SYNTHIA`, `C`, `CS`, `CSS` and `CSS_ft` 
models (see "Replicating our models" for a description). 
To use them,

- download 
[this file](https://drive.google.com/file/d/16rOMerQvUnj6UjGjMyQayC1GcqaRu44b/view?usp=sharing) 
and extract the contents to `<log_dir>/ex/`.

Now, you can evaluate and compare different models, e.g.

- `python eval_gui.py --ex C_SYNTHIA,C,CSS_ft`.

## Replicating our models

In the following, each list item gives an experiment name and parameters to
set in `./config.ini`. For each experiment,
first modify the configuration parameters as specified,
and then run `python run.py --ex experiment_name`.

First, create a series of experiments for the models pre-trained on `SYNTHIA`:
- C_SYNTHIA: `dataset = synthia`, `flownet = C`,
- CS_SYNTHIA: `dataset = synthia`, `flownet = CS`, `finetune = C_SYNTHIA`,
- CSS_SYNTHIA: `dataset = synthia`, `flownet = CSS`, `finetune = C_SYNTHIA,CS_SYNTHIA`.

Next, create a series of experiments for the models trained on `KITTI raw`:
- C: `dataset = kitti`, `flownet = C`, `finetune = C_SYNTHIA`,
- CS: `dataset = kitti`, `flownet = CS`, `finetune = C,CS_SYNTHIA`,
- CSS: `dataset = kitti`, `flownet = CSS`, `finetune = C,CS,CSS_SYNTHIA`.

Then, train the final fine-tuned model on `KITTI 2012 / 2015`:
- CSS_ft: `dataset = kitti_ft`, `flownet = CSS`, `finetune = C,CS,CSS`, `train_all = True`.
Please monitor the eval logs and stop the training as soon as the validation error
starts to increase (for CSS, it should be at about 70K iterations).

If you want to train on Cityscapes:
- C_Cityscapes: `dataset = cityscapes`, `flownet = C`, `finetune = C_SYNTHIA`.

Note that all models but `CSS_ft` were trained without ground truth optical flow,
using our unsupervised proxy loss only.

## Navigating the code
The core implementation of our method resides in [src/e2eflow/core](src/e2eflow/core). The key files are:
- [losses.py](src/e2eflow/core/losses.py): Proxy losses for unsupervised training,
- [flownet.py](src/e2eflow/core/flownet.py): [FlowNet](https://arxiv.org/abs/1612.01925) architectures with support for stacking,
- [image_warp.py](src/e2eflow/core/image_warp.py): Backward-warping via differentiable bilinear image sampling,
- [supervised.py](src/e2eflow/core/supervised.py): Supervised loss and network computation,
- [unsupervised.py](src/e2eflow/core/unsupervised.py): Unsupervised (bi-directional) multi-resolution network and loss computation,
- [train.py](src/e2eflow/core/train.py): Implementation of training with online evaluation,
- [augment.py](src/e2eflow/core/augment.py): Geometric and photometric data augmentations.
