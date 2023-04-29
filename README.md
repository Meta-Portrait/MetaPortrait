# MetaPortrait

![Teaser](./docs/Teaser.png)

This repo is the official implementation of "MetaPortrait: Identity-Preserving Talking Head Generation with Fast Personalized Adaptation" (CVPR 2023).

By [Bowen Zhang](http://home.ustc.edu.cn/~zhangbowen)\*, [Chenyang Qi](https://chenyangqiqi.github.io)\*, [Pan Zhang](https://panzhang0212.github.io), [Bo Zhang](https://bo-zhang.me/), [HsiangTao Wu](https://dl.acm.org/profile/81487650131), [Dong Chen](http://www.dongchen.pro/), [Qifeng Chen](https://cqf.io), [Yong Wang](http://en.auto.ustc.edu.cn/2021/0616/c26828a513186/page.htm) and [Fang Wen](https://www.microsoft.com/en-us/research/people/fangwen/).

[Paper](https://arxiv.org/abs/2212.08062) | [Project Page](https://meta-portrait.github.io/) | [Code](https://github.com/Meta-Portrait/MetaPortrait)

## Abstract

> In this work, we propose an ID-preserving talking head generation framework, which advances previous methods in two aspects. First, as opposed to interpolating from sparse flow, we claim that dense landmarks are crucial to achieving accurate geometry-aware flow fields. Second, inspired by face-swapping methods, we adaptively fuse the source identity during synthesis, so that the network better preserves the key characteristics of the image portrait. Although the proposed model surpasses prior generation fidelity on established benchmarks, to further make the talking head generation qualified for real usage, personalized fine-tuning is usually needed. However, this process is rather computationally demanding that is unaffordable to standard users. To solve this, we propose a fast adaptation model using a meta-learning approach. The learned model can be adapted to a high-quality personalized model as fast as 30 seconds. Last but not the least, a spatial-temporal enhancement module is proposed to improve the fine details while ensuring temporal coherency. Extensive experiments prove the significant superiority of our approach over the state of the arts in both one-shot and personalized settings.

## Todo

- [x] Release the inference code of base model and temporal super-resolution model
- [x] Release the training code of base model
- [x] Release the training code of super-resolution model

## Setup Environment

```bash
git clone https://github.com/Meta-Portrait/MetaPortrait.git
cd MetaPortrait
conda env create -f environment.yml
conda activate meta_portrait_base

# if you use GPU that only support cuda11, you may reinstall the torch build with cu11
# pip uninstall torch
# pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

## Base Model

### Inference Base Model

Download the [checkpoint of base model](https://drive.google.com/file/d/1Kmdv3w6N_we7W7lIt6LBzqRHwwy1dBxD/view?usp=share_link) and put it to `base_model/checkpoint`. We provide [preprocessed example data for inference](https://drive.google.com/file/d/166eNbabM6TeJVy7hxol2gL1kUGKHi3Do/view?usp=share_link), you could download the data, unzip and put it to `data`. The directory structure should like this:

```
data
├── 0
│   ├── imgs
│   │   ├── 00000000.png
│   │   ├── ...
│   ├── ldmks
│   │   ├── 00000000_ldmk.npy
│   │   ├── ...
│   └── thetas
│       ├── 00000000_theta.npy
│       ├── ...
├── src_0_id.npy
├── src_0_ldmk.npy
├── src_0.png
├── src_0_theta.npy
└── src_map_dict.pkl
```

You could generate results of self reconstruction on 256x256 resolution by running:
```bash
cd base_model
python inference.py --save_dir result --config config/meta_portrait_256_eval.yaml --ckpt checkpoint/ckpt_base.pth.tar
```

### Train Base Model from Scratch

Train the warping network first using the following command:
```bash
cd base_model
python main.py --config config/meta_portrait_256_pretrain_warp.yaml --fp16 --stage Warp --task Pretrain
```

Then, modify the path of `warp_ckpt` in `config/meta_portrait_256_pretrain_full.yaml` and joint train the warping network and refinement network using the following command:
```bash
python main.py --config config/meta_portrait_256_pretrain_full.yaml --fp16 --stage Full --task Pretrain
```

### Meta Training for Faster Personalization of Base Model

You could start from the standard pretrained checkpoint and further optimize the personalized adaptation speed of the model by utilizing meta-learning using the following command:
```bash
python main.py --config config/meta_portrait_256_pretrain_meta_train.yaml --fp16 --stage Full --task Meta --remove_sn --ckpt /path/to/standard_pretrain_ckpt
```

## Temporal Super-resolution Model

Set the root path to [sr_model](sr_model)

### Data and checkpoint

Download the [checkpoint](https://github.com/Meta-Portrait/MetaPortrait/releases/download/v0.0.1/temporal_gfpgan.pth) using the bash command

```bash
cd sr_model
bash download_sr.sh
```

Unzip the package and keep the file structure like

```
pretrained_ckpt
├── temporal_gfpgan.pth
├── GFPGANv1.3.pth
...
data
├── HDTF_warprefine
│   ├── gt
│   ├── lq
│   ├── ...
Basicsr
Experimental_root
options
```

### Installation Bash command

```bash

# Install a modified basicsr - https://github.com/xinntao/BasicSR

cd Basicsr
pip install -r requirements.txt
python setup.py develop

# Install facexlib - https://github.com/xinntao/facexlib
# We use face detection and face restoration helper in the facexlib package
pip install facexlib
cd ..
pip install -r requirements.txt
# python setup.py develop
```

### Quick Inference

ckpt for inference: pretrained_ckpt/temporal_gfpgan.pth
<!-- 
Example code to conduct face temporal super-resolution:
```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 Experimental_root/test.py -opt options/test/same_id.yml --launcher pytorch
``` -->

Enhance the result from our base model without calculating the metrics:
```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 Experimental_root/test.py -opt options/test/same_id_demo.yml --launcher pytorch
```
You may adjust the ```nproc_per_node``` to the number of GPUs on your own machine.
Finally, check the result at ```results/temporal_gfpgan_same_id_temporal_super_resolution```.

### Demo training

In the paper result, we train on the training split of hdtf dataset. Here we first provide a demo training code to train on the small demo dataset

```bash
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 Experimental_root/train.py -opt options/train/train_sr_hdtf.yml --launcher pytorch
```
The intermediate result can be check at `/home/cqiaa/talkinghead/MetaPortrait/sr_model/experiments/train_sr_hdtf/visualization/00000001/hdtf_random`

## Citing MetaPortrait

```
@misc{zhang2022metaportrait,
      title={MetaPortrait: Identity-Preserving Talking Head Generation with Fast Personalized Adaptation}, 
      author={Bowen Zhang and Chenyang Qi and Pan Zhang and Bo Zhang and HsiangTao Wu and Dong Chen and Qifeng Chen and Yong Wang and Fang Wen},
      year={2022},
      eprint={2212.08062},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements

This code borrows heavily from [FOMM](https://github.com/AliaksandrSiarohin/first-order-model), thanks the authors for sharing their code and models.

## Maintenance

This is the codebase for our research work. Please open a GitHub issue for any help. If you have any questions regarding the technical details, feel free to contact [zhangbowen@mail.ustc.edu.cn](zhangbowen@mail.ustc.edu.cn) or [cqiaa@connect.ust.hk](cqiaa@connect.ust.hk).