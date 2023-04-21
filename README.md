# MetaPortrait

![Teaser](./docs/Teaser.png)

This repo is the official implementation of "MetaPortrait: Identity-Preserving Talking Head Generation with Fast Personalized Adaptation" (CVPR 2023).

By [Bowen Zhang](http://home.ustc.edu.cn/~zhangbowen)\*, [Chenyang Qi](https://chenyangqiqi.github.io)\*, [Pan Zhang](https://panzhang0212.github.io), [Bo Zhang](https://bo-zhang.me/), [HsiangTao Wu](https://dl.acm.org/profile/81487650131), [Dong Chen](http://www.dongchen.pro/), [Qifeng Chen](https://cqf.io), [Yong Wang](http://en.auto.ustc.edu.cn/2021/0616/c26828a513186/page.htm) and [Fang Wen](https://www.microsoft.com/en-us/research/people/fangwen/).

[Paper](https://arxiv.org/abs/2212.08062) | [Project Page](https://meta-portrait.github.io/) | [Code](https://github.com/Meta-Portrait/MetaPortrait)

## Abstract

> In this work, we propose an ID-preserving talking head generation framework, which advances previous methods in two aspects. First, as opposed to interpolating from sparse flow, we claim that dense landmarks are crucial to achieving accurate geometry-aware flow fields. Second, inspired by face-swapping methods, we adaptively fuse the source identity during synthesis, so that the network better preserves the key characteristics of the image portrait. Although the proposed model surpasses prior generation fidelity on established benchmarks, to further make the talking head generation qualified for real usage, personalized fine-tuning is usually needed. However, this process is rather computationally demanding that is unaffordable to standard users. To solve this, we propose a fast adaptation model using a meta-learning approach. The learned model can be adapted to a high-quality personalized model as fast as 30 seconds. Last but not the least, a spatial-temporal enhancement module is proposed to improve the fine details while ensuring temporal coherency. Extensive experiments prove the significant superiority of our approach over the state of the arts in both one-shot and personalized settings.

## Todo

- [x] Release the inference code of base model
- [x] Release the training code of base model
- [ ] Release the training and inference code of sr model 

## Setup Environment

```bash
git clone https://github.com/Meta-Portrait/MetaPortrait.git
cd MetaPortrait
conda env create -f environment.yml
conda activate meta_portrait_base
```

## Inference Base Model

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
python inference.py --save_dir /path/to/output --config config/meta_portrait_256_eval.yaml --ckpt checkpoint/ckpt_base.pth.tar
```

## Train Base Model from Scratch

Train the warping network first using the following command:
```bash
cd base_model
python main.py --config config/meta_portrait_256_pretrain_warp.yaml --fp16 --stage Warp --task Pretrain
```

Then, modify the path of `warp_ckpt` in `config/meta_portrait_256_pretrain_full.yaml` and joint train the warping network and refinement network using the following command:
```bash
python main.py --config config/meta_portrait_256_pretrain_full.yaml --fp16 --stage Full --task Pretrain
```

## Meta Training for Faster Personalization of Base Model

You could start from the standard pretrained checkpoint and further optimize the personalized adaptation speed of the model by utilizing meta-learning using the following command:
```bash
python main.py --config config/meta_portrait_256_pretrain_meta_train.yaml --fp16 --stage Full --task Meta --remove_sn --ckpt /path/to/standard_pretrain_ckpt
```

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