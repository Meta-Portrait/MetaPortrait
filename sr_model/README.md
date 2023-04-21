
# Temporal Super Resolution in Metaportrait

## Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)
- System: Linux + NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

### Data and checkpoint

Download the [dataset](
https://hkustconnect-my.sharepoint.com/:f:/g/personal/cqiaa_connect_ust_hk/EuZ_hj6hcERKlDgajp-mhvwBxv4D1CX6_hPO4qJlSxK_cw?e=f4CnUI)
and [checkpoint](https://hkustconnect-my.sharepoint.com/:f:/g/personal/cqiaa_connect_ust_hk/EiV7jVV_YjJMtuZDsC8pjK4BmDEEPJ0h55NqLbPLcPbXIw?e=RlHXbd).

Then keep the file structure like

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
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
# Install a modified basicsr - https://github.com/xinntao/BasicSR

cd Basicsr
pip install -r requirements.txt
python setup.py develop

# Install facexlib - https://github.com/xinntao/facexlib
# We use face detection and face restoration helper in the facexlib package
pip install facexlib
cd ..
pip install -r requirements.txt
python setup.py develop
```

## Quick Inference
ckpt for inference: pretrained_ckpt/temporal_gfpgan.pth

Example code to conduct face temporal super-resolution

```bash
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 Experimental_root/test.py -opt options/test/same_id.yml --launcher pytorch
```

## Inference other data
Edit the dataroot_lq and dataroot_gt.
Also provide lq_temp and gt_temp if your data is not in compact folder
