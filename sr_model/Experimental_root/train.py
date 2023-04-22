# flake8: noqa
import sys
import os.path as osp
root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
sys.path.append(root_path)

from basicsr.train import train_pipeline


import Experimental_root.dataset
import Experimental_root.archs
import Experimental_root.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
