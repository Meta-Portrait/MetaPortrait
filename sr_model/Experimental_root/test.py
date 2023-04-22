import sys
import os.path as osp
root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
sys.path.append(root_path)
# import Experimental_root.ops
import Experimental_root.archs
import Experimental_root.models
import Experimental_root.dataset
# import Experimental_root.scripts
# import Experimental_root.losses
import Experimental_root.metrics

from basicsr import test_pipeline

if __name__ == '__main__':
    # if runtime_root is None:
        # runtime_root = osp.abspath(osp.join(__file__, osp.pardir, '..', '..'))
    # print(f'current root path: {runtime_root}')
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)