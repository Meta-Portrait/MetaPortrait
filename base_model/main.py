import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.utils.data import DataLoader

import utils
from dataset import PersonalDataset, PersonalMetaDataset
from modules.discriminator import MultiScaleDiscriminator
from modules.generator import Generator
from train_ddp import train_ddp


def get_dataset(conf, name, is_train=True):
    if name == 'personalized':
        return PersonalDataset(conf, name, is_train=is_train)
    elif name == 'meta':
        return PersonalMetaDataset(conf, name, is_train=is_train)
    else:
        raise Exception("Unsupported dataset type: {}".format(name))


def get_params():

    parser = argparse.ArgumentParser(description='Image Animation for Immersive Meeting Training Scripts',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ckpt', type=str, help='load checkpoint path')
    parser.add_argument("--config", type=str, default="config/test.yaml", help="path to config")
    parser.add_argument('--remove_sn', action='store_true')
    parser.add_argument("--fp16", action='store_true', help="Whether to use fp16")
    parser.add_argument("--stage", type=str, default="Full", help="Full | Warp")
    parser.add_argument("--task", type=str, default="Meta", help="Meta | Pretrain | Eval")
    parser.add_argument("--port", type=int, default=23456, help="Running port for DDP")

    args = parser.parse_args()

    return args

def main(rank, args):
    args = vars(args)
    args['local_rank'] = rank

    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=args['ngpus'],
        init_method="tcp://localhost:{}".format(args['port']),
    )
    torch.cuda.set_device(rank)

    args['device'] = rank if torch.cuda.is_available() else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(args['config']) as f:
        conf = yaml.safe_load(f)
        if rank == 0:
            print("Saving checkpoints to {}".format(conf['train']['ckpt_save_path']))

    utils.set_random_seed(conf['general']['random_seed'])
    conf['dataset']['ngpus'] = 1

    G = Generator(conf['model'].get('arch', None), **conf['model']['generator'], **conf['model']['common'])
    D = MultiScaleDiscriminator(**conf['model']['discriminator'], **conf['model']['common'])
    G = G.to(args['device'])
    D = D.to(args['device'])

    train_data_list = []
    for name in conf['dataset']['train_data']:
        data = get_dataset(conf, name)
        print("Dataset length: {}".format(len(data)))
        train_data_list.append(data)
    train_data = data

    sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=args['ngpus'], rank=rank,)
    batch_size = int(conf['train']['batch_size'] // args['ngpus']) if conf['train']['batch_size'] > 0 else None
    dataset_train = DataLoader(train_data, batch_size=batch_size, num_workers=1, 
                            drop_last=False, pin_memory=True, sampler=sampler)
    
    models = {'generator': G, 'discriminator': D}
    datasets = {'dataset_train': dataset_train}

    train_ddp(args, conf, models, datasets)
        

if __name__ == '__main__':
    params = get_params()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    params.ngpus = torch.cuda.device_count()
    mp.spawn(main, nprocs=params.ngpus, args=(params,))
