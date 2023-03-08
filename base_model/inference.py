import argparse
import os

import cv2
import numpy as np
import torch
import yaml
from moviepy.editor import ImageSequenceClip
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from dataset import PersonalDataset
from modules.discriminator import MultiScaleDiscriminator
from modules.generator import Generator
from modules.model import GeneratorFullModel


def build_model(args, conf):
    utils.set_random_seed(conf['general']['random_seed'])
    G = Generator(conf['model'].get('arch', None), **conf['model']['generator'], **conf['model']['common'])
    utils.load_ckpt(args['ckpt'], {'generator': G}, device=args['device'], strict=True)
    G.eval()
    
    D = MultiScaleDiscriminator(**conf['model']['discriminator'], **conf['model']['common'])

    G_full = GeneratorFullModel(None, G.cuda(), D.cuda(), conf['train'], conf['model'].get('arch', None), conf=conf)
    return G_full


def build_data_loader(conf, name='personal'):
    dataset = PersonalDataset(conf, name, is_train=False)
    sampler = torch.utils.data.SequentialSampler(dataset)
    test_dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=2, drop_last=False, pin_memory=True)
    return test_dataloader


def model_forward(G_full, data):
    for key, value in data.items():
        if isinstance(value, list):
            if isinstance(value[0], (str, list)):
                continue
            data[key] = [v.cuda() for v in value]
        elif isinstance(value, str):
            continue
        else:
            data[key] = value.cuda()
    
    generated = G_full(data, stage="Full", inference=True)
    return generated


def save_images(args, generated, data):
    for j in range(len(generated['prediction'])):
        final = np.transpose(generated['prediction'][j].data.cpu().numpy(), [1, 2, 0])
        final = np.clip(final * 255, 0, 255).astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(os.path.join(args["save_dir"], "output_256", str(data['driving_name'][j])), final)


def save_to_video(args, gt_path):
    img_list = []
    for file in tqdm(gt_path):
        img_name = file.split("/")[-1]
        final_img_path = os.path.join(args["save_dir"], 'output_256', img_name)
        final_img = cv2.resize(cv2.cvtColor(cv2.imread(final_img_path), cv2.COLOR_BGR2RGB), (256, 256))
        img_list.append(final_img)
    
    imgseqclip = ImageSequenceClip(img_list, 23.98)
    imgseqclip.write_videofile(os.path.join(args["save_dir"], "out_256.mp4"), logger=None)


def evaluation(args, conf):
    os.makedirs(os.path.join(args["save_dir"], "output_256"), exist_ok=True)
    
    G_full = build_model(args, conf)
    name = conf["dataset"]["train_data"][0]
    test_dataloader = build_data_loader(conf, name)

    print("Evaluation using {} images.".format(len(test_dataloader.dataset)))
    
    for data in tqdm(test_dataloader):
        with torch.inference_mode():
            generated = model_forward(G_full, data)

        save_images(args, generated, data)

    save_to_video(args, test_dataloader.dataset.data['imgs'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Animation for Immersive Meeting Evaluation Scripts',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--save_dir', type=str, default='../../', help='image save dir')
    parser.add_argument('--ckpt', type=str, help='load checkpoint path')
    parser.add_argument("--config", type=str, default="config/test.yaml", help="path to config")

    args = vars(parser.parse_args())
    args['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    with open(args['config']) as f:
        conf = yaml.safe_load(f)

    evaluation(args, conf)
