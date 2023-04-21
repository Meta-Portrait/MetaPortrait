#%%
"""
Input is several folder of gt, input and output from different model
output is a video, where each frame is the concatenation of same index image in each folder

"""
import os, sys
from moviepy.editor import VideoFileClip, ImageSequenceClip
import cv2
import numpy as np
from PIL import Image
from glob import glob
import yaml
# from dataset import DatasetRepeater, PersonalDataset, FramePairDataset
from tqdm import tqdm
#%%
def folder_to_concat_folder(folder_list, existing_dict=None, order_image=None, res=None, frame_num=-1):
    image_name_dict = {}
    for f in folder_list:
        image_list = sorted( glob(f+"/*.png"))
        if frame_num >0:
            image_list = image_list[0:frame_num]
        image_name_dict[f] = image_list
        print(f+str(len(image_list)))
    if existing_dict:
        image_name_dict.update(existing_dict)
    if order_image is None:
        order_image = sorted(image_name_dict.keys())
    print("order of video: ", order_image)
    first_image = cv2.cvtColor(cv2.imread(image_name_dict[order_image[0]][0]), cv2.COLOR_BGR2RGB)
    concat_frame_list = []
    if frame_num < 0:
        frame_num = len(image_name_dict[order_image[0]])
    for i in tqdm(np.arange(frame_num)):
        image_list_i = []
        for f in order_image:
            img_if = cv2.cvtColor(cv2.imread(image_name_dict[f][i]), cv2.COLOR_BGR2RGB)
        # final_img = cv2.cvtColor(cv2.imread(final_img_path), cv2.COLOR_BGR2RGB)
        # gt_img = cv2.cvtColor(cv2.imread(gt_img_path), cv2.COLOR_BGR2RGB)
            if res is not None:
                if img_if.shape[0]!=res:
                    # print(image_name_dict[f][i])
                    img_if = cv2.resize(img_if, (res, res))
            elif img_if.shape[0] != first_image.shape[0]:
                # print(image_name_dict[f][i])
                img_if = cv2.resize(img_if, (first_image.shape[0], first_image.shape[1]))
            image_list_i.append(img_if)
        concat_frame_list.append(np.concatenate(image_list_i, 1))
    return concat_frame_list

#%%

def folder_to_video(img_list, output_path):
    # img_list = []
    if isinstance(img_list[0], np.ndarray):
        img_list = img_list
    else:
        pass
        # Read the fraome from folder
        # img_list.append(np.concatenate([gt_img, deformed_img, final_img], 1))

    imgseqclip = ImageSequenceClip(img_list, 23.98)
    imgseqclip.write_videofile((output_path), logger=None)

#%%
if __name__:
    f_path = '/home/v-chenyangqi/v-chenyangqi/dense_motion_reg_general_clean/config/test/0920_GFPGAN_equalconv_256_test_10.yaml'
    with open(f_path) as f:
        conf = yaml.safe_load(f)
    # conf = yaml.safe_load(f)

    # testdata = PersonalDataset(conf, conf['dataset']['test_data'][0], is_train=False)

    input_gt_dict = {}
    # input_gt_dict['driving'] = testdata.data['imgs']
    # input_gt_dict['source'] = testdata.data['src_imgs']

    folder_list = [
            # './results/eval_Imgs/0923_GFPGAN_equalconv_256_test_10/final',
            # './results/eval_Imgs/0921_GFPGAN_equalconv_256_mulscale_lpips_amlt/final',
            # './results/eval_Imgs/0923_spade_256_perloss_warp_test_10/deformed',
            # '/home/v-chenyangqi/v-chenyangqi/GFPGAN/results/0924_test_spade_256_amlt_test_10/restored_imgs',
            '/home/v-chenyangqi/v-chenyangqi/BasicSR/datasets/data/REDS4/GT/000',
            # '/home/v-chenyangqi/v-chenyangqi/GFPGAN/results/0924_test_spade_256_amlt_test_10_woalign/restored_faces',
            # '/home/v-chenyangqi/v-chenyangqi/BasicVSR_PlusPlus/results/0924_0908_spade_256_amlt_test_10_1',
            # './results/eval_Imgs/0921_GFPGAN_equalconv_512_down_conv_amlt/final',
            # './results/eval_Imgs/0916_GFPGAN_bn_amlt/final',
            # './results/eval_Imgs/0924_0908_spade_256_amlt_test_10/final'
            ]
    # order_list = ['driving']+folder_list
    order_list = folder_list

    concat_frame_list = folder_to_concat_folder(folder_list, order_image=order_list, frame_num=99)



#%%
save_list = '/home/v-chenyangqi/v-chenyangqi/BasicSR/results/videos/REDS4_000.mp4'
folder_to_video(concat_frame_list, save_list)

# %%
