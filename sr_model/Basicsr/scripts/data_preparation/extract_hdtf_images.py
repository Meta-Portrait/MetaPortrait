#%%
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import cv2, glob
import torch, yaml
import numpy as np
import pickle
import zipfile
from io import BytesIO
from moviepy.editor import VideoFileClip, ImageSequenceClip
from PIL import Image, ImageDraw

# from demo_inpainting import ImmersiveMeeting
# from utils.inference_utils import BilateralSmoothOp_ldmks_np

videos_dir = '/home/v-chenyangqi/v-chenyangqi/data/HDTF'

videos = sorted(glob.glob(os.path.join(videos_dir, '*.mp4'), recursive=True))
print("Processing following videos:")
print(videos)
import os
# save_frame_id = 0
#%%
root = '/home/v-chenyangqi/v-chenyangqi/data/HDTF_images_100'
src_map_dict = {}
save_freq=1
save_format='raw'
zip_fd = None
resolution_list = [256,512]
list_info = []
for i, video in enumerate(videos):
    print(video)
    cap = cv2.VideoCapture(video)
    save_frame_id = 0

    # if save_format == 'raw':
        # save_session_dir = os.path.join(os.path.dirname(video), str(i))
        # if not os.path.exists(algo.save_session_dir):
        #     os.mkdir(algo.save_session_dir)

        # os.makedirs(os.path.join(algo.save_session_dir, 'imgs'), exist_ok=True)
        # os.makedirs(os.path.join(algo.save_session_dir, 'ldmks'), exist_ok=True)
        # os.makedirs(os.path.join(algo.save_session_dir, 'thetas'), exist_ok=True)
    # else:
    #     if i % 50 == 0:
    #         if zip_fd is not None:
    #             zip_fd.close()
    #         zip_fd = zipfile.ZipFile(os.path.join(os.path.dirname(video), "data_{}.zip".format(i // 50)), "w")

    while(cap.isOpened()):
        ret, frameInput = cap.read()
        if ret == True:

            print(save_frame_id)

            # with torch.no_grad():
            if save_frame_id % 10 == 0:
                print(save_frame_id)
            if save_frame_id % save_freq == 0:
                print(f"{root}/{i:03}/src/{save_frame_id:08}.png")
                os.makedirs(f"{root}/origin/{i:03}", exist_ok=True)
                cv2.imwrite(f"{root}/origin/{i:03}/{save_frame_id:08}.png", frameInput) # Save the first frame with full size
                for resolution in resolution_list:
                    image_resize = np.array(
                                Image.fromarray(frameInput).resize((resolution, resolution), Image.BICUBIC)
                                )
                    src_map_dict[f"{save_frame_id:08}.png"] = i
                    os.makedirs(f"{root}/{str(resolution)}/{i:03}", exist_ok=True)
                    # as source image for training and inference.
                    cv2.imwrite(f"{root}/{str(resolution)}/{i:03}/{save_frame_id:08}.png", image_resize) # Save the first frame with full size as source image for training and inference.
                # cv2.imwrite(f"{root}/src_{i}.png", cv2.resize(frame_info['img_c'], (args.imsize, args.imsize)))
            save_frame_id += 1
            if save_frame_id >=100:
                break

        else:
            print(ret)
            break



f = open(os.path.join(root, "src_map_dict.pkl"), "wb")
pickle.dump(src_map_dict, f)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--videos_dir",
#         type=str,
#         required=True,
#         help="assign videos directory to be processed",
#     )
#     parser.add_argument(
#         "--save_dir", type=str, required=True, help="path to save output frames"
#     )
#     parser.add_argument(
#         "--imsize", type=int, required=False, default=512, help="img size"
#     )

#     parser.add_argument(
#         "--ckpt",
#         required=False,
#         default=None,
#         help="ckpt path",
#     )
#     parser.add_argument(
#         "--config",
#         type=str,
#         required=False,
#         default="config/personal_inpainting_concatN.yaml",
#         help="config path",
#     )
#     parser.add_argument(
#         "--save_freq", type=int, required=False, default=1, help="save freq"
#     )
#     parser.add_argument(
#         "--save_format",
#         type=str,
#         required=False,
#         default="raw",
#         help="save data in raw | zip format",
#     )

#     args = parser.parse_args()
#     process_data(args=args)

# %%
