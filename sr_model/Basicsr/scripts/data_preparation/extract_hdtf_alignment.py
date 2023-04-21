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
from demo_inpainting import ImmersiveMeeting
from utils.inference_utils import BilateralSmoothOp_ldmks_np
from tqdm import tqdm

class Algorithm:
    def __init__(self, args):
        with open(args.config) as f:
            conf = yaml.safe_load(f)
        args = vars(args)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.algo = ImmersiveMeeting(conf, args, device)
        self.smoothOp = BilateralSmoothOp_ldmks_np(5, 5, 0.5)


def process_data(args):
    algo = Algorithm(args)
    videos = sorted(glob.glob(os.path.join(args.videos_dir, '*.mp4'), recursive=True))
    print("Processing following videos:")
    print(videos)
    algo.save_frame_id = 0
    
    root = args.save_dir
    src_map_dict = {}
    
    zip_fd = zipfile.ZipFile(os.path.join(root, "data_0.zip"), "w")
    list_info = []
    for i, video in enumerate(tqdm(videos)):
        cap = cv2.VideoCapture(video)
        cnt = 0
        
        if args.save_format == 'raw':
            algo.save_session_dir = os.path.join(root, 'data', str(i))
            os.makedirs(algo.save_session_dir, exist_ok=True)
            
            os.makedirs(os.path.join(algo.save_session_dir, 'imgs'), exist_ok=True)
            os.makedirs(os.path.join(algo.save_session_dir, 'ldmks'), exist_ok=True)
            os.makedirs(os.path.join(algo.save_session_dir, 'thetas'), exist_ok=True)
        else:
            if i % 50 == 0:
                if zip_fd is not None:
                    zip_fd.close()
                zip_fd = zipfile.ZipFile(os.path.join(root, "data_{}.zip".format(i // 50)), "w")
        
        while(cap.isOpened()):
            ret, frameInput = cap.read()
            if ret == True:
                algo.save_frame_id += 1
                with torch.no_grad():
                    if algo.save_frame_id % 10 == 0:
                        print(algo.save_frame_id)
                    if cnt == 0:
                        image_crop, bbox = algo.algo.preprocess_img(frameInput)
                        frame_info = algo.algo.predict_ldmk(image_crop, return_id=True, preprocess_personal_image=True)
                    elif cnt % args.save_freq == 0:
                        image_crop, _ = algo.algo.preprocess_img(frameInput, bbox)
                        frame_info = algo.algo.predict_ldmk(image_crop, return_id=True, preprocess_personal_image=True)
                        if args.save_format == 'raw':
                            src_map_dict[f"{algo.save_frame_id:08}.png"] = i
                        else:
                            if i in src_map_dict:
                                src_map_dict[i].append(f"{algo.save_frame_id:08}.png")
                            else:
                                src_map_dict[i] = [f"{algo.save_frame_id:08}.png"]
                
                if cnt == 0:
                    if args.save_format == 'raw':
                        cv2.imwrite(f"{root}/src.png", frameInput) # Save the first frame with full size as source image for training and inference.
                        cv2.imwrite(f"{root}/src_{i}.png", cv2.resize(frame_info['img_c'], (args.imsize, args.imsize)))
                        np.save(f"{root}/src_{i}_ldmk.npy", frame_info['ldmk'])
                        np.save(f"{root}/src_{i}_theta.npy", frame_info['theta'])
                    else:
                        _, buffer = cv2.imencode(".png", frameInput)
                        io_buf = BytesIO(buffer)
                        zip_fd.writestr("src.png", io_buf.getvalue())

                        _, buffer = cv2.imencode(".png", cv2.resize(frame_info['img_c'], (args.imsize, args.imsize)))
                        io_buf = BytesIO(buffer)
                        zip_fd.writestr(f"src_{i}.png", io_buf.getvalue())

                        buffer = BytesIO()
                        np.savez(buffer, ldmk=frame_info['ldmk'])
                        zip_fd.writestr(f"src_{i}_ldmk.npz", buffer.getvalue())

                        buffer = BytesIO()
                        np.savez(buffer, theta=frame_info['theta'])
                        zip_fd.writestr(f"src_{i}_theta.npz", buffer.getvalue())

                elif cnt % args.save_freq == 0:
                    if args.save_format == 'raw':
                        cv2.imwrite(f"{algo.save_session_dir}/imgs/{algo.save_frame_id:08}.png", cv2.resize(frame_info['img_c'], (args.imsize, args.imsize)))
                        np.save(f"{algo.save_session_dir}/ldmks/{algo.save_frame_id:08}_ldmk.npy", frame_info['ldmk'])
                        np.save(f"{algo.save_session_dir}/thetas/{algo.save_frame_id:08}_theta.npy", frame_info['theta'])
                    else:
                        _, buffer = cv2.imencode(".png", cv2.resize(frame_info['img_c'], (args.imsize, args.imsize)))
                        io_buf = BytesIO(buffer)
                        zip_fd.writestr(f"imgs/{algo.save_frame_id:08}.png", io_buf.getvalue())

                        buffer = BytesIO()
                        np.savez(buffer, ldmk=frame_info['ldmk'])
                        zip_fd.writestr(f"ldmks/{algo.save_frame_id:08}_ldmk.npz", buffer.getvalue())

                        buffer = BytesIO()
                        np.savez(buffer, theta=frame_info['theta'])
                        zip_fd.writestr(f"thetas/{algo.save_frame_id:08}_theta.npz", buffer.getvalue())

                        list_info.append(f"data_{i//50}.zip:imgs/{algo.save_frame_id:08}.png\n")
                cnt += 1
            else:
                break
    
    if args.save_format == 'zip':
        zip_fd.close()
        with open(os.path.join(root, "list_info.txt"), "w") as f:
            f.writelines(list_info)
    
    f = open(os.path.join(root, "src_map_dict.pkl"), "wb")
    pickle.dump(src_map_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos_dir",
        type=str,
        required=True,
        help="assign videos directory to be processed",
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="path to save output frames"
    )
    parser.add_argument(
        "--imsize", type=int, required=False, default=512, help="img size"
    )
    
    parser.add_argument(
        "--ckpt",
        required=False,
        default=None,
        help="ckpt path",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="config/personal_inpainting_concatN.yaml",
        help="config path",
    )
    parser.add_argument(
        "--save_freq", type=int, required=False, default=1, help="save freq"
    )
    parser.add_argument(
        "--save_format",
        type=str,
        required=False,
        default="raw",
        help="save data in raw | zip format",
    )

    args = parser.parse_args()
    process_data(args=args)
