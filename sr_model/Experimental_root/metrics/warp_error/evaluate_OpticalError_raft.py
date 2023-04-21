#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2
from datetime import datetime
import numpy as np
from tqdm import tqdm

### torch lib
import torch
import torch.nn as nn

## raft lib
sys.path.append('core')
from .core.raft import RAFT
from easydict import EasyDict as edict

### custom lib
# import Experimental_root.metrics.warp_error.utils_btc as utils
# import warp_error.utils_btc as utils
from .core import utils_btc as utils
# import utils_btc as utils
from tqdm import tqdm

from basicsr.utils.registry import METRIC_REGISTRY

def flow_warping(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.to(x.device)
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)

    output = nn.functional.grid_sample(x, vgrid)
    return output

def imread(path, size=None, size_multiplier=64, target_size=None):
    """Read image from path; resize the image to size, if image size is not the same

    Args:
        path (_type_): _description_
        target_size: Tuple e.g., (128, 128)
        size (_type_, optional): crop size Defaults to None.
        size_multiplier (int, optional): _description_. Defaults to 64.

    Returns:
        final image, and the resolution before size_multiplier
    """
    img = utils.read_img(path)
    if target_size:
        img = cv2.resize(img, target_size)
    if size is not None:
        h, w = size
        img = img[:h, :w, :]

    H_orig = img.shape[0]
    W_orig = img.shape[1]

    H_sc = int(math.ceil(float(H_orig) / size_multiplier) * size_multiplier)
    W_sc = int(math.ceil(float(W_orig) / size_multiplier) * size_multiplier)

    img = cv2.resize(img, (W_sc, H_sc))

    return img, (H_orig, W_orig)

def compute_warped_error(img_t, img_s_warp, occ_mask, threshold=1, crop_boarder=0):
    if occ_mask is not None:
        valid_pixels = np.sum(occ_mask == 1)
        error_map = np.absolute(occ_mask*img_t - occ_mask * img_s_warp)
        mean_error = np.sum(np.absolute(occ_mask*img_t - occ_mask * img_s_warp)) / (valid_pixels+1e-10)
    else:
        error_map = np.absolute(img_t - img_s_warp)
        confidence_mask = error_map < threshold
        valid_pixels = np.sum(confidence_mask)
        mean_error = np.sum(error_map * confidence_mask) / valid_pixels

    return mean_error, error_map

def save_absolute_flow(fw_flow, save_path):
    # scale grid to [-1,1]
    h,w,c = fw_flow.shape
    # vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    # vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    fw_flow[:, :, 0] = fw_flow[:, :, 0] / max(w-1,1)
    fw_flow[:, :, 1] = fw_flow[:, :, 1] / max(h-1,1)
    flow_color = flow_vis.flow_to_color(fw_flow, convert_to_bgr=False)
    cv2.imwrite(save_path, flow_color)




def get_warped_error(RAFTNet, input_list, result_list, interval=1, error_type="interval", threshold=1, result_dir=None, target_res=512):
    # input_folder: folder containing input images for flow computing
    # result_folder: folder containing deflicking result
    # caluate the error between the nth result image and the warped result of the (n+interval)th result image
    device = torch.device("cuda")
    folder_mean_error = 0
    cnt = 0
    num_images = len(result_list)
    warp_error_list = []
    for idx in tqdm(range(num_images-interval)):
        # fisrt calcuate the flow and the occ mask based on input images
        if error_type == "interval":
            input_img_s_path = input_list[idx]
            input_img_t_path = input_list[idx+interval]
            processed_img_s_path = result_list[idx]
            processed_img_t_path = result_list[idx+interval]
        else: # else with the first frame
            input_img_s_path = input_list[0]
            input_img_t_path = input_list[idx+1]
            processed_img_s_path = result_list[0]
            processed_img_t_path = result_list[idx+1]

        # read the result image
        processed_img_s = utils.read_img(processed_img_s_path)
        processed_img_s = cv2.resize(processed_img_s, (target_res, target_res))

        processed_img_t = utils.read_img(processed_img_t_path)
        processed_img_t = cv2.resize(processed_img_t, (target_res, target_res))
        h,w = processed_img_s.shape[0:2]
        # h,w = 256, 256
        # Read the Flow image pairs
        input_img_s, (H_orig, W_orig) = imread(input_img_s_path, target_size=(h,w))
        input_img_t, _ = imread(input_img_t_path, target_size=(h,w))

        with torch.no_grad():
            ### convert to tensor
            input_img_s = utils.img2tensor(input_img_s).to(device) * 255.
            input_img_t = utils.img2tensor(input_img_t).to(device) * 255.
            ### compute fw flow
            _, fw_flow_torch = RAFTNet(input_img_s, input_img_t, iters=10, test_mode=True) # Return flow from image1 to image2
            # flow in Absolute coordinated; e.g., shape=torch.Size([1, 2, 256, 256]), min=-2.50, max=0.10, var=0.37, 0.08840583264827728

            fw_flow = utils.tensor2img(fw_flow_torch)
            if result_dir:
                fw_flow_rgb = utils.flow_to_rgb(fw_flow, normalize=False)
                os.makedirs(f"{result_dir}/flow", exist_ok=True)
                cv2.imwrite(f"{result_dir}/flow/test_fw_{idx:08d}.png", fw_flow_rgb*255.0)
            ### compute bw flow
            _, bw_flow_torch = RAFTNet(input_img_t, input_img_s, iters=10, test_mode=True) # Return flow from image2 to image1
            bw_flow = utils.tensor2img(bw_flow_torch)
            if result_dir:
                bw_flow_rgb = utils.flow_to_rgb(bw_flow, normalize=False)
                os.makedirs(f"{result_dir}/flow", exist_ok=True)
                cv2.imwrite(f"{result_dir}/flow/test_bw_{idx:08d}.png", bw_flow_rgb*255.0)
                # cv2.imwrite("test_bw.png", bw_flow_rgb*255.0)
        ### resize flow
        fw_flow = utils.resize_flow(fw_flow, W_out = W_orig, H_out = H_orig)
        bw_flow = utils.resize_flow(bw_flow, W_out = W_orig, H_out = H_orig)
        # save_absolute_flow(fw_flow, f'./result/flow{idx}.png')

        ### compute occlusion
        occ_mask = utils.detect_occlusion(fw_flow, bw_flow)
        occ_mask = 1 - occ_mask # convert to noc_mask
        if result_dir:
            # bw_flow_rgb = utils.flow_to_rgb(bw_flow)
            os.makedirs(f"{result_dir}/occ", exist_ok=True)
            cv2.imwrite(f"{result_dir}/occ/occ_mask_{idx:08d}.png", occ_mask*255.0)
            # cv2.imwrite("occ_mask.png", occ_mask*255.0)
        # warp the result image
        with torch.no_grad():
            processed_img_s_tensor = utils.img2tensor(processed_img_s).to(device)
            processed_img_s_warp = flow_warping(processed_img_s_tensor, utils.img2tensor(bw_flow).to(device))
            processed_img_s_warp = utils.tensor2img(processed_img_s_warp)
        if result_dir:
            # bw_flow_rgb = utils.flow_to_rgb(bw_flow)
            os.makedirs(f"{result_dir}/warped", exist_ok=True)
            os.makedirs(f"{result_dir}/flow", exist_ok=True)
            cv2.imwrite(f"{result_dir}/warped/warped_{idx:08d}.png", processed_img_s_warp[:, :, ::-1]*255.0)
            # import pdb; pdb.set_trace()
            cv2.imwrite(f"{result_dir}/flow/masked_bw_flow_{idx:08d}.png", np.expand_dims(occ_mask, axis=2)*bw_flow_rgb[:, :, ::-1]*255.0)
        # cv2.imwrite("processed_img_s_warp.png", processed_img_s_warp*255.0)
        def crop_boarder(image, boarder=2):
            if len(image.shape)==3 or len(image.shape)==2:
                return image[boarder:-boarder, boarder:-boarder]
            else:
                raise NotImplementedError
        if interval == 1:
            warp_error, error_map = compute_warped_error(crop_boarder(processed_img_t), crop_boarder(processed_img_s_warp), np.expand_dims(crop_boarder(occ_mask), axis=2))
        else:
            warp_error, error_map = compute_warped_error(processed_img_t, processed_img_s_warp, None, threshold)
        # import numpy as np
        # import torchvision.utils as tvu
        if result_dir:
            # bw_flow_rgb = utils.flow_to_rgb(bw_flow)
            os.makedirs(f"{result_dir}/error", exist_ok=True)
            cv2.imwrite(f"{result_dir}/error/error_map_{idx:08d}.png", error_map*255.0)
        # os.makedirs(f'./result/self-consistency', exist_ok=True)
        # cv2.imwrite(f'./result/self-consistency/error_map_short{idx}.png',error_map*255.0)
        # print(f"warping error {warp_error}")
        folder_mean_error += warp_error
        warp_error_list.append(warp_error)
        cnt += 1
        torch.cuda.empty_cache()
    print("delete network and cache memory")
    del RAFTNet
    del input_img_s
    del input_img_t
    del fw_flow_torch
    del bw_flow_torch
    del _
    del processed_img_s_warp

    torch.cuda.empty_cache()
    folder_mean_error = folder_mean_error/float(cnt)
    return folder_mean_error, warp_error_list

def get_warped_error_torch(RAFTNet, input_list, result_list, interval=1, error_type="interval", threshold=1, result_dir=None, target_res=512):
    # input_folder: folder containing input images for flow computing
    # result_folder: folder containing deflicking result
    # caluate the error between the nth result image and the warped result of the (n+interval)th result image
    device = torch.device("cuda")
    folder_mean_error = 0
    cnt = 0
    num_images = len(result_list)
    warp_error_list = []
    for idx in tqdm(range(num_images-interval)):
        # fisrt calcuate the flow and the occ mask based on input images
        if error_type == "interval":
            input_img_s_path = input_list[idx]
            input_img_t_path = input_list[idx+interval]
            processed_img_s_path = result_list[idx]
            processed_img_t_path = result_list[idx+interval]
        else: # else with the first frame
            input_img_s_path = input_list[0]
            input_img_t_path = input_list[idx+1]
            processed_img_s_path = result_list[0]
            processed_img_t_path = result_list[idx+1]

        # read the result image
        processed_img_s = utils.read_img(processed_img_s_path)
        processed_img_s = cv2.resize(processed_img_s, (target_res, target_res))

        processed_img_t = utils.read_img(processed_img_t_path)
        processed_img_t = cv2.resize(processed_img_t, (target_res, target_res))
        h,w = processed_img_s.shape[0:2]
        # h,w = 256, 256
        # Read the Flow image pairs
        input_img_s, (H_orig, W_orig) = imread(input_img_s_path, target_size=(h,w))
        input_img_t, _ = imread(input_img_t_path, target_size=(h,w))

        with torch.no_grad():
            ### convert to tensor
            input_img_s = utils.img2tensor(input_img_s).to(device) * 255.
            input_img_t = utils.img2tensor(input_img_t).to(device) * 255.
            ### compute fw flow
            _, fw_flow_torch = RAFTNet(input_img_s, input_img_t, iters=10, test_mode=True) # Return flow from image1 to image2
            # flow in Absolute coordinated; e.g., shape=torch.Size([1, 2, 256, 256]), min=-2.50, max=0.10, var=0.37, 0.08840583264827728

            fw_flow = utils.tensor2img(fw_flow_torch)
            if result_dir:
                fw_flow_rgb = utils.flow_to_rgb(fw_flow, normalize=False)
                os.makedirs(f"{result_dir}/flow", exist_ok=True)
                cv2.imwrite(f"{result_dir}/flow/test_fw_{idx:08d}.png", fw_flow_rgb*255.0)
            ### compute bw flow
            _, bw_flow_torch = RAFTNet(input_img_t, input_img_s, iters=10, test_mode=True) # Return flow from image2 to image1
            bw_flow = utils.tensor2img(bw_flow_torch)
            if result_dir:
                bw_flow_rgb = utils.flow_to_rgb(bw_flow, normalize=False)
                os.makedirs(f"{result_dir}/flow", exist_ok=True)
                cv2.imwrite(f"{result_dir}/flow/test_bw_{idx:08d}.png", bw_flow_rgb*255.0)
                # cv2.imwrite("test_bw.png", bw_flow_rgb*255.0)
        ### resize flow
        fw_flow = utils.resize_flow(fw_flow, W_out = W_orig, H_out = H_orig)
        bw_flow = utils.resize_flow(bw_flow, W_out = W_orig, H_out = H_orig)
        # save_absolute_flow(fw_flow, f'./result/flow{idx}.png')

        ### compute occlusion
        occ_mask = utils.detect_occlusion(fw_flow, bw_flow)
        occ_mask = 1 - occ_mask # convert to noc_mask
        if result_dir:
            # bw_flow_rgb = utils.flow_to_rgb(bw_flow)
            os.makedirs(f"{result_dir}/occ", exist_ok=True)
            cv2.imwrite(f"{result_dir}/occ/occ_mask_{idx:08d}.png", occ_mask*255.0)
            # cv2.imwrite("occ_mask.png", occ_mask*255.0)
        # warp the result image
        with torch.no_grad():
            processed_img_s_tensor = utils.img2tensor(processed_img_s).to(device)
            processed_img_s_warp = flow_warping(processed_img_s_tensor, utils.img2tensor(bw_flow).to(device))
            processed_img_s_warp = utils.tensor2img(processed_img_s_warp)
        if result_dir:
            # bw_flow_rgb = utils.flow_to_rgb(bw_flow)
            os.makedirs(f"{result_dir}/warped", exist_ok=True)
            os.makedirs(f"{result_dir}/flow", exist_ok=True)
            cv2.imwrite(f"{result_dir}/warped/warped_{idx:08d}.png", processed_img_s_warp[:, :, ::-1]*255.0)
            # import pdb; pdb.set_trace()
            cv2.imwrite(f"{result_dir}/flow/masked_bw_flow_{idx:08d}.png", np.expand_dims(occ_mask, axis=2)*bw_flow_rgb[:, :, ::-1]*255.0)
        # cv2.imwrite("processed_img_s_warp.png", processed_img_s_warp*255.0)
        def crop_boarder(image, boarder=2):
            if len(image.shape)==3 or len(image.shape)==2:
                return image[boarder:-boarder, boarder:-boarder]
            else:
                raise NotImplementedError
        if interval == 1:
            warp_error, error_map = compute_warped_error(crop_boarder(processed_img_t), crop_boarder(processed_img_s_warp), np.expand_dims(crop_boarder(occ_mask), axis=2))
        else:
            warp_error, error_map = compute_warped_error(processed_img_t, processed_img_s_warp, None, threshold)
        # import numpy as np
        # import torchvision.utils as tvu
        if result_dir:
            # bw_flow_rgb = utils.flow_to_rgb(bw_flow)
            os.makedirs(f"{result_dir}/error", exist_ok=True)
            cv2.imwrite(f"{result_dir}/error/error_map_{idx:08d}.png", error_map*255.0)
        # os.makedirs(f'./result/self-consistency', exist_ok=True)
        # cv2.imwrite(f'./result/self-consistency/error_map_short{idx}.png',error_map*255.0)
        # print(f"warping error {warp_error}")
        folder_mean_error += warp_error
        warp_error_list.append(warp_error)
        cnt += 1
        torch.cuda.empty_cache()
    print("delete network and cache memory")
    del RAFTNet
    del input_img_s
    del input_img_t
    del fw_flow_torch
    del bw_flow_torch
    del _
    del processed_img_s_warp

    torch.cuda.empty_cache()
    folder_mean_error = folder_mean_error/float(cnt)
    return folder_mean_error, warp_error_list


@METRIC_REGISTRY.register()
def calculate_temp_warping_error(synthesized, flow, result_dir, frame_limit, **kwargs):
    # opts = parser.parse_args()
    # opts.cuda = True

    # print(opts)
    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
    # opts.device = device

    # load raft model
    raft_args = edict({'mixed_precision':True,
                    'small':False,
                    'model':f"{os.path.dirname(__file__)}/pretrained_models/raft-things.pth"})
    RAFTNet = RAFT(raft_args)
    checkpoint = torch.load(raft_args.model, map_location=lambda storage, loc: storage)
    new_checkpoint = {}
    for key, value in checkpoint.items():
        new_checkpoint[key.replace("module.",'')] = value
    RAFTNet.load_state_dict(new_checkpoint)
    del checkpoint
    del new_checkpoint
    RAFTNet = RAFTNet.to(device)
    RAFTNet.eval()

    output_dir = result_dir
    # input_root = flow
    ## print average if result already exists
    metric_filename = os.path.join(output_dir, "WarpError.txt")
    # if os.path.exists(metric_filename) and not redo:
    # # if os.path.exists(metric_filename):
    #     print("Output %s exists, skip..." % metric_filename)
    #     cmd = 'tail -n1 %s' % metric_filename
    #     utils.run_cmd(cmd)
    #     sys.exit()


    # video_full_path_list = glob.glob(synthesized+"/*")
    # clean_list = []
    # for p in video_full_path_list:
    #     if os.path.isdir(p):
    #         clean_list.append(p)
    # video_full_path_list = clean_list
    # video_list = sorted([ os.path.splitext(os.path.basename(p))[0] for p in video_full_path_list])
    # print(video_list)
    # ### start evaluation
    # err_all = np.zeros(len(video_list))
    # warp_error_dict = {}
    # for v in tqdm(range(len(video_list))):
    # video = video_list[v]

    # frame_dir = os.path.join(opts.data_dir, opts.method, opts.task, opts.dataset, video)
    # if opts.data is None:
        # frame_dir = os.path.join(opts.data_dir, opts.method, opts.task, video)
    # else:
    frame_dir = synthesized
    frame_list = sorted(glob.glob(os.path.join(frame_dir, "*g")))

    # set up flow for current video
    # input_dir = f
    input_dir = flow
    input_list = sorted(glob.glob(os.path.join(input_dir, "*g")))

    # Set up output video
    # os.path.join(opts.flow, video)
    output_dir = result_dir

    print("frame_dir", frame_dir)
    print("input_dir", input_dir)
    print("output_dir", output_dir)

    if frame_limit is not None:
        print("Limit the number of frames to %d" % frame_limit)
        frame_list = frame_list[:frame_limit]

    # compute error
    with torch.no_grad():
        error_interval, warp_error_list = get_warped_error(RAFTNet, input_list, frame_list, interval=1, error_type="interval", threshold=1, result_dir=output_dir)
    # error_long = get_warped_error(RAFTNet, input_list, frame_list, interval=1, error_type="0toT", threshold=1)

    # err_all[v] = error_interval + error_long
    # err_all[v] = error_interval
    # utils.make_video(output_dir, "*.png", os.path.join(opts.result_dir, f"{video}_video.mp4"))
    # warp_error_dict[video] = warp_error_list

    print("\nAverage Warping Error = %f\n" %(error_interval.mean()))

    # err_all = np.append(err_all, err_all.mean())
    print("Save %s" % metric_filename)
    np.savetxt(metric_filename, [error_interval], fmt="%f")
    del RAFTNet
    torch.cuda.empty_cache()
    return error_interval.mean(), warp_error_list



def calculate_temp_warping_error_video_frame(synthesized, flow, result_dir, frame_limit, **kwargs):
    # opts = parser.parse_args()
    # opts.cuda = True

    # print(opts)
    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
    # opts.device = device

    # load raft model
    raft_args = edict({'mixed_precision':False,
                    'small':False,
                    'model':f"{os.path.dirname(__file__)}/pretrained_models/raft-things.pth"})
    RAFTNet = RAFT(raft_args)
    checkpoint = torch.load(raft_args.model)
    new_checkpoint = {}
    for key, value in checkpoint.items():
        new_checkpoint[key.replace("module.",'')] = value
    RAFTNet.load_state_dict(new_checkpoint)
    del checkpoint
    RAFTNet = RAFTNet.to(device)
    RAFTNet.eval()

    output_dir = result_dir
    input_root = flow
    ## print average if result already exists
    metric_filename = os.path.join(output_dir, "WarpError.txt")
    # if os.path.exists(metric_filename) and not redo:
    # # if os.path.exists(metric_filename):
    #     print("Output %s exists, skip..." % metric_filename)
    #     cmd = 'tail -n1 %s' % metric_filename
    #     utils.run_cmd(cmd)
    #     sys.exit()


    video_full_path_list = glob.glob(synthesized+"/*")
    clean_list = []
    for p in video_full_path_list:
        if os.path.isdir(p):
            clean_list.append(p)
    video_full_path_list = clean_list
    video_list = sorted([ os.path.splitext(os.path.basename(p))[0] for p in video_full_path_list])
    print(video_list)
    ### start evaluation
    err_all = np.zeros(len(video_list))
    warp_error_dict = {}
    for v in tqdm(range(len(video_list))):
        video = video_list[v]

        # frame_dir = os.path.join(opts.data_dir, opts.method, opts.task, opts.dataset, video)
        # if opts.data is None:
            # frame_dir = os.path.join(opts.data_dir, opts.method, opts.task, video)
        # else:
        frame_dir = os.path.join(synthesized, video)
        frame_list = sorted(glob.glob(os.path.join(frame_dir, "*g")))

        # set up flow for current video
        input_dir = os.path.join(flow, video)
        input_list = sorted(glob.glob(os.path.join(input_dir, "*g")))

        # Set up output video
        # os.path.join(opts.flow, video)
        output_dir = os.path.join(result_dir, video)

        print("frame_dir", frame_dir)
        print("input_dir", input_dir)
        print("output_dir", output_dir)

        if frame_limit is not None:
            print("Limit the number of frames to %d" % frame_limit)
            frame_list = frame_list[:frame_limit]

        # compute error
        error_interval, warp_error_list = get_warped_error(RAFTNet, input_list, frame_list, interval=1, error_type="interval", threshold=1, result_dir=output_dir)
        # error_long = get_warped_error(RAFTNet, input_list, frame_list, interval=1, error_type="0toT", threshold=1)

        # err_all[v] = error_interval + error_long
        err_all[v] = error_interval
        # utils.make_video(output_dir, "*.png", os.path.join(opts.result_dir, f"{video}_video.mp4"))
        warp_error_dict[video] = warp_error_list

    print("\nAverage Warping Error = %f\n" %(err_all.mean()))

    err_all = np.append(err_all, err_all.mean())
    print("Save %s" % metric_filename)
    np.savetxt(metric_filename, err_all, fmt="%f")
    return err_all.mean(), warp_error_dict



