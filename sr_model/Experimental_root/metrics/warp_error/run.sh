# /home/zhanbo/remote/v-chenyangqi/GFPGAN/data/1023_train_warprefine_zerorecu_preGFPGANv13_hdtf_small_400v_10f_zero_out/hdtf_random


# GT2our-recurrent
python evaluate_OpticalError_raft.py --redo \
    --synthesized   /home/zhanbo/remote/v-chenyangqi/temporal_metric/data/1022_train_warprefine_zerorecu_preGFPGANv13_hdtf_small_400v_10f/hdtf_random \
    --flow          /home/zhanbo/remote/v-chenyangqi/temporal_metric/data/HDTF_warprefine/gt \
    --result_dir /home/zhanbo/remote/v-chenyangqi/temporal_metric/result/gt2gan2d_HDTF_warprefine_100 \
    --frame-limit 50
#     -> ./result/gt2gt/gt2gt.log
# # GT2input
# python evaluate_OpticalError_raft.py --redo \
#     --synthesized   /home/zhanbo/remote/v-chenyangqi/temporal_metric/data/HDTF_warprefine/lq \
#     --flow          /home/zhanbo/remote/v-chenyangqi/temporal_metric/data/HDTF_warprefine/gt \
#     --result_dir /home/zhanbo/remote/v-chenyangqi/temporal_metric/result/gt2input_HDTF_warprefine_100 \
#     --frame-limit 50 \ 
#     -> ./result/gt2gt/gt2gt.log


# python evaluate_WarpError_raft.py --redo \
#     --synthesized /home/zhanbo/remote/v-chenyangqi/temporal_metric/test_demo/imgs \
#     --flow /home/zhanbo/remote/v-chenyangqi/temporal_metric/test_demo/imgs \
#     --result_dir /home/zhanbo/remote/v-chenyangqi/temporal_metric/result/gt2gt_add_motion_bound_short 
# python evaluate_OpticalError_raft.py --redo \
#     --synthesized /home/zhanbo/remote/v-chenyangqi/temporal_metric/data/HDTF_warprefine/gt \
#     --flow /home/zhanbo/remote/v-chenyangqi/temporal_metric/data/HDTF_warprefine/gt \
#     --result_dir /home/zhanbo/remote/v-chenyangqi/temporal_metric/result/gt2gt_HDTF_warprefine_100 \
#     --frame-limit 50

# python evaluate_OpticalError_raft.py --redo \
#     --synthesized /home/zhanbo/remote/v-chenyangqi/temporal_metric/data/HDTF_warprefine/lq \
#     --flow /home/zhanbo/remote/v-chenyangqi/temporal_metric/data/HDTF_warprefine/lq \
#     --result_dir /home/zhanbo/remote/v-chenyangqi/temporal_metric/result/input2input_HDTF_warprefine_100 \
#     --frame-limit 50 

# python evaluate_OpticalError_raft.py --redo \
#     --synthesized /home/zhanbo/remote/v-chenyangqi/temporal_metric/test_demo/final \
#     --flow /home/zhanbo/remote/v-chenyangqi/temporal_metric/test_demo/imgs \
#     --result_dir /home/zhanbo/remote/v-chenyangqi/temporal_metric/result/gt2input_add_motion_bound_short 

# python evaluate_OpticalError_raft.py --redo \
#     --synthesized /home/zhanbo/remote/v-chenyangqi/temporal_metric/test_demo/final \
#     --flow /home/zhanbo/remote/v-chenyangqi/temporal_metric/test_demo/final \
#     --result_dir /home/zhanbo/remote/v-chenyangqi/temporal_metric/result/input2input_add_motion_bound_short 
# --task head \
# --method warprefine \
# --data_dir ./
# --list_dir
# --data




# cmd = "ffmpeg -y -loglevel error -framerate %s -i %s/%s -vcodec libx264 -pix_fmt yuv420p -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" %s" \
#             %(fps, input_dir, img_fmt, video_filename)