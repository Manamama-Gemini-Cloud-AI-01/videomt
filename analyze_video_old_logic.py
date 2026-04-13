
import argparse
import cv2
import os
import torch
import tqdm
import sys
import numpy as np

# Add project root and visualization folder to path for imports
sys.path.insert(1, os.path.join(sys.path[0], '.'))
sys.path.insert(1, os.path.join(sys.path[0], 'visualization'))

from videomt import DEVICE, add_videomt_config
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from visualization.predictor import VisualizationDemo_windows

def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_videomt_config(cfg)
    
    # Resolve paths relative to project root (sys.path[0])
    config_path = args.config_file
    if not os.path.isabs(config_path):
        config_path = os.path.join(sys.path[0], config_path)
        
    weights_path = args.weights
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(sys.path[0], weights_path)

    cfg.merge_from_file(config_path)
    cfg.merge_from_list(["MODEL.WEIGHTS", weights_path] + args.opts)
    
    # Use central DEVICE definition
    cfg.MODEL.DEVICE = DEVICE
    
    cfg.freeze()
    return cfg

def main():
    parser = argparse.ArgumentParser(description="Analyze any video file with VidEoMT")
    parser.add_argument("--config-file", default="configs/ytvis19/videomt/vit-base/videomt_online_ViTB.yaml")
    parser.add_argument("--input-video", required=True, help="Path to input video file")
    parser.add_argument("--output-video", required=False, help="Path to save output video")
    parser.add_argument("--weights", default="weights/yt_2019_vit_base_58.2.pth")
    parser.add_argument("--window-size", type=int, default=10, help="Temporal window size")
    parser.add_argument("--sample-rate", type=int, default=10, help="Process every Nth frame (decimation)")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[], help="Modify config options")
    
    args = parser.parse_args()

    if args.output_video is None:
        args.output_video = args.input_video + "_videomt_analyzed.mp4"
    
    
    setup_logger(name="fvcore")
    logger = setup_logger()
    
    cfg = setup_cfg(args)
    demo = VisualizationDemo_windows(cfg)

    # Open video
    cap = cv2.VideoCapture(args.input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate output FPS based on sampling
    output_fps = orig_fps / args.sample_rate
    
    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, output_fps, (width, height))

    processed_frames_total = num_frames // args.sample_rate
    print(f"Processing video: {args.input_video}")
    print(f"Original: {num_frames} frames @ {orig_fps:.2f} FPS")
    print(f"Sampling every {args.sample_rate}th frame -> Output @ {output_fps:.2f} FPS on {DEVICE}")
    
    vid_frames = []
    pbar = tqdm.tqdm(total=processed_frames_total)
    
    frame_idx = 0
    sampled_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % args.sample_rate == 0:
            vid_frames.append(frame) # frame is BGR
            sampled_count += 1
            
            # If window is full or it's the last possible sampled frame
            if len(vid_frames) == args.window_size or frame_idx + args.sample_rate >= num_frames:
                with torch.amp.autocast(device_type=DEVICE):
                    # keep=True propagates temporal memory across windows
                    predictions, visualized_output = demo.run_on_video(vid_frames, keep=(sampled_count > len(vid_frames)))
                
                for vis_frame in visualized_output:
                    res_frame = cv2.cvtColor(np.array(vis_frame.get_image()), cv2.COLOR_RGB2BGR)
                    out.write(res_frame)
                
                vid_frames = []
                pbar.update(sampled_count if sampled_count < args.window_size else args.window_size)
        
        frame_idx += 1

    cap.release()
    out.release()
    pbar.close()
    print(f"\nDone! Saved to {args.output_video}")

if __name__ == "__main__":
    main()
