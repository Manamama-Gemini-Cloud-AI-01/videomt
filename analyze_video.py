import argparse
import cv2
import os
import torch
import tqdm
import sys
import numpy as np


sys.path.insert(1, os.path.join(sys.path[0], '.'))
sys.path.insert(1, os.path.join(sys.path[0], 'visualization'))

from videomt import DEVICE, add_videomt_config
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from visualization.predictor import VisualizationDemo_windows
from huggingface_hub import hf_hub_download


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_videomt_config(cfg)

    config_path = args.config_file
    if not os.path.isabs(config_path):
        config_path = os.path.join(sys.path[0], config_path)

    weights_path = args.weights
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(sys.path[0], weights_path)

    # Auto-download if weight file is missing
    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}. Attempting to download from Hugging Face...")
        repo_id = "tue-mps/VidEoMT"
        filename = os.path.basename(weights_path)
        try:
            downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=os.path.dirname(weights_path))
            print(f"Successfully downloaded weights to {downloaded_path}")
            weights_path = downloaded_path
        except Exception as e:
            print(f"Error downloading weights: {e}")
            sys.exit(1)

    cfg.merge_from_file(config_path)
    cfg.merge_from_list(["MODEL.WEIGHTS", weights_path] + args.opts)
    cfg.MODEL.DEVICE = DEVICE
    cfg.freeze()
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="configs/ytvis19/videomt/vit-base/videomt_online_ViTB.yaml")
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output-video", required=False)
    parser.add_argument("--weights", default="weights/yt_2019_vit_base_58.2.pth")
    parser.add_argument("--sample-rate", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[])

    args = parser.parse_args()

    if args.output_video is None:
        args.output_video = args.input_video + "_videomt_analyzed.mp4"

    setup_logger(name="fvcore")
    logger = setup_logger()

    cfg = setup_cfg(args)
    demo = VisualizationDemo_windows(cfg)

    cap = cv2.VideoCapture(args.input_video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_fps = orig_fps / args.sample_rate if orig_fps > 0 else 0

    estimated_frames = (num_frames + args.sample_rate - 1) // args.sample_rate

    print(f"VideoMT Analyzer, version 2.2. Recent change: logic reworked to rationalize; pip install missing bits.")
    print(f"Processing video: {args.input_video}")
    print(f"Resolution: {width}x{height}")
    print(f"Original: {num_frames} frames @ {orig_fps:.2f} FPS")
    print(f"Sampling every {args.sample_rate}th frame")
    print(f"Estimated processed frames: {estimated_frames}")
    print(f"Effective output FPS: {output_fps:.2f}")
    print(f"Device: {DEVICE}")


    print(f"If needed:")
    print(f"pip install --user -r requirements.txt") 


    print(f"pip install --user --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'")                                                              
    print(f"pip install --user --no-build-isolation git+https://github.com/cocodataset/panopticapi.git")                                                                                                      

    print(f"")


    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        args.output_video,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps / args.sample_rate,
        (width, height)
    )





    pbar = tqdm.tqdm(total=estimated_frames)

    frame_idx = 0
    sampled_count = 0
    vid_frames = []

    with torch.inference_mode():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % args.sample_rate == 0:
                vid_frames.append(frame)
                sampled_count += 1

                if len(vid_frames) >= args.window_size:
                    with torch.amp.autocast(device_type=DEVICE):
                        preds, vis_outputs = demo.run_on_video(vid_frames, keep=(sampled_count > len(vid_frames)))
                    for vis in vis_outputs:
                        vis_frame = cv2.cvtColor(np.array(vis.get_image()), cv2.COLOR_RGB2BGR)
                        out.write(vis_frame)
                    pbar.update(len(vid_frames))
                    vid_frames = []

            frame_idx += 1

        if len(vid_frames) > 0:
            with torch.amp.autocast(device_type=DEVICE):
                preds, vis_outputs = demo.run_on_video(vid_frames, keep=(sampled_count > len(vid_frames)))
            for vis in vis_outputs:
                vis_frame = cv2.cvtColor(np.array(vis.get_image()), cv2.COLOR_RGB2BGR)
                out.write(vis_frame)
            pbar.update(len(vid_frames))

    cap.release()
    out.release()
    pbar.close()

    print(f"Done: {args.output_video}")


if __name__ == "__main__":
    main()
