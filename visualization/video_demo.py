# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py

import torch
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import time
import tqdm

# from torch.cuda.amp import autocast

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from videomt import add_videomt_config, DEVICE
from predictor import VisualizationDemo, VisualizationDemo_windows
from huggingface_hub import hf_hub_download
import cv2
import numpy as np


def setup_cfg(args):
	# load config from file and command-line arguments
	cfg = get_cfg()
	add_deeplab_config(cfg)
	add_videomt_config(cfg)

	weights_path = args.weights
	if not os.path.isabs(weights_path):
		weights_path = os.path.join(sys.path[0], "..", weights_path)

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

	cfg.merge_from_file(args.config_file)
	cfg.merge_from_list(["MODEL.WEIGHTS", weights_path] + args.opts)
	cfg.MODEL.DEVICE = DEVICE
	cfg.freeze()
	return cfg

def get_parser():
	parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
	parser.add_argument(
		"--config-file",
		default="configs/ytvis19/videomt/vit-base/videomt_online_ViTB.yaml",
		metavar="FILE",
		help="path to config file",
	)
	parser.add_argument(
		"--weights",
		default="weights/yt_2019_vit_base_58.2.pth",
		help="path to weights file",
	)
	parser.add_argument(
		"--input",
		help="directory of input video frames",
		required=True,
	)
	parser.add_argument(
		"--output",
		help="directory to save output frames",
		required=True,
	)
	parser.add_argument(
		"--confidence_threshold",
		type=float,
		default=0.5,
		help="Minimum score for instance predictions to be shown",
	)
	parser.add_argument(
		"--windows_size",
		type=int,
		default=20,
		help="Windows size for semi-offline mode",
	)
	parser.add_argument(
		"--opts",
		help="Modify config options using the command-line 'KEY VALUE' pairs",
		default=[],
		nargs=argparse.REMAINDER,
	)
	return parser

if __name__ == "__main__":
	mp.set_start_method("spawn", force=True)
	args = get_parser().parse_args()
	setup_logger(name="fvcore")
	logger = setup_logger()
	logger.info("Arguments: " + str(args))

	cfg = setup_cfg(args)

	demo = VisualizationDemo_windows(cfg)

	assert args.input and args.output

	video_root = args.input
	output_root = args.output
	score_threshold = args.confidence_threshold
	windows_size = args.windows_size

	start_time = time.time()
	vid_frames = []
	instances = set()

	if os.path.isdir(video_root):
		os.makedirs(output_root, exist_ok=True)
		frames_path = glob.glob(os.path.expanduser(os.path.join(video_root, '*.???')))
		frames_path.sort()
		if windows_size == -1:
			windows_size = len(frames_path)
		
		_frames_path = []
		for i, path in enumerate(tqdm.tqdm(frames_path)):
			img = read_image(path, format="BGR")
			_frames_path.append(path)
			vid_frames.append(img)
			if len(vid_frames) == windows_size or i == len(frames_path) - 1:
				with torch.amp.autocast(device_type=DEVICE):
					predictions, visualized_output = demo.run_on_video(vid_frames, keep=(i >= windows_size))
				
				for path, _vis_output in zip(_frames_path, visualized_output):
					out_filename = os.path.join(output_root, os.path.basename(path))
					_vis_output.save(out_filename)
				
				if 'pred_ids' in predictions.keys():
					for id in predictions['pred_ids']:
						instances.add(id)
				vid_frames = []
				_frames_path = []

	else:
		# Assume it's a video file
		cap = cv2.VideoCapture(video_root)
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = cap.get(cv2.CAP_PROP_FPS)
		num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		
		out = cv2.VideoWriter(
			output_root,
			cv2.VideoWriter_fourcc(*'mp4v'),
			fps,
			(width, height)
		)

		pbar = tqdm.tqdm(total=num_frames)
		frame_idx = 0
		while cap.isOpened():
			ret, frame = cap.read()
			if not ret:
				break
			vid_frames.append(frame)
			frame_idx += 1

			if len(vid_frames) == windows_size:
				with torch.amp.autocast(device_type=DEVICE):
					predictions, visualized_output = demo.run_on_video(vid_frames, keep=(frame_idx > windows_size))
				for vis in visualized_output:
					vis_frame = cv2.cvtColor(np.array(vis.get_image()), cv2.COLOR_RGB2BGR)
					out.write(vis_frame)
				if 'pred_ids' in predictions.keys():
					for id in predictions['pred_ids']:
						instances.add(id)
				pbar.update(len(vid_frames))
				vid_frames = []

		if len(vid_frames) > 0:
			with torch.amp.autocast(device_type=DEVICE):
				predictions, visualized_output = demo.run_on_video(vid_frames, keep=(frame_idx > windows_size))
			for vis in visualized_output:
				vis_frame = cv2.cvtColor(np.array(vis.get_image()), cv2.COLOR_RGB2BGR)
				out.write(vis_frame)
			if 'pred_ids' in predictions.keys():
				for id in predictions['pred_ids']:
					instances.add(id)
			pbar.update(len(vid_frames))
		
		cap.release()
		out.release()
		pbar.close()

	logger.info(
		"detected {} instances in {:.2f}s".format(
			len(instances), time.time() - start_time
		)
	)

