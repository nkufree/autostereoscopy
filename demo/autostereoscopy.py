import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
from tqdm import tqdm

from torch.cuda.amp import autocast

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from vita import add_vita_config
from genvis import add_genvis_config
from predictor import VisualizationDemo
from moviepy.editor import AudioFileClip, VideoFileClip

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_vita_config(cfg)
    add_genvis_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="genvis demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/genvis/youtubevis_2019/genvis_R50_bs8_online.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--video-input", help="Path to video file.")

    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--save-frames",
        default=False,
        help="Save frame level image outputs.",
    )
    
    parser.add_argument(
        "--mode",
        default="3d",
        help="可选值为usual和3d，前者用于输出识别的对象，后者用于输出裸眼3d视频",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
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
    
    demo = VisualizationDemo(cfg, conf_thres=args.confidence_threshold)

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    if args.video_input:
        video = cv2.VideoCapture(args.video_input)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_num = video.get(cv2.CAP_PROP_FRAME_COUNT)
        cap = cv2.VideoCapture(-1)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        batch_size = 50
        frame_offset = 0
        if args.output:
            out = cv2.VideoWriter(os.path.join(args.output, "visualization.mp4"), fourcc, fps, (W, H), True)
        with tqdm(total=int(frame_num)) as pbar:
            pbar.set_description('Processing')
            is_end = False
            while not is_end:
                num = 0
                vid_frames = []
                while num < 50:
                    num += 1
                    success, frame = video.read()
                    if success:
                        vid_frames.append(frame)
                    else:
                        is_end = True
                        break
            
                start_time = time.time()
                with autocast():
                    get_result = False
                    while not get_result:
                        try:
                            predictions, visualized_output = demo.run_on_video(vid_frames, mode=args.mode)
                            get_result = True
                        except Exception as e:
                            print(e)
                            continue
                # logger.info(
                #     "detected {} instances per frame in {:.2f}s".format(
                #         len(predictions["pred_scores"]), time.time() - start_time
                #     )
                # )
                if args.output:
                    if args.save_frames:
                        for idx, _vis_output in enumerate(visualized_output):
                            out_filename = os.path.join(args.output, f"{frame_offset + idx}.jpg")
                            _vis_output.save(out_filename)
                    for _vis_output in visualized_output:
                        frame = _vis_output.get_image()[:, :, ::-1]
                        out.write(frame)
                frame_offset += batch_size
                pbar.update(batch_size)
        cap.release()
        if args.output:
            out.release()
        # 合并音视频
        audio = AudioFileClip(args.video_input)
        video = VideoFileClip(os.path.join(args.output, "visualization.mp4"))
        video_merge = video.set_audio(audio)
        video_merge.write_videofile(os.path.join(args.output, "visualization-full.mp4"))