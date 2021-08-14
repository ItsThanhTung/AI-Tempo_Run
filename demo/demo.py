# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time

import cv2
import tqdm
from adet.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from predictor import VisualizationDemo
import numpy as np

# constants
WINDOW_NAME = "COCO detections"

def Get1LineResult(image_name, point):
  return str(image_name) + "," + str(point[0][0]) + ","  + str(point[0][1])  + "," + str(point[1][0]) + ","  + str(point[1][1])  + "," + str(point[2][0]) + ","  + str(point[2][1])  + "," + str(point[3][0]) + ","  + str(point[3][1]) + "\n"  

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return np.int32(rect)




def Submit(path, points, out_file):
  image_name = path.split("/")[-1]
  for point in points: 
    rect = order_points(point)
    out_file.write(Get1LineResult(image_name, rect))



def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output_file",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.37,
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
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    txt_file_path = args.output_file
    with open(txt_file_path, 'w') as file_result:
      if args.input:
          if os.path.isdir(args.input[0]):
              args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
          elif len(args.input) == 1:
              args.input = glob.glob(os.path.expanduser(args.input[0]))
              assert args.input, "The input path(s) was not found"
          for path in tqdm.tqdm(args.input, disable= None):
              # use PIL, to be consistent with evaluation
              img = read_image(path, format="BGR")
              predictions, points = demo.run_on_image(img)
              if len(predictions["instances"]) > 0:
                Submit(path, points, file_result)
            #   logger.info(
            #       "{}: detected {} instances".format(
            #           path, len(predictions["instances"]))
            #   )
    file_result.close()
