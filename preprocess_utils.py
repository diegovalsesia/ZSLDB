import torch
from PIL import Image
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd)
from aesthetic_scorer import AestheticScorerDiff
from tqdm import tqdm
import random
from collections import defaultdict
import prompts as prompts_file
import numpy as np
import torch.utils.checkpoint as checkpoint
import wandb
import contextlib
import torchvision
from transformers import AutoProcessor, AutoModel
import sys
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel,DDIMInverseScheduler, StableDiffusionControlNetPipeline, ControlNetModel
import datetime
import hpsv2
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from accelerate.logging import get_logger    
from accelerate import Accelerator
# from absl import app, flags
# from ml_collections import config_flags
# FLAGS = flags.FLAGS
# config_flags.DEFINE_config_file("config", "config/align_prop.py", "Training configuration.")
# from accelerate.utils import set_seed, ProjectConfiguration
# logger = get_logger(__name__)
import ml_collections
import matplotlib.pyplot as plt
import scipy.io
import cv2

tote = torchvision.transforms.ToTensor()
topil = torchvision.transforms.ToPILImage()


def depth_to_canny(depth, canny_kernel=8):
    image = topil(depth[0])
    image = np.array(image)[:,:,np.newaxis]
    image = np.repeat(image, 3 , axis=-1)
    # print(image.shape)
    image = cv2.Canny(image, canny_kernel, canny_kernel)    #for pandas
    # image = cv2.Canny(image, 5, 5)    #for books
    image = image[:, :, None]
    np_image = np.concatenate([image, image, image], axis=2)
    canny = Image.fromarray(np_image)

    return tote(canny).unsqueeze(0)

def image_to_canny(pimage, canny_kernel=90):
    image = topil(pimage[0])
    image = np.array(image)
    image = cv2.Canny(image, canny_kernel, canny_kernel)    #for books
    image = image[:, :, None]

    est_to_loaded_canny = np.concatenate([image, image, image], axis=2)
    # est_to_loaded_canny = Image.fromarray(np_image)

    est_to_loaded_canny = Image.fromarray(est_to_loaded_canny)

    return tote(est_to_loaded_canny).unsqueeze(0)

def process_depth(depth,H,W):
    to_blurred_depth = normalization(np.array(depth))
    to_blurred_depth = tote(to_blurred_depth).unsqueeze(0)
    print(to_blurred_depth.shape)

    to_blurred_depth = torch.nn.functional.interpolate(to_blurred_depth, size=[ H,W ], mode="nearest" )      #.permute(1,0,2,3)
    to_blurred_depth = torchvision.transforms.functional.rotate(to_blurred_depth,angle=-90,interpolation=torchvision.transforms.InterpolationMode.NEAREST, expand=True)

    return to_blurred_depth

def normalization(image):
    # minim = image.min()
    # maxim = image.max()
    
    minim = np.nanmin(image)
    maxim = np.nanmax(image)

    # im = np.nan_to_num(image)
    im = image
    im = image/np.nanmax(image)
    # im = (im-minim)/(maxim-minim)
    
    return im


def align_images_and_depth(img1, img2,depth,est_depth,threshold=2, thr_dist=0.75):

    img1 = np.array(img1) #cv.imread('image1.jpg', cv.IMREAD_GRAYSCALE)  # referenceImage
    img2 = np.array(img2) #cv.imread('image2.jpg', cv.IMREAD_GRAYSCALE)  # sensedImage
    depth = np.array(depth)
    est_depth = np.array(est_depth)
    # Initiate SIFT detector
    sift_detector = cv2.SIFT_create()
    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift_detector.detectAndCompute(img1, None)
    kp2, des2 = sift_detector.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Filter out poor matches
    good_matches = []
    for m,n in matches:
        if m.distance < thr_dist*n.distance:
            good_matches.append(m)

    matches = good_matches
            
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    # print('Good matched points', len(good_matches))

    if len(good_matches) > threshold:
        for i, match in enumerate(matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        # Find homography
        H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Warp image 1 to align with image 2
        img1Reg = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
        reg_depth = cv2.warpPerspective(depth, H, (img2.shape[1], img2.shape[0]))
        reg_est_depth = cv2.warpPerspective(est_depth, H, (img2.shape[1], img2.shape[0]))

        return img1Reg,reg_depth,reg_est_depth,len(good_matches)

    else:
        return None