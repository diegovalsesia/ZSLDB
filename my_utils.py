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

from aesthetic_scorer import AestheticScorerDiff

def show_multiple_images(images, titles, resolution=(15,10)):

    plt.figure(figsize=resolution)
    n_images = len(images)

    for j,image in enumerate(images):
        plt.subplot(1,n_images,j+1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(titles[j])
    plt.show()
    
        

def aesthetic_loss_fn(aesthetic_target=None,
                     grad_scale=0,
                     device=None,
                     accelerator=None,
                     torch_dtype=None):
    
    target_size = 224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    scorer = AestheticScorerDiff(dtype=torch_dtype).to(device, dtype=torch_dtype)
    scorer.requires_grad_(False)
    target_size = 224
    def loss_fn(im_pix_un):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        im_pix = torchvision.transforms.Resize(target_size)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)
        rewards = scorer(im_pix)
        if aesthetic_target is None: # default maximization
            loss = -1 * rewards
        else:
            # using L1 to keep on same scale
            loss = abs(rewards - aesthetic_target)
        return loss * grad_scale, rewards
    return loss_fn

def set_config_batch(config,total_samples_per_epoch, total_batch_size, per_gpu_capacity=1):
    #  Samples per epoch
    config.train.total_samples_per_epoch = total_samples_per_epoch  #(~~~~ this is desired ~~~~)
    config.train.num_gpus = 1
    # config.train.num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    
    assert config.train.total_samples_per_epoch%config.train.num_gpus==0, "total_samples_per_epoch must be divisible by num_gpus"
    config.train.samples_per_epoch_per_gpu = config.train.total_samples_per_epoch//config.train.num_gpus
    
    #  Total batch size
    config.train.total_batch_size = total_batch_size  #(~~~~ this is desired ~~~~)
    assert config.train.total_batch_size%config.train.num_gpus==0, "total_batch_size must be divisible by num_gpus"
    config.train.batch_size_per_gpu = config.train.total_batch_size//config.train.num_gpus
    config.train.batch_size_per_gpu_available = per_gpu_capacity    #(this quantity depends on the gpu used)
    assert config.train.batch_size_per_gpu%config.train.batch_size_per_gpu_available==0, "batch_size_per_gpu must be divisible by batch_size_per_gpu_available"
    config.train.gradient_accumulation_steps = config.train.batch_size_per_gpu//config.train.batch_size_per_gpu_available
    
    assert config.train.samples_per_epoch_per_gpu%config.train.batch_size_per_gpu_available==0, "samples_per_epoch_per_gpu must be divisible by batch_size_per_gpu_available"
    config.train.data_loader_iterations  = config.train.samples_per_epoch_per_gpu//config.train.batch_size_per_gpu_available    
    return config

def gen_config():

    config = ml_collections.ConfigDict()

    ###### General ######    
    config.eval_prompt_fn = ''
    config.soup_inference = False
    config.save_freq = 4
    config.resume_from = ""
    config.resume_from_2 = ""
    config.vis_freq = 1
    config.max_vis_images = 2
    config.only_eval = False
    config.run_name = ""

    # prompting
    config.prompt_fn = "simple_animals"
    config.reward_fn = "aesthetic"
    config.debug =False
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    config.mixed_precision  = "fp32"
    # number of checkpoints to keep before overwriting old ones.
    config.num_checkpoint_limit = 10
    # run name for wandb logging and checkpoint saving -- if not provided, will be auto-generated based on the datetime.
    config.run_name = ""
    # top-level logging directory for checkpoint saving.
    config.logdir = "logs"
    # random seed for reproducibility.
    config.seed = 42    
    # number of epochs to train for. each epoch is one round of sampling from the model followed by training on those
    # samples.
    config.num_epochs = 50    

    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True

    config.visualize_train = True
    config.visualize_eval = False

    # config.truncated_backprop = False
    # config.truncated_backprop_rand = False
    # config.truncated_backprop_minmax = (35,45)
    # config.trunc_backprop_timestep = 100

    config.grad_checkpoint = True
    config.same_evaluation = True


    ###### Training ######    
    config.train = train = ml_collections.ConfigDict()
    config.train.loss_coeff = 1.0
    # whether to use the 8bit Adam optimizer from bitsandbytes.
    train.use_8bit_adam = False
    # learning rate.
    train.learning_rate = 3e-4
    # Adam beta1.
    train.adam_beta1 = 0.9
    # Adam beta2.
    train.adam_beta2 = 0.999
    # Adam weight decay.
    train.adam_weight_decay = 1e-4
    # Adam epsilon.
    train.adam_epsilon = 1e-8 
    # maximum gradient norm for gradient clipping.
    train.max_grad_norm = 1.0    
    config.aesthetic_target = 10
    config.grad_scale = 1
    config.sd_guidance_scale = 7.5
    config.steps = 10

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    pretrained.model = "runwayml/stable-diffusion-v1-5"
    # revision of the model to load.
    pretrained.revision = "main"

    config.prompt_fn = "simple_animals"
    config.eval_prompt_fn = "eval_simple_animals"

    config.reward_fn = 'aesthetic' # CLIP or imagenet or .... or .. 
    config.train.max_grad_norm = 5.0    
    config.train.loss_coeff = 0.01
    config.train.learning_rate = 1e-1
    config.max_vis_images = 4
    config.train.adam_weight_decay = 0.1

    config.save_freq = 1
    config.num_checkpoint_limit = 3
    config.truncated_backprop_rand = True
    config.truncated_backprop_minmax = (0,50)
    config.trunc_backprop_timestep = 40
    config.truncated_backprop = True
    config = set_config_batch(config,total_samples_per_epoch=1,total_batch_size= 1, per_gpu_capacity=1)

    return config


def matlab_style_gauss2D(shape=(3,3),sigmax=5,sigmay=5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x)/(2.*sigmax*sigmax) - (y*y) / (2.*sigmay*sigmay) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
