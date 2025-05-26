# %%
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', help = "INT ID of CUDA DEVICE (ex 2)", type=int)
parser.add_argument('--reg_lpips',default=1, help = "Regularization using lpips(blurred inp, blurring(solution))", type=float)
parser.add_argument('--reg_aest_loss',default=0.01, help = "Regularization using aestetic distanc with LION dataset", type=float)
parser.add_argument('--reg_tv',default=0.0, help = "Regularization using total variation", type=float)
parser.add_argument('--label',default='scooter2', help = "label to load and save files", type=str)
parser.add_argument('--controlnet',default='depth', help = "controlnet to use (canny or depth)", type=str)
parser.add_argument('--my_controlnet_conditional_scale',default=0.2, help = "hyper par to control the influence od depth", type=float)
parser.add_argument('--latent_space_lr',default=0.003, help = "LR for the latent space", type=float)
parser.add_argument('--blurring_kernel_lr',default=0.0, help = "LR for the blur kernel", type=float)
parser.add_argument('--gamma_step',default=0.5, help = "LR for the blur kernel", type=float)
parser.add_argument('--do_inversion',default=False, help = "do inversion with DDIM", type=bool)
parser.add_argument('--flag_true_kernel',default=False, help = "using the estim kernel from JMKD", type=bool)
parser.add_argument('--num_epochs',default=200, help = "number of epochs", type=int)
parser.add_argument('--epoch_step',default=200, help = "number of epochs", type=int)
parser.add_argument('--inference_steps',default=10, help = "inference steps of diffusion", type=int)


args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

from my_utils import *
from my_diffusion import *
from preprocess_utils import *
import time
import argparse
import lpips
from absl import app, flags
from ml_collections import config_flags



FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/align_prop.py:evaluate", "Training configuration.")
from accelerate.utils import set_seed, ProjectConfiguration
logger = get_logger(__name__)


def total_variation_loss(img):
     bs_img, c_img, h_img, w_img = img.size()
     tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
     tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
     return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img)


# %%
config = gen_config()
##### IMPORTANT HYPER-PARAMETER TO SET UP ######

my_controlnet_conditional_scale=args.my_controlnet_conditional_scale

gamma_step = args.gamma_step
epoch_step = args.epoch_step

do_inversion=args.do_inversion
latent_space_lr = args.latent_space_lr

label = args.label

reg_lpips = args.reg_lpips
reg_aest_loss = args.reg_aest_loss
reg_tv = args.reg_tv

blurring_kernel_lr = args.blurring_kernel_lr

first_epoch = 0
ker_size = 33  # selected_kernel.shape[-1]

NUM_EPOCHS=args.num_epochs
print('Number of epochs: ', NUM_EPOCHS)
freq_to_plot = 10

flag_true_kernel=args.flag_true_kernel

if args.controlnet =='depth':
    # lab = 'cannyg'
    lab = 'depthg'
    depth_folder = "/media/HDD2/valsesia/ZSLDB_arkit/testset/depth/"+ label +'.png'

elif args.controlnet =='canny':
    lab = 'cannyg'
    depth_folder = "/media/HDD2/valsesia/ZSLDB_arkit/testset/canny/"+ label +'.png'

else:
    lab = 'base'
    depth_folder = "/media/HDD2/valsesia/ZSLDB_arkit/testset/canny/"+ label +'.png'

# %%

accelerator_config = ProjectConfiguration(
    project_dir=os.path.join(config.logdir, config.run_name),
    automatic_checkpoint_naming=True,
    total_limit=config.num_checkpoint_limit,
)

accelerator = Accelerator(
    log_with="wandb",
    # mixed_precision=config.mixed_precision,
    project_config=accelerator_config,
    gradient_accumulation_steps=config.train.gradient_accumulation_steps,
)

# tim = time.time()
# tim = round(tim,0)
see = torch.randint(0,1000,(1,)).item()

result_folder = lab +'_'+ str(my_controlnet_conditional_scale)+'_rlpips'+ str(reg_lpips)+ '_raest'+str(reg_aest_loss)+ '_rtv' + str(reg_tv) +'_epochs'+ str(NUM_EPOCHS) + '_lr' + str(latent_space_lr) + '_kerlr' + str(blurring_kernel_lr) +'_epochsched' + str(epoch_step)+'_gam'+str(gamma_step) + '_infsteps' + str(args.inference_steps)


if accelerator.is_main_process:
    wandb_args = {}
    if config.debug:
        wandb_args = {'mode':"disabled"}        
    accelerator.init_trackers(
        project_name="align-prop", config=config.to_dict(), init_kwargs={"wandb": wandb_args}
    )

    wandb.run.name = label +'_'+ result_folder #+ '_seed'+str(see)

    accelerator.project_configuration.project_dir = os.path.join(config.logdir, wandb.run.name)
    accelerator.project_configuration.logging_dir = os.path.join(config.logdir, wandb.run.name)    

# accelerator.device='cuda:3'
print(wandb.run.name)
logger.info(f"\n{config}")

# set seed (device_specific is very important to get different prompts on different devices)
set_seed(config.seed, device_specific=True)



# pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, revision=config.pretrained.revision)
accelerator.mixed_precision


# %%
inference_dtype = torch.float32

if args.controlnet =='depth':
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=inference_dtype)
else:
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=inference_dtype)

pipeline = StableDiffusionControlNetPipeline.from_pretrained(config.pretrained.model,controlnet=controlnet,revision=config.pretrained.revision)

# %%
# freeze parameters of models to save more memory
pipeline.vae.requires_grad_(False)
pipeline.text_encoder.requires_grad_(False)
pipeline.unet.requires_grad_(False)


# disable safety checker
pipeline.safety_checker = None    

# make the progress bar nicer
pipeline.set_progress_bar_config(
    position=1,
    disable=not accelerator.is_local_main_process,
    leave=False,
    desc="Timestep",
    dynamic_ncols=True,
)    

# switch to DDIM scheduler
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler.set_timesteps(config.steps)

 

# Move unet, vae and text_encoder to device and cast to inference_dtype
pipeline.vae.to(accelerator.device, dtype=inference_dtype)
pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)

pipeline.unet.to(accelerator.device, dtype=inference_dtype) 

# Set correct lora layers
lora_attn_procs = {}
for name in pipeline.unet.attn_processors.keys():
    cross_attention_dim = (
        None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
    )
    if name.startswith("mid_block"):
        hidden_size = pipeline.unet.config.block_out_channels[-1]
    elif name.startswith("up_blocks"):
        block_id = int(name[len("up_blocks.")])
        hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
    elif name.startswith("down_blocks"):
        block_id = int(name[len("down_blocks.")])
        hidden_size = pipeline.unet.config.block_out_channels[block_id]

    lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
pipeline.unet.set_attn_processor(lora_attn_procs)

# this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
# this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
# `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.

class _Wrapper(AttnProcsLayers):
    def forward(self, *args, **kwargs):
        return pipeline.unet(*args, **kwargs)

unet = _Wrapper(pipeline.unet.attn_processors) 

# class _Wrapper(torch.nn.parameter.Parameter):
#     def forward(self, *args, **kwargs):
#         return pipeline.unet(*args, **kwargs)
# unet = _Wrapper(starting_latent) 

# unet = starting_latent


# generate negative prompt embeddings
neg_prompt_embed = pipeline.text_encoder(
    pipeline.tokenizer(
        [""],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(accelerator.device)
)[0]

train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size_per_gpu_available, 1, 1)

autocast = contextlib.nullcontext



keep_input = True
timesteps = pipeline.scheduler.timesteps

# eval_prompts, eval_prompt_metadata = zip(
#     *[eval_prompt_fn() for _ in range(config.train.batch_size_per_gpu_available * config.max_vis_images)]
# )    

global_step = 0

# %%
pipeline.to(accelerator.device)


prompt_fn = getattr(prompts_file, config.prompt_fn)
eval_prompt_fn = prompt_fn
 
eval_prompts, eval_prompt_metadata = zip(
    *[eval_prompt_fn() for _ in range(config.train.batch_size_per_gpu_available * config.max_vis_images)]
)  

eval_prompts = list(eval_prompts)
# eval_prompts = ['a realistic high resolution photo of a ' for i,_ in enumerate(eval_prompts)]
eval_prompts = ['' for i,_ in enumerate(eval_prompts)]
# eval_prompts[-1] = 'a realistic photo of a panda'
eval_prompts = tuple(eval_prompts)

latents_folder = '/media/HDD2/valsesia/ZSLDB_arkit/output/inverted_latents/'
# base_folder = '/home/montanaro/AlignProp/real_dataset/processed_dataset/latents/'
blurred_folder = "/media/HDD2/valsesia/ZSLDB_arkit/testset/blurred/"+ label +'.png'
gt_folder = "/media/HDD2/valsesia/ZSLDB_arkit/testset/rgb/"+ label +'.png'
kernel_folder = "/media/HDD2/valsesia/ZSLDB_arkit/output/estimated_kernels/" + label +'.npy'


true_kernel = np.load( kernel_folder )    
estim_kernel = matlab_style_gauss2D(shape=(ker_size,ker_size), sigmax=3,sigmay=3)


conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=ker_size, stride=1, padding=ker_size//2, bias=False)  

kernel = true_kernel
kernel_torch = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).float()
conv.weight = torch.nn.Parameter(kernel_torch)
conv.weight.shape

# estim_kernel = matlab_style_gauss2D(shape=(ker_size,ker_size), sigmax=3,sigmay=3)

unknown_conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=ker_size, stride=1, padding=ker_size//2, bias=False).to(accelerator.device)  
# kernel = my_f
# kernel = np.zeros((ker_size,ker_size))
if flag_true_kernel:
    estim_kernel = kernel
    kernel_torch = torch.from_numpy(estim_kernel).unsqueeze(0).unsqueeze(0).float()
else:
    kernel_torch = torch.from_numpy(estim_kernel).unsqueeze(0).unsqueeze(0).float()

true_ker_im = kernel
init_ker_im = estim_kernel

unknown_conv.weight = torch.nn.Parameter(kernel_torch)
unknown_conv.requires_grad = True

# test_image = Image.open('/home/montanaro/AlignProp/GOPR0869_11_00-000098.png')

test_image = Image.open(blurred_folder) 

test_image = tote(test_image).unsqueeze(0)

canny_image = Image.open( depth_folder)

if args.controlnet =='depth':
    canny_test_image = tote(canny_image).unsqueeze(0).repeat(1,3,1,1)
    # print('\n\n\n\n\n\n\n\n\n\nn\n')
    # print(canny_test_image.shape)
else:
    canny_test_image = tote(canny_image).unsqueeze(0)

test_image =test_image.view(-1,1,512,512)

raw_blur_image = test_image
raw_ukn_image = unknown_conv(raw_blur_image).view(1,-1,512,512)
raw_blur_image = raw_blur_image.view(1,-1,512,512)
raw_test_image = test_image.view(1,-1,512,512)

images = [topil(raw_test_image[0]), topil(raw_blur_image[0]), topil(raw_ukn_image[0]), topil(canny_test_image[0])]

# range the test image between -1 and 1
test_image = 2*(raw_test_image-0.5)
test_image = test_image.repeat(config.train.batch_size_per_gpu_available ,1,1,1)

blur_image = 2*(raw_blur_image-0.5)
blur_image = blur_image.repeat(config.train.batch_size_per_gpu_available ,1,1,1)


# %%

conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=ker_size, stride=1, padding=ker_size//2, bias=False)  

kernel_torch = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).float().to(accelerator.device)
conv.weight = torch.nn.Parameter(kernel_torch)
conv.weight.shape


unknown_conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=ker_size, stride=1, padding=ker_size//2, bias=False).to(accelerator.device)  
# kernel = my_f
# kernel = np.zeros((ker_size,ker_size))
kernel_torch = torch.from_numpy(estim_kernel).unsqueeze(0).unsqueeze(0).float().to(accelerator.device)
unknown_conv.weight = torch.nn.Parameter(kernel_torch)
unknown_conv.requires_grad = True

l = img_to_latents(blur_image, pipeline.to(accelerator.device), accelerator.device)


if os.path.exists(latents_folder + '%s.pt'%(label)):
    inv_latents = torch.load(latents_folder + '%s.pt'%(label))
else:
    with torch.no_grad():
        inv_latents = invert(l, "",pipeline,accelerator.device, config, num_inference_steps=200, negative_prompt="")
    
    torch.save(inv_latents, latents_folder + '%s.pt'%(label))


config.steps=args.inference_steps
pipeline.scheduler.set_timesteps(config.steps)

lpips_criterion = lpips.LPIPS(net='alex').to(accelerator.device)

loss_fn = torch.nn.MSELoss()
# loss_fn =torch.nn.L1Loss()


aesthetic_loss = aesthetic_loss_fn(grad_scale=config.grad_scale,
                            aesthetic_target=config.aesthetic_target,
                            accelerator = accelerator,
                            torch_dtype = inference_dtype,
                            device = accelerator.device)

guess_mode=False,
control_guidance_start : Union[float, List[float]] = 0.0
control_guidance_end: Union[float, List[float]] = 1.0
controlnet_conditioning_scale: Union[float, List[float]] = 1.0
guidance_scale=7.5
device = accelerator.device
timesteps = pipeline.scheduler.timesteps
num_inference_steps=50
num_images_per_prompt=1
do_classifier_free_guidance=True


controlnet = pipeline.controlnet._orig_mod if is_compiled_module(pipeline.controlnet) else pipeline.controlnet

batch_size=1

image = pipeline.prepare_image(
        image=canny_test_image,
        width=None,
        height=None,
        batch_size=batch_size * num_images_per_prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        dtype=controlnet.dtype,
        do_classifier_free_guidance=do_classifier_free_guidance,
        guess_mode=guess_mode,
    )

height, width = image.shape[-2:]

timestep_cond = None
if pipeline.unet.config.time_cond_proj_dim is not None:
    guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
    timestep_cond = pipeline.get_guidance_scale_embedding(
        guidance_scale_tensor, embedding_dim=pipeline.unet.config.time_cond_proj_dim
    ).to(device=device)


num_warmup_steps = len(timesteps) - num_inference_steps * pipeline.scheduler.order

control_guidance_start =  [control_guidance_start]
control_guidance_end =  [control_guidance_end]

controlnet_keep = []
controlnet_conditioning_scale = [controlnet_conditioning_scale] 
for i in range(len(timesteps)):
    keeps = [
        1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
        for s, e in zip(control_guidance_start, control_guidance_end)
    ]
    controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

# %%
##### IMPORTANT HYPER-PARAMETER TO SET UP ######

config.num_epochs=NUM_EPOCHS

#################### TRAINING ####################

do_classifier_free_guidance=True
prompts = eval_prompts

prompt_ids = pipeline.tokenizer(
    "",
    # "de-blurred, high details, sharp",
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=pipeline.tokenizer.model_max_length,
).input_ids.to(accelerator.device)   

pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)
prompt_embeds = pipeline.text_encoder(prompt_ids)[0]         
prompt_embeds.requires_grad = True
# prompt_embeds.requires_grad
timesteps = pipeline.scheduler.timesteps

# Initialize the optimizer
# optimizer_cls = torch.optim.AdamW
optimizer_cls = torch.optim.Adam
# starting_latent = torch.nn.parameter.Parameter( z.to(accelerator.device)  )
# starting_latent = torch.nn.parameter.Parameter( inv_latents[-7].unsqueeze(0).to(accelerator.device)  )
starting_latent = torch.nn.parameter.Parameter( inv_latents[-20].unsqueeze(0).to(accelerator.device)  )

optimizer = optimizer_cls(
    # [{'params':starting_latent,'lr':0.1}],            #best parameter for l2 in latent space
    # [{'params':starting_latent,'lr':0.001}],            #best parameter for l2 in latent space
    [{'params':starting_latent,'lr':latent_space_lr}, {'params':unknown_conv.weight,'lr':blurring_kernel_lr}],
    # [{'params':starting_latent,'lr':0.001},{'params':prompt_embeds,'lr':0.001}],
    # latent.parameters(),
    lr=config.train.learning_rate,
)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoch_step, gamma=gamma_step)

images_list=[]

for epoch in list(range(first_epoch, config.num_epochs)):
    # unet.train()
    info = defaultdict(list)
    info_vis = defaultdict(list)
    image_vis_list = []

    # for inner_iters in tqdm(list(range(config.train.data_loader_iterations)),position=0,disable=not accelerator.is_local_main_process):
    for inner_iters in range(1):
        latent = starting_latent

        if accelerator.is_main_process:

            logger.info(f"{wandb.run.name} Epoch {epoch}.{inner_iters}: training")
        
        prompt_ids = pipeline.tokenizer(
            "",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)   

        pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)

        # Prepare everything with our `accelerator`.
        # unet = accelerator.prepare(unet)
        # unet,starting_latent,optimizer = accelerator.prepare(unet,starting_latent,optimizer)
        unet,starting_latent,prompt_embeds, optimizer, lpips = accelerator.prepare(unet,starting_latent,prompt_embeds,optimizer,lpips)
        # unet,starting_latent,prompt_embeds, optimizer = accelerator.prepare(unet,starting_latent,prompt_embeds,optimizer)    

    
        with accelerator.accumulate(unet):
            with autocast():
                with torch.enable_grad(): # important b/c don't have on by default in module                        

                    keep_input = True
                    for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
                        # print('Timestep: ', t)
                        latent_model_input = torch.cat([latent] * 2) if do_classifier_free_guidance else latent
                        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
                        control_model_input = latent_model_input
                        # controlnet_prompt_embeds = prompt_embeds
                        # if isinstance(controlnet_keep[i], list):
                        #     cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                        # else:
                        #     controlnet_cond_scale = controlnet_conditioning_scale
                        #     if isinstance(controlnet_cond_scale, list):
                        #         controlnet_cond_scale = controlnet_cond_scale[0]
                        #     cond_scale = controlnet_cond_scale * controlnet_keep[i]
                        cond_scale = 1.0
                        t = torch.tensor([t],
                                            dtype=inference_dtype,
                                            device=latent.device)
                        t = t.repeat(config.train.batch_size_per_gpu_available)

                        if config.grad_checkpoint:

                            down_block_res_samples, mid_block_res_sample =  checkpoint.checkpoint( pipeline.controlnet,
                                                                                        control_model_input,
                                                                                        t,
                                                                                        prompt_embeds.repeat(2,1,1),
                                                                                        image,
                                                                                        my_controlnet_conditional_scale,None,None,None,None,False,
                                                                                        False,
                                                                                    )
                            # unet,starting_latent,prompt_embeds, optimizer = accelerator.prepare(unet,starting_latent,prompt_embeds,optimizer)
                            

                            # checkpoint.checkpoint( pipeline.controlnet, latent,t,prompt_embeds,image)
                            # print(down_block_res_samples[0].shape, mid_block_res_sample[0].shape)
                            
                            # noise_pred = checkpoint.checkpoint( unet, latent_model_input,t,prompt_embeds.repeat(2,1,1),timestep_cond,
                            #                            down_block_res_samples,mid_block_res_sample)[0]

                            # noise_pred_uncond = checkpoint.checkpoint(pipeline.unet,latent, t, train_neg_prompt_embeds, use_reentrant=False).sample
                            noise_p = checkpoint.checkpoint( unet,
                                                latent_model_input,
                                                t,
                                                prompt_embeds.repeat(2,1,1),
                                                None,None,None,None,None,
                                                down_block_res_samples,
                                                mid_block_res_sample,
                                                None,
                                                False,)[0]
                                                # use_reentrant=False)
                            
                            # noise_pred_cond, noise_pred_uncond = checkpoint.checkpoint(  pipeline.unet,latent, t, prompt_embeds, down_block_res_samples, mid_block_res_sample, use_reentrant=False).sample
                            noise_pred_uncond, noise_pred_cond = noise_p.chunk(2)
                            # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                            # noise_pred_uncond = checkpoint.checkpoint(unet, latent, t, train_neg_prompt_embeds, use_reentrant=False).sample
                            # noise_pred_cond = checkpoint.checkpoint(unet, latent, t, prompt_embeds, use_reentrant=False).sample
                        else:
                            down_block_res_samples, mid_block_res_sample =  pipeline.controlnet(control_model_input,t,encoder_hidden_states=controlnet_prompt_embeds.repeat(2,1,1),
                                                                                            controlnet_cond=image,conditioning_scale=cond_scale,guess_mode=guess_mode,return_dict=False,)

                            noise_pred = pipeline.unet( latent_model_input,t,encoder_hidden_states=prompt_embeds.repeat(2,1,1),timestep_cond=timestep_cond,
                                                       down_block_additional_residuals=down_block_res_samples,mid_block_additional_residual=mid_block_res_sample,return_dict=False,)[0]

                            # noise_pred_uncond = unet(latent, t, train_neg_prompt_embeds).sample
                            # noise_pred_cond = unet(latent, t, prompt_embeds).sample
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                                                        
                        if config.truncated_backprop:
                            if config.truncated_backprop_rand:
                                timestep = random.randint(config.truncated_backprop_minmax[0],config.truncated_backprop_minmax[1])
                                if i < timestep:
                                    noise_pred_uncond = noise_pred_uncond.detach()
                                    noise_pred_cond = noise_pred_cond.detach()
                            else:
                                # if i < config.trunc_backprop_timestep:
                                if i < config.trunc_backprop_timestep:
                                    noise_pred_uncond = noise_pred_uncond.detach()
                                    noise_pred_cond = noise_pred_cond.detach()

                        grad = (noise_pred_cond - noise_pred_uncond)
                        noise_pred = noise_pred_uncond + config.sd_guidance_scale * grad                
                        latent = pipeline.scheduler.step(noise_pred, t[0].long(), latent).prev_sample
                                            
                    ims = pipeline.vae.decode(latent.to(pipeline.vae.dtype) / 0.18215).sample
                    
                    # if "hps" in config.reward_fn:
                    #     loss, rewards = loss_fn(ims, prompts)
                    # else:
                    #     loss, rewards = loss_fn(ims)
                    # print(ims.min(), ims.max(),blur_image.min(), blur_image.max(), )
                    # ims = (ims/ 2 + 0.5).clamp(0, 1)
                    # print(ims.min(), ims.max(),blur_image.min(), blur_image.max(), )
                    b,c,h,w = ims.shape

                    tv_reg = total_variation_loss(ims)
                    
                    aest_loss ,rew = aesthetic_loss(ims)
                    aest_loss = aest_loss.sum()     
                    
                    # blurred_latent = latent
                    
                    conv_ims = unknown_conv(ims.view(b*c,1,h,w)).view(b,c,h,w)
                    
                    perc_loss = lpips_criterion(conv_ims,blur_image.to(accelerator.device)).to(accelerator.device).sum()            

                    # conv_ims = conv(ims.view(b*c,1,h,w)).view(b,c,h,w)
                    blurred_latent = pipeline.vae.encode(conv_ims.to(accelerator.device) )
                    blurred_latent = 0.18215 * blurred_latent.latent_dist.sample()
                    # blurred_latent = unknown_conv(latent.view(b*c,1,h,w)).view(b,c,h,w)

                    loss = loss_fn(blurred_latent,l.to(accelerator.device)).sum()                    
                    # loss = loss_fn(ims,test_image[:2].to(accelerator.device))                    
                    # loss = loss_fn(ims,blur_image[:2].to(accelerator.device))                    
                    # loss = perc_dist(ims,blur_image[:2].to(accelerator.device))                    
                    loss =    loss + reg_lpips*perc_loss + reg_aest_loss*aest_loss + reg_tv*tv_reg  # + 10*tv_reg #+ 0.1*perc_loss.sum() + 0.01*aest_loss.sum() #+ 0.1*perc_loss.sum()
                    # loss = loss/config.train.batch_size_per_gpu_available
                    # loss = loss * config.train.loss_coeff
                    info["total_loss"].append(loss)
                    info["lpips"].append(perc_loss)
                    info["aesthetic_loss"].append(aest_loss)
                    info["total_variation"].append(tv_reg)
                    # backward pass
                    accelerator.backward(loss ,retain_graph=True)
                    # loss.backward()
                    # print('BACKWARD DONE ')
                    # if accelerator.sync_gradients:
                    #     accelerator.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()  
                    scheduler.step()                      

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            assert (
                inner_iters + 1
            ) % config.train.gradient_accumulation_steps == 0
            # log training and evaluation 
            
            logger.info("Logging")
            
            info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
            info = accelerator.reduce(info, reduction="mean")
            logger.info(f"total_loss: {info['total_loss']}")
            logger.info(f"lpips: {info['lpips']}")
            logger.info(f"aesthetic_loss: {info['aesthetic_loss']}")
            logger.info(f"total_variation: {info['total_variation']}")

            info.update({"epoch": epoch, "inner_epoch": inner_iters})
            # info.update({"epoch": epoch, "inner_epoch": inner_iters, "eval_rewards":eval_reward_mean,"eval_rewards_std":eval_reward_std})
            accelerator.log(info, step=global_step)

            if config.visualize_train:
                images  = []
                pil_im = []
                if epoch % freq_to_plot == 0:
                    # print('Epoch %d, Training Loss %.4f, Lr  '  %(epoch, loss.item()),  scheduler.get_last_lr() )
                    # print( 'Aestetic and Perceptual loss'    , aest_loss.sum().item(), perc_loss.sum().item())
                    # print('Total variation', tv_reg)
                    images_list.append(ims.detach().cpu())

                accelerator.log(
                    {"images": images},
                    step=global_step,
                )

            global_step += 1
            info = defaultdict(list)

    # make sure we did an optimization step at the end of the inner epoch
    assert accelerator.sync_gradients
    
    if epoch % config.save_freq == 0 and accelerator.is_main_process:
        # accelerator.wait_for_everyone()
        # accelerator.save_iteration +=1000
        accelerator.save_state()


# %%
lpips_crit = lpips.LPIPS(net='alex').to('cpu')
l2_crit = torch.nn.MSELoss()

gt = Image.open( gt_folder )
to_gt = tote(gt).unsqueeze(0)   #.to(accelerator.device)


img_sol = torch.cat(images_list, dim=0)

distances = []
l2_distances=[]

best_dist=100
best_l2=100

for i in range(img_sol.shape[0]):
    logger.info("Logging")
    
    image = (img_sol[i].unsqueeze(0).detach().cpu() / 2 + 0.5).clamp(0, 1)
    # plt.imshow(topil(image[0]))
    perc_dist = lpips_crit(image, to_gt)
    l2_dist = l2_crit(image, to_gt)
    
    if l2_dist < best_l2:
        best_l2 = l2_dist
        best_restored_image_l2 = image

    if perc_dist < best_dist:
        best_dist = perc_dist
        best_restored_image = image
    distances.append(perc_dist.detach().numpy().item())
    l2_distances.append(l2_dist.detach().numpy().item())


data = [[x, y] for (x, y) in zip(np.arange(len(distances)),np.array(distances))]

table = wandb.Table(data=data, columns=["iterations", "LPIPS(GT,Sol_it)"])
wandb.log(
    {
        "my_custom_plot_lpips": wandb.plot.line(
            table, "iterations", "LPIPS(GT,Sol_it)", title="LPIPS evaluation"
        )
    }
)

data = [[x, y] for (x, y) in zip(np.arange(len(l2_distances)),np.array(l2_distances))]

table = wandb.Table(data=data, columns=["iterations", "L2(GT,Sol_it)"])
wandb.log(
    {
        "my_custom_plot_l2": wandb.plot.line(
            table, "iterations", "L2(GT,Sol_it)", title="L2 evaluation"
        )
    }
)


path_to_save = '/media/HDD2/valsesia/ZSLDB_arkit/output/kernels/' 

if not os.path.exists(path_to_save):
    os.mkdir(path_to_save)

learnt_ker_im = unknown_conv.weight[0,0].detach().cpu().numpy()

images = [true_ker_im, init_ker_im, learnt_ker_im]
titles = ['True kernel', 'Initialized kernel', 'Learnt kernel (lr %.5f)'%(blurring_kernel_lr)]

plt.figure(figsize=(15,10))
n_images = len(images)

for j,imag in enumerate(images):
    plt.subplot(1,n_images,j+1)
    plt.imshow(imag)
    plt.axis('off')
    plt.title(titles[j])
    plt.colorbar()
plt.savefig(path_to_save + label +'.png')
plt.close()
# %%

results_name = wandb.run.name

path_to_save_best = '/media/HDD2/valsesia/ZSLDB_arkit/output/best_lpips_' + result_folder + '/'
path_to_save_best_l2 = '/media/HDD2/valsesia/ZSLDB_arkit/output/best_l2_' + result_folder + '/'
path_to_save_last = '/media/HDD2/valsesia/ZSLDB_arkit/output/last_' + result_folder + '/'

if not os.path.exists(path_to_save_best):
    os.mkdir(path_to_save_best)
    # os.mkdir(path_to_save_best_l2)
    # os.mkdir(path_to_save_last)

path_to_save_best += label + '_restored.png'
path_to_save_best_l2 += label + '_restored.png'
path_to_save_last += label + '_restored.png'

topil(best_restored_image[0]).save(path_to_save_best )
# topil(best_restored_image_l2[0]).save(path_to_save_best_l2 )
# topil(image[0]).save(path_to_save_last )

# write lpips and l2 values
with open(path_to_save_best.replace('.png','_lpips.txt'), 'w') as f:
    f.write(str(best_dist))
with open(path_to_save_best.replace('.png','_psnr.txt'), 'w') as f:
    best_psnr = -10*np.log10(best_l2)
    f.write(str(best_psnr))