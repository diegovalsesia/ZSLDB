from my_utils import *
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
# from diffusers.image_processor import PipelineImageInput


def img_to_latents(x: torch.Tensor, pipeline, device):
    with torch.no_grad():
        inv_latent = pipeline.vae.encode(x.to(device) )
    latents = 0.18215 * inv_latent.latent_dist.sample()
    return latents


def evaluate(latent,train_neg_prompt_embeds,prompts, pipeline, accelerator, inference_dtype, config,device):
    prompt_ids = pipeline.tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(device)       
    pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(device)
    prompt_embeds = pipeline.text_encoder(prompt_ids)[0]         
    
    all_rgbs_t = []
    # timesteps = pipeline.scheduler.timesteps
    # timesteps = reversed(pipeline.scheduler.timesteps)
    for i, t in tqdm(enumerate(pipeline.scheduler.timesteps), total=len(pipeline.scheduler.timesteps)):
        # t = torch.tensor([timesteps[i]],
        t = torch.tensor([t],
                            dtype=inference_dtype,
                            device=latent.device)
        t = t.repeat(config.train.batch_size_per_gpu_available)

        noise_pred_uncond = pipeline.unet(latent, t, train_neg_prompt_embeds).sample
        noise_pred_cond = pipeline.unet(latent, t, prompt_embeds).sample
                
        grad = (noise_pred_cond - noise_pred_uncond)
        noise_pred = noise_pred_uncond + config.sd_guidance_scale * grad
        latent = pipeline.scheduler.step(noise_pred, t[0].long(), latent).prev_sample
    ims = pipeline.vae.decode(latent.to(pipeline.vae.dtype) / 0.18215).sample
    # if "hps" in config.reward_fn:
    #     loss, rewards = loss_fn(ims, prompts)
    # else:    
    #     _, rewards = loss_fn(ims)
    return ims


def controlnet_evaluate(image,latents,train_neg_prompt_embeds,prompts,negative_prompt_embeds, 
                        pipeline, accelerator, inference_dtype, config,device,guess_mode=False,
                        control_guidance_start : Union[float, List[float]] = 0.0,
                        control_guidance_end: Union[float, List[float]] = 1.0,
                        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
                        guidance_scale=7.5,my_cond_scale=1.0,
                        ):
        
        prompt_ids = pipeline.tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(device)       
        pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(device)
        prompt_embeds = pipeline.text_encoder(prompt_ids)[0]   

        # 8. Denoising loop
        timesteps = pipeline.scheduler.timesteps
        num_inference_steps=50
        num_images_per_prompt=1
        do_classifier_free_guidance=True
        # callback_on_step_end_tensor_inputs: List[str] = ["latents"]
        # # ip_adapter_image: Optional[PipelineImageInput] = None
        # ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None
        # negative_prompt: Optional[Union[str, List[str]]] = None


        controlnet = pipeline.controlnet._orig_mod if is_compiled_module(pipeline.controlnet) else pipeline.controlnet

        batch_size=1

        image = pipeline.prepare_image(
                image=image,
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
            ).to(device=device, dtype=latents.dtype)


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

        with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in tqdm(enumerate(timesteps)):
                # Relevant thread:
                # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                # if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                #     torch._inductor.cudagraph_mark_step_begin()
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)


                control_model_input = latent_model_input
                controlnet_prompt_embeds = prompt_embeds

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]
                # print(cond_scale)
                cond_scale=my_cond_scale
                # print(control_model_input.shape, controlnet_prompt_embeds.shape, image.shape)

                down_block_res_samples, mid_block_res_sample = pipeline.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds.repeat(2,1,1),
                    controlnet_cond=image,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )

                # print(down_block_res_samples[0].shape, mid_block_res_sample[0].shape)

                if guess_mode and do_classifier_free_guidance:
                    # Inferred ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # predict the noise 
                # print(latent_model_input.shape, prompt_embeds.shape)
                        # 7.1 Add image embeds for IP-Adapter
                ip_adapter_image = None
                image_embeds = None
                ip_adapter_image_embeds= None
                added_cond_kwargs = (
                    {"image_embeds": image_embeds}
                    if ip_adapter_image is not None or ip_adapter_image_embeds is not None
                    else None
                )
                
                noise_pred = pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds.repeat(2,1,1),
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=None,
                    return_dict=False,
                )[0]

                # print(noise_pred.shape)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        ims = pipeline.vae.decode(latents.to(pipeline.vae.dtype) / 0.18215).sample

        return latents, ims


## Inversion
@torch.no_grad()
def invert(
    start_latents,
    prompt,
    pipeline,
    device,
    config,
    num_inference_steps=200,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
):


    guidance_scale=config.sd_guidance_scale


    # Encode prompt
    text_embeddings = pipeline._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipeline.scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipeline.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # latents = pipeline.scheduler.step(noise_pred, t, latents).prev_sample


        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipeline.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipeline.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)