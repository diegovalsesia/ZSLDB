#!/bin/bash

export WANDB_API_KEY=YOUR_KEY_HERE

labels=`ls output/estimated_kernels/`

for label in $labels
do

    netlabel=`basename $label | cut -f1-2 -d'.'`

    echo "Processing label: $netlabel"

    # No depth
    #python main.py --label $netlabel --device 0 --flag_true_kernel True --my_controlnet_conditional_scale 0.0 --num_epochs 200  --epoch_step 100 --gamma_step 0.5 --latent_space_lr 0.005 --blurring_kernel_lr 0.0001 --reg_lpips 1.5 --reg_aest_loss 0.1

    # Depth
    python main.py --label $netlabel --device 0 --flag_true_kernel True --my_controlnet_conditional_scale 0.2 --num_epochs 200 --epoch_step 100 --gamma_step 0.5 --latent_space_lr 0.005 --blurring_kernel_lr 0.0001 --reg_lpips 1.5 --reg_aest_loss 0.1


done


