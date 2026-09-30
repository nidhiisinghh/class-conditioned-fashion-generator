#!/usr/bin/env python3
"""
Weave - Fashion LoRA Fine-Tuning Script for Stable Diffusion 1.5
Targeted for Google Colab (T4 / A100 GPU) with automatic checkpointing to Google Drive,
PEFT low-rank adaptation, gradient checkpointing, and mixed precision.
"""

import os
import sys
import glob
import math
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from PIL import Image

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model, PeftModel

# Local dataset loader
try:
    from ml.training.dataset_utils import FashionDiffusionDataset
except ImportError:
    from dataset_utils import FashionDiffusionDataset

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SD1.5 on Fashion Concepts with LoRA")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained SD1.5 model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to fashion dataset containing images and metadata.jsonl / metadata.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/weave/checkpoints/fashion_lora",
        help="Directory to save checkpoints and final LoRA adapter weights.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution for input images (SD1.5 standard is 512).",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for training. Use 1 or 2 on T4 GPU.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=3000,
        help="Total number of training steps. If provided, overrides epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for LoRA parameters.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help="Scheduler type ('linear', 'cosine', 'constant').",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in lr scheduler.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank dimension r.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout rate.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision (fp16 is optimal for Colab T4).",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        help="Whether training should be restored from a checkpoint. 'latest' auto-discovers most recent.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible training.",
    )
    return parser.parse_args()


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Finds the most recent checkpoint folder in output_dir."""
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    # Sort by step number
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0)
    return checkpoints[-1]


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    # 1. Load Base Models (SD1.5 components)
    logger.info(f"Loading pretrained pipeline components from {args.pretrained_model_name_or_path}...")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Freeze base models: VAE, Text Encoder, and base UNet weights
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Enable gradient checkpointing to save VRAM on Colab T4
    unet.enable_gradient_checkpointing()

    # 2. Inject LoRA adapters into UNet cross-attention modules
    logger.info(f"Setting up LoRA adapters (rank={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=args.lora_dropout,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Setup cast precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # 3. Setup Optimizer & Dataset
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    train_dataset = FashionDiffusionDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Calculate steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare for accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # 4. Handle Auto-Resume from Checkpoint (Google Drive resilience)
    global_step = 0
    first_epoch = 0
    resume_path = None

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            resume_path = find_latest_checkpoint(args.output_dir)
        elif os.path.exists(args.resume_from_checkpoint):
            resume_path = args.resume_from_checkpoint

        if resume_path:
            logger.info(f"Resuming training from checkpoint: {resume_path}")
            accelerator.load_state(resume_path)
            global_step = int(os.path.basename(resume_path).split("-")[-1])
            first_epoch = global_step // num_update_steps_per_epoch
            logger.info(f"Resumed at global step {global_step} (Epoch {first_epoch})")
        else:
            logger.info("No checkpoint found; starting training from scratch.")

    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training Fashion LoRA",
    )

    # 5. Training Loop
    logger.info("***** Running Weave Fashion LoRA Training *****")
    logger.info(f"  Num samples = {len(train_dataset)}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total Optimization steps = {args.max_train_steps}")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Encode pixel values to latent space with frozen VAE
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Compute loss (MSE between noise and prediction)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss += loss.detach().item()

                # Checkpointing
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved training state checkpoint to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # 6. Save Final Trained LoRA Adapter
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        final_lora_dir = os.path.join(args.output_dir, "final_fashion_lora")
        os.makedirs(final_lora_dir, exist_ok=True)
        unwrapped_unet.save_pretrained(final_lora_dir)
        logger.info(f"Successfully saved final Fashion LoRA adapter to {final_lora_dir}")
        print(f"\n[DONE] Fashion LoRA trained and saved to: {final_lora_dir}")
        print(f"File contents in final dir: {os.listdir(final_lora_dir)}")

if __name__ == "__main__":
    main()
