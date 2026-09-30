#!/usr/bin/env python3
"""
Weave - Comparative Baseline: From-Scratch AC-GAN (WGAN-GP + SAGAN + Spectral Norm)
Strictly adheres to PRD Document 9.6:
- Generator: z in R^100 + class embedding -> Transposed-Conv stack with Spectral Normalization -> RGB Output
- Discriminator: Shared Conv trunk (Spectral Normalization) -> Two Heads:
    1. WGAN Critic scalar score D(x)
    2. AC-GAN Auxiliary classification head C(x)
- Stabilization: Self-Attention block (SAGAN-style), EMA generator weights (decay 0.999)
- Loss: WGAN-GP critic loss (lambda=10 gradient penalty) + AC-GAN auxiliary cross-entropy
- Optimizer: Adam, lr=2e-4 (G) / 1e-4 (D), beta1=0.0, beta2=0.9
- Evaluation-only: Internal baseline benchmark to compare against fine-tuned diffusion.
"""

import os
import copy
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

try:
    from ml.training.dataset_utils import (
        FashionDiffusionDataset,
        FASHION_CATEGORIES,
    )
except ImportError:
    from dataset_utils import (
        FashionDiffusionDataset,
        FASHION_CATEGORIES,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Self-Attention Block (SAGAN) ---
class SelfAttention(nn.Module):
    """Self-Attention block as used in SAGAN (Zhang et al. 2019)"""
    def __init__(self, in_dim: int):
        super().__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.utils.spectral_norm(nn.Conv2d(in_dim, in_dim // 8, kernel_size=1))
        self.key_conv = nn.utils.spectral_norm(nn.Conv2d(in_dim, in_dim // 8, kernel_size=1))
        self.value_conv = nn.utils.spectral_norm(nn.Conv2d(in_dim, in_dim, kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, w, h = x.size()
        proj_query = self.query_conv(x).view(b, -1, w * h).permute(0, 2, 1) # B x (W*H) x C'
        proj_key = self.key_conv(x).view(b, -1, w * h)                     # B x C' x (W*H)
        energy = torch.bmm(proj_query, proj_key)                            # B x (W*H) x (W*H)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(b, -1, w * h)                  # B x C x (W*H)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, c, w, h)
        return self.gamma * out + x


# --- Generator ---
class ACGANGenerator(nn.Module):
    """
    Generator: Latent z (100-d) + label-embedding conditioning ->
    transposed-conv upsampling stack with spectral normalization -> RGB output (128x128)
    """
    def __init__(self, z_dim: int = 100, num_classes: int = len(FASHION_CATEGORIES), emb_dim: int = 50, ngf: int = 64):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, emb_dim)
        in_dim = z_dim + emb_dim

        self.project = nn.utils.spectral_norm(nn.Linear(in_dim, ngf * 8 * 4 * 4))
        self.bn0 = nn.BatchNorm2d(ngf * 8)
        self.relu = nn.ReLU(True)
        self.ngf = ngf

        self.block1 = nn.Sequential(
            # 4x4 -> 8x8
            nn.utils.spectral_norm(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        self.block2 = nn.Sequential(
            # 8x8 -> 16x16
            nn.utils.spectral_norm(nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.attn = SelfAttention(ngf * 2) # Self-attention at 16x16
        self.block3 = nn.Sequential(
            # 16x16 -> 32x32
            nn.utils.spectral_norm(nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        self.block4 = nn.Sequential(
            # 32x32 -> 64x64
            nn.utils.spectral_norm(nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True)
        )
        self.to_rgb = nn.Sequential(
            # 64x64 -> 128x128
            nn.utils.spectral_norm(nn.ConvTranspose2d(ngf // 2, 3, 4, 2, 1, bias=False)),
            nn.Tanh() # Output in [-1, 1]
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        c_emb = self.label_emb(labels)
        x = torch.cat([z, c_emb], dim=1)
        x = self.relu(self.bn0(self.project(x).view(-1, self.ngf * 8, 4, 4)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.attn(x)
        x = self.block3(x)
        x = self.block4(x)
        out = self.to_rgb(x)
        return out


# --- Discriminator with Spectral Norm & Two Heads ---
class ACGANDiscriminator(nn.Module):
    """
    Discriminator: Shared conv trunk (spectral norm) -> two heads:
    1. WGAN critic score D(x)
    2. AC-GAN auxiliary 10-way classification head C(x)
    """
    def __init__(self, num_classes: int = len(FASHION_CATEGORIES), ndf: int = 64):
        super().__init__()
        # Shared trunk: 128x128 -> 4x4
        self.trunk = nn.Sequential(
            # 128 -> 64
            nn.utils.spectral_norm(nn.Conv2d(3, ndf, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 -> 32
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(ndf * 2),
            # 32 -> 16
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 -> 8
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 -> 4
            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        # Head 1: WGAN Critic score
        self.adv_head = nn.utils.spectral_norm(nn.Linear(ndf * 8 * 4 * 4, 1))
        # Head 2: AC-GAN auxiliary classification head
        self.cls_head = nn.Sequential(
            nn.Linear(ndf * 8 * 4 * 4, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor):
        feat = self.trunk(x).view(x.size(0), -1)
        critic_score = self.adv_head(feat)
        cls_logits = self.cls_head(feat)
        return critic_score, cls_logits


# --- Gradient Penalty for WGAN-GP ---
def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = discriminator(interpolates)
    fake = torch.ones((real_samples.size(0), 1), device=device, requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def parse_args():
    parser = argparse.ArgumentParser(description="Train AC-GAN Comparative Baseline")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/weave/checkpoints/baseline_acgan")
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--lr_g", type=float, default=2e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--lambda_gp", type=float, default=10.0)
    parser.add_argument("--n_critic", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    sample_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    device = torch.device(args.device)

    logger.info("Initializing AC-GAN Comparative Baseline (WGAN-GP + Spectral Norm + Self-Attention)...")

    # Dataset at 128x128
    dataset = FashionDiffusionDataset(data_dir=args.data_dir, size=128)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    num_classes = len(FASHION_CATEGORIES)
    generator = ACGANGenerator(z_dim=args.z_dim, num_classes=num_classes).to(device)
    discriminator = ACGANDiscriminator(num_classes=num_classes).to(device)

    # EMA Generator setup (decay 0.999)
    ema_generator = copy.deepcopy(generator)
    ema_decay = 0.999

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    cls_criterion = nn.CrossEntropyLoss()

    fixed_z = torch.randn(min(16, num_classes), args.z_dim, device=device)
    fixed_labels = torch.arange(min(16, num_classes), device=device) % num_classes

    step = 0
    for epoch in range(1, args.num_epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.num_epochs}")
        for batch in pbar:
            step += 1
            real_imgs = batch["pixel_values"].to(device)
            real_labels = batch["category_idx"].to(device)
            bsz = real_imgs.size(0)

            # ---------------------
            # Train Discriminator
            # ---------------------
            opt_d.zero_grad()
            z = torch.randn(bsz, args.z_dim, device=device)
            gen_labels = torch.randint(0, num_classes, (bsz,), device=device)
            fake_imgs = generator(z, gen_labels)

            real_critic, real_cls = discriminator(real_imgs)
            fake_critic, fake_cls = discriminator(fake_imgs.detach())

            # WGAN-GP Adversarial loss
            wgan_loss = fake_critic.mean() - real_critic.mean()
            gp = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, device)
            
            # AC-GAN auxiliary classification loss
            cls_loss = cls_criterion(real_cls, real_labels) + cls_criterion(fake_cls, gen_labels)

            d_loss = wgan_loss + args.lambda_gp * gp + cls_loss
            d_loss.backward()
            opt_d.step()

            # -----------------
            # Train Generator (every n_critic steps)
            # -----------------
            if step % args.n_critic == 0:
                opt_g.zero_grad()
                z = torch.randn(bsz, args.z_dim, device=device)
                gen_labels = torch.randint(0, num_classes, (bsz,), device=device)
                gen_imgs = generator(z, gen_labels)

                critic_score, cls_pred = discriminator(gen_imgs)
                g_adv_loss = -critic_score.mean()
                g_cls_loss = cls_criterion(cls_pred, gen_labels)

                g_loss = g_adv_loss + g_cls_loss
                g_loss.backward()
                opt_g.step()

                # Update EMA generator
                with torch.no_grad():
                    for p_ema, p in zip(ema_generator.parameters(), generator.parameters()):
                        p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            pbar.set_postfix({
                "D_loss": f"{d_loss.item():.3f}",
                "WGAN": f"{wgan_loss.item():.3f}",
                "GP": f"{gp.item():.3f}"
            })

        # Save visual sample grid at end of each epoch
        with torch.no_grad():
            ema_generator.eval()
            sample_imgs = ema_generator(fixed_z, fixed_labels)
            # Denormalize [-1, 1] -> [0, 1]
            sample_imgs = (sample_imgs + 1.0) / 2.0
            sample_file = os.path.join(sample_dir, f"epoch_{epoch:03d}.png")
            save_image(sample_imgs, sample_file, nrow=4)

    # Save final baseline generator model
    final_path = os.path.join(args.output_dir, "acgan_baseline_final.pt")
    torch.save({
        "generator_state_dict": ema_generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "z_dim": args.z_dim,
        "num_classes": num_classes,
        "categories": FASHION_CATEGORIES
    }, final_path)

    logger.info(f"AC-GAN Baseline training complete. Checkpoint saved to: {final_path}")
    print(f"\n[DONE] Comparative AC-GAN Baseline saved to {final_path}")

if __name__ == "__main__":
    main()
