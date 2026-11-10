"""LightlySSL SimCLR + SEANA, using a small synthetic dataset to show the integration.

From the repository root:
    uv run --python 3.12 --extra experiments examples/ssl_pipeline.py

Add --device cuda for GPU execution, or --smoke for a download-free check with
a toy denoiser. Replace the synthetic loader with your noisy training images
and increase the fitting/training budgets for an actual experiment.

Lightly's SimCLR components:
https://docs.lightly.ai/self-supervised-learning/examples/simclr.html
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms import SimCLRViewTransform

from seana import SEANA


class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=None)
        # Small-image stem for this 32x32 fixture. Use the standard stem for 224x224 images.
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SimCLRProjectionHead(512, 512, 128)

    def forward(self, images):
        features = self.backbone(images).flatten(1)
        return self.projection_head(features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--fit-steps", type=int, default=2)
    parser.add_argument(
        "--smoke", action="store_true", help="Use a toy averaging denoiser instead of pretrained DRUNet"
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/library_example/seana.pt"))
    args = parser.parse_args()
    if args.epochs < 1 or args.fit_steps < 1:
        parser.error("--epochs and --fit-steps must be positive")
    if args.checkpoint.exists():
        parser.error("Checkpoint already exists. Choose a new --checkpoint path.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu.")
    torch.manual_seed(0)
    torch.set_num_threads(2)

    # Replace these toy tensors with acquired/noisy training images in [0, 1].
    # Keep the loader free of SSL crops, color transforms, and normalization.
    # ImageFolder or LightlyDataset batches may also include labels/filenames,
    # which this example ignores. Batch images must have matching dimensions.
    loader = DataLoader(TensorDataset(torch.rand(16, 3, 32, 32)), batch_size=8, shuffle=True, drop_last=True)

    # 1. Fit SEANA once on the noisy training split, without labels or clean targets.
    # SEANA() uses pretrained DRUNet. Pass your own nn.Module to use a custom denoiser.
    if args.smoke:
        augmenter = SEANA(nn.AvgPool2d(3, stride=1, padding=1), sigma_vst=0.1, device=args.device)
    else:
        augmenter = SEANA(sigma_d=0.08, sigma_vst=0.1, device=args.device)
    augmenter.fit(loader, steps=args.fit_steps, patch_size=16, patch_batch_size=2, seed=0)

    # 2. Apply a SINGLE-view Lightly transform to each independent SEANA source.
    # Cropping, flipping, color jitter, grayscale and normalization happen here.
    # Gaussian blur is disabled for this small-image recipe.
    ssl_view = SimCLRViewTransform(input_size=32, cj_strength=0.5, gaussian_blur=0.0)
    model = SimCLR().to(args.device).train()
    criterion = NTXentLoss(temperature=0.5)
    # The optimizer contains only the encoder and Lightly projection head.
    # The fitted SEANA module, including its denoiser, stays frozen.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        for step, (images, *_) in enumerate(loader, start=1):
            images = images.to(args.device)
            # Denoise once at the original geometry, then resample independent sources.
            source_a, source_b = augmenter(images)
            # Draw transforms independently for every image and for both branches.
            view_a = torch.stack([ssl_view(image) for image in source_a])
            view_b = torch.stack([ssl_view(image) for image in source_b])
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(view_a), model(view_b))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite SimCLR loss")
            loss.backward()
            optimizer.step()
            print(
                f"epoch={epoch + 1}, step={step}, loss={float(loss.detach()):.4f}, "
                f"view_shape={tuple(view_a.shape)}"
            )

    # Keep model.backbone for downstream feature extraction (discard the projection head).
    # Save/reload the fitted augmentation module for reuse in another SSL pipeline.
    augmenter.save(args.checkpoint)
    loaded = SEANA.load(
        args.checkpoint,
        device=args.device,
        **({"denoiser": nn.AvgPool2d(3, stride=1, padding=1)} if args.smoke else {}),
    )
    torch.testing.assert_close(augmenter.denoise(images), loaded.denoise(images))
    print(f"Saved and reloaded fitted augmenter: {args.checkpoint}")


if __name__ == "__main__":
    main()
