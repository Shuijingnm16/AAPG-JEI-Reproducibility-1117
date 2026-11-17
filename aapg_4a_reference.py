import csv
import time
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# Experiment 2B-1
# Full Physics Spatially Varying PSF Validation + Benchmark
# ============================================================

SEED = 2026

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DIV2K_DIR = PROJECT_ROOT / "data" / "DIV2K_train_HR"

PARAM_CSV = (
    PROJECT_ROOT
    / "data"
    / "aberration_parameters"
    / "train_parameters.csv"
)

FIG_DIR = (
    PROJECT_ROOT
    / "figures"
    / "fig7_full_physics_dataloader_v1"
)

FIG_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 256

# Only a few samples are needed for runtime benchmarking
N_TEST_IMAGES = 4

# Two configurations to compare
CONFIGS = [
    {
        "name": "Fast",
        "grid_size": 3,
        "psf_size": 33,
        "pupil_size": 64,
    },
    {
        "name": "HighFidelity",
        "grid_size": 5,
        "psf_size": 65,
        "pupil_size": 128,
    },
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Device
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ============================================================
# Basic utilities
# ============================================================

def load_csv(path):

    if not path.exists():
        raise FileNotFoundError(path)

    rows = []

    with open(path, "r", newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            parsed = {}

            for key, value in row.items():

                try:
                    parsed[key] = float(value)
                except (ValueError, TypeError):
                    parsed[key] = value

            rows.append(parsed)

    return rows


def get_numeric(row, names, default):

    for name in names:

        if name in row:

            try:
                return float(row[name])
            except Exception:
                pass

    return float(default)


def pil_to_tensor(img):

    arr = np.asarray(img).astype(np.float32) / 255.0

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    arr = np.transpose(arr, (2, 0, 1))

    return torch.from_numpy(arr.copy()).float()


def tensor_to_numpy(x):

    x = x.detach().cpu().clamp(0, 1)

    return np.transpose(
        x.numpy(),
        (1, 2, 0)
    )


def psnr(x, y):

    mse = torch.mean(
        (x.float() - y.float()) ** 2
    ).item()

    if mse < 1e-12:
        return 100.0

    return 10.0 * np.log10(1.0 / mse)


def mae(x, y):

    return torch.mean(
        torch.abs(x.float() - y.float())
    ).item()


# ============================================================
# Zernike-like wavefront model
# ============================================================

def make_pupil_coordinates(n):

    coords = torch.linspace(
        -1.0,
        1.0,
        n,
        device=DEVICE
    )

    yy, xx = torch.meshgrid(
        coords,
        coords,
        indexing="ij"
    )

    rho = torch.sqrt(xx ** 2 + yy ** 2)
    theta = torch.atan2(yy, xx)

    pupil = (rho <= 1.0).float()

    return xx, yy, rho, theta, pupil


def zernike_modes(rho, theta):

    # Defocus-like
    z_defocus = (
        2.0 * rho ** 2 - 1.0
    )

    # Astigmatism
    z_astig_0 = (
        rho ** 2 * torch.cos(2.0 * theta)
    )

    z_astig_45 = (
        rho ** 2 * torch.sin(2.0 * theta)
    )

    # Coma
    z_coma_x = (
        (3.0 * rho ** 3 - 2.0 * rho)
        * torch.cos(theta)
    )

    z_coma_y = (
        (3.0 * rho ** 3 - 2.0 * rho)
        * torch.sin(theta)
    )

    # Spherical
    z_spherical = (
        6.0 * rho ** 4
        - 6.0 * rho ** 2
        + 1.0
    )

    return (
        z_defocus,
        z_astig_0,
        z_astig_45,
        z_coma_x,
        z_coma_y,
        z_spherical,
    )


# ============================================================
# Field-dependent aberration coefficients
# ============================================================

def field_coefficients(
    param,
    field_x,
    field_y
):

    r = np.sqrt(
        field_x ** 2 + field_y ** 2
    )

    r = min(r / np.sqrt(2.0), 1.0)

    phi = np.arctan2(
        field_y,
        field_x
    )

    field_power = get_numeric(
        param,
        ["field_power"],
        1.7
    )

    defocus_max = get_numeric(
        param,
        ["defocus_max"],
        0.12
    )

    astig_max = get_numeric(
        param,
        ["astig_max"],
        0.10
    )

    coma_max = get_numeric(
        param,
        ["coma_max"],
        0.12
    )

    spherical_center = get_numeric(
        param,
        ["spherical_center"],
        0.04
    )

    spherical_edge_add = get_numeric(
        param,
        ["spherical_edge_add"],
        0.03
    )

    radial = r ** field_power

    defocus = defocus_max * radial

    astig_0 = (
        astig_max
        * radial
        * np.cos(2.0 * phi)
    )

    astig_45 = (
        astig_max
        * radial
        * np.sin(2.0 * phi)
    )

    coma_x = (
        coma_max
        * radial
        * np.cos(phi)
    )

    coma_y = (
        coma_max
        * radial
        * np.sin(phi)
    )

    spherical = (
        spherical_center
        + spherical_edge_add * radial
    )

    return (
        defocus,
        astig_0,
        astig_45,
        coma_x,
        coma_y,
        spherical,
    )


# ============================================================
# Physics PSF
# ============================================================

def generate_psf(
    param,
    field_x,
    field_y,
    pupil_size,
    psf_size,
):
    """
    Generate one physics-based PSF.

    Important for older macOS + Apple Silicon:
    PyTorch MPS does not support the complex tensors required by
    exp(1j * phase) / FFT on some macOS versions.  Therefore the
    complex optical calculation is performed on CPU.  The final
    real-valued PSF is moved back to DEVICE so the spatial
    convolution can still run on MPS/CUDA when available.
    """

    # Complex-valued pupil/FFT calculation is deliberately kept on CPU.
    psf_device = torch.device("cpu")

    coords = torch.linspace(
        -1.0,
        1.0,
        pupil_size,
        device=psf_device,
        dtype=torch.float32,
    )

    yy, xx = torch.meshgrid(
        coords,
        coords,
        indexing="ij",
    )

    rho = torch.sqrt(xx ** 2 + yy ** 2)
    theta = torch.atan2(yy, xx)
    pupil = (rho <= 1.0).float()

    modes = zernike_modes(
        rho,
        theta,
    )

    coeffs = field_coefficients(
        param,
        field_x,
        field_y,
    )

    wavefront = torch.zeros_like(rho)

    for coefficient, mode in zip(
        coeffs,
        modes,
    ):
        wavefront = (
            wavefront
            + float(coefficient) * mode
        )

    wavefront = wavefront * pupil
    phase = 2.0 * np.pi * wavefront

    # Explicit complex64 construction on CPU avoids MPS complex errors.
    complex_pupil = torch.complex(
        pupil * torch.cos(phase),
        pupil * torch.sin(phase),
    )

    amplitude = torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.ifftshift(
                complex_pupil
            )
        )
    )

    intensity = torch.abs(amplitude) ** 2
    intensity = intensity / torch.sum(intensity).clamp_min(1e-12)

    # Crop PSF around optical axis
    n = pupil_size
    c = n // 2
    half = psf_size // 2

    y0 = c - half
    y1 = c + half + 1
    x0 = c - half
    x1 = c + half + 1

    psf = intensity[
        y0:y1,
        x0:x1,
    ]

    # Safety normalization after cropping
    psf = psf / torch.sum(psf).clamp_min(1e-12)

    # Only the real-valued PSF is transferred to the acceleration device.
    return psf.real.float().to(DEVICE)


# ============================================================
# PSF anchor generation
# ============================================================

def generate_psf_grid(
    param,
    grid_size,
    pupil_size,
    psf_size,
):

    field_positions = np.linspace(
        -1.0,
        1.0,
        grid_size
    )

    psfs = []

    for fy in field_positions:

        row = []

        for fx in field_positions:

            kernel = generate_psf(
                param,
                fx,
                fy,
                pupil_size,
                psf_size
            )

            row.append(kernel)

        psfs.append(row)

    return psfs


# ============================================================
# Spatial blending weights
# ============================================================

def make_blending_weights(
    h,
    w,
    grid_size
):

    ys = torch.linspace(
        0,
        h - 1,
        grid_size,
        device=DEVICE
    )

    xs = torch.linspace(
        0,
        w - 1,
        grid_size,
        device=DEVICE
    )

    yy, xx = torch.meshgrid(
        torch.arange(
            h,
            device=DEVICE,
            dtype=torch.float32
        ),
        torch.arange(
            w,
            device=DEVICE,
            dtype=torch.float32
        ),
        indexing="ij"
    )

    spacing_y = (
        (h - 1)
        / max(grid_size - 1, 1)
    )

    spacing_x = (
        (w - 1)
        / max(grid_size - 1, 1)
    )

    sigma_y = max(
        spacing_y * 0.75,
        1.0
    )

    sigma_x = max(
        spacing_x * 0.75,
        1.0
    )

    weights = []

    for y in ys:

        for x in xs:

            d = (
                ((yy - y) / sigma_y) ** 2
                +
                ((xx - x) / sigma_x) ** 2
            )

            weight = torch.exp(
                -0.5 * d
            )

            weights.append(weight)

    weights = torch.stack(
        weights,
        dim=0
    )

    weights = (
        weights
        / torch.sum(
            weights,
            dim=0,
            keepdim=True
        ).clamp_min(1e-8)
    )

    return weights


# ============================================================
# RGB convolution
# ============================================================

def apply_psf(
    image,
    kernel
):

    # image: C,H,W

    c, h, w = image.shape

    k = kernel.shape[-1]

    pad = k // 2

    kernel = kernel.view(
        1,
        1,
        k,
        k
    )

    kernel = kernel.repeat(
        c,
        1,
        1,
        1
    )

    x = image.unsqueeze(0)

    # Reflect padding avoids dark boundary artifacts
    x = F.pad(
        x,
        (pad, pad, pad, pad),
        mode="reflect"
    )

    y = F.conv2d(
        x,
        kernel,
        groups=c
    )

    return y.squeeze(0)


# ============================================================
# Full spatially varying degradation
# ============================================================

@torch.no_grad()
def full_physics_degradation(
    gt,
    param,
    grid_size,
    pupil_size,
    psf_size,
):

    gt = gt.to(DEVICE)

    _, h, w = gt.shape

    # Generate PSF anchors
    psfs = generate_psf_grid(
        param,
        grid_size,
        pupil_size,
        psf_size
    )

    # Spatial blending weights
    weights = make_blending_weights(
        h,
        w,
        grid_size
    )

    output = torch.zeros_like(
        gt
    )

    weight_index = 0

    for row in psfs:

        for kernel in row:

            blurred = apply_psf(
                gt,
                kernel
            )

            weight = weights[
                weight_index
            ].unsqueeze(0)

            output = (
                output
                + blurred * weight
            )

            weight_index += 1

    output = output.clamp(
        0.0,
        1.0
    )

    return output


# ============================================================
# Image loading
# ============================================================

def load_center_patch(
    image_path,
    patch_size
):

    img = Image.open(
        image_path
    ).convert("RGB")

    w, h = img.size

    if (
        w < patch_size
        or h < patch_size
    ):

        scale = max(
            patch_size / w,
            patch_size / h
        )

        img = img.resize(
            (
                int(np.ceil(w * scale)),
                int(np.ceil(h * scale)),
            ),
            Image.Resampling.BICUBIC
        )

        w, h = img.size

    left = (
        w - patch_size
    ) // 2

    top = (
        h - patch_size
    ) // 2

    img = img.crop(
        (
            left,
            top,
            left + patch_size,
            top + patch_size
        )
    )

    return pil_to_tensor(img)




# ============================================================
# Experiment 4A
# Final Spatial Aberration-Aware AAPG Full Training
# ============================================================

import json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

TRAIN_SAMPLES = 800
VAL_SAMPLES = 100
EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

GRID_SIZE = 5
PSF_SIZE = 65
PUPIL_SIZE = 128

VAL_DIR = PROJECT_ROOT / "data" / "DIV2K_valid_HR"
ID_PARAM_CSV = PROJECT_ROOT / "data" / "aberration_parameters" / "id_test_parameters.csv"
OUT_DIR = PROJECT_ROOT / "figures" / "fig12_final_aapg_training_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CKPT = OUT_DIR / "final_aapg_best.pth"
LAST_CKPT = OUT_DIR / "final_aapg_last.pth"
HISTORY_CSV = OUT_DIR / "training_history.csv"
CONFIG_TXT = OUT_DIR / "training_config.txt"

PHYSICS_KEYS = [
    "defocus_max", "astig_max", "coma_max",
    "spherical_center", "spherical_edge_add", "field_power",
]


def parameter_vector(param):
    defaults = {
        "defocus_max": 0.12, "astig_max": 0.10, "coma_max": 0.12,
        "spherical_center": 0.04, "spherical_edge_add": 0.03,
        "field_power": 1.7,
    }
    return np.asarray(
        [get_numeric(param, [k], defaults[k]) for k in PHYSICS_KEYS],
        dtype=np.float32,
    )


def compute_parameter_stats(parameters):
    x = np.stack([parameter_vector(p) for p in parameters], axis=0)
    mean = x.mean(axis=0).astype(np.float32)
    std = np.maximum(x.std(axis=0).astype(np.float32), 1e-6)
    return mean, std


def make_physics_maps(param, mean, std, h, w):
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    fy, fx = np.meshgrid(yy, xx, indexing="ij")
    r = np.clip(np.sqrt(fx**2 + fy**2) / np.sqrt(2.0), 0.0, 1.0)
    phi = np.arctan2(fy, fx)

    raw = parameter_vector(param)
    dmax, amax, cmax, sc, se, fp = raw
    radial = r ** fp

    maps = np.stack([
        dmax * radial,
        amax * radial * np.cos(2.0 * phi),
        amax * radial * np.sin(2.0 * phi),
        cmax * radial * np.cos(phi),
        cmax * radial * np.sin(phi),
        sc + se * radial,
    ], axis=0).astype(np.float32)

    scale = np.asarray([
        max(float(mean[0] + 3*std[0]), 0.05),
        max(float(mean[1] + 3*std[1]), 0.05),
        max(float(mean[1] + 3*std[1]), 0.05),
        max(float(mean[2] + 3*std[2]), 0.05),
        max(float(mean[2] + 3*std[2]), 0.05),
        max(float(mean[3] + mean[4] + 3*(std[3] + std[4])), 0.03),
    ], dtype=np.float32).reshape(6, 1, 1)

    maps = np.clip(maps / scale, -2.0, 2.0)
    return torch.from_numpy(maps.copy()).float()


class FinalAAPGDataset(Dataset):
    """
    Training:
      - full 800-sample epoch
      - epoch-dependent random image permutation/crop/aberration pairing
    Validation:
      - all 100 DIV2K validation images
      - deterministic center crop
      - deterministic ID aberration pairing
    """
    def __init__(self, image_paths, parameters, n_samples, training, seed, pmean, pstd):
        self.image_paths = list(image_paths)
        self.parameters = list(parameters)
        self.n_samples = n_samples
        self.training = training
        self.seed = seed
        self.pmean = pmean
        self.pstd = pstd
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.n_samples

    def _load_patch(self, path, index, rng):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if w < PATCH_SIZE or h < PATCH_SIZE:
            scale = max(PATCH_SIZE / w, PATCH_SIZE / h)
            img = img.resize(
                (int(np.ceil(w*scale)), int(np.ceil(h*scale))),
                Image.Resampling.BICUBIC,
            )
            w, h = img.size

        if self.training:
            left = rng.randint(0, w - PATCH_SIZE)
            top = rng.randint(0, h - PATCH_SIZE)
        else:
            left = (w - PATCH_SIZE) // 2
            top = (h - PATCH_SIZE) // 2

        return pil_to_tensor(
            img.crop((left, top, left + PATCH_SIZE, top + PATCH_SIZE))
        )

    def __getitem__(self, index):
        if self.training:
            rng = random.Random(
                self.seed + self.epoch * 10000019 + index * 100003
            )
            # 800 samples -> every DIV2K train image participates once/epoch,
            # while epoch-dependent offset changes the ordering.
            image_idx = (index + self.epoch * 137) % len(self.image_paths)
            param_idx = rng.randrange(len(self.parameters))
        else:
            rng = random.Random(self.seed + index * 100003)
            image_idx = index % len(self.image_paths)
            param_idx = index % len(self.parameters)

        image_path = self.image_paths[image_idx]
        param = self.parameters[param_idx]
        gt = self._load_patch(image_path, index, rng)

        degraded = full_physics_degradation(
            gt, param,
            grid_size=GRID_SIZE,
            pupil_size=PUPIL_SIZE,
            psf_size=PSF_SIZE,
        ).detach().cpu()

        physics_maps = make_physics_maps(
            param, self.pmean, self.pstd, PATCH_SIZE, PATCH_SIZE
        )
        return degraded, gt, physics_maps, image_path.name


class PhysicsGuidedBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        h = F.relu(self.conv1(x), inplace=False)
        h = self.conv2(F.relu(h, inplace=False))
        return x + 0.10 * h


class FinalAAPG(nn.Module):
    """Final model selected by Experiments 3A-2/3A-3: spatial maps ON, global FiLM OFF."""
    def __init__(self, channels=56, blocks=5):
        super().__init__()
        self.head = nn.Conv2d(9, channels, 3, padding=1)  # RGB + 6 physics maps
        self.blocks = nn.ModuleList(
            [PhysicsGuidedBlock(channels) for _ in range(blocks)]
        )
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, degraded, physics_maps):
        h = F.relu(self.head(torch.cat([degraded, physics_maps], dim=1)), inplace=False)
        for block in self.blocks:
            h = block(h)
        return degraded + self.tail(h)


def per_image_psnr(pred, target):
    pred = pred.detach().float().clamp(0, 1)
    target = target.detach().float()
    mse = ((pred-target)**2).flatten(1).mean(1).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0/mse)


def batch_ssim(pred, target):
    """Differentiation-free validation SSIM, 11x11 local-window approximation."""
    x = pred.detach().float().clamp(0, 1)
    y = target.detach().float().clamp(0, 1)
    k = 11
    mu_x = F.avg_pool2d(x, k, stride=1, padding=k//2)
    mu_y = F.avg_pool2d(y, k, stride=1, padding=k//2)
    sigma_x = F.avg_pool2d(x*x, k, stride=1, padding=k//2) - mu_x*mu_x
    sigma_y = F.avg_pool2d(y*y, k, stride=1, padding=k//2) - mu_y*mu_y
    sigma_xy = F.avg_pool2d(x*y, k, stride=1, padding=k//2) - mu_x*mu_y
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu_x*mu_y+c1)*(2*sigma_xy+c2)) / (
        (mu_x*mu_x+mu_y*mu_y+c1)*(sigma_x+sigma_y+c2)
    )
    return ssim_map.flatten(1).mean(1)


@torch.no_grad()
def evaluate(model, loader, collect_examples=False):
    model.eval()
    loss_sum = pin_sum = pout_sum = ssim_sum = 0.0
    n = 0
    examples = []

    for degraded, gt, pmaps, names in loader:
        degraded = degraded.to(DEVICE)
        gt = gt.to(DEVICE)
        pmaps = pmaps.to(DEVICE)

        pred = model(degraded, pmaps)
        bs = gt.size(0)

        loss_sum += F.l1_loss(pred, gt).item() * bs
        pin_sum += per_image_psnr(degraded, gt).sum().item()
        pout_sum += per_image_psnr(pred, gt).sum().item()
        ssim_sum += batch_ssim(pred, gt).sum().item()
        n += bs

        if collect_examples and len(examples) < 4:
            for j in range(min(bs, 4-len(examples))):
                examples.append((
                    degraded[j].cpu(),
                    pred[j].clamp(0,1).cpu(),
                    gt[j].cpu(),
                    names[j],
                ))

    return {
        "val_l1": loss_sum/n,
        "input_psnr": pin_sum/n,
        "val_psnr": pout_sum/n,
        "psnr_gain": (pout_sum-pin_sum)/n,
        "val_ssim": ssim_sum/n,
        "examples": examples,
    }


def save_history(history):
    fields = [
        "epoch", "learning_rate", "train_l1", "val_l1",
        "input_psnr", "val_psnr", "psnr_gain", "val_ssim", "epoch_time"
    ]
    with open(HISTORY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(history)


def save_training_curves(history):
    if not history:
        return
    ep = [r["epoch"] for r in history]

    plt.figure(figsize=(7,5))
    plt.plot(ep, [r["train_l1"] for r in history], marker="o", label="Train L1")
    plt.plot(ep, [r["val_l1"] for r in history], marker="s", label="Validation L1")
    plt.xlabel("Epoch"); plt.ylabel("L1 loss")
    plt.title("Final AAPG Training Loss")
    plt.grid(alpha=0.25); plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR/"training_loss.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7,5))
    plt.plot(ep, [r["val_psnr"] for r in history], marker="o", label="Validation PSNR")
    plt.xlabel("Epoch"); plt.ylabel("PSNR (dB)")
    plt.title("Final AAPG Validation PSNR")
    plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(OUT_DIR/"training_psnr.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7,5))
    plt.plot(ep, [r["val_ssim"] for r in history], marker="o", label="Validation SSIM")
    plt.xlabel("Epoch"); plt.ylabel("SSIM")
    plt.title("Final AAPG Validation SSIM")
    plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(OUT_DIR/"training_ssim.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_examples(examples):
    if not examples:
        return
    fig, axes = plt.subplots(3, len(examples), figsize=(4*len(examples), 11))
    if len(examples) == 1:
        axes = np.asarray(axes).reshape(3,1)

    for j, (deg, pred, gt, name) in enumerate(examples):
        pin = per_image_psnr(deg[None], gt[None]).item()
        pout = per_image_psnr(pred[None], gt[None]).item()
        imgs = [gt, deg, pred]
        titles = [
            f"GT\n{name}",
            f"Degraded\nPSNR={pin:.2f} dB",
            f"Final AAPG\nPSNR={pout:.2f} dB",
        ]
        for i in range(3):
            axes[i,j].imshow(imgs[i].permute(1,2,0).numpy().clip(0,1))
            axes[i,j].set_title(titles[i])
            axes[i,j].axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR/"validation_examples.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_psnr,
                    history, pmean, pstd):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_psnr": best_psnr,
        "history": history,
        "parameter_mean": pmean,
        "parameter_std": pstd,
        "physics_keys": PHYSICS_KEYS,
        "architecture": "FinalAAPG-SpatialOnly",
        "grid_size": GRID_SIZE,
        "psf_size": PSF_SIZE,
        "pupil_size": PUPIL_SIZE,
        "patch_size": PATCH_SIZE,
        "seed": SEED,
    }, path)


def write_config(model):
    cfg = {
        "experiment": "4A Final AAPG Full Training",
        "seed": SEED,
        "device": str(DEVICE),
        "architecture": "Spatial-Only AAPG",
        "spatial_maps": True,
        "global_film": False,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "train_samples_per_epoch": TRAIN_SAMPLES,
        "validation_samples": VAL_SAMPLES,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR",
        "loss": "L1",
        "patch_size": PATCH_SIZE,
        "psf_grid": GRID_SIZE,
        "psf_size": PSF_SIZE,
        "pupil_fft": PUPIL_SIZE,
    }
    CONFIG_TXT.write_text(
        "\n".join(f"{k}: {v}" for k,v in cfg.items()) + "\n",
        encoding="utf-8"
    )


def reset_rng():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


if __name__ == "__main__":
    reset_rng()

    print("\n=== Experiment 4A: Final Spatial Aberration-Aware AAPG Full Training ===")
    print(f"Random seed       : {SEED}")
    print(f"Device            : {DEVICE}")
    print(f"Architecture      : Spatial-Only AAPG")
    print(f"Spatial maps      : ON")
    print(f"Global FiLM       : OFF")
    print(f"Patch size        : {PATCH_SIZE} x {PATCH_SIZE}")
    print(f"Train samples     : {TRAIN_SAMPLES}/epoch")
    print(f"Validation samples: {VAL_SAMPLES}")
    print(f"Batch size        : {BATCH_SIZE}")
    print(f"Epochs            : {EPOCHS}")
    print(f"Physics PSF       : {GRID_SIZE}x{GRID_SIZE}, PSF {PSF_SIZE}, pupil {PUPIL_SIZE}")

    train_images = sorted(DIV2K_DIR.glob("*.png"))
    val_images = sorted(VAL_DIR.glob("*.png"))
    train_params = load_csv(PARAM_CSV)
    val_params = load_csv(ID_PARAM_CSV)

    if len(train_images) != 800:
        raise RuntimeError(f"Expected 800 DIV2K train images, found {len(train_images)}")
    if len(val_images) != 100:
        raise RuntimeError(f"Expected 100 DIV2K validation images, found {len(val_images)}")
    if not train_params or not val_params:
        raise RuntimeError("Aberration parameter CSV is empty.")

    pmean, pstd = compute_parameter_stats(train_params)

    train_set = FinalAAPGDataset(
        train_images, train_params, TRAIN_SAMPLES, True, SEED, pmean, pstd
    )
    val_set = FinalAAPGDataset(
        val_images, val_params, VAL_SAMPLES, False, SEED+500000, pmean, pstd
    )

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, generator=g
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = FinalAAPG().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    print(f"Model parameters  : {sum(p.numel() for p in model.parameters()):,}")
    write_config(model)

    start_epoch = 1
    best_psnr = -float("inf")
    history = []

    # Automatic resume from last checkpoint.
    if LAST_CKPT.exists():
        print(f"\nResume checkpoint found: {LAST_CKPT}")
        ckpt = torch.load(LAST_CKPT, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_psnr = float(ckpt.get("best_val_psnr", -float("inf")))
        history = ckpt.get("history", [])
        print(f"Resuming from epoch {start_epoch} | best PSNR={best_psnr:.3f} dB")

    initial = evaluate(model, val_loader)
    print(
        f"\nValidation before training/resume: "
        f"input={initial['input_psnr']:.3f} dB | "
        f"restored={initial['val_psnr']:.3f} dB | "
        f"SSIM={initial['val_ssim']:.4f}"
    )

    if start_epoch > EPOCHS:
        print("Training is already complete. Evaluating best checkpoint only.")
    else:
        total_start = time.perf_counter()

        for epoch in range(start_epoch, EPOCHS+1):
            train_set.set_epoch(epoch)
            model.train()
            running = 0.0
            seen = 0
            epoch_start = time.perf_counter()

            for degraded, gt, pmaps, _ in train_loader:
                degraded = degraded.to(DEVICE)
                gt = gt.to(DEVICE)
                pmaps = pmaps.to(DEVICE)

                optimizer.zero_grad(set_to_none=True)
                pred = model(degraded, pmaps)
                loss = F.l1_loss(pred, gt)
                loss.backward()
                optimizer.step()

                bs = gt.size(0)
                running += loss.item() * bs
                seen += bs

            train_l1 = running / seen
            metrics = evaluate(model, val_loader)
            epoch_time = time.perf_counter() - epoch_start
            lr_now = optimizer.param_groups[0]["lr"]

            row = {
                "epoch": epoch,
                "learning_rate": lr_now,
                "train_l1": train_l1,
                "val_l1": metrics["val_l1"],
                "input_psnr": metrics["input_psnr"],
                "val_psnr": metrics["val_psnr"],
                "psnr_gain": metrics["psnr_gain"],
                "val_ssim": metrics["val_ssim"],
                "epoch_time": epoch_time,
            }
            history.append(row)

            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"lr={lr_now:.6f} | "
                f"train L1={train_l1:.6f} | "
                f"val L1={metrics['val_l1']:.6f} | "
                f"input={metrics['input_psnr']:.3f} dB | "
                f"restored={metrics['val_psnr']:.3f} dB | "
                f"gain={metrics['psnr_gain']:+.3f} dB | "
                f"SSIM={metrics['val_ssim']:.4f} | "
                f"{epoch_time:.1f} s"
            )

            # Best model is selected on validation PSNR before scheduler update.
            if metrics["val_psnr"] > best_psnr:
                best_psnr = metrics["val_psnr"]
                save_checkpoint(
                    BEST_CKPT, model, optimizer, scheduler, epoch,
                    best_psnr, history, pmean, pstd
                )
                print(f"  -> New best checkpoint: {best_psnr:.3f} dB")

            scheduler.step()

            save_checkpoint(
                LAST_CKPT, model, optimizer, scheduler, epoch,
                best_psnr, history, pmean, pstd
            )
            save_history(history)
            save_training_curves(history)

        print(f"\nTotal training time: {(time.perf_counter()-total_start)/3600:.2f} h")

    # Final evaluation always uses the best checkpoint.
    if not BEST_CKPT.exists():
        raise RuntimeError("Best checkpoint was not created.")

    best = torch.load(BEST_CKPT, map_location="cpu")
    model.load_state_dict(best["model_state_dict"])
    model.to(DEVICE)

    final = evaluate(model, val_loader, collect_examples=True)
    save_examples(final["examples"])
    save_history(history)
    save_training_curves(history)

    print("\n=== Experiment 4A Final Results ===")
    print(f"Best epoch        : {best['epoch']}")
    print(f"Input PSNR        : {final['input_psnr']:.3f} dB")
    print(f"Restored PSNR     : {final['val_psnr']:.3f} dB")
    print(f"PSNR gain         : {final['psnr_gain']:+.3f} dB")
    print(f"Validation SSIM   : {final['val_ssim']:.4f}")
    print(f"Best checkpoint   : {BEST_CKPT}")
    print(f"Last checkpoint   : {LAST_CKPT}")
    print(f"Results saved to  : {OUT_DIR}")
    print("Experiment 4A completed.")
