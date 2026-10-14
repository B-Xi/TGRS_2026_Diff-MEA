import os
import sys
import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent / "mamba_encoder_pretraining"))
import backbone
from utils import (
    load_HSI,
    choose_train_and_test_point,
    choose_train_and_test_point_t,
    mirror_hsi,
    train_and_test_data,
    train_and_test_data_t,
    train_and_test_label,
    train_and_test_label_t,
    output_metric,
)
from unet3d import SimpleUnet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEFAULT_SEEDS = ( 94320)


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError('Must be at least 1')
    return parsed


def seed_everything(seed):
    """Initialize all random number generators for a reproducible run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Disable cuDNN benchmarking and request deterministic kernels.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def flip_augmentation(data):
    horizontal = np.random.random() > 0.5
    vertical = np.random.random() > 0.5
    if horizontal:
        data = torch.flip(data, dims=[-2])
    if vertical:
        data = torch.flip(data, dims=[-3])
    return data

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def set_requires_grad(models, requires_grad=True):
    if not isinstance(models, (list, tuple)):
        models = [models]
    for model in models:
        if model is not None:
            for param in model.parameters():
                param.requires_grad = requires_grad

def extract(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape):
    if arr.device != timesteps.device:
        arr = arr.to(timesteps.device)
    res = arr[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2):
    if schedule == "linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = np.arange(n_timestep + 1, dtype=np.float64) / n_timestep + 0.008
        alphas = np.cos((timesteps / 1.008) * np.pi / 2) ** 2
        alphas = alphas / alphas[0]
        betas = 1 - (alphas[1:] / alphas[:-1])
        betas = np.clip(betas, 0, 0.999)
    else:
        raise NotImplementedError(f"Unknown beta schedule: {schedule}")
    return betas

class SSBCE:
    def __init__(self, unet3d_model, K=600, schedule="linear", device=device,
                 parameterization="eps", clip_denoised=True, loss_type="l2"):
        self.model = unet3d_model
        self.K = K
        self.device = device
        self.parameterization = parameterization
        self.clip_denoised = clip_denoised
        self.loss_type = loss_type

        self.betas = make_beta_schedule(schedule, n_timestep=K, linear_start=5e-5, linear_end=1e-2)
        self.betas = torch.from_numpy(self.betas).float().to(device)

        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), self.alphas_cumprod[:-1]]).to(device)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(device)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod).to(device)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        ).to(device)
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).to(device)
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        ).to(device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        shape = x_start.shape
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t, t, noise):
        shape = x_t.shape
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, shape)
        sqrt_recipm1_alphas_t = extract(self.sqrt_one_minus_alphas_cumprod, t, shape) / extract(
            self.sqrt_alphas_cumprod, t, shape
        )
        return sqrt_recip_alphas_t * x_t - sqrt_recipm1_alphas_t * noise

    def p_mean_variance(self, x, t, t_index=None):
        device = x.device
        t = t.to(device)
        model_output = self.model(x, t)
        if self.parameterization == "eps":
            pred_noise = model_output
            x_recon = self.predict_start_from_noise(x, t, pred_noise)
        elif self.parameterization == "x0":
            x_recon = model_output
            pred_noise = self.predict_start_from_noise(x, t, x_recon)
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")
        if self.clip_denoised:
            x_recon = torch.clamp(x_recon, -1.0, 1.0)
        model_mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * x_recon +
            extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        posterior_variance = extract(self.posterior_variance, t, x.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x.shape)
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    def p_sample(self, x, t, t_index=None):
        model_mean, _, model_log_variance, x_recon = self.p_mean_variance(x, t, t_index)
        if t_index == 0:
            return x_recon
        noise = torch.randn_like(x)
        return model_mean + torch.exp(0.5 * model_log_variance) * noise

    def denoise_single_step(self, x_noisy, k):
        t = torch.full((x_noisy.shape[0],), k, device=self.device, dtype=torch.long)
        model_mean, _, _, x_recon = self.p_mean_variance(x_noisy, t)
        return x_recon

    def get_loss(self, x_start, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise=noise)
        self.model = self.model.to(device)
        predicted = self.model(x_noisy, t)
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")
        if self.loss_type == "l1":
            loss = torch.abs(predicted - target).mean()
        elif self.loss_type == "l2":
            loss = F.mse_loss(predicted, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        return loss, x_start, x_noisy, predicted

def log_metrics(seed, epoch, k, oa, aa, kappa, metrics_path):
    log_line = (
        f"Seed {seed}\tEpoch {epoch + 1}\tk {k}\t"
        f"OA: {oa * 100:.2f}%\tAA: {aa * 100:.2f}%\tKappa: {kappa:.4f}\n"
    )
    print(log_line, end="")
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(log_line)

def save_transitional_features(ssbce, source_features, current_k, save_dir, device, batch_size):
    os.makedirs(save_dir, exist_ok=True)
    ssbce.model.eval()
    dataset = TensorDataset(source_features)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for i in range(current_k + 1):
        save_path = os.path.join(save_dir, f"features_k{i}.pt")
        if os.path.exists(save_path):
            continue

        all_features = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Generating transitional features for i={i}"):
                source_batch = batch[0].to(device)
                source_3d = source_batch.permute(0, 3, 1, 2).unsqueeze(1)

                if i == 0:
                    all_features.append(source_batch.cpu())
                else:
                    t_i = torch.full((source_batch.size(0),), i, dtype=torch.long, device=device)
                    x_noisy_i = ssbce.q_sample(source_3d, t_i)
                    denoised_i = ssbce.denoise_single_step(x_noisy_i, i)
                    denoised_i = denoised_i.squeeze(1).permute(0, 2, 3, 1)
                    all_features.append(denoised_i.cpu())

        features_i = torch.cat(all_features, dim=0)
        torch.save(features_i, save_path)
        print(f"Saved transitional features for i={i} to {save_path}")

def train_SubPhase_SA(ssbce, unet3d, s2mamba_model, dataloader, criterion, optimizer, k, epoch_idx):
    set_requires_grad([s2mamba_model.mtda_memory, s2mamba_model.mtda_ssdbranch, s2mamba_model.norm, s2mamba_model.avgpool, s2mamba_model.head], False)
    set_requires_grad(unet3d, True)
    
    total_semantic_loss = 0.0
    total_diffusion_loss = 0.0
    total_loss = 0.0
    
    num_batches = len(dataloader)
    
    for source_batch, label_batch, _ in dataloader:
        source_batch = source_batch.to(device, non_blocking=True)
        source_batch = flip_augmentation(source_batch)
        label_batch = label_batch.to(device, non_blocking=True).long()
        
        optimizer.zero_grad()
        
        source_3d = source_batch.permute(0, 3, 1, 2).unsqueeze(1)
        t = torch.full((source_batch.size(0),), k, dtype=torch.long, device=device)
        diffusion_loss, _, _, _ = ssbce.get_loss(source_3d, t)
        
        x_noisy = ssbce.q_sample(source_3d, t)
        denoised_features = ssbce.denoise_single_step(x_noisy, k)
        
        with torch.no_grad():
            denoised_3d = denoised_features
            denoised_2d = denoised_3d.squeeze(1)
            x_memory = s2mamba_model.mtda_memory(denoised_2d.unsqueeze(2))
            x_ssdbranch = x_memory.permute(0, 3, 1, 2)
            x_ssdbranch = s2mamba_model.mtda_ssdbranch(x_ssdbranch)
            x_ssdbranch = x_ssdbranch.permute(0, 2, 3, 1)
            x_fused = x_memory * 0.5 + x_ssdbranch * 0.5
            x_flat = torch.flatten(x_fused, 1, 2)
            feats = s2mamba_model.norm(x_flat)
            x = s2mamba_model.avgpool(feats.transpose(1, 2))
            agg_feats = torch.flatten(x, 1)
            outputs_source = s2mamba_model.head(agg_feats)
        
        num_classes = outputs_source.size(1)
        label_batch = torch.clamp(label_batch, min=0, max=num_classes-1)
        
        loss_cls = criterion(outputs_source, label_batch)
        
        semantic_loss_weight = 0.1
        total_loss_batch = diffusion_loss + semantic_loss_weight * loss_cls
        
        total_loss_batch.backward()
        optimizer.step()
        
        total_semantic_loss += loss_cls.item()
        total_diffusion_loss += diffusion_loss.item()
        total_loss += total_loss_batch.item()
    
    avg_semantic_loss = total_semantic_loss / num_batches
    avg_diff_loss = total_diffusion_loss / num_batches
    avg_total_loss = total_loss / num_batches
    
    return avg_total_loss, avg_semantic_loss, avg_diff_loss

def train_SubPhase_BA(ssbce, unet3d, s2mamba_model, dataloader, criterion, optimizer, k, epoch_idx, feature_dir):
    set_requires_grad(unet3d, False)
    set_requires_grad([s2mamba_model.mtda_memory, s2mamba_model.mtda_ssdbranch, s2mamba_model.norm, s2mamba_model.avgpool, s2mamba_model.head], True)
    
    total_loss = 0.0
    num_batches = len(dataloader)
    
    for source_batch, label_batch, indices in dataloader:
        source_batch = source_batch.to(device, non_blocking=True)
        source_batch = flip_augmentation(source_batch)
        label_batch = label_batch.to(device, non_blocking=True).long()
        
        current_batch_size = source_batch.size(0)
        
        historical_losses = []
        for i in range(k):
            if i == 0:
                feats_i = source_batch
            else:
                feature_path = os.path.join(feature_dir, f"features_k{i}.pt")
                if os.path.exists(feature_path):
                    saved_feature = torch.load(feature_path, map_location='cpu')
                    feats_i = saved_feature[indices].to(device)
                else:
                    t_i = torch.full((current_batch_size,), i, dtype=torch.long, device=device)
                    source_3d = source_batch.permute(0, 3, 1, 2).unsqueeze(1)
                    x_noisy_i = ssbce.q_sample(source_3d, t_i)
                    denoised_i = ssbce.denoise_single_step(x_noisy_i, i)
                    feats_i = denoised_i.squeeze(1).permute(0, 2, 3, 1)
            
            feats_i_2d = feats_i.permute(0, 3, 1, 2)
            x_memory = s2mamba_model.mtda_memory(feats_i_2d.unsqueeze(2))
            x_ssdbranch = x_memory.permute(0, 3, 1, 2)
            x_ssdbranch = s2mamba_model.mtda_ssdbranch(x_ssdbranch)
            x_ssdbranch = x_ssdbranch.permute(0, 2, 3, 1)
            x_fused = x_memory * 0.5 + x_ssdbranch * 0.5
            x_flat = torch.flatten(x_fused, 1, 2)
            feats_i_norm = s2mamba_model.norm(x_flat)
            x_i = s2mamba_model.avgpool(feats_i_norm.transpose(1, 2))
            agg_feats_i = torch.flatten(x_i, 1)
            outputs_i = s2mamba_model.head(agg_feats_i)
            loss_i = criterion(outputs_i, label_batch)
            historical_losses.append(loss_i)
        
        avg_historical_loss = sum(historical_losses) / len(historical_losses) if historical_losses else 0.0
        
        feature_path_k = os.path.join(feature_dir, f"features_k{k}.pt")
        if os.path.exists(feature_path_k):
            saved_feature_k = torch.load(feature_path_k, map_location='cpu')
            feats_k = saved_feature_k[indices].to(device)
        else:
            t_k = torch.full((current_batch_size,), k, dtype=torch.long, device=device)
            source_3d = source_batch.permute(0, 3, 1, 2).unsqueeze(1)
            x_noisy_k = ssbce.q_sample(source_3d, t_k)
            denoised_k = ssbce.denoise_single_step(x_noisy_k, k)
            feats_k = denoised_k.squeeze(1).permute(0, 2, 3, 1)
        
        feats_k_2d = feats_k.permute(0, 3, 1, 2)
        x_memory_k = s2mamba_model.mtda_memory(feats_k_2d.unsqueeze(2))
        x_ssdbranch_k = x_memory_k.permute(0, 3, 1, 2)
        x_ssdbranch_k = s2mamba_model.mtda_ssdbranch(x_ssdbranch_k)
        x_ssdbranch_k = x_ssdbranch_k.permute(0, 2, 3, 1)
        feats_k_fused = x_memory_k * 0.5 + x_ssdbranch_k * 0.5
        feats_k_flat = torch.flatten(feats_k_fused, 1, 2)
        feats_k_norm = s2mamba_model.norm(feats_k_flat)
        x_k = s2mamba_model.avgpool(feats_k_norm.transpose(1, 2))
        agg_feats_k = torch.flatten(x_k, 1)
        outputs_k = s2mamba_model.head(agg_feats_k)
        loss_k = criterion(outputs_k, label_batch)
        
        total_loss_batch = avg_historical_loss + loss_k
        
        optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
    
    avg_loss = total_loss / num_batches
    return avg_loss, avg_loss, 0.0

def extract_features(model, loader):
    model.eval()
    features = []
    labels = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for img, label in loader:
            img = img.to(device)
            model = model.to(device)
            _, feat = model(img)
            features.append(feat.cpu())
            labels.append(label)
    return torch.cat(features), torch.cat(labels)

def evaluate_s2mamba(model, dataloader, device, num_classes):
    model.eval()
    all_preds, all_targets = [], []
    with torch.inference_mode():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            outputs, _ = model(images)
            all_preds.append(outputs.argmax(dim=1).cpu())
            all_targets.append(targets)
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    return output_metric(all_targets, all_preds, num_classes=num_classes)

def prepare_data(args):
    """Preprocess data once; construct a seed-specific data loader for each run."""
    src_args = type('Args', (), {'dataset': 'Houston'})()
    inp_s, lab_s, cls_s, TR_s, TE_mask_s, TE_class_s, _, _ = load_HSI(src_args)
    h_s, w_s, band_s = inp_s.shape
    total_pos_tr_s, total_pos_te_s, total_pos_true_s, num_tr_s, num_te_s, num_true_s = (
        choose_train_and_test_point(TR_s, TE_mask_s, lab_s, cls_s)
    )
    mir_s = mirror_hsi(h_s, w_s, band_s, inp_s, patch=args.patch_size)
    x_tr_s, _ = train_and_test_data(
        mir_s, band_s, total_pos_tr_s, total_pos_te_s, total_pos_true_s,
        patch=args.patch_size
    )
    y_tr_s, _, _ = train_and_test_label(
        num_tr_s, num_te_s, num_true_s, cls_s
    )
    x_tr_s = torch.from_numpy(x_tr_s.transpose(0, 3, 1, 2)).float()
    y_tr_s = torch.from_numpy(y_tr_s).long()

    target_args = type('Args', (), {'dataset': 'Houston18'})()
    input_normalize, label, num_classes, TR, TE_mask, TE_class, _, _ = load_HSI(target_args)
    height, width, band = input_normalize.shape
    total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = (
        choose_train_and_test_point_t(TR, TE_mask, label, num_classes)
    )
    mirror_image = mirror_hsi(
        height, width, band, input_normalize, patch=args.patch_size
    )
    _, x_test = train_and_test_data_t(
        mirror_image, band, total_pos_train, total_pos_test, total_pos_true,
        patch=args.patch_size
    )
    _, y_test, _ = train_and_test_label_t(
        number_train, number_test, number_true, num_classes
    )
    x_test = torch.from_numpy(x_test.transpose(0, 3, 1, 2)).float()
    y_test = torch.from_numpy(y_test).long()

    print(f"Source domain training samples: {len(y_tr_s)}")
    print(f"Test samples: {len(y_test)}")
    return {
        'x_train': x_tr_s,
        'y_train': y_tr_s,
        'x_test': x_test,
        'y_test': y_test,
        'band': band,
        'num_classes': num_classes,
    }


def choose_seeds(args):
    """Return the predefined or explicitly supplied set of random seeds."""
    seeds = args.seeds
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique")
    return seeds


def write_rows_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_best_and_final(rows, metric_suffix=''):
    """Select the highest-OA epoch and the final epoch."""
    best = max(rows, key=lambda row: (row[f'oa{metric_suffix}'], -row['epoch']))
    final = max(rows, key=lambda row: row['epoch'])
    return [('best_epoch', best), ('final_epoch', final)]


def format_class_accuracies(row, class_fields, aggregate=False):
    """Format class-wise accuracies using one-based class indices."""
    parts = []
    for class_index, field in enumerate(class_fields, start=1):
        value = row[f'{field}_mean'] if aggregate else row[field]
        if value is None:
            parts.append(f'Class {class_index}: N/A')
        elif aggregate:
            std = row[f'{field}_std']
            parts.append(f'Class {class_index}: {value * 100:.2f}% ± {std * 100:.2f}%')
        else:
            parts.append(f'Class {class_index}: {value * 100:.2f}%')
    return ' | '.join(parts)


def run_one_seed(args, seed, data):
    seed_everything(seed)

    seed_name = f"seed_{seed}"
    seed_feature_dir = os.path.join(args.feature_dir, seed_name)
    seed_output_dir = os.path.join(args.output_dir, seed_name)
    seed_model_dir = os.path.join(args.final_model_dir, seed_name)
    os.makedirs(seed_feature_dir, exist_ok=True)
    os.makedirs(seed_output_dir, exist_ok=True)
    os.makedirs(seed_model_dir, exist_ok=True)

    old_feature_files = [
        name for name in os.listdir(seed_feature_dir)
        if name.startswith("features_k") and name.endswith(".pt")
    ]
    if old_feature_files and not args.overwrite_seed_features:
        raise FileExistsError(
            f"{seed_feature_dir} already contains intermediate features. "
            "Use --overwrite_seed_features to rerun these seeds from scratch."
        )
    if args.overwrite_seed_features:
        for name in old_feature_files:
            os.remove(os.path.join(seed_feature_dir, name))

    metrics_text_path = os.path.join(seed_output_dir, "epoch_metrics.txt")
    with open(metrics_text_path, "w", encoding="utf-8") as f:
        f.write("")

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    train_set = TensorDataset(data['x_train'], data['y_train'])
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        generator=loader_generator,
    )
    test_dataset = TensorDataset(data['x_test'], data['y_test'])
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"\n{'=' * 70}")
    print(f"Starting seed {seed}; features -> {seed_feature_dir}")
    print(f"{'=' * 70}")

    # Reload the same pretrained checkpoint independently for each seed.
    s2mamba_model = torch.load(args.s2mamba_model, map_location=device, weights_only=False)
    if not isinstance(s2mamba_model, backbone.S2Mamba):
        raise TypeError(f"Expected a full S2Mamba model, got {type(s2mamba_model).__name__}")
    s2mamba_model.to(device)
    s2mamba_model.eval()
    set_requires_grad([s2mamba_model.patch_embed, s2mamba_model.layers], False)

    unet3d = SimpleUnet(_image_channels=1).to(device)
    unet_state = torch.load(args.unet3d_model, map_location=device)
    unet3d.load_state_dict(unet_state)
    ssbce = SSBCE(
        unet3d_model=unet3d,
        K=args.diffusion_steps,
        device=device,
    )

    print("Extracting source features...")
    source_features, source_labels = extract_features(s2mamba_model, train_loader)
    if source_features.size(0) != source_labels.size(0):
        raise ValueError(
            f"Source features ({source_features.size(0)}) and "
            f"labels ({source_labels.size(0)}) mismatch"
        )
    print("Source features shape:", source_features.shape)

    source_indices = torch.arange(source_features.size(0))
    feature_dataset = TensorDataset(
        source_features, source_labels, source_indices
    )
    feature_generator = torch.Generator()
    feature_generator.manual_seed(seed + 1)
    feature_loader = DataLoader(
        feature_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        generator=feature_generator,
    )

    criterion = nn.CrossEntropyLoss()
    xuexilv = args.adaptation_lr
    seed_rows = []
    best_epoch_oa = -float('inf')
    class_fields = [
        f'class_{class_index}_accuracy'
        for class_index in range(1, data['num_classes'] + 1)
    ]

    for epoch in range(args.num_epochs):
        print(f"\n=== Seed {seed} | Epoch {epoch + 1}/{args.num_epochs} ===")
        k = min(epoch + 1, args.diffusion_steps)
        print(f"Processing k={k}")

        print(f"\n--- Starting SubPhase_SA for k={k} ---")
        sa_optim = optim.Adam(
            unet3d.parameters(),
            lr=xuexilv,
            weight_decay=args.weight_decay,
        )
        sa_losses = []
        sa_counter = 0
        sa_stopped_early = False
        for mini_epoch in range(args.max_sa_epochs):
            sa_loss, semantic_loss, diff_loss = train_SubPhase_SA(
                ssbce, unet3d, s2mamba_model, feature_loader,
                criterion, sa_optim, k, epoch
            )
            if mini_epoch == 0:
                initial_sa_loss = sa_loss
                sa_losses.append(sa_loss)
                continue

            if sa_loss < sa_losses[-1]:
                sa_counter += 1
            else:
                sa_counter = 0
            sa_losses.append(sa_loss)
            if (
                sa_counter >= args.consecutive_drops
                and sa_loss < initial_sa_loss
            ):
                sa_stopped_early = True
                break
        print(
            f"SubPhase_SA final iteration {mini_epoch + 1}/{args.max_sa_epochs}: "
            f"loss={sa_loss:.4f}, semantic_loss={semantic_loss:.4f}, "
            f"diffusion_loss={diff_loss:.4f}, initial_loss={initial_sa_loss:.4f}, "
            f"consecutive_drops={sa_counter}, "
            f"stop={'early' if sa_stopped_early else 'max_iterations'}"
        )

        save_transitional_features(
            ssbce, source_features, k, seed_feature_dir,
            device, args.batch_size
        )

        print(f"\n--- Starting SubPhase_BA for k={k} ---")
        ba_optim = optim.Adam(
            list(s2mamba_model.mtda_memory.parameters())
            + list(s2mamba_model.mtda_ssdbranch.parameters())
            + list(s2mamba_model.norm.parameters())
            + list(s2mamba_model.avgpool.parameters())
            + list(s2mamba_model.head.parameters()),
            lr=xuexilv,
            weight_decay=args.weight_decay,
        )
        ba_losses = []
        ba_counter = 0
        ba_stopped_early = False
        for mini_epoch in range(args.max_ba_epochs):
            ba_loss, _, _ = train_SubPhase_BA(
                ssbce, unet3d, s2mamba_model, feature_loader,
                criterion, ba_optim, k, epoch,
                feature_dir=seed_feature_dir,
            )
            if mini_epoch == 0:
                initial_ba_loss = ba_loss
                ba_losses.append(ba_loss)
                continue

            if ba_loss < ba_losses[-1]:
                ba_counter += 1
            else:
                ba_counter = 0
            ba_losses.append(ba_loss)
            if (
                ba_counter >= args.consecutive_drops
                and ba_loss < initial_ba_loss
            ):
                ba_stopped_early = True
                break
        print(
            f"SubPhase_BA final iteration {mini_epoch + 1}/{args.max_ba_epochs}: "
            f"loss={ba_loss:.4f}, initial_loss={initial_ba_loss:.4f}, "
            f"consecutive_drops={ba_counter}, "
            f"stop={'early' if ba_stopped_early else 'max_iterations'}"
        )

        # Evaluate every epoch to identify the highest-OA checkpoint.
        oa, aa, kappa, class_accuracies = evaluate_s2mamba(
            s2mamba_model, test_loader, device, data['num_classes']
        )
        if len(class_accuracies) != len(class_fields):
            raise ValueError("The number of class accuracies does not match the target classes")
        log_metrics(
            seed, epoch, k, oa, aa, kappa, metrics_text_path
        )
        row = {
            'seed': seed,
            'epoch': epoch + 1,
            'k': k,
            'oa': float(oa),
            'aa': float(aa),
            'kappa': float(kappa),
            'sa_loss': float(sa_loss),
            'ba_loss': float(ba_loss),
        }
        row.update({
            field: float(value) if np.isfinite(value) else None
            for field, value in zip(class_fields, class_accuracies)
        })
        seed_rows.append(row)
        write_rows_csv(
            os.path.join(seed_output_dir, "epoch_metrics.csv"),
            seed_rows,
            list(row.keys()),
        )

        if oa > best_epoch_oa:
            best_epoch_oa = oa
            torch.save(
                s2mamba_model.state_dict(),
                os.path.join(seed_model_dir, "best_epoch_model.pth"),
            )

    final_model_path = os.path.join(seed_model_dir, "final_model.pth")
    torch.save(s2mamba_model.state_dict(), final_model_path)
    print(f"Seed {seed} final model saved to {final_model_path}")

    selected_rows = [
        {
            'selection': selection,
            'epoch': row['epoch'],
            'oa': row['oa'],
            'aa': row['aa'],
            'kappa': row['kappa'],
            **{field: row[field] for field in class_fields},
        }
        for selection, row in select_best_and_final(seed_rows)
    ]
    write_rows_csv(
        os.path.join(seed_output_dir, 'performance_summary.csv'),
        selected_rows,
        ('selection', 'epoch', 'oa', 'aa', 'kappa', *class_fields),
    )
    print(f"\nSeed {seed} classification metrics:")
    for row in selected_rows:
        print(
            f"{row['selection']} | Epoch {row['epoch']:02d} | "
            f"OA {row['oa'] * 100:.2f}% | AA {row['aa'] * 100:.2f}% | "
            f"Kappa {row['kappa']:.4f}"
        )
        print(f"Per-class accuracy | {format_class_accuracies(row, class_fields)}")

    del ssbce, unet3d, s2mamba_model, source_features, source_labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return seed_rows


def aggregate_seed_results(all_rows, seeds, output_dir, rank_by):
    expected = len(seeds)
    class_fields = [
        field for field in all_rows[0] if field.startswith('class_')
    ]
    summary = []
    epochs = sorted({row['epoch'] for row in all_rows})
    for epoch in epochs:
        epoch_rows = [row for row in all_rows if row['epoch'] == epoch]
        if len(epoch_rows) != expected:
            raise RuntimeError(
                f"Epoch {epoch} has {len(epoch_rows)} seed results; "
                f"expected {expected}"
            )

        result = {'epoch': epoch, 'num_seeds': len(epoch_rows)}
        for metric in ('oa', 'aa', 'kappa'):
            values = np.asarray(
                [row[metric] for row in epoch_rows], dtype=np.float64
            )
            result[f'{metric}_mean'] = float(values.mean())
            result[f'{metric}_std'] = float(
                values.std(ddof=1) if len(values) > 1 else 0.0
            )
        for field in class_fields:
            values = [row[field] for row in epoch_rows if row[field] is not None]
            result[f'{field}_mean'] = float(np.mean(values)) if values else None
            result[f'{field}_std'] = (
                float(np.std(values, ddof=1)) if len(values) > 1
                else 0.0 if values else None
            )
        summary.append(result)

    other_metric = 'aa_mean' if rank_by == 'oa' else 'oa_mean'
    summary.sort(
        key=lambda row: (row[f'{rank_by}_mean'], row[other_metric]),
        reverse=True,
    )
    for rank, row in enumerate(summary, start=1):
        row['rank'] = rank

    raw_fields = [
        'seed', 'epoch', 'k', 'oa', 'aa', 'kappa', 'sa_loss', 'ba_loss',
        *class_fields,
    ]
    write_rows_csv(
        os.path.join(output_dir, "all_seed_epoch_metrics.csv"),
        all_rows,
        raw_fields,
    )
    summary_fields = [
        'rank', 'epoch', 'num_seeds',
        'oa_mean', 'oa_std', 'aa_mean', 'aa_std',
        'kappa_mean', 'kappa_std',
        *(field for class_field in class_fields
          for field in (f'{class_field}_mean', f'{class_field}_std')),
    ]
    write_rows_csv(
        os.path.join(output_dir, "epoch_ranking.csv"),
        summary,
        summary_fields,
    )

    selected_mean_rows = [
        {'selection': selection, **{
            field: row[field]
            for field in summary_fields if field != 'rank'
        }}
        for selection, row in select_best_and_final(summary, '_mean')
    ]
    write_rows_csv(
        os.path.join(output_dir, 'performance_summary.csv'),
        selected_mean_rows,
        ('selection', *(field for field in summary_fields if field != 'rank')),
    )

    report = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'selected_seeds': seeds,
        'num_epochs': len(epochs),
        'rank_by': f'{rank_by}_mean_desc',
        'std': 'sample standard deviation (ddof=1)',
        'ranking': summary,
        'best_and_final': selected_mean_rows,
    }
    with open(
        os.path.join(output_dir, "experiment_summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nClassification performance across {expected} seeds (mean ± sample SD):")
    for row in selected_mean_rows:
        print(
            f"{row['selection']} | Epoch {row['epoch']:02d} | "
            f"OA {row['oa_mean'] * 100:.2f}% ± {row['oa_std'] * 100:.2f}% | "
            f"AA {row['aa_mean'] * 100:.2f}% ± {row['aa_std'] * 100:.2f}% | "
            f"Kappa {row['kappa_mean']:.4f} ± {row['kappa_std']:.4f}"
        )
        print(f"Per-class accuracy | {format_class_accuracies(row, class_fields, aggregate=True)}")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description='10-seed, 20-epoch Houston domain adaptation experiment'
    )
    parser.add_argument(
        '--encoder_model', '--s2mamba_model',
        dest='s2mamba_model',
        type=str,
        default='/mnt/HDD/data/wxyxier/jiaotidiff/20260915_开源模型/backbone_pretrain_migrated.pth',
    )
    parser.add_argument(
        '--unet3d_model',
        type=str,
        default='/mnt/HDD/data/wxyxier/jiaotidiff/20260915_开源模型/feature_diffusion_epoch_100.pt',
    )
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=512)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--adaptation_lr', type=float, default=1e-3)
    parser.add_argument('--patch_size', type=int, default=9)
    parser.add_argument(
        '--num_epochs', type=int, default=20, choices=(20,),
        help='Fixed at 20 for this experiment',
    )
    parser.add_argument('--diffusion_steps', type=int, default=600)
    parser.add_argument('--max_sa_epochs', type=positive_int, default=100)
    parser.add_argument('--max_ba_epochs', type=positive_int, default=20)
    parser.add_argument('--consecutive_drops', type=int, default=2)
    parser.add_argument(
        '--feature_dir',
        type=str,
        default='/mnt/HDD/data/wxyxier/jiaotidiff/20260915_开源模型/10seed_20epoch_features',
        help='Root directory; actual features are saved under seed_<seed>/',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/mnt/HDD/data/wxyxier/jiaotidiff/20260915_开源模型/10seed_20epoch_results',
    )
    parser.add_argument(
        '--final_model_dir',
        type=str,
        default='/mnt/HDD/data/wxyxier/jiaotidiff/20260915_开源模型/10seed_20epoch_models',
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=DEFAULT_SEEDS,
        help='Optional explicit seed list; defaults to the 10 specified seeds',
    )
    parser.add_argument('--rank_by', choices=('oa', 'aa'), default='oa')
    parser.add_argument(
        '--overwrite_seed_features',
        action='store_true',
        help='Delete existing features_k*.pt in selected seed folders first',
    )
    return parser


def main():
    args = build_parser().parse_args()
    os.makedirs(args.feature_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.final_model_dir, exist_ok=True)

    seeds = choose_seeds(args)
    print(f"Selected {len(seeds)} seeds: {seeds}")
    with open(
        os.path.join(args.output_dir, "selected_seeds.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                'seeds': seeds,
                'num_epochs': args.num_epochs,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    data = prepare_data(args)
    all_rows = []
    for run_index, seed in enumerate(seeds, start=1):
        print(f"\nRunning seed {run_index}/{len(seeds)}: {seed}")
        all_rows.extend(run_one_seed(args, seed, data))

    aggregate_seed_results(
        all_rows, seeds, args.output_dir, args.rank_by
    )


if __name__ == "__main__":
    main()
