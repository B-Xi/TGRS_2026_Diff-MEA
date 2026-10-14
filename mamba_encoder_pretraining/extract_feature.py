import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for directory in (SCRIPT_DIR, PROJECT_ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from backbone import S2Mamba
from Diff_MAE.utils import (
    load_HSI,
    choose_train_and_test_point_t,
    mirror_hsi,
    train_and_test_data_t,
    train_and_test_label_t,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(model, loader):
    """Extract spatial latent features from a pretrained mamba-based encoder.

    Returns feature tensors of shape [N, H, W, C] and the corresponding labels.
    """
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for img, label in loader:
            img = img.to(device)
            _, feat_before = model(img)

            features.append(feat_before.cpu())
            labels.append(label)
    features = torch.cat(features)
    labels = torch.cat(labels)
    return features, labels

def save_features(features, labels, save_dir, dataset_name="Houston18"):
    """Save latent features to the fixed path used for diffusion pretraining.

    The save_dir argument is retained for compatibility and is not used.
    """
    fixed_save_dir = "."
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_filename = f"_features_Houston.npy"

    os.makedirs(fixed_save_dir, exist_ok=True)

    feature_path = os.path.join(fixed_save_dir, feature_filename)

    np.save(feature_path, features.numpy())
    print(f"Features saved to: {feature_path}")
    print(f"Feature tensor shape: {features.shape}")
    print(f"Label tensor shape: {labels.shape}")

def main():
    parser = argparse.ArgumentParser(description='Extract Houston18 latent features with a mamba-based encoder')
    parser.add_argument('--model_path', type=str, 
                        default=str(PROJECT_ROOT / 'backbone_pretrain_migrated.pth'),
                        help='Path to the pretrained encoder checkpoint')
    parser.add_argument('--batch_size', type=int, default=65, help='Inference batch size')
    parser.add_argument('--patch_size', type=int, default=9, help='Spatial patch size')
    parser.add_argument('--output_dir', type=str, 
                        default='.',
                        help='Retained for compatibility; output uses the fixed path in save_features')
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    print("Loading the Houston18 target-domain dataset...")
    input_normalize, label, num_classes, TR, TE_mask, TE_class, _, _ = \
        load_HSI(type('Args', (), {'dataset': 'Houston18'})())
    height, width, band = input_normalize.shape
    total_pos_train, total_pos_test, total_pos_true, \
    number_train, number_test, number_true = \
        choose_train_and_test_point_t(TR, TE_mask, label, num_classes)
    mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patch_size)
    x_train, x_test = train_and_test_data_t(
        mirror_image, band,
        total_pos_train, total_pos_test, total_pos_true,
        patch=args.patch_size
    )
    _, y_test, y_true = train_and_test_label_t(number_train, number_test, number_true, num_classes)

    x_test = torch.from_numpy(x_test.transpose(0, 3, 1, 2)).float()
    y_test = torch.from_numpy(y_test).long()
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Target-domain samples: {len(test_loader.dataset)}")

    print("Loading the pretrained mamba-based encoder...")
    model = torch.load(args.model_path, map_location=device, weights_only=False)
    if not isinstance(model, S2Mamba):
        raise TypeError(f"Expected a full mamba-based encoder checkpoint, got {type(model).__name__}")
    model.to(device)
    model.eval()
    print("Encoder set to evaluation mode.")

    print("Extracting target-domain latent features...")
    features, labels = extract_features(model, test_loader)

    print("Saving target-domain latent features...")
    save_features(features, labels, args.output_dir)

if __name__ == "__main__":
    main()
