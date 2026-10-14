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

from checkpoint_compat import load_backbone_checkpoint
from Diff_MAE.utils import (
    load_HSI,
    choose_train_and_test_point_t,
    mirror_hsi,
    train_and_test_data_t,
    train_and_test_label_t,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 特征提取函数
def extract_features(model, loader):
    """
    从S2Mamba模型中提取中间层特征（agg_feats），即分类头前的特征向量。
    
    Args:
        model: 预训练的S2Mamba模型
        loader: 数据加载器，包含Houston18的测试样本
    
    Returns:
        features: 提取的特征，形状为 [N, C]
        labels: 对应的标签，形状为 [N]
    """
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for img, label in loader:
            img = img.to(device)
            # 获取features_before_classification [B, H, W, C]
            _, feat_before = model(img)
            # 应用后续处理以获得agg_feats [B, C]

            features.append(feat_before.cpu())
            labels.append(label)
    features = torch.cat(features)
    labels = torch.cat(labels)
    return features, labels

# 保存特征函数
def save_features(features, labels, save_dir, dataset_name="Houston18"):
    """
    将提取的特征和标签保存为numpy文件。
    
    Args:
        features: 提取的特征
        labels: 对应的标签
        save_dir: 保存目录（此参数将被忽略，文件将保存到固定路径）
        dataset_name: 数据集名称，用于文件名
    """
    # 指定固定的保存目录
    fixed_save_dir = "/mnt/HDD/data/wxyxier/jiaotidiff/20260915_开源模型"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_filename = f"_features_final_20260915_Houston.npy"
    # label_filename = f"labels_final_0727_houston30.npy"
    
    # 确保目录存在
    os.makedirs(fixed_save_dir, exist_ok=True)
    
    # 构造文件路径
    feature_path = os.path.join(fixed_save_dir, feature_filename)
    # label_path = os.path.join(fixed_save_dir, label_filename)
    
    # 保存文件
    np.save(feature_path, features.numpy())
    # np.save(label_path, labels.numpy())
    
    # 打印保存信息
    print(f"特征已保存至: {feature_path}")
    # print(f"标签已保存至: {label_path}")
    print(f"特征形状: {features.shape}")
    print(f"标签形状: {labels.shape}")

# PCA降维函数

# 主函数
def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='使用S2Mamba提取Houston18的中间层特征')
    parser.add_argument('--model_path', type=str, 
                        default='/mnt/HDD/data/wxyxier/jiaotidiff/20260915_开源模型/backbone_pretrain.pth',
                        help='预训练模型路径')
    parser.add_argument('--batch_size', type=int, default=65, help='批大小')
    parser.add_argument('--patch_size', type=int, default=9, help='图像块大小')
    parser.add_argument('--output_dir', type=str, 
                        default='/mnt/HDD/data/wxyxier/TIP_返稿',
                        help='特征保存目录')
    args = parser.parse_args()

    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    # 加载Houston18数据
    print("加载Houston18数据集...")
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

    # 准备测试数据加载器
    x_test = torch.from_numpy(x_test.transpose(0, 3, 1, 2)).float()
    y_test = torch.from_numpy(y_test).long()
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"测试样本数量: {len(test_loader.dataset)}")

    # 初始化并加载预训练S2Mamba模型
    print("初始化S2Mamba模型...")
    # 请根据实际训练时的参数调整以下值
    # model = S2Mamba(patch=9, in_chans=48, num_classes=7, depths=[1], dims=[64], d_state=16).to(device)
    # model.load_state_dict(torch.load(args.model_path))
    model = load_backbone_checkpoint(args.model_path, map_location=device)
    model.to(device)  # Ensure the model is on "cuda:1"
    model.eval()
    print("模型已加载并设置为评估模式")

    # 提取特征
    print("提取中间层特征...")
    features, labels = extract_features(model, test_loader)

    # 保存特征
    print("保存特征和标签...")
    save_features(features, labels, args.output_dir)

if __name__ == "__main__":
    main()
