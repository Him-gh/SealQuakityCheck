import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

class SegmentationDataset(Dataset):
    def __init__(self, tif_files, png_files, transform=None):
        """
        tif_files: tif文件路径列表
        png_files: png文件路径列表
        transform: albumentations transform
        """
        self.tif_files = tif_files
        self.png_files = png_files
        self.transform = transform
    
    def __len__(self):
        return len(self.tif_files)
    
    def __getitem__(self, idx):
        # 读取图像 (伪彩色tif，3通道)
        image = cv2.imread(self.tif_files[idx], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图像: {self.tif_files[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取标签 (二值png)
        mask = cv2.imread(self.png_files[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法读取标签: {self.png_files[idx]}")
        mask = (mask == 255).astype(np.float32)  # 转为0/1
        
        # 应用增广
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)  # 添加通道维度 (1, H, W)
        
        return image, mask


def get_transforms(target_size=(256, 256), augment=False):
    """
    获取数据转换
    
    target_size: 目标尺寸 (height, width)
    augment: 是否启用数据增广
    """
    # 基础转换：调整尺寸和归一化
    base_transforms = [
        A.Resize(height=target_size[0], width=target_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    
    if augment:
        # 训练时的数据增广
        transforms = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), 
                     rotate=(-30, 30), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        # 验证时只调整尺寸和归一化
        transforms = A.Compose(base_transforms)
    
    return transforms


def get_loaders(data_dir, batch_size=4, val_split=0.2, augment=True, 
                num_workers=0, target_size=(256, 256)):
    """
    获取训练和验证数据加载器
    
    data_dir: data文件夹路径
    batch_size: 批次大小
    val_split: 验证集比例
    augment: 是否对训练集做数据增广
    num_workers: 数据加载线程数
    target_size: 统一的目标尺寸 (height, width)
    """
    # 获取所有文件
    tif_files = sorted(glob(os.path.join(data_dir, "*.tif")))
    png_files = sorted(glob(os.path.join(data_dir, "*.png")))
    
    assert len(tif_files) == len(png_files), f"图片和标签数量不匹配: {len(tif_files)} vs {len(png_files)}"
    print(f"找到 {len(tif_files)} 对图像")
    
    # 划分训练集和验证集
    train_tif, val_tif, train_png, val_png = train_test_split(
        tif_files, png_files,
        test_size=val_split,
        random_state=42,
        stratify=None
    )
    
    print(f"训练集: {len(train_tif)} 对, 验证集: {len(val_tif)} 对")
    
    # 创建数据集
    train_transform = get_transforms(target_size, augment=augment)
    val_transform = get_transforms(target_size, augment=False)
    
    train_dataset = SegmentationDataset(train_tif, train_png, transform=train_transform)
    val_dataset = SegmentationDataset(val_tif, val_png, transform=val_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True  # 丢弃最后一个不完整的batch
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


# 检查数据集的辅助函数
def check_dataset(data_dir):
    """检查数据集状态"""
    tif_files = sorted(glob(os.path.join(data_dir, "*.tif")))
    png_files = sorted(glob(os.path.join(data_dir, "*.png")))
    
    print(f"TIF文件数量: {len(tif_files)}")
    print(f"PNG文件数量: {len(png_files)}")
    
    if len(tif_files) > 0:
        # 检查第一张图片的尺寸
        img = cv2.imread(tif_files[0])
        mask = cv2.imread(png_files[0], cv2.IMREAD_GRAYSCALE)
        print(f"示例图片尺寸: {img.shape}")
        print(f"示例标签尺寸: {mask.shape}")
        
        # 检查所有图片尺寸是否一致
        sizes = set()
        for f in tif_files[:10]:  # 只检查前10张
            img = cv2.imread(f)
            sizes.add(img.shape[:2])
        
        if len(sizes) > 1:
            print(f"⚠️ 警告: 发现不同尺寸的图片: {sizes}")
            print("建议: 使用 target_size 参数统一尺寸")
        else:
            print(f"✓ 所有图片尺寸一致: {sizes.pop()}")
    
    return tif_files, png_files

# 使用示例
if __name__ == "__main__":
    # 检查数据集
    check_dataset('data')
    
    # 创建数据加载器
    train_loader, val_loader = get_loaders(
        data_dir='data',
        batch_size=4,
        val_split=0.2,
        augment=True,
        num_workers=2,  # 可以设置 >0 了
        target_size=(256, 256)  # 统一尺寸
    )
    
    # 测试数据加载
    for images, masks in train_loader:
        print(f"Batch images shape: {images.shape}")
        print(f"Batch masks shape: {masks.shape}")
        break