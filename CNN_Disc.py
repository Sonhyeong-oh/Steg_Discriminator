import zipfile
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import optuna
from collections import Counter
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageChops
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from math import log10
import random
import cv2
from torchvision.models import resnet18
import torch.nn.functional as F
from shutil import copyfile

# 오분류 이미지 저장용 폴더 생성
os.makedirs("C:/Users/Admin/Desktop/Image_data/Confusion", exist_ok=True)

# 3. transform 설정
transform = transforms.Compose([
    transforms.ToTensor()
])

criterion = nn.BCEWithLogitsLoss()

# 4방향 라플라시안 필터 + 가우시안 블러

def pil_to_cv2(pil_img):
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

class Dir4LaplacianBlur(nn.Module):
    def __init__(self):
        super().__init__()

        # Laplacian kernel
        lap_kernel = torch.tensor([[0,  -1,  0],
                                   [-1,  4, -1],
                                   [0,  -1,  0]], dtype=torch.float32)
        lap_kernel = lap_kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)  # (3,1,3,3)
        self.register_buffer('lap_kernel', lap_kernel)

        # Gaussian kernel (fixed 3x3, sigma=1)
        gauss_kernel = self._create_gaussian_kernel(3, sigma=1.0)
        gauss_kernel = gauss_kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)  # (3,1,3,3)
        self.register_buffer('gauss_kernel', gauss_kernel)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = x.to(self.lap_kernel.device)
        lap = F.conv2d(x, self.lap_kernel, padding=1, groups=3)  # Laplacian
        blur = F.conv2d(lap, self.gauss_kernel, padding=1, groups=3)  # Gaussian Blur
        hybrid = lap +  blur
        return hybrid

    def _create_gaussian_kernel(self, kernel_size=3, sigma=1.0):
        """Generates a 2D Gaussian kernel."""
        ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / torch.sum(kernel)
        return kernel

class Dir8LaplacianBlur(nn.Module):
    def __init__(self):
        super().__init__()

        # Laplacian kernel
        lap_kernel = torch.tensor([[-1,  -1, -1],
                                   [-1,   8, -1],
                                   [-1,  -1, -1]], dtype=torch.float32)
        lap_kernel = lap_kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)  # (3,1,3,3)
        self.register_buffer('lap_kernel', lap_kernel)

        # Gaussian kernel (fixed 3x3, sigma=1)
        gauss_kernel = self._create_gaussian_kernel(3, sigma=1.0)
        gauss_kernel = gauss_kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)  # (3,1,3,3)
        self.register_buffer('gauss_kernel', gauss_kernel)

    def forward(self, x):
        # x: (B, 3, H, W)
        lap = F.conv2d(x.to(self.lap_kernel.device), self.lap_kernel, padding=1, groups=3)  # Laplacian
        blur = F.conv2d(lap, self.gauss_kernel, padding=1, groups=3)  # Gaussian Blur
        hybrid = lap +  blur
        return hybrid

    def _create_gaussian_kernel(self, kernel_size=3, sigma=1.0):
        """Generates a 2D Gaussian kernel."""
        ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / torch.sum(kernel)
        return kernel

def to_tensor_from_bgr(np_img, device):
    # np_img: (H, W, 3), BGR, uint8
    rgb_img = np_img[..., ::-1].copy()  # BGR → RGB
    tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0  # (3, H, W)
    return tensor.unsqueeze(0).to(device)  # (1, 3, H, W)

def tensor_to_numpy(tensor):
    """PyTorch tensor를 NumPy array로 변환 (CPU로 이동 후)"""
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.detach().numpy()

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === ResNet + 4방향 라플라시안 필터 + 가우시안 블러 모델 ===
class ResNet(nn.Module):
    def __init__(self, in_channels=3, feature_dim=300, mode='discriminator'):
        super().__init__()
        self.lb4 = Dir4LaplacianBlur()
        self.mode = mode
        self.feature_dim = feature_dim

        base = resnet18(weights=None)
        if in_channels != 3:
            base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if mode == 'discriminator':
            base.fc = nn.Linear(512, 1)
        else:
            base.fc = nn.Identity()  # feature 추출용

        self.base = base
        if mode == 'feature':
            self.feature_proj = nn.Linear(512, feature_dim)

    def forward(self, x, center=None):
        x = self.lb4.forward(x)
        features = self.base(x)
        if self.mode == 'discriminator':
            return features
        else:
            return self.feature_proj(features)  # (B, feature_dim)

class StegoDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB') # 이미지를 RGB로 가지고 옴
        if self.transform:
            img = self.transform(img)
        return img, label

# 3. transform 설정
transform = transforms.Compose([
    transforms.ToTensor()
])

def test_dataset(cover_dir, stego_dir, seed=42):
    cover_paths = [os.path.join(cover_dir, f) for f in os.listdir(cover_dir) if f.endswith('.png')]
    stego_paths = [os.path.join(stego_dir, f) for f in os.listdir(stego_dir) if f.endswith('.png')]

    # 라벨링
    cover_labeled = [(path, 0) for path in cover_paths]
    stego_labeled = [(path, 1) for path in stego_paths]

    # Cover + Stego 합치고 셔플
    full_data = cover_labeled + stego_labeled
    random.seed(seed)
    random.shuffle(full_data)

    return full_data


cover_dir = 'C:/Users/Admin/Desktop/Image_data/Test/cover'
stego_dir = 'C:/Users/Admin/Desktop/Image_data/Test/stego'
test_data = test_dataset(cover_dir, stego_dir)
test_data1 = StegoDataset(test_data, transform)
test_loader = DataLoader(test_data1, batch_size=16, shuffle=False)

def evaluate_model(model, dataloader, criterion, device, dataset):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    misclassified_files = []

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(tqdm(dataloader, desc="테스트 진행 중")):
            imgs = imgs.to(device)
            labels = labels.float().to(device)

            outputs = model(imgs).squeeze()
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            preds = (outputs > 0.5).long()

            # 정답 비교
            correct += (preds == labels.long()).sum().item()
            total += imgs.size(0)

            # 전체 정답 및 예측 저장
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 오분류 이미지 파일명 수집
            for i in range(len(preds)):
                if preds[i].item() != labels[i].long().item():
                    index_in_dataset = batch_idx * dataloader.batch_size + i
                    if index_in_dataset < len(dataset.samples):
                        path, _ = dataset.samples[index_in_dataset]
                        misclassified_files.append(os.path.basename(path))  # 또는 전체 경로 저장
            
            # 오분류 이미지 저장
            for i in range(len(preds)):
                if preds[i].item() != labels[i].long().item():
                    index_in_dataset = batch_idx * dataloader.batch_size + i
                    if index_in_dataset < len(dataset.samples):
                        path, _ = dataset.samples[index_in_dataset]
                        misclassified_files.append(os.path.basename(path))  # 파일명 기록

                        # Confusion 폴더로 복사
                        dest_path = os.path.join("C:/Users/Admin/Desktop/Image_data/Confusion", os.path.basename(path))
                        copyfile(path, dest_path)


    avg_loss = total_loss / total
    accuracy = correct / total * 100

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return avg_loss, accuracy, precision, recall, f1, misclassified_files

resmodel_path = "C:/Users/Admin/Desktop/기계학습 프로젝트 (201813784 손형오)/resnet_trained.pth"

# === 모델 불러오기 ===
resmodel = ResNet().to(device)
resmodel.load_state_dict(torch.load(resmodel_path, map_location=device))
# resmodel.eval()

# === 테스트 수행 ===
# avg_loss, accuracy, precision, recall, f1, misclassified_files = evaluate_model(
#     resmodel, test_loader, criterion, device, test_data1
# )

# print(f"""
# === 테스트 결과 ===
# - 평균 Loss : {avg_loss:.4f}
# - 정확도    : {accuracy:.2f}%
# - 정밀도    : {precision:.2f}
# - 민감도    : {recall:.2f}
# - F1 점수   : {f1:.2f}
# - 오분류 이미지 수 : {len(misclassified_files)}
# """)

# # 오분류 파일 저장
# with open("misclassified_images.txt", "w") as f:
#     for fname in misclassified_files:
#         f.write(f"{fname}\n")
