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
from optuna.trial import TrialState
import json
from torch.utils.data import ConcatDataset, random_split
import time

# === 순수 ResNet18 모델 정의 ===
class ResNet_vanilla(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(weights=None)
        base.fc = nn.Linear(512, 1)
        self.base = base

    def forward(self, x):
        return self.base(x)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)

# 3. transform 설정
transform = transforms.Compose([
    transforms.ToTensor()
])

criterion = nn.BCEWithLogitsLoss()

# 4방향 라플라시안 필터 + 가우시안 블러

def pil_to_cv2(pil_img):
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

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

# 모델 평가 함수 (f1, 정확도, 평균 Loss 반환)
def evaluate_model(model, val_loader, criterion, device):
    """모델 평가 함수"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            outputs = model(imgs).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 예측값 계산 (sigmoid 적용 후 0.5 기준으로 분류)
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # F1 스코어 계산
    f1 = f1_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(val_loader)

    return f1, accuracy, avg_loss

def evaluate_model_for_real(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    misclassified_files = []

    with torch.no_grad():
        for batch_idx, (imgs, labels, paths) in enumerate(tqdm(dataloader, desc="테스트 진행 중")):
            imgs = imgs.to(device)
            labels = labels.float().to(device)
        
            outputs = model(imgs).squeeze()
            loss = criterion(outputs, labels)
        
            total_loss += loss.item() * imgs.size(0)
            preds = (outputs > 0.5).long()
        
            correct += (preds == labels.long()).sum().item()
            total += imgs.size(0)
        
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
            # 오분류 이미지 경로 저장
            for pred, label, path in zip(preds.cpu().numpy(), labels.cpu().numpy(), paths):
                if pred != int(label):
                    misclassified_files.append(path)

    avg_loss = total_loss / total
    accuracy = correct / total * 100

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return avg_loss, accuracy, precision, recall, f1, misclassified_files

class PTStegoDataset(Dataset):
    def __init__(self, pt_path, transform=None):
        data = torch.load(pt_path, map_location='cpu')
        self.images = data['images']
        self.labels = data['labels']
        self.filenames = data['filenames']
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        filename = self.filenames[idx]
        if self.transform:
            img = self.transform(img)
        return img, label, filename

# 모델 평가 함수 (f1, 정확도, 평균 Loss 반환)
def evaluate_model(model, val_loader, criterion, device):
    """모델 평가 함수"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels, filenames in val_loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            outputs = model(imgs).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 예측값 계산 (sigmoid 적용 후 0.5 기준으로 분류)
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # F1 스코어 계산
    f1 = f1_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(val_loader)

    return f1, accuracy, avg_loss


# 병합
cover_dataset = PTStegoDataset("C:/Users/Admin/Desktop/Python/Attention_Disc/adaptive_train_cover.pt")
stego_dataset = PTStegoDataset("C:/Users/Admin/Desktop/Python/Attention_Disc/adaptive_train_stego.pt")
full_dataset = ConcatDataset([cover_dataset, stego_dataset])

# train/val 분할 (예: 80/20)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Early Stopping 클래스 (재사용)
class EarlyStopping:
    """Early Stopping 구현"""
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True, verbose=True):
        """
        Parameters:
        - patience: 개선되지 않아도 기다릴 에포크 수
        - min_delta: 개선으로 간주할 최소 변화량
        - restore_best_weights: 최고 성능 모델로 복원할지 여부
        - verbose: 출력 여부
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_score, model=None):
        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """최고 성능 모델 저장"""
        if model and self.restore_best_weights:
            self.best_weights = model.state_dict().copy()

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    model = ResNet_vanilla().to(device)
    model.apply(init_weights)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    best_f1 = 0
    early_stopping = EarlyStopping(patience=5, min_delta=0.005, restore_best_weights=True, verbose=False)

    for epoch in range(15):
        model.train()
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            outputs = model(imgs).squeeze()
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        f1, acc, val_loss = evaluate_model(model, val_loader, criterion, device)
        early_stopping(-val_loss, model)
        if f1 > best_f1:
            best_f1 = f1
        if early_stopping.early_stop:
            break

    if early_stopping.best_weights:
        model.load_state_dict(early_stopping.best_weights)

    final_f1, final_acc, _ = evaluate_model(model, val_loader, criterion, device)
    return final_f1, final_acc

total_start = time.time()
study = optuna.create_study(directions=["maximize", "maximize"])
study.optimize(objective, n_trials=10)

# 최적 하이퍼파라미터 저장
best_trial = study.best_trials[0]
with open("best_hyperparams.json", "w") as f:
    json.dump(best_trial.params, f)

with open("best_hyperparams.json") as f:
    best_params = json.load(f)

model = ResNet_vanilla().to(device)
model.apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
batch_size = best_params["batch_size"]

# === 최종 모델 학습 ===
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

early_stopping = EarlyStopping(patience=5, min_delta=0.005, restore_best_weights=True, verbose=True)

for epoch in range(20):  # 최대 20 에포크
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for imgs, labels, _ in tqdm(train_loader, desc=f"[Final Train] Epoch {epoch+1}"):
        imgs, labels = imgs.to(device), labels.float().to(device)
        outputs = model(imgs).squeeze()
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    train_f1 = f1_score(all_labels, all_preds)
    val_f1, val_acc, val_loss = evaluate_model(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1} - Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.2f}")

    early_stopping(-val_loss, model)
    if early_stopping.early_stop:
        print("🛑 Early stopping triggered")
        break

# 최고 성능 모델 복원
if early_stopping.best_weights:
    model.load_state_dict(early_stopping.best_weights)
    print("✅ Best model weights restored")

# Epoch loop (same as above)
test_cover = PTStegoDataset("C:/Users/Admin/Desktop/Python/Attention_Disc/adaptive_test_cover.pt")
test_stego = PTStegoDataset("C:/Users/Admin/Desktop/Python/Attention_Disc/adaptive_test_stego.pt")
test_dataset = ConcatDataset([test_cover, test_stego])
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 평가 함수
avg_loss, acc, prec, recall, f1, misclassified = evaluate_model_for_real(model, test_loader, criterion, device)
print(f"Test F1: {f1:.4f}, Accuracy: {acc:.2f}%, Precision: {prec:.2f}, Recall: {recall:.2f}")
