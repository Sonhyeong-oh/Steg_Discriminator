import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

# 📌 1. Dir4LaplacianBlur 정의
class Dir4LaplacianBlur(nn.Module):
    def __init__(self):
        super().__init__()
        lap_kernel = torch.tensor([[0, -1, 0],
                                   [-1, 4, -1],
                                   [0, -1, 0]], dtype=torch.float32)
        lap_kernel = lap_kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.register_buffer('lap_kernel', lap_kernel)

        gauss_kernel = self._create_gaussian_kernel(3, sigma=1.0)
        gauss_kernel = gauss_kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.register_buffer('gauss_kernel', gauss_kernel)

    def forward(self, x):
        lap = F.conv2d(x, self.lap_kernel, padding=1, groups=3)
        blur = F.conv2d(lap, self.gauss_kernel, padding=1, groups=3)
        hybrid = lap + blur
        return hybrid

    def _create_gaussian_kernel(self, kernel_size=3, sigma=1.0):
        ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
        kernel = kernel / torch.sum(kernel)
        return kernel

# 📌 2. 이미지 로드 함수
transform = transforms.Compose([transforms.ToTensor()])

def load_image(path):
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)  # (1, 3, H, W)

# 📌 3. 텐서를 이미지로 저장하고 용량 측정
def tensor_to_png_and_get_size(tensor, save_path):
    img = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(save_path, format='PNG')
    return os.path.getsize(save_path) / 1024  # KB

# 📌 4. 경로 설정
cover_dir = "C:/Users/Admin/Desktop/Image_data/Original/train"
stego_dir = "C:/Users/Admin/Desktop/Image_data/Stegno/stg_class"

# 📌 5. 필터 모델 준비
filter_model = Dir4LaplacianBlur()

# 📌 6. 통계 계산용 리스트
orig_cover_sizes, orig_stego_sizes = [], []
filt_cover_sizes, filt_stego_sizes = [], []
orig_diff_list, filt_diff_list = [], []

# 📌 7. 반복 처리
file_list = sorted(os.listdir(cover_dir))
for fname in file_list:
    cover_path = os.path.join(cover_dir, fname)
    stego_path = os.path.join(stego_dir, fname.replace(".png", "_encoded.png"))
    
    if not os.path.exists(stego_path):
        continue

    # 원본 이미지
    cover_tensor = load_image(cover_path)
    stego_tensor = load_image(stego_path)

    # 용량 측정 (필터 전)
    cover_orig_size = os.path.getsize(cover_path) / 1024
    stego_orig_size = os.path.getsize(stego_path) / 1024
    orig_cover_sizes.append(cover_orig_size)
    orig_stego_sizes.append(stego_orig_size)
    orig_diff_list.append(abs(cover_orig_size - stego_orig_size))

    # 필터 적용
    with torch.no_grad():
        filt_cover_tensor = filter_model(cover_tensor)
        filt_stego_tensor = filter_model(stego_tensor)

    # 필터 적용 후 저장 & 용량 측정 (임시 파일)
    tmp_cov = f"./tmp_cov.png"
    tmp_stg = f"./tmp_stg.png"
    cov_filt_size = tensor_to_png_and_get_size(filt_cover_tensor, tmp_cov)
    stg_filt_size = tensor_to_png_and_get_size(filt_stego_tensor, tmp_stg)
    filt_cover_sizes.append(cov_filt_size)
    filt_stego_sizes.append(stg_filt_size)
    filt_diff_list.append(abs(cov_filt_size - stg_filt_size))

# 📌 8. 결과 출력
def mean(lst): return np.mean(lst) if lst else 0

print("📊 필터 적용 전:")
print(f"  - 커버 평균 용량: {mean(orig_cover_sizes):.2f} KB")
print(f"  - 스테고 평균 용량: {mean(orig_stego_sizes):.2f} KB")
print(f"  - 커버-스테고 용량 차이 평균: {mean(orig_diff_list):.2f} KB")

print("\n📊 필터 적용 후:")
print(f"  - 커버 평균 용량: {mean(filt_cover_sizes):.2f} KB")
print(f"  - 스테고 평균 용량: {mean(filt_stego_sizes):.2f} KB")
print(f"  - 커버-스테고 용량 차이 평균: {mean(filt_diff_list):.2f} KB")

# 📌 9. 임시 파일 삭제
try:
    os.remove("./tmp_cov.png")
    os.remove("./tmp_stg.png")
except:
    pass