import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CNN_Disc import ResNet, Dir4LaplacianBlur, Dir8LaplacianBlur
import cv2
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any
import random
from torchvision import transforms
import os
from PIL import Image, ImageChops
from PPOTrainer import PPOTrainer

# -------------------------------
# 1. Hard Attention Mask Generator
# -------------------------------
def generate_attention_mask(height, width, center, radius=100, device='cuda'):
    """
    하드 원형 어텐션 마스크 (값은 1 또는 0)
    """
    if center.ndim == 2:
        center = center[0]
    cy, cx = center[0].item(), center[1].item()

    # 좌표 그리드 생성
    y = torch.arange(0, height, device=device).view(-1, 1).float()
    x = torch.arange(0, width, device=device).view(1, -1).float()
    dist = torch.sqrt((x - cx)**2 + (y - cy)**2)

    # 하드 마스크: 원 내부는 1, 외부는 0
    mask = (dist <= radius).float()

    return mask.unsqueeze(0)  # (1, H, W)


class SteganalysisEnv:
    """
    스테가분석을 위한 강화학습 환경
    논문의 "Self-Seeking Steganalysis" 환경 구현
    """
    
    def __init__(self, 
                 image_dataset,
                 labels_dataset, 
                 discriminant_model,
                 max_steps=5,           # 논문에서 T=5
                 image_size=(300, 300),
                 initial_center_range=0.3):
        """
        Args:
            image_dataset: 이미지 데이터셋 (torch.Tensor or list)
            labels_dataset: 라벨 데이터셋 (0: cover, 1: stego)
            discriminant_model: 미리 학습된 스테가분석 모델
            max_steps: 최대 스텝 수
            image_size: 이미지 크기
            initial_center_range: 초기 중심점 범위 (0.3 = 중앙 30% 영역)
        """
        self.image_dataset = image_dataset
        self.labels_dataset = labels_dataset
        self.discriminant_model = discriminant_model
        self.max_steps = max_steps
        self.image_size = image_size
        self.initial_center_range = initial_center_range
        
        # 환경 상태
        self.current_image = None
        self.current_label = None
        self.current_center = None
        self.step_count = 0
        self.episode_reward = 0
        
        # 성능 추적
        self.attention_history = []
        self.reward_history = []
        
        # 디바이스 설정
        self.device = next(discriminant_model.parameters()).device
        
        # 필터 설정
        self.filter = Dir8LaplacianBlur().to(self.device)  # 환경 초기화 시 한 번만 생성
        self.blur = Dir4LaplacianBlur().to(self.device)
        
    def reset(self, image_idx=None):
        """
        환경 초기화
        Returns:
            state: 초기 상태 (image, center)
        """
        # 랜덤하게 이미지 선택 (또는 지정된 인덱스)

        if image_idx is None:
            idx = random.randint(0, len(self.image_dataset) - 1)
        else:
            idx = image_idx
            
        self.current_image = self.image_dataset[idx]
        self.current_label = self.labels_dataset[idx]
        
        # 텐서 변환
        if isinstance(self.current_image, np.ndarray):
            self.current_image = torch.from_numpy(self.current_image).float()

        # (H, W) ➜ (1, 1, H, W)
        if self.current_image.ndim == 2:
            self.current_image = self.current_image.unsqueeze(0).unsqueeze(0)

        # (H, W, 3) ➜ (1, 3, H, W)
        if self.current_image.ndim == 3 and self.current_image.shape[-1] == 3:
            self.current_image = self.current_image.permute(2, 0, 1).unsqueeze(0)

        # (3, H, W) or (1, H, W) ➜ (1, C, H, W)
        if self.current_image.ndim == 3 and self.current_image.shape[0] in [1, 3]:
            self.current_image = self.current_image.unsqueeze(0)

        # (1, 1, H, W) ➜ (1, 3, H, W)
        if self.current_image.ndim == 4 and self.current_image.shape[1] == 1:
            self.current_image = self.current_image.repeat(1, 3, 1, 1)

        # 최종 확인
        assert self.current_image.shape[1] == 3, f"이미지는 RGB 3채널이어야 합니다. 현재 shape: {self.current_image.shape}"
        
        # 중심점 먼저 설정
        H, W = self.image_size
        center_range_h = int(H * self.initial_center_range)
        center_range_w = int(W * self.initial_center_range)

        center_y = random.randint(H//2 - center_range_h//2, H//2 + center_range_h//2)
        center_x = random.randint(W//2 - center_range_w//2, W//2 + center_range_w//2)

        self.current_center = torch.tensor([[center_y, center_x]], dtype=torch.long)

        # 상태 초기화
        self.step_count = 0
        self.episode_reward = 0
        self.attention_history = [self.current_center.clone()]
        self.reward_history = []
        
        return self._get_state()
    
    def step(self, action):
        """
        환경에서 한 스텝 진행
        Args:
            action: 행동 (0~15, 16방향 움직임)
        Returns:
            next_state: 다음 상태
            reward: 보상
            done: 에피소드 종료 여부
            info: 추가 정보
        """
        self.step_count += 1
        
        # 액션에 따라 중심점 이동
        self.current_center = self._move_center(self.current_center, action)
        
        # 보상 계산
        reward = self._compute_reward()
        self.episode_reward += reward
        self.reward_history.append(reward)
        
        # 주의 위치 기록
        self.attention_history.append(self.current_center.clone())
        
        # 종료 조건 확인
        done = self.step_count >= self.max_steps
        
        # 추가 정보
        info = {
            'step': self.step_count,
            'episode_reward': self.episode_reward,
            'center': self.current_center.cpu().numpy(),
            'classification_accuracy': self._get_classification_accuracy()
        }
        
        return self._get_state(), reward, done, info
    
    def _get_state(self):
        """
        현재 상태 반환
        Returns:
            dict: {'image': tensor, 'center': tensor}
        """
        return {
            'image': self.current_image,
            'center': self.current_center,
            'step': self.step_count
        }
    
    def _move_center(self, current_center, action):
        """
        액션에 따라 중심점 이동
        """
        # 16방향 움직임 정의 (8방향 × 2거리)
        action_space_size = 32  # 픽셀 단위 이동 거리
        angles = np.linspace(0, 2*np.pi, 9)[:-1]  # 8 directions
        
        action_vectors = []
        for angle in angles:
            # Short distance
            dx_short = int(action_space_size * 0.5 * np.cos(angle))
            dy_short = int(action_space_size * 0.5 * np.sin(angle))
            action_vectors.append((dy_short, dx_short))
            
            # Long distance  
            dx_long = int(action_space_size * np.cos(angle))
            dy_long = int(action_space_size * np.sin(angle))
            action_vectors.append((dy_long, dx_long))
        
        # 현재 중심점
        center_y, center_x = current_center[0, 0].item(), current_center[0, 1].item()
        
        # 액션 적용
        if action < len(action_vectors):
            dy, dx = action_vectors[action]
            new_y = max(0, min(self.image_size[0]-1, center_y + dy))
            new_x = max(0, min(self.image_size[1]-1, center_x + dx))
        else:
            # 잘못된 액션인 경우 현재 위치 유지
            new_y, new_x = center_y, center_x
        
        return torch.tensor([[new_y, new_x]], dtype=torch.long)
    
    def _compute_reward(self):
        """
        보상 계산 (BCE 기반 실수형 보상 + 엔트로피 + 경계 패널티)
        """
        with torch.no_grad():
            # 현재 주의 영역에 마스크 적용
            masked_image = self._apply_dual_filter_mask(self.current_image, self.current_center)
            masked_image = masked_image.to(self.device)

            # 스테가분석 예측
            logits = self.discriminant_model(masked_image)  # shape: (1,) or (B,)
            logits = logits.view(-1)  # ensure shape is (1,)

            label_tensor = torch.tensor([self.current_label], dtype=torch.float32, device=self.device)

            # BCEWithLogitsLoss로 실수형 loss 계산
            loss = F.binary_cross_entropy_with_logits(logits, label_tensor)
            accuracy_reward = 1.0 / (1.0 + loss.item())  # 낮은 loss → 높은 보상

            # 확률 계산 (sigmoid)
            probs = torch.sigmoid(logits)
            entropy = - (probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))
            entropy_reward = 1.0 / (1.0 + entropy.item())

            # 경계 페널티
            boundary_penalty = self._compute_boundary_penalty()

            # 총 보상 계산
            total_reward = accuracy_reward + 0.1 * entropy_reward - 0.05 * boundary_penalty

        return total_reward
    
    def _apply_dual_filter_mask(self, image, center, radius=100):
        """
        어텐션 영역은 Dir8Laplacian 필터,
        비어텐션 영역은 Dir4Laplacian 적용
        """
        B, C, H, W = image.shape
        output_images = []

        for b in range(B):
            # 하드 마스크 생성 (1 또는 0)
            mask = generate_attention_mask(H, W, center[b], radius, device=self.device)
            mask = mask.to(dtype=image.dtype, device=self.device)  # (1, H, W)
            mask = mask.expand(1, C, H, W)  # (1, C, H, W)
            inverted_mask = 1 - mask

            image_b = image[b:b+1].to(self.device)

            # 필터 적용
            focused = self.filter(image_b)  # Dir4Laplacian

            blurred = self.blur(image_b)
            # blurred_tensor = torch.from_numpy(blurred.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)

            # 필터 결합
            combined = mask * focused + inverted_mask * blurred
            output_images.append(combined)

        return torch.cat(output_images, dim=0)
    
    def _compute_boundary_penalty(self):
        """
        경계 페널티 계산 (이미지 가장자리로 갈수록 페널티)
        """
        center_y, center_x = self.current_center[0, 0].item(), self.current_center[0, 1].item()
        H, W = self.image_size
        
        # 경계로부터의 최소 거리
        min_dist_to_boundary = min(center_y, center_x, H - center_y, W - center_x)
        
        # 경계 근처 (50픽셀 이내)면 페널티
        boundary_threshold = 50
        if min_dist_to_boundary < boundary_threshold:
            penalty = (boundary_threshold - min_dist_to_boundary) / boundary_threshold
        else:
            penalty = 0
            
        return penalty
    
    def _get_classification_accuracy(self):
        """
        현재 위치에서의 분류 정확도
        """
        with torch.no_grad():          
            masked_image = self._apply_dual_filter_mask(self.current_image, self.current_center).to(self.device)
            # 이미지 저장
            img_np = masked_image[0].permute(1, 2, 0).cpu().numpy()
            cv2.imwrite("masked_output.png", (img_np * 255).astype(np.uint8))
            logits = self.discriminant_model(masked_image)
            pred_label = torch.argmax(logits, dim=1)
            accuracy = (pred_label == self.current_label).float().item()
        return accuracy
    
    def render(self, save_path=None):
        """
        현재 상태 시각화
        """
        # 이미지를 numpy로 변환
        img = self.current_image[0].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img_color = img.copy()
        
        # 0-255 범위로 정규화
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        # 컬러로 변환
        if img.ndim == 2 or img.shape[2] == 1:
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_color = img.copy()  # 이미 RGB면 그대로 사용
        
        # 주의 중심점 표시
        center_y, center_x = self.current_center[0, 0].item(), self.current_center[0, 1].item()
        
        # 주의 영역 원 그리기
        cv2.circle(img_color, (center_x, center_y), 10, (0, 255, 0), 2)  # 초록색 원
        cv2.circle(img_color, (center_x, center_y), 5, (0, 0, 255), -1)   # 빨간색 중심점
        
        # 주의 이동 경로 표시
        if len(self.attention_history) > 1:
            for i in range(1, len(self.attention_history)):
                prev_center = self.attention_history[i-1][0]
                curr_center = self.attention_history[i][0]
                
                pt1 = (prev_center[1].item(), prev_center[0].item())
                pt2 = (curr_center[1].item(), curr_center[0].item())
                
                cv2.arrowedLine(img_color, pt1, pt2, (255, 0, 0), 2)  # 파란색 화살표
        
        # 텍스트 정보 추가
        info_text = f"Step: {self.step_count}, Reward: {self.episode_reward:.3f}"
        cv2.putText(img_color, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, img_color)
        
        return img_color
    
    def get_attention_summary(self):
        """
        주의 영역 요약 정보 (SoAFR - Summary of Attention-Focused Regions)
        """
        return {
            'attention_centers': [center.cpu().numpy() for center in self.attention_history],
            'rewards': self.reward_history,
            'total_reward': self.episode_reward,
            'final_accuracy': self._get_classification_accuracy()
        }


# -------------------------------
# 4. Complete ActorCritic Network (마음 역할)
# -------------------------------
class ActorCritic(nn.Module):
    def __init__(self, input_shape=(3, 300, 300), num_actions=16, feature_dim=300, 
                 action_space_size=32):
        """
        Complete Actor-Critic for steganalysis with visual attention
        
        Args:
            input_shape: (C, H, W) input image shape
            num_actions: number of movement actions (논문에서 16개 방향)
            feature_dim: feature dimension
            action_space_size: movement step size in pixels
        """
        super(ActorCritic, self).__init__()
        
        
        # Visual attention CNN (눈)
        self.attention_cnn = ResNet(in_channels=input_shape[0], 
                                        feature_dim=feature_dim,
                                        mode = 'feature')
        
        # Discriminant model (뇌)
        self.discriminant = ResNet(in_channels=input_shape[0])
        
        # Actor network (정책 네트워크)
        self.actor = nn.Sequential(
            nn.Linear(feature_dim + 2, 64),  # feature_dim=300 + 2 for center
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        
        # Critic network (가치 네트워크)
        self.critic = nn.Sequential(
            nn.Linear(feature_dim + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Action space configuration
        self.num_actions = num_actions
        self.action_space_size = action_space_size
        self.input_shape = input_shape
        
        # Define 16 movement directions (8 directions * 2 distances)
        angles = np.linspace(0, 2*np.pi, 9)[:-1]  # 8 directions
        self.action_vectors = []
        for angle in angles:
            # Short distance
            dx_short = int(action_space_size * 0.5 * np.cos(angle))
            dy_short = int(action_space_size * 0.5 * np.sin(angle))
            self.action_vectors.append((dy_short, dx_short))
            
            # Long distance  
            dx_long = int(action_space_size * np.cos(angle))
            dy_long = int(action_space_size * np.sin(angle))
            self.action_vectors.append((dy_long, dx_long))
    
    def forward(self, x, center):
        """
        Forward pass
        Args:
            x: input image (B, C, H, W)
            center: current attention center (B, 2) - (y, x) coordinates
        """
        # Extract features using attention CNN
        attention_features = self.attention_cnn(x, center)
        
        # Normalize center coordinates to [-1, 1]
        H, W = x.shape[2], x.shape[3]
        normalized_center = center.float()
        normalized_center[:, 0] = (normalized_center[:, 0] / H) * 2 - 1  # y
        normalized_center[:, 1] = (normalized_center[:, 1] / W) * 2 - 1  # x
        
        # Combine attention features with center information
        combined_features = torch.cat([attention_features, normalized_center], dim=1)
        
        # Actor output (action probabilities)
        action_logits = self.actor(combined_features)
        
        # Critic output (state value)
        state_value = self.critic(combined_features)
        
        return action_logits, state_value.squeeze(-1)
    
    def get_stego_prediction(self, x, center):
        """
        Get steganalysis prediction using discriminant model
        """
        with torch.no_grad():
            # Apply attention mask
            B, C, H, W = x.shape
            attention_masks = []
            
            for b in range(B):
                mask = generate_attention_mask(H, W, center[b])
                attention_masks.append(torch.from_numpy(mask))
            
            masks = torch.stack(attention_masks).unsqueeze(1).to(x.device)
            attended_x = x * masks
            
            # Get classification logits
            logits = self.discriminant(attended_x)
            probs = F.softmax(logits, dim=-1)
            
            return logits, probs
    
    def move_attention(self, current_center, action, image_shape):
        """
        Move attention center based on action
        Args:
            current_center: (B, 2) current center coordinates
            action: (B,) action indices
            image_shape: (H, W) image dimensions
        """
        H, W = image_shape
        new_center = current_center.clone()
        
        for b in range(len(action)):
            if action[b] < len(self.action_vectors):
                dy, dx = self.action_vectors[action[b]]
                new_y = max(0, min(H-1, current_center[b, 0] + dy))
                new_x = max(0, min(W-1, current_center[b, 1] + dx))
                new_center[b] = torch.tensor([new_y, new_x])
        
        return new_center
    
    def compute_reward(self, x, center, true_labels):
        """
        Compute reward based on steganalysis accuracy and information entropy
        논문의 보상 함수 구현
        """
        logits, probs = self.get_stego_prediction(x, center)
        
        # Classification accuracy reward
        pred_labels = torch.argmax(logits, dim=1)
        accuracy_reward = (pred_labels == true_labels).float()
        
        # Information entropy penalty (lower entropy = higher reward)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        entropy_reward = 1.0 / (1.0 + entropy)  # Inverse entropy
        
        # Combined reward
        total_reward = accuracy_reward + 0.1 * entropy_reward
        
        return total_reward


def evaluate_all_images_best_attention_with_render(env, full_data, max_steps=20, save_path="C:/Users/Admin/Desktop/Image_data/Adaptive"):
    """
    각 이미지에서 best attention 위치를 찾아 필터 이미지 저장 (파일명 유지)

    Args:
        env: SteganalysisEnv 객체
        full_data: [(path, label)] 리스트
        max_steps: attention 이동 횟수
        save_path: 이미지 저장 폴더
    """
    os.makedirs(save_path, exist_ok=True)
    results = []

    for i in range(len(env.image_dataset)):
        state = env.reset(image_idx=i)
        best_reward = -float('inf')
        best_center = None
        best_accuracy = 0.0

        for step in range(max_steps):
            action = random.randint(0, 15)
            next_state, reward, done, info = env.step(action)

            if reward > best_reward:
                best_reward = reward
                best_center = env.current_center.clone()
                best_accuracy = info['classification_accuracy']

            if done:
                break

        # 원본 파일명에서 확장자 제거
        original_path = full_data[i][0]
        filename = os.path.basename(original_path)
        filename_noext = os.path.splitext(filename)[0]

        # 필터 적용 이미지 저장 (그대로 곱하기 255 후 uint8 변환)
        filtered_tensor = env._apply_dual_filter_mask(env.current_image, best_center)
        img_np = filtered_tensor[0].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        filtered_path = os.path.join(save_path, f"Adaptive_{filename_noext}.png")
        cv2.imwrite(filtered_path, img_np)

        results.append({
            'image_idx': i,
            'filename': filename,
            'best_center': best_center.cpu().numpy().tolist(),
            'best_reward': best_reward,
            'best_accuracy': best_accuracy,
            'filtered_path': filtered_path
        })

    return results

# ========================================
# 4. 메인 실행 함수
# ========================================

def main():
    """메인 실행 함수"""
    print("스테가분석 환경 초기화 중...")
    
    # 1. ResNet 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}")
    
    # ResNet 모델 초기화 및 가중치 로드
    model = ResNet().to(device)
    
    # 가중치 파일이 존재하는지 확인
    weight_path = "C:/Users/Admin/Desktop/기계학습 프로젝트 (201813784 손형오)/resnet_trained.pth"
    if os.path.exists(weight_path):
        try:
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=False))
            print("사전 훈련된 가중치 로드 완료")
        except Exception as e:
            print(f"가중치 로드 실패: {e}")
            print("랜덤 가중치로 진행합니다.")
    else:
        print(f"가중치 파일을 찾을 수 없습니다: {weight_path}")
        print("랜덤 가중치로 진행합니다.")
    
    model.train()
    
    # 2. 샘플 데이터셋 생성
    print("샘플 데이터셋 생성 중...")

    # ========================================
    # 3. 샘플 데이터셋 생성 함수
    # ========================================

    # 데이터셋 생성
    def make_dataset(confusion_dir, split_ratio=0.8, seed=42):
        cover_paths = [os.path.join(confusion_dir, f) for f in os.listdir(confusion_dir) if f.endswith('.png')]
        stego_paths = [os.path.join(confusion_dir, f) for f in os.listdir(confusion_dir) if f.endswith('_encoded.png')]

        # 라벨링
        cover_labeled = [(path, 0) for path in cover_paths]
        stego_labeled = [(path, 1) for path in stego_paths]

        # Cover + Stego 합치고 셔플
        full_data = cover_labeled + stego_labeled
        random.seed(seed)
        random.shuffle(full_data)

        # Split
        split_idx = int(len(full_data) * split_ratio)
        train_data = full_data[:split_idx]
        val_data = full_data[split_idx:]

        return full_data

    # class StegoDataset(Dataset):
    #     def __init__(self, samples, transform=None):
    #         self.samples = samples
    #         self.transform = transform

    #     def __len__(self):
    #         return len(self.samples)

    #     def __getitem__(self, idx):
    #         path, label = self.samples[idx]
    #         img = Image.open(path).convert('RGB') # 이미지를 RGB로 가지고 옴
    #         if self.transform:
    #             img = self.transform(img)
    #         return img, label

    # 3. transform 설정
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    confusion_dir = "C:/Users/Admin/Desktop/Image_data/Confusion"
    full_data = make_dataset(confusion_dir)

    # 이미지와 라벨을 리스트로 저장
    images = [transform(Image.open(path).convert("RGB")) for path, _ in full_data]
    labels = [torch.tensor(label, dtype=torch.long) for _, label in full_data]
        
    print(f"이미지 개수: {len(images)}")
    print(f"라벨 개수: {len(labels)}")

    # 3. 환경 생성
    print("환경 생성 중...")

    actor_critic_model = ActorCritic(input_shape=(3, 300, 300), num_actions=16).to(device)

    env = SteganalysisEnv(
        image_dataset=images,
        labels_dataset=labels,
        discriminant_model=model,
        max_steps=5
    )

    trainer = PPOTrainer(model=actor_critic_model, env=env, device=device)

    # 4. 환경 테스트
    print("환경 테스트 시작...")
    state = env.reset()
    print(f"초기 중심점: {state['center'].cpu().numpy()}")
    
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)
    trainer.train(epochs=5, episodes_per_epoch=5)
    
    # # 5. 시각화
    # print("결과 시각화...")
    # env.render("attention_visualization.png")
    # print("시각화 결과가 'attention_visualization.png'에 저장되었습니다.")
    
    print("Best attention 위치 탐색 및 시각화 시작...")
    results = evaluate_all_images_best_attention_with_render(env, full_data, max_steps=5)

    # 결과 출력
    for res in results:
        print(f"[Image {res['image_idx']}] Center: {res['best_center']}, "
              f"Reward: {res['best_reward']:.3f}, Accuracy: {res['best_accuracy']:.2f}, "
              f"Saved at: {res['filtered_path']}")

    # 6. 요약 정보 출력
    # summary = env.get_attention_summary()
    # print(f"\n=== 에피소드 요약 ===")
    # print(f"총 보상: {summary['total_reward']:.3f}")
    # print(f"최종 정확도: {summary['final_accuracy']:.3f}")
    # print(f"주의 이동 경로 수: {len(summary['attention_centers'])}")

if __name__ == "__main__":
    main()