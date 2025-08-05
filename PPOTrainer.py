import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from tqdm import tqdm
import gc
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class PPOTrainer:
    def __init__(self, model, env, lr=1e-4, gamma=0.99, eps_clip=0.2, epochs=4, batch_size=8, device='cuda'):
        self.model = model
        self.env = env
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        image = state['image'].to(self.device)
        center = state['center'].to(self.device)
        logits, value = self.model(image, center)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def compute_returns(self, rewards, dones, values, next_value):
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    torch.autograd.set_detect_anomaly(False)
    def train(self, epochs=5, num_rollout_episodes=5):
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch + 1} ===")
            memory = {
                'states': [], 'centers': [], 'actions': [], 'log_probs': [],
                'rewards': [], 'dones': [], 'values': []
            }

            all_preds = []
            all_targets = []

            # 🔄 tqdm 에피소드 진행률 바 적용
            episode_bar = tqdm(range(num_rollout_episodes), desc=f"[Epoch {epoch+1}] Episodes")
            for ep in episode_bar:
                state = self.env.reset()
                done = False

                while not done:
                    action, log_prob, value = self.select_action(state)
                    next_state, reward, done, _ = self.env.step(action)

                    memory['states'].append(state['image'])
                    memory['centers'].append(state['center'])
                    memory['actions'].append(torch.tensor([action]))
                    memory['log_probs'].append(log_prob)
                    memory['rewards'].append(reward)
                    memory['dones'].append(done)
                    memory['values'].append(value)

                    # ✅ 예측값 계산
                    with torch.no_grad():
                        logits, _ = self.model.get_stego_prediction(
                            state['image'].to(self.device), state['center'].to(self.device)
                        )
                        prob = torch.sigmoid(logits).squeeze().item()  # (B=1,)일 때 가능
                        pred = int(prob > 0.5)
                        target = int(self.env.current_label.item())

                    all_preds.append(pred)
                    all_targets.append(target)

                    state = next_state

            # ✅ 에포크 끝나고 정확도/F1 score 평가
            acc = accuracy_score(all_targets, all_preds)
            f1 = f1_score(all_targets, all_preds, average='macro')
            print(f"[Epoch {epoch+1}] Accuracy: {acc:.4f} | F1: {f1:.4f}")

            # ✅ 다음 상태 값 계산
            with torch.no_grad():
                _, next_value = self.model(memory['states'][-1].to(self.device),
                                           memory['centers'][-1].to(self.device))

            # ✅ PPO 학습
            returns = self.compute_returns(memory['rewards'], memory['dones'], memory['values'], next_value).detach()
            values = torch.stack(memory['values']).squeeze().to(self.device).detach()
            log_probs_old = torch.stack(memory['log_probs']).detach().to(self.device)
            actions = torch.cat(memory['actions']).to(self.device)
            advantages = (returns - values).detach()

            for _ in range(self.epochs):
                for i in range(0, len(returns), self.batch_size):
                    batch_slice = slice(i, i + self.batch_size)
                    image_batch = torch.cat(memory['states'][batch_slice]).to(self.device)
                    center_batch = torch.cat(memory['centers'][batch_slice]).to(self.device)

                    logits, value = self.model(image_batch, center_batch)
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    log_probs = dist.log_prob(actions[batch_slice])
                    ratio = torch.exp(log_probs - log_probs_old[batch_slice])

                    adv = advantages[batch_slice]
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value = value.view_as(returns[batch_slice])
                    value_loss = F.mse_loss(value, returns[batch_slice])
                    loss = policy_loss + 0.5 * value_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    gc.collect()
                    torch.cuda.empty_cache()

            print(f"훈련 완료 - 에포크 {epoch+1}: 평균 리턴 {returns.mean().item():.4f}")

            # # ✅ 전체 평가 데이터셋으로 다시 평가
            # acc, f1 = evaluate_model_on_all_data(self.model, self.env.image_dataset, self.env.labels_dataset, self.device)
            # print(f"[Epoch {epoch+1}] 전체 평가 - Accuracy: {acc:.4f}, F1: {f1:.4f}")
