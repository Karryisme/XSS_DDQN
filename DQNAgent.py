import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from DuelingDQN import DuelingDQN
from Sumtree import PrioritizedReplayMemory

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, 
                 target_update_freq=1000, memory_capacity=10000):
        """
        :param state_dim: 狀態向量維度
        :param action_dim: 動作數量
        :param lr: 學習率
        :param gamma: 折扣因子
        :param epsilon_start: 初始 epsilon（ε-貪婪中的隨機探索率）
        :param epsilon_end: 最小 epsilon
        :param epsilon_decay: 每步衰減率
        :param target_update_freq: 目標網路更新頻率（步）
        :param memory_capacity: 經驗回放記憶體容量
        """
        self.action_dim = action_dim
        self.gamma = gamma
        # 建立主網路和目標網路
        self.policy_net = DuelingDQN(state_dim, action_dim)
        self.target_net = DuelingDQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # 初始時目標網路權重與主網路相同
        self.target_net.eval()  # 目標網路只用於推斷，不訓練
        # 優化器
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        # 經驗回放記憶體（使用優先經驗回放）
        self.memory = PrioritizedReplayMemory(memory_capacity)
        # ε-貪婪參數
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step_count = 0
        self.target_update_freq = target_update_freq

    def select_action(self, state, eval_mode=False):
        """
        根據當前策略選擇動作。使用 ε-貪婪：以 epsilon 機率隨機選動作，以 (1-epsilon) 選擇 Q 值最高的動作:contentReference[oaicite:29]{index=29}。
        若 eval_mode=True，則總是選擇最優動作（測試/驗證時使用）。
        """
        if not eval_mode and np.random.rand() < self.epsilon:
            # 探索：隨機選擇一個動作
            return np.random.randint(0, self.action_dim)
        else:
            # 利用：選擇 Q 值最大的動作
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # 加批次維度
            q_values = self.policy_net(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
            return action

    def store_transition(self, state, action, reward, next_state, done):
        """將交互過程中的 (s, a, r, s', done) 經驗存入回放記憶體中。"""
        self.memory.add(state, action, reward, next_state, done)

    def train_step(self, batch_size):
        """從回放記憶體取樣一批經驗，更新網路參數。"""
        # 若經驗不滿一個批次，則不訓練
        if self.memory.tree.size < batch_size:
            return 0.0
        # 從優先回放記憶體取樣一批
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(batch_size)
        # 將 numpy 轉為 tensor
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)
        # 計算當前狀態的 Q(s,a)
        all_Q = self.policy_net(states)             # (batch, action_dim)
        # 挑出每個樣本對應執行動作的 Q(s, a)值
        Qsa = all_Q.gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch,)
        # 計算目標 Q 值 (Double DQN)：利用主網路決定下一步最佳動作，再由目標網路估算其Q值:contentReference[oaicite:30]{index=30}
        with torch.no_grad():
            # 主網路選擇下一狀態最大 Q 的動作
            next_Q = self.policy_net(next_states)               # 主網路對下一狀態的 Q(s', a')
            next_actions = torch.argmax(next_Q, dim=1)          # 每個樣本最佳動作索引
            # 目標網路估計相應的 Q 值
            target_Q = self.target_net(next_states)             # 目標網路對下一狀態的 Q*(s', a')
            target_Qsa = target_Q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # 計算TD目標：若done，則目標為當前獎勵；否則 = r + γ * target_Q
            Y = rewards + (1 - dones) * self.gamma * target_Qsa
        # 計算 TD 誤差
        td_errors = Y - Qsa
        # 計算加權損失：使用重要性採樣權重加權的 MSE:contentReference[oaicite:31]{index=31}
        loss = torch.mean(weights * (td_errors ** 2))
        # 反向傳播和更新參數
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # 更新優先級 (Prioritized Replay Buffer) 中取樣經驗的優先值
        errors = td_errors.detach().numpy()
        self.memory.update_priorities(indices, errors)
        # 每隔一定步數更新目標網路參數
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        # 逐步衰減探索率 epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min
        return float(loss.item())
