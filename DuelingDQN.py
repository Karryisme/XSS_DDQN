import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    """Dueling DQN 網路，輸入狀態特徵，輸出每個動作的 Q 值。"""
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        # 公共特徵提取層
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 128)
        # 狀態價值分支
        self.value_fc = nn.Linear(128, 128)
        self.value_out = nn.Linear(128, 1)
        # 優勢分支
        self.adv_fc = nn.Linear(128, 128)
        self.adv_out = nn.Linear(128, action_dim)
        # 初始化網路權重（可選，也可使用 PyTorch 預設初始化）
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)
        nn.init.kaiming_uniform_(self.value_fc.weight)
        nn.init.kaiming_uniform_(self.adv_fc.weight)
        # 偏置初始化為0
        for layer in [self.fc1, self.fc2, self.fc3, self.value_fc, self.value_out, self.adv_fc, self.adv_out]:
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # 前向傳播：計算各動作的Q值
        # 公共層
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # 價值與優勢分支
        value = F.relu(self.value_fc(x))
        value = self.value_out(value)            # 尺寸: (batch, 1)
        adv = F.relu(self.adv_fc(x))
        adv = self.adv_out(adv)                 # 尺寸: (batch, action_dim)
        # Dueling 組合： Q = V(s) + (A(s,a) - 平均A)
        adv_mean = torch.mean(adv, dim=1, keepdim=True)
        Q = value + (adv - adv_mean)
        return Q
