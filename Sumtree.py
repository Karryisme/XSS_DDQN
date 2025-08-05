import numpy as np

class SumTree:
    """SumTree 資料結構：以樹陣列存儲優先級和，用於高效隨機抽樣。"""
    def __init__(self, capacity):
        self.capacity = capacity               # 最大容量（葉節點數量）
        self.tree = np.zeros(2 * capacity - 1) # 樹陣列（葉子+內部節點）
        self.data = np.zeros(capacity, dtype=object)  # 儲存實際經驗的陣列
        self.size = 0       # 當前存儲的經驗數量
        self.next_index = 0 # 下一個插入位置索引（循環覆蓋舊經驗）

    def add(self, priority, data):
        """添加一筆新經驗到樹中，同時更新對應優先級。"""
        leaf_index = self.next_index + self.capacity - 1  # 找到對應葉節點在 tree 陣列中的索引
        self.data[self.next_index] = data  # 保存經驗數據
        # 更新該葉節點的優先級值差異，並向上更新樹
        self.update(leaf_index, priority)
        # 移動索引，如超過容量則回到頭部（覆蓋最舊經驗）
        self.next_index = (self.next_index + 1) % self.capacity
        # 更新當前數量（不超過容量上限）
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_index, new_priority):
        """更新樹中某一節點（葉子）的優先級，並調整父節點權重。"""
        # 計算優先級改變量 Δ
        delta = new_priority - self.tree[tree_index]
        self.tree[tree_index] = new_priority
        # 往上更新到根節點
        parent = (tree_index - 1) // 2
        while parent >= 0:
            self.tree[parent] += delta
            if parent == 0:  # 根節點更新完畢
                break
            parent = (parent - 1) // 2

    def total_priority(self):
        """返回所有葉節點優先級總和（根節點值）。"""
        return self.tree[0]

    def get(self, value):
        """
        根據給定的優先級累積值，從樹中查找對應的葉節點。
        `value` 為一隨機數（0 ~ 總優先級），返回 (葉子索引, 優先級, 經驗)。
        """
        parent_index = 0
        # 迭代往下搜尋直到到達葉節點
        while True:
            left_child = 2 * parent_index + 1
            right_child = left_child + 1
            # 如果越過葉節點範圍，跳出
            if left_child >= len(self.tree):
                break
            # 比較 value 與左子節點權重，決定往左或往右
            if value <= self.tree[left_child]:
                parent_index = left_child
            else:
                value -= self.tree[left_child]
                parent_index = right_child
        # 找到葉子索引
        leaf_index = parent_index
        data_index = leaf_index - (self.capacity - 1)
        # 返回葉節點對應的優先級和值
        return leaf_index, self.tree[leaf_index], self.data[data_index]


class PrioritizedReplayMemory:
    """優先體驗回放記憶體，內含 SumTree 用於根據 TD 誤差取樣經驗。"""
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        :param capacity: 最大經驗儲存容量
        :param alpha: 決定取樣分佈與優先級關係的參數（0表示均勻取樣，1表示按優先級嚴格取樣）
        :param beta_start: 初始的 beta 值（修正取樣偏差用）
        :param beta_frames: 完全補償偏差所需的時間步數（逐漸增大 beta 用）
        """
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # 用於追蹤已過去的訓練時間步，以增大 beta
        self.max_priority = 1.0  # 初始化最大優先級（新樣本預設優先級）

    def add(self, state, action, reward, next_state, done):
        """添加一筆新的回放經驗，同時以當前最大優先級插入。"""
        # 將狀態等數據打包成元組
        experience = (state, action, reward, next_state, done)
        # 使用目前最大優先級，確保新樣本在至少一次取樣中被選到
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)

    def sample(self, batch_size):
        """按照優先級隨機取樣一批經驗，返回樣本及其權重和索引。"""
        batch = []
        indices = []
        priorities = []
        # 將總優先級值劃分為 batch_size 個區間，每個區間內均勻取一個隨機值
        total_priority = self.tree.total_priority()
        segment = total_priority / batch_size
        for i in range(batch_size):
            # 在該區間內選一隨機值
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)
            # 透過 SumTree 尋找對應的葉子節點
            leaf_index, priority, data = self.tree.get(value)
            batch.append(data)
            indices.append(leaf_index)
            priorities.append(priority)
        # 計算重要性採樣權重
        self.frame += 1  # 增加時間步計數
        beta = self._get_beta()
        # 將取樣優先級轉為概率分佈 P(i) = p_i / sum(p)
        probabilities = np.array(priorities) / self.tree.total_priority()
        # w_i = (N * P(i))^(-beta)，N使用緩衝區大小來計算偏差校正:contentReference[oaicite:13]{index=13}
        N = self.tree.size
        weights = (N * probabilities) ** (-beta)
        # 規範化 weights，使最大值=1，穩定訓練:contentReference[oaicite:14]{index=14}
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        # 將 batch 拆解為各項 list
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(next_states), np.array(dones, dtype=np.float32), indices, weights

    def update_priorities(self, indices, errors):
        """根據給定樣本的TD誤差更新其優先級（影響後續抽樣概率）。"""
        # 新的優先級 p = (|δ| + ε)^α :contentReference[oaicite:15]{index=15}
        # ε 取一個很小的值，避免優先級為0（保證每個樣本都有非零機率被抽到）
        eps = 1e-6
        for idx, err in zip(indices, errors):
            p = (abs(err) + eps) ** self.alpha
            # 更新 SumTree 中對應葉子節點優先級
            self.tree.update(idx, p)
            # 更新當前最大優先級值（供新插入樣本用）
            self.max_priority = max(self.max_priority, p)

    def _get_beta(self):
        """隨時間步數增長逐漸增加 beta，最終趨近 1（完全補償取樣偏差）。"""
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
