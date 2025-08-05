import random
from MyHTMLParse import MyHTMLParser

from Encode import encode_state
from ActionTable import ACTIONS


# Levenshtein距離計算函式，用於計算輔助獎勵
def levenshtein_dist(s1, s2):
    """計算字串 s1 和 s2 的Levenshtein編輯距離"""
    len1, len2 = len(s1), len(s2)
    # 動態規劃DP矩陣
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    # 填表
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,      # 刪除
                           dp[i][j-1] + 1,      # 插入
                           dp[i-1][j-1] + cost) # 替換
    return dp[len1][len2]

class XSSEnv:
    """XSS 攻擊向量生成環境，負責將智能體動作作用於攻擊字串，並計算回報。"""
    def __init__(self, initial_samples, max_steps=5):
        """
        :param initial_samples: 初始攻擊向量樣本列表（每回合從中選一作起點）
        :param max_steps: 每回合允許的最大變異步數
        """
        self.initial_samples = initial_samples
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        """重置環境至新回合初始狀態，返回初始 state 向量。"""
        # 隨機選一個初始攻擊字串（或循環選取）
        self.current_payload = random.choice(self.initial_samples)
        self.steps_taken = 0
        # 狀態表示（Byte Histogram 特徵向量）
        return encode_state(self.current_payload)

    def simulate_filter(self, payload):
        """
        模擬目標系統對輸入payload的過濾處理，返回過濾後實際出現在響應中的字串。
        （這裡簡化使用一些替換規則，模擬WAF）
        """
        filtered = payload
        # 簡單模擬若干過濾規則：
        # 1. 移除<script>及其內容
        import re
        filtered = re.sub(r"<script.*?>.*?</script>", "", filtered, flags=re.IGNORECASE)
        # 2. 去掉 javascript: 協議前綴
        filtered = filtered.replace("javascript:", "")
        # 3. 過濾事件屬性 onerror, onload 等，替換為空
        filtered = re.sub(r"onerror\s*=", "", filtered, flags=re.IGNORECASE)
        filtered = re.sub(r"onload\s*=", "", filtered, flags=re.IGNORECASE)
        # （更多過濾規則可按需添加）
        return filtered

    def step(self, action_idx):
        """
        根據當前payload和智能體選擇的動作，產生下一狀態、獎勵及是否結束。
        """
        action_fn = ACTIONS[action_idx]
        # 施加變異操作
        new_payload = action_fn(self.current_payload)
        # 模擬目標網站對payload的響應
        response_payload = self.simulate_filter(new_payload)
        # 計算獎勵
        reward = 0.0
        done = False
        info = {}  # 可用於存放額外資訊
        # 使用HTML解析和檢查彈窗（靜態分析）
        if self._static_check_trigger(response_payload):
            # 靜態分析發現有 alert/prompt/confirm 存在於響應DOM中，視為可能成功
            # 為了慎重，我們進一步使用動態WebDriver驗證
            if self._dynamic_verify(new_payload):
                # 觸發彈窗確認成功
                reward = 10.0  # 主要獎勵
                done = True
                info['triggered'] = True
            else:
                # WebDriver未監測到彈窗，視為假陽性，當作未成功觸發繼續
                info['triggered'] = False
        # 若尚未成功觸發彈窗，給予輔助獎勵
        if not done:
            # 計算編輯距離並轉化為相似度獎勵
            L = len(new_payload) if len(new_payload) > 0 else 1
            dist = levenshtein_dist(new_payload, response_payload)
            similarity_reward = 2 * (1 - dist / L)
            reward = similarity_reward
        # 更新環境狀態
        self.current_payload = new_payload
        self.steps_taken += 1
        # 檢查是否達到最大步數，強制結束
        if self.steps_taken >= self.max_steps:
            done = True
        # 準備下一狀態（如果未結束）
        next_state = encode_state(self.current_payload) if not done else None
        return next_state, reward, done, info

    def _static_check_trigger(self, html_content):
        """
        靜態分析HTML內容，構建DOM樹並檢查是否存在可觸發彈窗的關鍵節點。
        若發現alert/prompt/confirm 在有效的位置，返回 True。
        """
        parser = MyHTMLParser()
        parser.feed(html_content)
        root = parser.root
        # 深度優先遍歷DOM樹，查找含 alert/prompt/confirm 的節點
        stack = [root]
        found = False
        while stack:
            node = stack.pop()
            if node.node_type == 'element':
                # 檢查 element 節點的屬性
                for attr, val in node.attrs.items():
                    if attr.lower().startswith("on"):  # 事件屬性
                        if any(fn in val for fn in ["alert", "prompt", "confirm"]):
                            # 若事件屬性中包含彈窗函數名，認為可能觸發
                            found = True
                            break
                # 檢查 <script> 節點內容
                if node.tag.lower() == "script":
                    if any(fn + "(" in node.text_content for fn in ["alert", "prompt", "confirm"]):
                        found = True
            elif node.node_type == 'text':
                # 對於 <a href="javascript:..."> 這類協議，也算一種可能
                text = node.text_content
                if any(fn + "(" in text for fn in ["alert", "prompt", "confirm"]):
                    found = True
            if found:
                break
            # 將子節點入堆疊
            if node.node_type in ['element', 'document']:
                for child in node.children:
                    stack.append(child)
        return found

    def _dynamic_verify(self, payload):
        """
        動態使用 Selenium WebDriver 驗證payload是否真正觸發瀏覽器彈窗。
        （需要Selenium和瀏覽器環境支持）
        """
        try:
            from selenium import webdriver
            from selenium.common.exceptions import NoAlertPresentException
        except ImportError:
            # Selenium未安裝，無法動態驗證，這裡返回False或假定
            return False
        # 啟動瀏覽器（使用無頭模式避免干擾）
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        driver = webdriver.Chrome(options=options)
        # 將payload寫入簡單的HTML模板，加載頁面
        html_data_url = "data:text/html,<html><body>" + payload + "</body></html>"
        driver.get(html_data_url)
        triggered = False
        try:
            # 等待一下，看看是否有alert跳出
            # Selenium在有alert時 driver.switch_to.alert 可獲取
            alert = driver.switch_to.alert
            alert.accept()  # 關閉alert
            triggered = True
        except Exception as e:
            triggered = False
        driver.quit()
        return triggered
