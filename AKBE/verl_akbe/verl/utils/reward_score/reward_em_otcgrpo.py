import re
import sys
import string
import math
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
from collections import Counter
from transformers import AutoTokenizer
import json

# --- OTC-PO 全局状态追踪 ---
# 键为 Prompt 的文本（或哈希），值为该 Prompt 目前见过的最少工具调用次数 (n)
# 注意：在多进程/分布式训练中，这个字典在每个 Worker 内部维护一份局部最优，
# 只有在同步时才能实现真正的全局最优，但对于 GRPO 这种 Sample-based 方法，局部维护通常足够。
GLOBAL_MIN_TOOL_COUNTS = {}

def count_tool_calls(text: str) -> int:
    """
    统计工具调用次数 (m)
    根据你的格式，工具标签为 <search> 和 <python>
    """
    search_count = text.count('<search>')
    python_count = text.count('<python>')
    return search_count + python_count

def calculate_otc_multiplier(m: int, n: int, c: int = 5) -> float:
    """
    OTC-PO 核心公式：计算工具效率系数
    Args:
        m: 当前轨迹的工具调用次数
        n: 历史最优工具调用次数 (Optimal / Minimal)
        c: 衰减常数，论文建议设为允许的最大交互轮数 (如 5)
    Returns:
        float: 0.0 到 1.0 之间的系数
    """
    # 如果当前调用次数小于等于历史最优，直接给满分系数
    if m <= n:
        return 1.0
    
    # 计算差异
    diff = m - n
    
    # 使用 Cosine 衰减 (论文公式 5/6 的变体)
    # 当 m 远大于 n 时，奖励趋近于 0
    # 避免分母为0
    denominator = 2 * (diff + c)
    if denominator == 0:
        return 0.0
        
    angle = (diff * math.pi) / denominator
    reward_scale = math.cos(angle)
    
    # 截断到 [0, 1] 之间
    return max(0.0, min(1.0, reward_scale))

# ... [保留原有的 validate_format, validate_format_python, extract_answer, remove_boxed, last_boxed_only_string, normalize_answer, get_f1_score, get_em_score 函数不变] ...
# 为了节省篇幅，这里省略这部分未修改的代码，请保留你原文件中的这些辅助函数
# ... [Start of skipped functions] ...
def validate_format(text: str) -> Tuple[bool, str]:
    # ... (Keep existing code)
    if text.count('<think>') != text.count('</think>'): return False, "<think> </think> not paired"
    if text.count('<think>') == 0 or text.count('</think>') == 0: return False, "<think> or </think> not found"
    if text.count('<answer>') != 1 or text.count('</answer>') != 1: return False, "<answer> or </answer> not found"
    # ... (simplified for brevity, assume original logic is here)
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end: return False, "<answer> must be before </answer>"
    answer_content = text[answer_start:answer_end]
    if '\\boxed{' not in answer_content or '}' not in answer_content: return False, "answer is missing \\boxed{} format"
    return True, "format is correct"

def extract_answer(text: str) -> Optional[str]:
    text = text.strip()
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if not match: return None
    return match.group(1)

def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        return s[len(left):]
    left = "\\boxed{"
    if s[:len(left)] == left and s[-1] == "}":
        return s[len(left):-1]
    return s

def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string: return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0: return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{": num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None: return None
    return string[idx:right_brace_idx + 1]

def normalize_answer(s: str) -> str:
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_f1_score(prediction: str, ground_truths: Union[str, List[str]]) -> float:
    if isinstance(ground_truths, str): ground_truths = [ground_truths]
    final_metric = {"f1": 0}
    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth: continue
        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth: continue
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0: continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        final_metric["f1"] = max(f1, final_metric["f1"])
    return final_metric['f1']

def get_em_score(prediction: str, ground_truths: Union[str, List[str]]) -> float:
    if isinstance(ground_truths, str): ground_truths = [ground_truths]
    normalized_prediction = normalize_answer(prediction)
    for ground_truth in ground_truths:
        if normalized_prediction == normalize_answer(ground_truth): return 1.0
    return 0.0
# ... [End of skipped functions] ...


def compute_score(
    data_source: str, 
    prompt_str: str,  # <--- 新增: 必须传入 Prompt 用于记录每个问题的最优状态
    solution_str: str, 
    ground_truth: Any, 
    extra_info: Optional[Dict[str, Any]] = None, 
    is_validate = False
) -> Dict[str, Any]:
    """
    计算 OTC-PO 奖励分数
    """
    
    # 初始化
    result = {
        "score": 0,
        "em_score": 0,
        "reason": "",
        "answer": "",
        "f1_score": 0,
        "tool_calls": 0, # 新增返回信息
        "otc_multiplier": 0 # 新增返回信息
    }
    result["ground_truth"] = json.dumps(ground_truth['target'])
    
    response = solution_str
    
    # 1. 计算当前回答的工具调用次数 m
    m_calls = count_tool_calls(response)
    result["tool_calls"] = m_calls

    valid_template, reason = validate_format(response)
    
    if not valid_template and not is_validate:
        result["score"] = -1
        result["em_score"] = 0
        result["reason"] = f"bad format: {reason}"
        return result
    
    # Remove EOS token
    if extra_info is not None and "tokenizer" in extra_info and extra_info["tokenizer"].eos_token and response.endswith(extra_info["tokenizer"].eos_token):
        response = response[:-len(extra_info["tokenizer"].eos_token)]
    
    answer_part = extract_answer(response)
    if answer_part is None:
        result["score"] = -1
        result["em_score"] = 0
        result["reason"] = "cannot extract answer"
        return result
    
    try:
        answer = remove_boxed(last_boxed_only_string(answer_part))
        result["answer"] = answer
    except Exception as e:
        result["score"] = -1
        result["em_score"] = 0
        result["reason"] = f"find box error: {e}"
        return result
    
    # 2. 计算正确性基础分数 (Base Reward)
    f1_score = get_f1_score(answer, ground_truth['target'])
    em_score = get_em_score(answer, ground_truth['target'])
    result["f1_score"] = f1_score
    result["em_score"] = em_score
    
    base_correctness = em_score

    # 3. OTC-PO 逻辑核心
    otc_multiplier = 0.0
    
    if base_correctness > 0: # 只有回答正确才进行工具奖励计算和状态更新
        # 获取该 Prompt 目前已知的最少工具调用次数 n
        # 如果 prompt 为空（未传入），则默认 n=0，退化为只要用了工具就扣分
        current_n = 0
        prompt_key = prompt_str if prompt_str else "default_unknown_prompt"
        
        if prompt_key in GLOBAL_MIN_TOOL_COUNTS:
            current_n = GLOBAL_MIN_TOOL_COUNTS[prompt_key]
        else:
            # 第一次遇到这个问题，初始化为无穷大或当前的 m
            current_n = float('inf')

        # 更新全局最优 n
        # 如果当前的 m 比历史最优还小，说明发现了更高效的路径，更新 n
        if m_calls < current_n:
            GLOBAL_MIN_TOOL_COUNTS[prompt_key] = m_calls
            current_n = m_calls # 使用更新后的 n 进行计算
        
        # 计算系数
        otc_multiplier = calculate_otc_multiplier(m=m_calls, n=current_n, c=6)
        
        # 4. 合成最终奖励 (Correctness * OTC_Multiplier)
        # alpha 系数通常设为 1.0 (论文中)，此处直接乘
        final_score = 0.6 * base_correctness * otc_multiplier
        
        result["score"] = final_score
        result["otc_multiplier"] = otc_multiplier
        result["reason"] = f"Correct. Base: {base_correctness}, Tools: {m_calls} (Best: {current_n}), OTC Factor: {otc_multiplier:.4f}"
        
    else:
        # 回答错误，奖励归零（防止 Reward Hacking）
        result["score"] = 0
        result["otc_multiplier"] = 0.0
        result["reason"] = f"Wrong answer. Answer: {answer}"
    
    return result

if __name__ == "__main__":
    # 测试代码
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct") # 替换为你的路径
    
    # 模拟数据
    prompt_text = "In each cell, a strip of length $100$ is worth a chip..."
    # 假设这是用户的问题部分
    full_prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
    
    response_high_tools = "<think>...</think><search>q1</search><result>..</result><think>..</think><python>code</python><result>..</result><think>..</think><answer> \\boxed{12} </answer>"
    response_low_tools = "<think>...</think><python>code_solve_all</python><result>..</result><think>..</think><answer> \\boxed{12} </answer>"
    
    answer = "12"
    ground_truth = {'target': answer}
    extra_info = {"tokenizer": tokenizer}

    print("--- 第一轮：高工具调用 (m=2), 历史最优未知 ---")
    res1 = compute_score("test", response_high_tools, ground_truth, prompt=prompt_text, extra_info=extra_info)
    print(f"Score: {res1['score']}, Reason: {res1['reason']}")
    # 此时 Global Min 应该更新为 2

    print("\n--- 第二轮：低工具调用 (m=1), 历史最优为 2 ---")
    res2 = compute_score("test", response_low_tools, ground_truth, prompt=prompt_text, extra_info=extra_info)
    print(f"Score: {res2['score']}, Reason: {res2['reason']}")
    # 此时 Global Min 更新为 1，且因为 m(1) < old_n(2)，系数为 1.0，分数最高
    
    print("\n--- 第三轮：高工具调用 (m=2), 历史最优已变为 1 ---")
    res3 = compute_score("test", response_high_tools, ground_truth, prompt=prompt_text, extra_info=extra_info)
    print(f"Score: {res3['score']}, Reason: {res3['reason']}")
    # 此时 m(2) > n(1)，会有惩罚，分数应低于 1.0
