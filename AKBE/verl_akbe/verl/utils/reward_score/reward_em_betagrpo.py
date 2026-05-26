import re
import sys
import string
import math
from typing import Union, List, Dict, Any, Optional, Tuple
from collections import Counter
from transformers import AutoTokenizer
import json

# =============================================================================
# 新增/修改的辅助函数
# =============================================================================

def calculate_search_confidence(
    token_ids: List[int], 
    log_probs: List[float], 
    tokenizer: Any
) -> float:
    """
    计算搜索片段的置信度 (Search-R1 Beta-GRPO的核心逻辑)
    逻辑: 找到所有 <search>...</search> 片段，取这些片段中所有 token 概率的最小值。
    如果没有搜索片段，返回 1.0 (高置信度)。
    """
    if not token_ids or not log_probs or len(token_ids) != len(log_probs):
        # 如果没有提供概率数据，默认不进行惩罚
        return 1.0

    # 获取标签的 token id
    # 注意：不同 tokenizer 对特殊 token 的处理可能不同，这里假设是单独的 token
    # 实际使用中建议先 print(tokenizer.encode("<search>")) 确认
    try:
        search_start_id = tokenizer.encode("<search>", add_special_tokens=False)[0]
        search_end_id = tokenizer.encode("</search>", add_special_tokens=False)[0]
    except:
        # 如果 tokenizer 无法编码这些 tag，可能需要 fallback 到字符串匹配（较难精确对齐概率）
        return 1.0

    min_prob = 1.0
    in_search = False
    found_search = False
    
    current_segment_probs = []

    for i, tid in enumerate(token_ids):
        if tid == search_start_id:
            in_search = True
            current_segment_probs = []
            continue
        
        if tid == search_end_id:
            if in_search and current_segment_probs:
                found_search = True
                # 计算该片段内的最小概率
                segment_min = min(current_segment_probs)
                # 更新全局最小概率
                if segment_min < min_prob:
                    min_prob = segment_min
            in_search = False
            continue
        
        if in_search:
            # log_prob 转 prob
            prob = math.exp(log_probs[i])
            current_segment_probs.append(prob)
    
    # 如果处于搜索状态但没有结束标签（截断情况），也可以计算当前累积的概率
    if in_search and current_segment_probs:
        found_search = True
        segment_min = min(current_segment_probs)
        if segment_min < min_prob:
            min_prob = segment_min

    return min_prob if found_search else 1.0

# =============================================================================
# 原有辅助函数 (保持不变)
# =============================================================================

def validate_format(text: str) -> Tuple[bool, str]:
    """Validate if the text follows the required format with paired tags."""
    if text.count('<think>') != text.count('</think>'):
        return False, "<think> </think> not paired"

    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return False, "<think> or </think> not found"

    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "<answer> or </answer> not found"

    # Check the order of search/result
    current_pos = 0
    while True:
        search_pos = text.find('<search>', current_pos)
        if search_pos == -1:
            break

        result_pos = text.find('<result>', search_pos)
        search_end_pos = text.find('</search>', search_pos)
        result_end_pos = text.find('</result>', result_pos)

        if -1 in (result_pos, search_end_pos, result_end_pos):
            return False, "search/result tags are incomplete"

        if not (search_pos < search_end_pos < result_pos < result_end_pos):
            return False, "search/result tags are nested in the wrong order"

        current_pos = result_end_pos

    # Check the order of python/result
    current_pos = 0
    while True:
        python_pos = text.find('<python>', current_pos)
        if python_pos == -1:
            break

        result_pos = text.find('<result>', python_pos)
        python_end_pos = text.find('</python>', python_pos)
        result_end_pos = text.find('</result>', result_pos)

        if -1 in (result_pos, python_end_pos, result_end_pos):
            return False, "python/result tags are incomplete"

        if not (python_pos < python_end_pos < result_pos < result_end_pos):
            return False, "python/result tags are nested in the wrong order"

        current_pos = result_end_pos

    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"
    answer_content = text[answer_start:answer_end]
    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False, "answer is missing \\boxed{} format"

    return True, "format is correct"

def extract_answer(text: str) -> Optional[str]:
    text = text.strip()
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    return match.group(1)

def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]
    left = "\\boxed{"
    assert s[:len(left)] == left
    assert s[-1] == "}"
    return s[len(left):-1]

def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_f1_score(prediction: str, ground_truths: Union[str, List[str]]) -> float:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    final_metric = {"f1": 0, "precision": 0, "recall": 0}
    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue
        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        final_metric["precision"] = max(precision, final_metric["precision"])
        final_metric["recall"] = max(recall, final_metric["recall"])
        final_metric["f1"] = max(f1, final_metric["f1"])
    return final_metric['f1']

def get_em_score(prediction: str, ground_truths: Union[str, List[str]]) -> float:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    normalized_prediction = normalize_answer(prediction)
    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_prediction == normalized_ground_truth:
            return 1.0
    return 0.0 

# =============================================================================
# 核心修改后的 compute_score
# =============================================================================

def compute_score(data_source: str, prompt_str: str, solution_str: str, ground_truth: Any, extra_info: Optional[Dict[str, Any]] = None, is_validate = False) -> Dict[str, Any]:
    """
    Compute reward score for a solution based on the ground truth.
    Supports beta-GRPO (Search-R1) logic by checking search confidence.
    """
    
    # 1. 基础初始化
    result = {
        "score": 0,
        "em_score": 0,
        "reason": "",
        "answer": "",
        "f1_score": 0,
    }
    # 兼容处理 ground_truth 格式
    if isinstance(ground_truth, dict) and 'target' in ground_truth:
        ground_truths = ground_truth['target']
    else:
        ground_truths = ground_truth
        
    result["ground_truth"] = json.dumps(ground_truths) if not isinstance(ground_truths, str) else ground_truths
    
    response = solution_str
    
    # 2. 格式校验
    valid_template, reason = validate_format(response)
    if not valid_template and not is_validate:
        result["score"] = -1
        result["em_score"] = 0
        result["reason"] = f"bad format: {reason}"
        return result
    
    # Remove EOS token if present
    if extra_info is not None and "tokenizer" in extra_info and extra_info["tokenizer"].eos_token and response.endswith(extra_info["tokenizer"].eos_token):
        response = response[:-len(extra_info["tokenizer"].eos_token)]
    
    # 3. 提取答案
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
    
    # 4. 计算基础得分 (F1 / EM)
    f1_score = get_f1_score(answer, ground_truths)
    em_score = get_em_score(answer, ground_truths)
    
    result["f1_score"] = f1_score
    result["em_score"] = em_score

    # 5. Search-R1 Beta-GRPO 核心修改: 不确定性惩罚
    # 需要从 extra_info 中获取 token_ids 和 log_probs
    # 只有当答案基本正确(f1>0)时，才进行惩罚计算，节省算力
    if f1_score > 0 and extra_info:
        token_ids = extra_info.get("response_ids", [])
        log_probs = extra_info.get("log_probs", [])
        tokenizer = extra_info.get("tokenizer", None)
        
        # 论文中的阈值 beta，通常设为 0.4
        BETA_THRESHOLD = 0.4 
        
        if token_ids and log_probs and tokenizer:
            search_confidence = calculate_search_confidence(token_ids, log_probs, tokenizer)
            
            # 如果置信度低于阈值，奖励归零
            if search_confidence < BETA_THRESHOLD:
                # 保留原始的 em/f1 记录用于统计，但最终 score 判为 0
                result["score"] = 0.0
                result["reason"] = f"Correct answer but low search confidence ({search_confidence:.3f} < {BETA_THRESHOLD})"
                return result
            else:
                # 置信度足够高，继续后续的 bonus 计算
                pass
        
    # 6. 计算最终奖励 (含 Bonus)
    # Bonus for using multiple tools correctly
    if f1_score > 0 and "</search>" in response and "</python>" in response:
        result["score"] = f1_score + 0.1
        result["reason"] = f"correct answer and calling search and python at the same time, get score: {f1_score + 0.1}"
    elif f1_score > 0:
        result["score"] = em_score
        result["reason"] = f"correct answer, get f1 score: {f1_score}"
    else:
        result["score"] = 0
        result["reason"] = f"wrong answer but good format: {answer}"
    
    return result


if __name__ == "__main__":
    # 模拟测试
    # 注意：实际运行需要安装 transformers 并提供模型路径，或者 mock tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    except:
        print("Warning: Tokenizer not found, using mock for demonstration.")
        class MockTokenizer:
            eos_token = "<|im_end|>"
            def encode(self, text, add_special_tokens=False):
                # 简单的 mock，实际必须用真实 tokenizer
                if text == "<search>": return [101]
                if text == "</search>": return [102]
                return [0]
        tokenizer = MockTokenizer()

    response_text = "<|im_start|>system...<search> query </search><result> result </result><answer>The value is \\boxed{12}</answer><|im_end|>"
    ground_truth = {'target': "12"}
    
    # 模拟数据：假设这是对应 response_text 的 token 和 log_prob
    # 这里 mock 一个高置信度的搜索 (prob=0.9 -> log_prob ~ -0.1)
    mock_token_ids = [101, 200, 201, 102, 300, 301] # 101=<search>, 102=</search>
    mock_log_probs = [-0.1, -0.1, -0.2, -0.1, -0.05, -0.05] 
    
    extra_info = {
        "tokenizer": tokenizer,
        "token_ids": mock_token_ids,
        "log_probs": mock_log_probs
    }
    
    res = compute_score("test_data_source", response_text, ground_truth, extra_info)
    print("Test Result (High Confidence):", res)
    
    # 模拟低置信度 (prob=0.1 -> log_prob ~ -2.3)
    mock_log_probs_low = [-0.1, -2.3, -0.2, -0.1, -0.05, -0.05]
    extra_info["log_probs"] = mock_log_probs_low
    res_low = compute_score("test_data_source", response_text, ground_truth, extra_info)
    print("Test Result (Low Confidence):", res_low)
