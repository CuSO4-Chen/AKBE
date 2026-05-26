import re
import sys
import string
from typing import Union, List, Dict, Any, Optional, Tuple
from collections import Counter
from transformers import AutoTokenizer
import json

def validate_format(text: str) -> Tuple[bool, str]:
    """
    Validate if the text follows the required format with paired tags.
    
    Args:
        text: The text to validate
        
    Returns:
        tuple: (is_valid, reason)
    """
    # Check if <think></think>, <answer></answer> is paired
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

    # Check if \boxed{} is in the answer
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"
    answer_content = text[answer_start:answer_end]
    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False, "answer is missing \\boxed{} format"

    return True, "format is correct"
    

def extract_answer(text: str) -> Optional[str]:
    """
    Extract answer content from the text within <answer> tags.
    
    Args:
        text: The text to extract answer from
        
    Returns:
        Optional[str]: The extracted answer or None if no match
    """
    text = text.strip()

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    return match.group(1)


def remove_boxed(s: str) -> str:
    """
    Remove the LaTeX \boxed{} wrapper from the string.
    
    Args:
        s: String potentially containing \boxed{}
        
    Returns:
        str: String with \boxed{} removed
    """
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string: str) -> Optional[str]:
    """
    Extract the last \boxed{} content from the string.
    
    Args:
        string: String to extract \boxed{} from
        
    Returns:
        Optional[str]: The extracted \boxed{} content or None if not found
    """
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
    """
    Normalize the answer string by removing articles, white spaces, punctuation and converting to lowercase.
    
    Args:
        s: String to normalize
        
    Returns:
        str: Normalized string
    """
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
    """
    Calculate F1 score between prediction and ground truths.
    
    Args:
        prediction: The predicted answer
        ground_truths: The ground truth answer(s)
        
    Returns:
        float: F1 score
    """
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
    """
    Calculate Exact Match (EM) score between prediction and ground truths.
    Returns 1.0 if exact match, 0.0 otherwise.
    
    Args:
        prediction: The predicted answer
        ground_truths: The ground truth answer(s)
        
    Returns:
        float: EM score (0.0 or 1.0)
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    
    normalized_prediction = normalize_answer(prediction)
    
    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_prediction == normalized_ground_truth:
            return 1.0
    
    return 0.0

def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info: Optional[Dict[str, Any]] = None, is_validate = False) -> Dict[str, Any]:
    """
    Compute reward score for a solution based on the ground truth.
    Modified to include penalties for tool usage and bonuses for no-tool solutions.
    """
    
    # 初始化统一的返回结构
    result = {
        "score": 0,
        "em_score": 0,     
        "reason": "",
        "answer": "",
        "f1_score": 0,
    }
    # 处理 ground_truth 可能的格式差异，确保能 json dump
    target = ground_truth.get('target', ground_truth) if isinstance(ground_truth, dict) else ground_truth
    result["ground_truth"] = json.dumps(target)
    
    response = solution_str
    
    # 1. 格式检查 (Format Check) - 优先级最高
    valid_template, reason = validate_format(response)
    
    if not valid_template and not is_validate:
        result["score"] = -1
        result["em_score"] = 0
        result["reason"] = f"bad format: {reason}"
        return result
    
    # Remove EOS token if present
    if extra_info is not None and "tokenizer" in extra_info and extra_info["tokenizer"].eos_token and response.endswith(extra_info["tokenizer"].eos_token):
        response = response[:-len(extra_info["tokenizer"].eos_token)]
    
    # 2. 答案提取 (Answer Extraction)
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
    
    # 3. 正确性计算 (Correctness Calculation)
    # 注意：这里使用 ground_truth['target'] 适配您的输入格式
    gt_target = ground_truth['target'] if isinstance(ground_truth, dict) else ground_truth
    
    f1_score = get_f1_score(answer, gt_target)
    result["f1_score"] = f1_score

    em_score = get_em_score(answer, gt_target)
    result["em_score"] = em_score
    
    # ----------------------------------------------------------------------
    # 4. Reward Shaping (您的新需求)
    # ----------------------------------------------------------------------
    
    # 计算搜索工具调用次数
    # 简单的字符串匹配即可，因为已经通过了 validate_format，XML 结构应该是相对完整的
    search_count = response.count("<search>")
    
    is_correct = (em_score == 1.0) # 定义完全答对

    free_quota = 1       
    bonus_no_tool = 1.5   
    penalty_search = 0.05
    
    if is_correct:
        if search_count == 0:
            # === No-Tool Trajectory (Correct) ===
            # Requirement: 1.5 score
            result["score"] = bonus_no_tool
            result["reason"] = f"correct answer without tool usage. EM: {em_score}"
        elif search_count <= free_quota:
            # === With-Tool Trajectory (Correct) ===
            result["score"] = 1.0
            result["reason"] = "Correct & Within Quota."
        else:
            # === With-Tool Trajectory (Correct but Over Quota) ===
            excess = search_count - free_quota
            penalty = excess * penalty_search
            final = max(0.1, 1.0 - penalty)
            result["score"] = final
            result["reason"] = f"Correct but Efficiency Penalty: -{penalty}"
            # 稍微做一点数值保护，防止浮点数精度问题
            final = float(f"{final:.4f}")
            result["score"] = final
            result["reason"] = f"Correct but Efficiency Penalty: -{penalty}"
    else:
        # === Incorrect Answer ===
        # Requirement: 0 score
        result["score"] = 0
        result["reason"] = f"wrong answer. EM: {em_score}, F1: {f1_score}"

    return result
