"""v9 多模态雷达感知 QA 评价指标.

评价维度:
1. 类别判断 (4-选 1): A/B/C/D 答对比例
2. 目标存在判断: "无目标" / "有目标"
3. 目标计数: 模型答 N 个 vs 真实 N 个
4. 数值精度 (距离/多普勒): ±10% 容差
5. 边界框精度: 距离像素 5 内
6. 答案完整度: ROUGE-L 简化版 (token overlap)
"""
import re
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


# 类别映射 (assistant 第一问答 类别)
CLASS_LETTER_MAP = {"A": "汽车", "B": "行人", "C": "骑行者", "D": "无目标"}


def parse_first_class_answer(text: str) -> str:
    """从 assistant 答案中提取 类别 (A/B/C/D)."""
    text = text.strip()
    m = re.search(r"\b([A-D])\b", text)
    return m.group(1) if m else ""


def parse_target_count(text: str) -> int:
    """从文本提取目标数量. '0 个目标' -> 0, '1 个目标' -> 1."""
    # 匹配 "N 个目标"
    m = re.search(r"(\d+)\s*个\s*目标", text)
    if m: return int(m.group(1))
    m = re.search(r"没有\s*目标|无目标|没有\s*任何\s*目标|不存在", text)
    if m: return 0
    m = re.search(r"检测到\s*(\d+)", text)
    if m: return int(m.group(1))
    return -1


def parse_has_target(text: str) -> bool:
    """判断文本是否说'有目标' (True) 或 '无目标' (False). 模糊时返回 None."""
    has_keywords = ["检测到", "有目标", "存在", "目标1:", "图中检测到"]
    no_keywords = ["无目标", "没有目标", "没有检测到", "不存在", "未检测到", "没有"]
    text_lower = text.lower()
    has_count = sum(1 for k in has_keywords if k in text)
    no_count = sum(1 for k in no_keywords if k in text)
    if no_count > has_count: return False
    if has_count > no_count: return True
    return None  # ambiguous


def parse_numbers(text: str) -> List[float]:
    """提取文本中的所有数字 (含负号/小数)."""
    # 排除坐标/ID 类的数字 (如 "[10, 20]" 可能是坐标)
    nums = []
    # 匹配 ±X.X 或 X 等数字
    for m in re.finditer(r"(-?\d+\.?\d*)", text):
        s = m.group(1)
        try:
            nums.append(float(s))
        except ValueError:
            pass
    return nums


def parse_ranges(text: str) -> List[tuple]:
    """提取文本中的范围 (如 '38.5-40.0' 或 '38.5-40.0 m')."""
    ranges = []
    # 匹配 'X-Y' 或 'X.X-Y.Y'
    for m in re.finditer(r"(-?\d+\.?\d*)\s*[-到至~]\s*(-?\d+\.?\d*)", text):
        a = float(m.group(1)); b = float(m.group(2))
        if abs(a) < 1000 and abs(b) < 1000:  # 过滤年份等大数
            ranges.append((min(a, b), max(a, b)))
    return ranges


def number_match_score(ans_nums: List[float], gt_nums: List[float], tol=0.1) -> float:
    """比较两个数字列表, 容差 ±10%, 返回匹配率 (0-1)."""
    if not gt_nums: return 1.0 if not ans_nums else 0.5
    if not ans_nums: return 0.0
    matched = 0
    for g in gt_nums:
        for a in ans_nums:
            if abs(g) < 0.01:  # 接近 0
                if abs(a - g) < 0.1: matched += 1; break
            elif abs(a - g) / abs(g) < tol:
                matched += 1; break
    return matched / len(gt_nums)


def range_match_score(ans_ranges: List[tuple], gt_ranges: List[tuple], tol=0.1) -> float:
    """比较两个范围列表, 返回匹配率 (IoU-based)."""
    if not gt_ranges: return 1.0 if not ans_ranges else 0.5
    if not ans_ranges: return 0.0
    matched = 0
    for gr in gt_ranges:
        for ar in ans_ranges:
            # 计算 IoU
            inter_lo = max(gr[0], ar[0]); inter_hi = min(gr[1], ar[1])
            if inter_hi <= inter_lo: continue
            inter = inter_hi - inter_lo
            union = max(gr[1], ar[1]) - min(gr[0], ar[0])
            iou = inter / union if union > 0 else 0
            if iou > 0.5:  # 至少 50% overlap
                matched += 1; break
    return matched / len(gt_ranges)


def eval_qa_pair(question: str, true_answer: str, gen_answer: str) -> Dict:
    """评价单对 QA, 返回多个维度的指标."""
    metrics = {}
    # 1. 类别判断 (question 含 "是什么类别?" 则评价)
    if "是什么类别" in question or "类别" in question and "A." in question:
        gt_letter = parse_first_class_answer(true_answer)
        gen_letter = parse_first_class_answer(gen_answer)
        metrics["class_match"] = int(gt_letter == gen_letter) if gt_letter else None
        metrics["gt_letter"] = gt_letter
        metrics["gen_letter"] = gen_letter
    # 2. 目标存在判断
    if any(k in question for k in ["几个目标", "目标", "有行人", "有汽车", "目标数量"]):
        gt_has = parse_has_target(true_answer)
        gen_has = parse_has_target(gen_answer)
        if gt_has is not None and gen_has is not None:
            metrics["has_target_match"] = int(gt_has == gen_has)
        else:
            metrics["has_target_match"] = None
        # 3. 目标计数
        gt_cnt = parse_target_count(true_answer)
        gen_cnt = parse_target_count(gen_answer)
        if gt_cnt >= 0 and gen_cnt >= 0:
            metrics["count_match"] = int(gt_cnt == gen_cnt)
            metrics["count_diff"] = abs(gt_cnt - gen_cnt)
        else:
            metrics["count_match"] = None
    # 4. 数值精度 (距离/多普勒/角度)
    if any(k in question for k in ["距离", "速度", "多普勒", "角度", "范围"]):
        gt_nums = parse_numbers(true_answer)
        gen_nums = parse_numbers(gen_answer)
        metrics["num_match_score"] = number_match_score(gen_nums, gt_nums, tol=0.1)
        gt_ranges = parse_ranges(true_answer)
        gen_ranges = parse_ranges(gen_answer)
        metrics["range_match_score"] = range_match_score(gen_ranges, gt_ranges, tol=0.15)
    # 5. 边界框 (含 [row_min=...] 格式)
    if "边界框" in question or "bbox" in question.lower():
        gt_nums = parse_numbers(true_answer)
        gen_nums = parse_numbers(gen_answer)
        if len(gt_nums) >= 4 and len(gen_nums) >= 4:
            # 4 个数: row_min, col_min, row_max, col_max
            diff = sum(abs(a - g) for a, g in zip(gen_nums[:4], gt_nums[:4]))
            metrics["bbox_diff"] = diff
            metrics["bbox_match"] = int(diff < 20)  # 20 像素内算对
        else:
            metrics["bbox_match"] = None
    # 6. 简单 token overlap (ROUGE-1 简化)
    gt_tokens = set(true_answer.replace(" ", ""))
    gen_tokens = set(gen_answer.replace(" ", ""))
    if gt_tokens:
        overlap = len(gt_tokens & gen_tokens) / len(gt_tokens)
        metrics["token_overlap"] = overlap
    return metrics


def aggregate_metrics(metrics_list: List[Dict]) -> Dict:
    """聚合多个 sample 的评价指标."""
    agg = {}
    for k in ["class_match", "has_target_match", "count_match",
              "num_match_score", "range_match_score",
              "bbox_match", "token_overlap"]:
        vals = [m[k] for m in metrics_list if m.get(k) is not None]
        if vals:
            agg[f"avg_{k}"] = sum(vals) / len(vals)
            agg[f"n_{k}"] = len(vals)
    agg["n_samples"] = len(metrics_list)
    return agg


# 简单 test
if __name__ == "__main__":
    # Test
    metrics = eval_qa_pair(
        "图中目标是什么类别?\nA. 汽车 B. 行人 C. 骑行者 D. 无目标",
        "A",
        "A"
    )
    print("class_match:", metrics)
    metrics = eval_qa_pair(
        "图中目标有几个?",
        "当前帧共有 1 个目标, 1 个汽车。距离 2.9-3.5 m, 多普勒 9.7-10.5 m/s.",
        "当前帧共有 0 个目标"
    )
    print("count_match:", metrics)
    metrics = eval_qa_pair(
        "RD 视图 bbox?",
        "RD 视图 bbox: [row_min=237, col_min=55, row_max=240, col_max=57]",
        "RD 视图 bbox: [row_min=238, col_min=55, row_max=240, col_max=57]"
    )
    print("bbox_match:", metrics)