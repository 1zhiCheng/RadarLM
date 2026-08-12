"""v12: PKC 分割解码器
从 PKC 输出的 4 类 segmentation logits 解码出 object list:
  [{class: 'car', bbox_rd: [r0,c0,r1,c1], bbox_ra: [...],
    range_m, angle_deg, doppler_ms, box_rd, box_ra}, ...]

PKC_CLS_LABELS = {0: 'background', 1: 'pedestrian', 2: 'cyclist', 3: 'car'}
"""
import numpy as np
from scipy.ndimage import label, find_objects
from pathlib import Path

# 物理量映射常量 (与 generate_captions.py 一致)
DOPPLER_RES = 0.41968030701528203  # m/s / pixel
ANGLE_RES = 0.01227184630308513    # rad / pixel
RANGE_RES = 0.1953125              # m / pixel
RD_SHAPE = (256, 64)
RA_SHAPE = (256, 256)


CLASS_NAMES_EN = {0: 'background', 1: 'pedestrian', 2: 'cyclist', 3: 'car'}
CLASS_NAMES_CN = {0: '背景', 1: '行人', 2: '骑行者', 3: '汽车'}


def row_to_range_m(row, view_rows=256):
    """行转距离 (m): 第 0 行最远, 第 255 行最近"""
    return (view_rows - 1 - row) * RANGE_RES


def col_to_angle_deg(col, view_cols=256):
    """列转角度 (度): 第 128 列为 0°, 中心左右对称"""
    return (col - view_cols // 2) * (180 / 3.141592653589793) * ANGLE_RES


def col_to_doppler_ms(col, view_cols=64):
    """列转多普勒速度 (m/s): 第 32 列为 0"""
    return (col - view_cols // 2) * DOPPLER_RES


def decode_pkc_segmentation(pkc_logits_rd, pkc_logits_ra, pkc_logits_ad=None,
                              min_area=4, threshold=0.5):
    """从 PKC x9 logits 解码出 object list.

    Args:
        pkc_logits_rd: (4, 256, 64) 4 类分割 logits (RD)
        pkc_logits_ra: (4, 256, 256) 4 类分割 logits (RA)
        pkc_logits_ad: (4, 256, 64) 4 类分割 logits (AD)
        min_area: 最小连通域像素数 (过滤噪声)
        threshold: 类别概率阈值 (argmax 后 softmax max > threshold)

    Returns:
        objects: [{
            'class': int (1/2/3),
            'class_en': 'pedestrian'/'cyclist'/'car',
            'class_cn': '行人'/'骑行者'/'汽车',
            'bbox_rd': [r0, c0, r1, c1],
            'bbox_ra': [r0, c0, r1, c1],
            'bbox_ad': [r0, c0, r1, c1] (可选),
            'range_m': float, 'angle_deg': float, 'doppler_ms': float,
            'area': int,
            'confidence': float,
        }, ...]
    """
    objects = []

    # 1. argmax → 预测类别
    pred_rd = pkc_logits_rd.argmax(axis=0)  # (256, 64)
    pred_ra = pkc_logits_ra.argmax(axis=0)  # (256, 256)
    pred_ad = pkc_logits_ad.argmax(axis=0) if pkc_logits_ad is not None else None

    # 2. 找各类别的连通域
    for cls in [1, 2, 3]:  # skip 0=background
        # RD 视图
        mask_rd = (pred_rd == cls)
        if mask_rd.sum() < min_area:
            continue
        labeled_rd, n_rd = label(mask_rd)
        for inst_id in range(1, n_rd + 1):
            inst_mask = (labeled_rd == inst_id)
            if inst_mask.sum() < min_area:
                continue
            rows, cols = np.where(inst_mask)
            bbox_rd = [int(rows.min()), int(cols.min()),
                        int(rows.max()), int(cols.max())]
            # RA 视图同位置 (用 bounding box 中心找对应区域)
            # 简化: 用 RD bbox 推断 RA 区域
            r_center = (rows.min() + rows.max()) // 2
            c_center = (cols.min() + cols.max()) // 2
            # 在 RA 视图找同样类别连通域
            mask_ra = (pred_ra == cls)
            if mask_ra.sum() < min_area:
                bbox_ra = [r_center * 256 // 256, c_center * 256 // 256,
                            r_center * 256 // 256, c_center * 256 // 256]
            else:
                labeled_ra, n_ra = label(mask_ra)
                # 找 RA bbox: 选最大连通域
                sizes = [(labeled_ra == i).sum() for i in range(1, n_ra + 1)]
                largest = np.argmax(sizes) + 1
                r_rows, r_cols = np.where(labeled_ra == largest)
                bbox_ra = [int(r_rows.min()), int(r_cols.min()),
                            int(r_rows.max()), int(r_cols.max())]
            # 物理量
            range_m = row_to_range_m(r_center)
            angle_deg = col_to_angle_deg((bbox_ra[1] + bbox_ra[3]) // 2)
            doppler_ms = col_to_doppler_ms(c_center)
            # 置信度 (softmax 后)
            conf = float(np.exp(pkc_logits_rd[cls, r_center, c_center]) /
                          np.exp(pkc_logits_rd[:, r_center, c_center]).sum())

            obj = {
                'class': int(cls),
                'class_en': CLASS_NAMES_EN[cls],
                'class_cn': CLASS_NAMES_CN[cls],
                'bbox_rd': bbox_rd,
                'bbox_ra': bbox_ra,
                'range_m': round(range_m, 2),
                'angle_deg': round(angle_deg, 2),
                'doppler_ms': round(doppler_ms, 2),
                'area': int(mask_rd.sum()),
                'confidence': round(conf, 3),
            }
            if pred_ad is not None:
                mask_ad = (pred_ad == cls)
                if mask_ad.sum() > 0:
                    labeled_ad, n_ad = label(mask_ad)
                    sizes = [(labeled_ad == i).sum() for i in range(1, n_ad + 1)]
                    largest = np.argmax(sizes) + 1
                    a_rows, a_cols = np.where(labeled_ad == largest)
                    obj['bbox_ad'] = [int(a_rows.min()), int(a_cols.min()),
                                       int(a_rows.max()), int(a_cols.max())]
            objects.append(obj)

    # 按 range 排序 (近的在前)
    objects.sort(key=lambda o: o['range_m'])
    return objects


def objects_to_text(objects, level=3):
    """把 object list 转成结构化文本 (curriculum 不同 level 不同 detail)

    Args:
        level: 1=完整 (Stage 1), 2=部分 (Stage 2), 3=仅图像 (Stage 3)
    """
    if not objects:
        text = "雷达检测结果: 未检测到目标.\n"
    else:
        n = len(objects)
        lines = [f"雷达检测结果: 检测到 {n} 个目标."]
        for i, obj in enumerate(objects, 1):
            head = f"  目标{i}: {obj['class_cn']} ({obj['class_en']})"
            if level >= 1:
                head += f", 距离={obj['range_m']:.1f}m, 角度={obj['angle_deg']:.1f}°"
                head += f", 多普勒={obj['doppler_ms']:.1f}m/s"
            if level >= 2:
                head += f", RD区域=[{obj['bbox_rd'][0]},{obj['bbox_rd'][1]}]-[{obj['bbox_rd'][2]},{obj['bbox_rd'][3]}]"
                head += f", RA区域=[{obj['bbox_ra'][0]},{obj['bbox_ra'][1]}]-[{obj['bbox_ra'][2]},{obj['bbox_ra'][3]}]"
            if 'bbox_ad' in obj:
                head += f", AD区域=[{obj['bbox_ad'][0]},{obj['bbox_ad'][1]}]-[{obj['bbox_ad'][2]},{obj['bbox_ad'][3]}]"
            lines.append(head)
        text = "\n".join(lines) + "\n"

    if level == 3:
        text = ""  # Stage 3 完全靠图像
    return text


# QA 模板生成器 (基于 object list)
def generate_qa_from_objects(objects, qa_type):
    """根据检测结果生成 QA 答案"""
    if not objects:
        no_target = "图中无目标"
    else:
        n = len(objects)

    templates = {
        "是否有目标": "有" if objects else "无",
        "目标数量": str(len(objects)) if objects else no_target,
        "是什么类别": ", ".join(o['class_cn'] for o in objects) if objects else no_target,
        "距离": objects[0]['range_m'] if objects else no_target,
        "角度": objects[0]['angle_deg'] if objects else no_target,
        "多普勒": objects[0]['doppler_ms'] if objects else no_target,
        "有几个汽车": str(sum(1 for o in objects if o['class'] == 3)) if objects else no_target,
        "有几个行人": str(sum(1 for o in objects if o['class'] == 1)) if objects else no_target,
        "有骑行者吗": "有" if any(o['class'] == 2 for o in objects) else "无",
        "是汽车吗": "是" if any(o['class'] == 3 for o in objects) else "否",
    }
    return templates.get(qa_type, "未知")


if __name__ == "__main__":
    # 测试 (mock pkc output)
    import torch
    pkc_rd = torch.randn(4, 256, 64)
    pkc_ra = torch.randn(4, 256, 256)
    pkc_ad = torch.randn(4, 256, 64)
    objs = decode_pkc_segmentation(pkc_rd.numpy(), pkc_ra.numpy(), pkc_ad.numpy())
    print(f"decoded {len(objs)} objects")
    for o in objs:
        print(f"  {o['class_cn']} ({o['class_en']}) range={o['range_m']}m angle={o['angle_deg']}°")
    print("\n--- Stage 1 (full) ---")
    print(objects_to_text(objs, level=1))
    print("\n--- Stage 2 (partial) ---")
    print(objects_to_text(objs, level=2))
    print("\n--- Stage 3 (image only) ---")
    print(repr(objects_to_text(objs, level=3)))