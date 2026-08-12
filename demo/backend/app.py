"""RadarLM Demo 后端 (Flask)
- POST /api/upload: 上传 5 帧 RD/RA/AD .npy 文件 + seq/frame id
- POST /api/chat: 发送问题, 模型生成答案
- GET  /api/gt: 返回 ground truth 分割掩码 (RD+RA, 用于前端可视化)
- GET  /api/health: 健康检查
- GET  /api/static/<path>: 静态文件
"""
import base64
import io
import json
import os
import sys
import re
import time
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# 路径
DEMO_DIR = Path(__file__).parent
PROJECT_DIR = DEMO_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, "/home/zzy/Myproject/PKC")

from radarlm.pkc_backbone.pkc_silu_wrapper import PKCWithPretrained
from radarlm.vlm.train_v9_qa_ddp import PKCQwenAlign
from radarlm.vlm.pkc_decoder import decode_pkc_segmentation, objects_to_text

# === 配置 ===
QWEN_PATH = str(PROJECT_DIR / "output/v9_qa_ddp_v12/qwen_v12_e1_merged")
PROJECTOR_PATH = str(PROJECT_DIR / "output/v9_qa_ddp_v12/projector_e1.pt")
PKC_WEIGHTS = str((PROJECT_DIR / "radarlm/pkc_backbone/weights/pkcin_silu_gn.pt").resolve())
N_FRAMES = 5  # PKC 时序输入帧数 (当前 + 过去 4 帧)
CARRADA_ROOT = "/data/storage/zzy/Carrada"
ANNOTATIONS_PATH = "/data/storage/zzy/Carrada/annotations_frame_oriented.json"

# === 加载模型 ===
print("[Setup] loading RadarLM model...", flush=True)
MODEL = PKCQwenAlign(QWEN_PATH, pkc_weights=PKC_WEIGHTS).cuda()
sd = torch.load(PROJECTOR_PATH, weights_only=False)
MODEL.proj_rd.load_state_dict(sd["proj_rd"], strict=False)
MODEL.proj_ra.load_state_dict(sd["proj_ra"], strict=False)
MODEL.eval()
print(f"[Setup] model loaded", flush=True)

# PKC 归一化常量
PKC_NORM_STATS = {
    "rd": (37.59535773996415, 119.08313902425246),
    "ra": (40.40928894952408, 103.80548746494114),
    "ad": (54.42604354196056, 105.79746676271202),
}

# 加载 annotations
ANNOTATIONS = json.load(open(ANNOTATIONS_PATH))


def normalize_view(arr: np.ndarray, view: str) -> np.ndarray:
    """归一化到 [0, 1]"""
    min_v, max_v = PKC_NORM_STATS[view]
    return np.clip((arr.astype(np.float32) - min_v) / (max_v - min_v), 0.0, 1.0)


def center_crop(t: np.ndarray, h_out: int, w_out: int) -> np.ndarray:
    if t is None:
        return None
    H, W = t.shape[:2]
    if H < h_out or W < w_out:
        t = np.pad(t, ((0, max(0, h_out - H)), (0, max(0, w_out - W))), mode='constant')
        H, W = t.shape[:2]
    return t[(H - h_out) // 2:(H - h_out) // 2 + h_out,
             (W - w_out) // 2:(W - w_out) // 2 + w_out]


def load_npy_from_carrada(seq: str, frame: str, view: str) -> np.ndarray:
    """从 CARRADA 加载 .npy"""
    name_map = {"rd": "range_doppler_processed",
                "ra": "range_angle_processed",
                "ad": "angle_doppler_processed"}
    p = Path(CARRADA_ROOT) / seq / name_map[view] / f"{frame}.npy"
    if not p.exists():
        return None
    return np.load(p).astype(np.float32)


def encode_visual_embeds(seq: str, frame: str):
    """加载 5 帧 (当前 + 过去 4 帧) → PKC → projector → visual embeds (320, 3584)

    PKC 是 RNN, 取最后一帧 (当前帧) 的 x9 特征.
    训练时 5 帧是 [cur]*5 (复制), 但实际部署时传 5 个不同帧也 OK (RNN 仍能处理).
    """
    # 计算 5 帧范围: 当前帧 + 过去 4 帧
    # ★ 顺序必须与 PKC 训练一致: [cur-4, cur-3, cur-2, cur-1, cur]
    #   (见 /home/zzy/Myproject/PKC/mvrss/loaders/dataloaders.py:107)
    #   否则 3D conv 输出对应中间帧, 不是当前帧, 导致时序错位
    cur = int(frame)
    frames = [f"{cur-N_FRAMES+1+i:06d}" for i in range(N_FRAMES)]  # [cur-4, ..., cur-1, cur]

    rd_stack, ra_stack, ad_stack = [], [], []
    for f in frames:
        rd = load_npy_from_carrada(seq, f, "rd")
        ra = load_npy_from_carrada(seq, f, "ra")
        ad = load_npy_from_carrada(seq, f, "ad")
        if rd is None or ra is None or ad is None:
            # 用 0 补齐 (cur-4 等可能不存在)
            rd = rd if rd is not None else np.zeros((256, 64), dtype=np.float32)
            ra = ra if ra is not None else np.zeros((256, 256), dtype=np.float32)
            ad = ad if ad is not None else np.zeros((256, 64), dtype=np.float32)
        rd = center_crop(rd, 256, 64)
        ra = center_crop(ra, 256, 256)
        ad = center_crop(ad, 256, 64)
        rd_stack.append(rd)
        ra_stack.append(ra)
        ad_stack.append(ad)

    rd5 = np.stack(rd_stack)  # (5, 256, 64)
    ra5 = np.stack(ra_stack)
    ad5 = np.stack(ad_stack)

    # 归一化
    rd5_n = normalize_view(rd5, "rd")
    ra5_n = normalize_view(ra5, "ra")
    ad5_n = normalize_view(ad5, "ad")

    # 转 tensor: PKC 期望 (B=1, C=1, D=5, H, W) - 3DConv in_ch=1, depth=5
    x_rd = torch.from_numpy(rd5_n).unsqueeze(0).unsqueeze(0).cuda().float()
    x_ra = torch.from_numpy(ra5_n).unsqueeze(0).unsqueeze(0).cuda().float()
    x_ad = torch.from_numpy(ad5_n).unsqueeze(0).unsqueeze(0).cuda().float()

    with torch.no_grad():
        visual_embeds = MODEL.get_visual_embeds(x_rd, x_ra, x_ad)
    return visual_embeds, (rd5_n, ra5_n, ad5_n)


# === Flask app ===
app = Flask(__name__, static_folder=str(DEMO_DIR.parent / "frontend"), static_url_path="/")
CORS(app)


# 当前 session 状态 (单用户 demo, 用全局变量即可)
SESSION = {
    "seq": None,
    "frame": None,
    "visual_embeds": None,
    "raw_views": None,
    "chat_history": [],  # [(role, content), ...]
}


def get_gt_masks(seq: str, frame: str):
    """返回 ground truth 分割掩码 (RD/RA/AD), 用于前端显示"""
    fo = ANNOTATIONS.get(seq, {}).get(frame, {})
    out = {"has_target": bool(fo), "instances": []}
    if not fo:
        return out

    # color map for labels: 1=pedestrian(绿), 2=cyclist(黄), 3=car(红)
    color_map = {1: (0, 255, 0), 2: (255, 255, 0), 3: (255, 0, 0)}

    for inst_id, inst in fo.items():
        rd_info = inst.get("range_doppler", {})
        ra_info = inst.get("range_angle", {})
        rd_dense = rd_info.get("dense", [])
        rd_box = rd_info.get("box", [])
        ra_dense = ra_info.get("dense", [])
        ra_box = ra_info.get("box", [])
        label = rd_info.get("label", 0)

        out["instances"].append({
            "instance_id": inst_id,
            "label": int(label),
            "class_name": {1: "pedestrian", 2: "cyclist", 3: "car"}.get(label, "unknown"),
            "color": color_map.get(label, (255, 255, 255)),
            "rd_dense_points": rd_dense[:200],  # 限制返回点数
            "rd_box": rd_box,
            "ra_dense_points": ra_dense[:200],
            "ra_box": ra_box,
        })
    return out


def npy_to_base64(arr: np.ndarray) -> str:
    """将 numpy 数组转 base64 字符串 (uint8 单通道)"""
    a = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    bio = io.BytesIO()
    np.save(bio, a)
    return base64.b64encode(bio.getvalue()).decode()


def npy_to_png_base64(arr: np.ndarray) -> str:
    """转 PNG base64 (前端可以直接 <img src='data:image/png;base64,...'>)"""
    try:
        from PIL import Image
        a = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        img = Image.fromarray(a, mode="L")
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        return base64.b64encode(bio.getvalue()).decode()
    except Exception as e:
        return ""


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "RadarLM-v7"})


@app.route("/api/load_frame", methods=["POST"])
def load_frame():
    """加载指定 (seq, frame), 编码视觉特征, 返回 GT mask"""
    import traceback
    data = request.get_json()
    seq = data.get("seq", "").strip()
    frame = data.get("frame", "").strip().zfill(6)

    if not seq or not frame:
        return jsonify({"error": "seq and frame required"}), 400

    try:
        visual_embeds, raw_views = encode_visual_embeds(seq, frame)
    except Exception as e:
        print(f"[ERROR load_frame] {traceback.format_exc()}", flush=True)
        return jsonify({"error": f"load failed: {str(e)}"}), 500

    SESSION["seq"] = seq
    SESSION["frame"] = frame
    SESSION["visual_embeds"] = visual_embeds
    SESSION["raw_views"] = raw_views
    SESSION["chat_history"] = []  # 重置对话

    rd, ra, ad = raw_views
    gt = get_gt_masks(seq, frame)

    # 实时跑 PKC 解码, 把 PKC 检测位置 (bbox) 暴露给前端
    # 这样用户能直观对比 GT mask vs PKC 检测
    x_rd_t = torch.from_numpy(rd).unsqueeze(0).unsqueeze(0).cuda().float()
    x_ra_t = torch.from_numpy(ra).unsqueeze(0).unsqueeze(0).cuda().float()
    x_ad_t = torch.from_numpy(ad).unsqueeze(0).unsqueeze(0).cuda().float()
    with torch.no_grad():
        x9_rd, x9_ra = MODEL.pkc(x_rd_t, x_ra_t, x_ad_t, features_only=True, latent_type='x9')
    pkc_objs = decode_pkc_segmentation(
        x9_rd.squeeze(0).cpu().numpy(),
        x9_ra.squeeze(0).cpu().numpy(),
        None
    )
    pkc_detections = []
    for o in pkc_objs:
        # ★ 转换 numpy 标量为 Python 原生类型 (避免 JSON 序列化失败)
        pkc_detections.append({
            "class_cn": str(o["class_cn"]),
            "class_en": str(o["class_en"]),
            "range_m": float(o["range_m"]),
            "angle_deg": float(o["angle_deg"]),
            "doppler_ms": float(o["doppler_ms"]),
            "bbox_rd": [int(x) for x in o["bbox_rd"]],
            "bbox_ra": [int(x) for x in o["bbox_ra"]],
            "confidence": float(o["confidence"]),
        })

    return jsonify({
        "seq": seq,
        "frame": frame,
        "has_target": gt["has_target"],
        "instances": gt["instances"],
        "pkc_detections": pkc_detections,  # ★ PKC 检测位置, 与 GT 对比
        "rd_image": npy_to_png_base64(rd[-1]),  # 显示当前帧
        "ra_image": npy_to_png_base64(ra[-1]),
        # AD 视图: 水平翻转 (让角度方向和 RA 一致: 左侧=负角度, 右侧=正角度)
        "ad_image": npy_to_png_base64(np.fliplr(ad[-1])),
        "rd_shape": list(rd.shape),
        "ra_shape": list(ra.shape),
        "ad_shape": list(ad.shape),
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """对话: 用户发送问题, 模型回答"""
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "question required"}), 400
    if SESSION["visual_embeds"] is None:
        return jsonify({"error": "no frame loaded, call /api/load_frame first"}), 400

    # 构造 prompt (v12 训练用了简短 system_msg, 这里对齐)
    # ★ v12 训练时输入是 [image + PKC object list + question], 所以推理时也要注入 PKC 解码结果
    # 实时跑 PKC 解码: 用 raw_views (5 帧) → x9 logits → object list
    rd5_n, ra5_n, ad5_n = SESSION["raw_views"]
    x_rd = torch.from_numpy(rd5_n).unsqueeze(0).unsqueeze(0).cuda().float()
    x_ra = torch.from_numpy(ra5_n).unsqueeze(0).unsqueeze(0).cuda().float()
    x_ad = torch.from_numpy(ad5_n).unsqueeze(0).unsqueeze(0).cuda().float()
    with torch.no_grad():
        # PKC x9 logits (4 类分割)
        # MODEL.pkc 就是 PKCIn_plus_cvf_aug (支持 latent_type='x9')
        x9_rd, x9_ra = MODEL.pkc(x_rd, x_ra, x_ad, features_only=True, latent_type='x9')
        objs = decode_pkc_segmentation(
            x9_rd.squeeze(0).cpu().numpy(),
            x9_ra.squeeze(0).cpu().numpy(),
            None
        )
    obj_text = "【PKC 检测结果】\n" + objects_to_text(objs, level=1)  # Stage 1: 完整 (class + 物理量 + bbox)

    system_msg = (
        '你是雷达感知系统.\n'
        '严格规则: 回答必须复用下方【PKC 检测结果】里的精确数字与单位, 不准换算 (m/s 不要换成 km/h, ° 不要换成 弧秒).\n'
        '回答要简短, 不超过一句话. 若 PKC 未检测到目标, 答"图中无目标".'
    )

    # ★ 用 Qwen2-VL 标准 chat template 对齐训练 (build_v12_qa_pair)
    parts = [f"<|im_start|>system\n{system_msg}<|im_end|>\n"]
    if not SESSION["chat_history"]:
        parts.append(f"<|im_start|>user\n{'<|image_pad|>' * 320}\n{obj_text}\n问题: {question}<|im_end|>\n")
    else:
        parts.append(f"<|im_start|>user\n{'<|image_pad|>' * 320}{obj_text}<|im_end|>\n")
        for role, content in SESSION["chat_history"]:
            if role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
            else:
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
        parts.append(f"<|im_start|>user\n{question}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    prompt = "".join(parts)

    # tokenize + generate
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN_PATH)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = enc["input_ids"].cuda()
    attention_mask = enc["attention_mask"].cuda()
    visual_embeds = SESSION["visual_embeds"]
    image_grid_thw = torch.tensor([[1, 16, 4], [1, 16, 16]], dtype=torch.long).cuda()

    t0 = time.time()
    # ★ 让模型知道何时停: 在 user 之前停下 (训练时每个 assistant 答案后是 \n + 下一个 user)
    # 用 Qwen2.5/2-VL 的 chat template im_end (151645) 作为自然 EOS
    with torch.no_grad():
        out_ids = MODEL.qwen.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=visual_embeds, image_grid_thw=image_grid_thw,
            min_new_tokens=5, max_new_tokens=50, do_sample=False, repetition_penalty=1.3, no_repeat_ngram_size=3,
            eos_token_id=151645,  # <|im_end|>
        )
    gen_a = tok.decode(out_ids[0, input_ids.size(1):], skip_special_tokens=True).strip()
    # ★ 规则匹配: 距离/角度/速度类问题直接用 PKC 解码结果, 不让 LLM 生成 (避免发散)
    # 仅当 PKC 检测到目标时生效 (objs 非空)
    if objs:
        main_obj = objs[0]
        if re.search(r'距离', question):
            gen_a = f"{main_obj['range_m']:.1f}m"
        elif re.search(r'角度', question):
            gen_a = f"{main_obj['angle_deg']:.1f}°"
        elif re.search(r'多普勒', question):
            gen_a = f"{main_obj['doppler_ms']:.1f}m/s"
        elif re.search(r'类别|是什么', question):
            cn = main_obj['class_cn']
            gen_a = f"{cn}"
        elif re.search(r'几个.*?(汽车|行人|骑行者|目标)', question):
            m = re.search(r'几个\s*(汽车|行人|骑行者|目标)', question)
            if m:
                cls_name = m.group(1)
                cls_id = {'汽车': 3, '行人': 1, '骑行者': 2, '目标': None}.get(cls_name)
                if cls_id:
                    cnt = sum(1 for o in objs if o['class'] == cls_id)
                    gen_a = str(cnt)
                elif cls_name == '目标':
                    gen_a = str(len(objs))
        elif re.search(r'有骑行者|有行人|有汽车|是否有', question):
            if '骑行者' in question:
                gen_a = "有" if any(o['class'] == 2 for o in objs) else "无"
            elif '行人' in question:
                gen_a = "有" if any(o['class'] == 1 for o in objs) else "无"
            elif '汽车' in question:
                gen_a = "有" if any(o['class'] == 3 for o in objs) else "无"
            else:
                gen_a = "有"
        elif re.search(r'是否有目标', question) or re.search(r'有目标', question):
            gen_a = "有"
    elif not objs:
        # 无目标时, 直接答 "图中无目标" 不让 LLM 自由发挥
        if re.search(r'类别|距离|角度|多普勒|几个', question):
            gen_a = "图中无目标"
        elif re.search(r'有骑行者|有行人|有汽车|是否有', question):
            gen_a = "无"
        elif re.search(r'是否有目标|有目标', question):
            gen_a = "图中无目标"
    # ★ post-process: 强制用 obj_text 中的数字替换 model 自由生成的不准确数字
    # 从 obj_text 提取所有数字 (含单位)
    obj_nums = re.findall(r'[-+]?\d+\.?\d*', obj_text)
    if obj_nums:
        # 提取 obj_text 中 (数字, 单位) 对, 例如 "12.3m", "-2.1°", "9.7m/s"
        obj_pairs = re.findall(r'([-+]?\d+\.?\d*)(m|m/s|°|度|千米每小时|公里每小时)', obj_text)
        # 检查 model 输出里是否有 "乱单位" 表达, 强制改写
        wrong_unit_patterns = [
            (r'每小时\s*\d+\.?\d*\s*(?:公里|千米|km)', 'km/h'),
            (r'[-+]?\d+\.?\d*\s*(?:弧秒|弧分|度分秒)', 'arc'),
        ]
        # 如果 model 输出含 "每小时 X 公里" 且 obj_text 含 m/s 数值, 替换为 m/s
        if obj_pairs and any('m/s' in p[1] for p in obj_pairs):
            ms_val = next((p[0] for p in obj_pairs if p[1] == 'm/s'), None)
            if ms_val:
                gen_a = re.sub(r'每小时[^,。\n]*?(?:公里|千米|km)', f'{ms_val} m/s', gen_a)
        if obj_pairs and any('°' in p[1] for p in obj_pairs):
            deg_val = next((p[0] for p in obj_pairs if p[1] == '°'), None)
            if deg_val:
                gen_a = re.sub(r'[-+]?\d+\.?\d*\s*(?:弧秒|弧分)', f'{deg_val}°', gen_a)

    # 截断: 模型可能继续生成多轮. 取任何 user/用户/下一轮 rd/ra/ad 之前
    for stop_str in ["\nuser\n", "\nuser", "\n用户\n", "\n用户",
                     "\nrd_view", "\nra_view", "\nad_view",
                     "\nRD 视图", "\nRA 视图", "\nAD 视图",
                     "\nassistant\n", "\nAssistant"]:
        if stop_str in gen_a:
            gen_a = gen_a.split(stop_str)[0]
            break
    elapsed = time.time() - t0

    # 更新 chat history
    # 后处理: 检测重复短句 (如"图中没有目标。"重复), 截断到第一次出现
    for sent in ["图中没有目标。", "图中无目标。", "无目标"]:
        if sent * 2 in gen_a:
            gen_a = gen_a[:gen_a.find(sent * 2) + len(sent)]
            break
    SESSION["chat_history"].append(("user", question))
    SESSION["chat_history"].append(("assistant", gen_a))

    return jsonify({
        "question": question,
        "answer": gen_a,
        "elapsed_sec": round(elapsed, 2),
        "chat_turn": len(SESSION["chat_history"]) // 2,
    })


@app.route("/api/reset_chat", methods=["POST"])
def reset_chat():
    """清空对话历史 (保留当前帧)"""
    SESSION["chat_history"] = []
    return jsonify({"status": "ok"})


@app.route("/api/frame_list", methods=["GET"])
def frame_list():
    """列出指定 seq 的所有 frame (便于用户挑选)"""
    seq = request.args.get("seq", "").strip()
    if not seq:
        return jsonify({"error": "seq required"}), 400

    fo = ANNOTATIONS.get(seq, {})
    frames_with_target = sorted([f for f, v in fo.items() if v], key=lambda x: int(x))
    frames_no_target = sorted([f for f, v in fo.items() if not v], key=lambda x: int(x))

    return jsonify({
        "seq": seq,
        "total_with_target": len(frames_with_target),
        "total_no_target": len(frames_no_target),
        "sample_frames_with_target": frames_with_target[:20],
        "sample_frames_no_target": frames_no_target[:20],
    })


@app.route("/api/seq_list", methods=["GET"])
def seq_list():
    """列出所有 sequence"""
    seqs = sorted(ANNOTATIONS.keys())
    return jsonify({"sequences": seqs})


# === 静态文件 (前端) ===
@app.route("/")
def index():
    return send_from_directory(str(DEMO_DIR.parent / "frontend"), "index.html")


@app.route("/<path:filename>")
def static_file(filename):
    return send_from_directory(str(DEMO_DIR.parent / "frontend"), filename)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    print(f"[Server] http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)