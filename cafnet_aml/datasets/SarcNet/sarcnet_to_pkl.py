"""
将 SarcNet 的 parquet 数据转为现有加载流程兼容的 pkl。
生成的结构与 MOSI/MUStARD 等数据一致：包含 text_bert、audio、vision、raw_text、id、regression_labels 等字段。

用法示例：
python datasets/sarcnet_to_pkl.py --parquet-dir datasets/SarcNet --output datasets/SarcNet/Processed/sarcnet_features.pkl
"""

import argparse
import io
import os
import pickle
from typing import Dict, List

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from transformers import AutoTokenizer


def image_to_feature(img_bytes: bytes) -> np.ndarray:
    """将图片缩放为 32x16 后展平，视觉维度由像素数量自动确定。"""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((32, 16))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (16,32,3)
    return arr.reshape(-1)


def convert_split(
    path: str,
    tokenizer,
    max_len: int = 50,
    vision_dim: int = None,
    audio_dim: int = 5,
) -> Dict[str, np.ndarray]:
    # 合并 chunk，避免 Struct 列在 chunked array 上不能直接 field()
    table = pq.read_table(path).combine_chunks()
    texts: List[str] = table["text"].to_pylist()
    labels = table["label"].to_numpy()
    ids = table["id"].to_pylist()
    image_struct = table["image"].combine_chunks()
    img_bytes = image_struct.field("bytes").to_pylist()

    # 文本：BERT 分词
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    input_ids = np.array(enc["input_ids"], dtype=np.int64)
    attn_mask = np.array(enc["attention_mask"], dtype=np.int64)
    token_type = np.array(enc["token_type_ids"], dtype=np.int64)
    text_bert = np.stack([input_ids, attn_mask, token_type], axis=1)  # (N,3,L)

    # 视觉：简单的像素均值特征，形状 (N,1,vision_dim)
    vision_flat = [image_to_feature(b) for b in img_bytes]
    if vision_dim is None:
        vision_dim = vision_flat[0].shape[0]
    vision = np.stack([v.reshape(vision_dim) for v in vision_flat], axis=0)
    vision = vision[:, None, :].astype(np.float32)
    vision_lengths = [1] * len(texts)

    # 音频占位：数据集中无音频，填充 0
    audio = np.zeros((len(texts), 1, audio_dim), dtype=np.float32)
    audio_lengths = [1] * len(texts)

    data = {
        "text_bert": text_bert.astype(np.int64),
        "audio": audio,
        "vision": vision,
        "audio_lengths": audio_lengths,
        "vision_lengths": vision_lengths,
        "raw_text": texts,
        "id": ids,
        "regression_labels": labels.astype(np.float32),
    }
    return data


def build_pkl(parquet_dir: str, output: str, max_len: int = 50):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    splits = {"train": "train-00000-of-00001.parquet",
              "valid": "validation-00000-of-00001.parquet",
              "test": "test-00000-of-00001.parquet"}
    result = {}
    for mode, fname in splits.items():
        path = os.path.join(parquet_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} 不存在")
        result[mode] = convert_split(path, tokenizer, max_len=max_len)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(result, f)
    print(f"已保存到 {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", type=str, default="datasets/SarcNet")
    parser.add_argument("--output", type=str, default="datasets/SarcNet/Processed/sarcnet_features.pkl")
    parser.add_argument("--max-len", type=int, default=50)
    args = parser.parse_args()
    build_pkl(args.parquet_dir, args.output, max_len=args.max_len)


if __name__ == "__main__":
    main()
