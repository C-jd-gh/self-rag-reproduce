import os
import json
# 开启 HuggingFace 镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset

print("🚀 正在从 HuggingFace 下载 HotpotQA (distractor 验证集)...")
dataset = load_dataset("hotpot_qa", "distractor", split="validation")

output_dir = "eval_data_official"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "hotpotqa_dev.jsonl")

print(f"✍️ 正在将数据转换为 Self-RAG 格式并写入 {output_file}...")
with open(output_file, "w", encoding="utf-8") as f:
    for item in dataset:
        ctxs = []
        # HotpotQA 官方数据中，context 包含了多篇文档的标题和句子列表
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            ctxs.append({"title": title, "text": "".join(sentences)})
            
        # Self-RAG 需要的 JSONL 格式
        self_rag_item = {
            "question": item["question"],
            "answers": [item["answer"]], # 标准答案
            "ctxs": ctxs # 包含 10 篇左右的混合文档（含干扰项和支撑项）
        }
        f.write(json.dumps(self_rag_item, ensure_ascii=False) + "\n")

print(f"✅ 转换完成！共提取 {len(dataset)} 条多跳推理测试数据。")