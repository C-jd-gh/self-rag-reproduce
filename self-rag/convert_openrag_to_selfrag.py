import json
from pathlib import Path
import argparse

def convert_openrag_to_selfrag(input_file, output_file):
    input_path = Path(input_file)
    output_path = Path(output_file)

    with input_path.open('r', encoding='utf-8') as f_in, \
         output_path.open('w', encoding='utf-8') as f_out:
        
        for line in f_in:
            data = json.loads(line)
            new_ctxs = []
            
            # 遍历 Open-RAG 的 ctxs 列表
            for c in data.get("ctxs", []):
                # 检查是否存在 [SEP]，如果存在，说明是拼接的证据链
                full_text = c.get("text", "")
                if "[SEP]" in full_text:
                    parts = full_text.split("[SEP]")
                    for part in parts:
                        # 尝试提取标题（Knowledge N: Title\nText）
                        # 如果没有明显的标题，则按照你的“最小修复”给个空标题
                        new_ctxs.append({
                            "title": "", 
                            "text": part.strip()
                        })
                else:
                    # 单个段落的情况
                    new_ctxs.append({
                        "title": "",
                        "text": full_text.strip()
                    })
            
            # 更新数据
            data["ctxs"] = new_ctxs
            
            # 写入新文件
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

def resolve_input_path(raw_name):
    candidate = Path(raw_name)
    if candidate.exists():
        return candidate

    fallback = Path("eval_data_official/inputs") / raw_name
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"找不到输入文件: {raw_name}\n"
        f"已尝试: {candidate} 和 {fallback}"
    )


def main():
    parser = argparse.ArgumentParser(description="Convert Open-RAG JSONL to Self-RAG format.")
    parser.add_argument("--input", help="Input JSONL file path")
    parser.add_argument("--output", help="Output JSONL file path")
    args = parser.parse_args()

    if args.input and args.output:
        in_path = resolve_input_path(args.input)
        convert_openrag_to_selfrag(in_path, args.output)
        print(f"转换完成: {in_path} -> {args.output}")
        return

    # 默认处理两个常用文件
    defaults = [
        ("2wikimultihop-dev-br-top3.jsonl", "2wiki_for_selfrag.jsonl"),
        ("hotpot_dev_distractor_v1-br-top3.jsonl", "hotpot_for_selfrag.jsonl"),
    ]
    for src, dst in defaults:
        in_path = resolve_input_path(src)
        convert_openrag_to_selfrag(in_path, dst)
        print(f"转换完成: {in_path} -> {dst}")


if __name__ == "__main__":
    main()
