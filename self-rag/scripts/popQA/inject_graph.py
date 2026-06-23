import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from symbolic_graph import extract_mini_graph

print("🚀 正在为测试集注入神经符号图谱，请稍候...")

# 我们先用之前切出来的 10 条 tiny 数据做微型实验！
input_file = "eval_data_official/inputs/popqa_longtail_w_gs.jsonl"
output_file = "eval_data_official/inputs/popqa_longtail_w_gs_graph_all.jsonl"

enhanced_data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        
        # 提取上下文，如果没有检索文本，就从原问题中抽提图谱
        passages = []
        if 'ctxs' in data:
            passages = [ctx['text'] for ctx in data['ctxs'][:5]]
        else:
            passages = [data['question']]
            
        # 调用你的创新引擎生成图谱
        graph_str = extract_mini_graph(passages)
        
        # ⚔️ 核心黑科技：加了极其严厉的“短答案”约束！
        original_q = data['question']
        enhanced_q = f"{original_q}\n\n{graph_str}\n[System Hint]: Please carefully trace the logical paths in the Knowledge Graph Memory above to find the exact answer.\nWARNING: YOU MUST OUTPUT ONLY THE EXACT SHORT ANSWER (1-3 WORDS). DO NOT EXPLAIN YOUR REASONING. DO NOT WRITE FULL SENTENCES."
        data['question'] = enhanced_q
        
        enhanced_data.append(data)
        print(f"完成第 {i+1} 条数据的注入...")

with open(output_file, 'w', encoding='utf-8') as f:
    for data in enhanced_data:
        f.write(json.dumps(data) + '\n')

print(f"✅ 注入完成！带外挂的测试卷已生成：{output_file}")