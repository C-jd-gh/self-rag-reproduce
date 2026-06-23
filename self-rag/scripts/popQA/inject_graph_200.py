import json
import re
from collections import Counter
from symbolic_graph import extract_mini_graph, extract_entities

print("🚀 正在为 200 条中型测试集注入神经符号图谱，请稍候...")

MIN_Q_ENTS_FOR_GRAPH = 2
SINGLE_HOP_TEMPLATE_RE = re.compile(
    r"^(who|what|when|where)\s+(is|was|are|were)\s+.+\??$",
    re.IGNORECASE,
)
MULTIHOP_CUE_RE = re.compile(
    r"\b(same|both|either|between|or|and|whose|which|compared to)\b",
    re.IGNORECASE,
)


def should_inject_graph(question):
    q_entities = extract_entities(question)

    # 对典型一跳问法做保护，避免无效图谱干扰
    if SINGLE_HOP_TEMPLATE_RE.match(question.strip()) and len(q_entities) <= 1:
        return False, q_entities, "single_hop_template"

    if len(q_entities) >= MIN_Q_ENTS_FOR_GRAPH:
        return True, q_entities, "enough_entities"

    if MULTIHOP_CUE_RE.search(question) and len(q_entities) >= 1:
        return True, q_entities, "multihop_cue"

    return False, q_entities, "default_skip"

# 输入是刚才切出来的 200 条标准卷
input_file = "eval_data_official/hotpotqa_dev_200.jsonl"
output_file = "eval_data_official/hotpotqa_dev_200_graph.jsonl"

enhanced_data = []
inject_counter = 0
reason_counter = Counter()
with open(input_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        
        passages = []
        if 'ctxs' in data:
            passages = [ctx['text'] for ctx in data['ctxs'][:5]]
        else:
            passages = [data['question']]
            
        original_q = data['question']
        do_inject, q_entities, reason = should_inject_graph(original_q)
        reason_counter[reason] += 1

        if do_inject:
            graph_str = extract_mini_graph(passages, focus_entities=q_entities)
            enhanced_q = f"{original_q}\n\n{graph_str}\n[System Hint]: Please carefully trace the logical paths in the Knowledge Graph Memory above to find the exact answer.\nWARNING: YOU MUST OUTPUT ONLY THE EXACT SHORT ANSWER (1-3 WORDS). DO NOT EXPLAIN YOUR REASONING. DO NOT WRITE FULL SENTENCES."
            data['question'] = enhanced_q
            inject_counter += 1
        
        enhanced_data.append(data)
        
        if (i + 1) % 50 == 0:
            print(f"已完成 {i+1}/200 条数据的注入...")

with open(output_file, 'w', encoding='utf-8') as f:
    for data in enhanced_data:
        f.write(json.dumps(data) + '\n')

print(f"✅ 注入完成！200条外挂卷已生成：{output_file}")
print(f"📊 注入比例: {inject_counter}/{len(enhanced_data)} = {inject_counter/max(len(enhanced_data),1):.2%}")
print(f"📊 门控原因统计: {dict(reason_counter)}")