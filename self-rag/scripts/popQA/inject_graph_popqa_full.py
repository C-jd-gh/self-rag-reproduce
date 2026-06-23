import json
import re
from collections import Counter
from symbolic_graph import extract_mini_graph, extract_entities

print("🚀 正在为 PopQA 【全量】测试集注入神经符号图谱，请耐心等待...")

MIN_Q_ENTS_FOR_GRAPH = 2
SIMPLE_LOOKUP_PROPS = {
    "occupation", "country", "sport", "genre", "religion",
    "capital", "author", "director", "producer", "composer",
    "screenwriter", "father", "mother", "place of birth"
}
SINGLE_HOP_TEMPLATE_RE = re.compile(
    r"^what\s+is\s+.+?'s\s+(occupation|country|sport|genre|religion|capital|author|director|producer|composer|screenwriter|father|mother|place of birth)\??$",
    re.IGNORECASE,
)
MULTIHOP_CUE_RE = re.compile(
    r"\b(based on|derived from|in the country of|in the city of|capital of|whose)\b",
    re.IGNORECASE,
)


def should_inject_graph(question, prop=None):
    q_entities = extract_entities(question)
    lowered_prop = (prop or "").lower().strip()

    # 强规则：典型单跳模板直接跳过，保护单跳准确率
    if SINGLE_HOP_TEMPLATE_RE.match(question.strip()):
        return False, q_entities, "single_hop_template"

    # 强规则：属性型单跳问题且实体数<=1时不注入
    if lowered_prop in SIMPLE_LOOKUP_PROPS and len(q_entities) <= 1:
        return False, q_entities, "simple_lookup_prop"

    # 放行条件1：问题中实体较多，通常是复合关系查询
    if len(q_entities) >= MIN_Q_ENTS_FOR_GRAPH:
        return True, q_entities, "enough_entities"

    # 放行条件2：显式多跳提示词 + 至少1个实体
    if MULTIHOP_CUE_RE.search(question) and len(q_entities) >= 1:
        return True, q_entities, "multihop_cue"

    return False, q_entities, "default_skip"

# 这里直接读取完整的 PopQA 原版数据集
input_file = "eval_data_official/popqa_longtail_w_gs.jsonl"
output_file = "eval_data_official/popqa_longtail_w_gs_graph.jsonl"

enhanced_data = []
inject_counter = 0
reason_counter = Counter()
with open(input_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        
        passages = []
        # PopQA 的上下文结构可能与 HotpotQA 略有不同，安全提取
        if 'ctxs' in data:
            passages = [ctx['text'] for ctx in data['ctxs'][:5]]
        else:
            passages = [data['question']]
            
        original_q = data['question']
        do_inject, q_entities, reason = should_inject_graph(original_q, data.get("prop"))
        reason_counter[reason] += 1
        if do_inject:
            graph_str = extract_mini_graph(passages, focus_entities=q_entities)
            enhanced_q = f"{original_q}\n\n{graph_str}\n[System Hint]: Please carefully trace the logical paths in the Knowledge Graph Memory above to find the exact answer.\nWARNING: YOU MUST OUTPUT ONLY THE EXACT SHORT ANSWER (1-3 WORDS). DO NOT EXPLAIN YOUR REASONING. DO NOT WRITE FULL SENTENCES."
            data['question'] = enhanced_q
            inject_counter += 1
        
        enhanced_data.append(data)
        
        # 每处理 500 条打印一次进度
        if (i + 1) % 500 == 0:
            print(f"已完成 {i+1} 条 PopQA 数据的注入...")

with open(output_file, 'w', encoding='utf-8') as f:
    for data in enhanced_data:
        f.write(json.dumps(data) + '\n')

print(f"✅ 全量注入完成！PopQA 终极外挂卷已生成：{output_file}")
print(f"📊 注入比例: {inject_counter}/{len(enhanced_data)} = {inject_counter/max(len(enhanced_data),1):.2%}")
print(f"📊 门控原因统计: {dict(reason_counter)}")