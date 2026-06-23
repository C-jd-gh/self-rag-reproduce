import spacy

# 加载轻量级英文 NLP 引擎 (纯 CPU 运行，零显存占用)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

ENTITY_LABELS = {"PERSON", "ORG", "GPE", "DATE", "EVENT"}


def extract_entities(text):
    entities = []
    seen = set()
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ENTITY_LABELS:
            normalized = ent.text.strip()
            normalized_key = normalized.lower()
            if normalized_key and normalized_key not in seen:
                seen.add(normalized_key)
                entities.append(normalized)
    return entities

def extract_mini_graph(passages, max_edges=15, focus_entities=None):
    """
    核心创新模块：动态微型图谱提取
    输入：多篇检索到的文档片段 (List of strings)
    输出：符号化的知识图谱字符串 (String)
    """
    graph_triplets = []
    
    focus_set = None
    if focus_entities:
        focus_set = {ent.lower() for ent in focus_entities if ent}

    # 遍历每一篇文档
    for text in passages:
        # 只处理前 500 个字符，保证极速运行
        doc = nlp(text[:500]) 
        
        # 按照句子切分
        for sent in doc.sents:
            # 提取句子中的核心实体 (如人名、地名、组织、时间)
            entities = [ent.text.strip() for ent in sent.ents if ent.label_ in ENTITY_LABELS]
            
            # 符号逻辑假定：如果两个实体出现在同一个句子里，它们大概率有逻辑关联
            if len(entities) >= 2:
                # 取前两个最主要的实体构建三元组
                head = entities[0]
                tail = entities[1]
                if head != tail:
                    if focus_set is not None:
                        head_hit = head.lower() in focus_set
                        tail_hit = tail.lower() in focus_set
                        if not (head_hit or tail_hit):
                            continue
                    graph_triplets.append(f"({head}) --[relates_to]--> ({tail})")
    
    # 去重并限制最大边数，防止图谱过大把大模型绕晕
    unique_triplets = list(set(graph_triplets))[:max_edges]
    
    if not unique_triplets:
        return "[Knowledge Graph]: No distinct logical paths found."
    
    # 拼装成提示词友好的格式
    graph_string = "[Knowledge Graph Memory]:\n" + " | ".join(unique_triplets)
    return graph_string

# 沙盒测试代码 (只有直接运行本脚本时才会执行)
if __name__ == "__main__":
    test_docs = [
        "Tim Cook is the current CEO of Apple Inc. He was born in Alabama.",
        "Apple Inc. is headquartered in Cupertino, California, which was founded by Steve Jobs."
    ]
    print("===== 测试: 传统检索到的纯文本 =====")
    print(test_docs)
    print("\n===== 创新: 提取出的动态符号图谱 =====")
