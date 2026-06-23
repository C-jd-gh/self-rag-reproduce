import json
import re
import string
from collections import Counter
import sys

def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def main(pred_file, gold_file):
    print("⏳ 正在扒开原作者的 JSON 外壳，提取答题卡...")
    
    # 1. 专门针对原作者的字典格式进行解析，提取 preds 数组
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
        if isinstance(pred_data, dict) and 'preds' in pred_data:
            preds = pred_data['preds']
        else:
            preds = pred_data
            
    # 2. 读取我们准备好的原试卷
    with open(gold_file, 'r', encoding='utf-8') as f:
        golds = [json.loads(line) for line in f if line.strip()]
        
    print(f"✅ 提取成功！预测结果: {len(preds)} 条，标准答案: {len(golds)} 条")
    
    em_total = 0
    f1_total = 0
    valid_count = 0
    
    for p_data, g_data in zip(preds, golds):
        # 兼容 p_data 是纯字符串还是字典的情况
        if isinstance(p_data, str):
            raw_pred = p_data
        elif isinstance(p_data, dict):
            raw_pred = p_data.get('pred', p_data.get('preds', p_data.get('output', '')))
        else:
            raw_pred = str(p_data)
            
        if isinstance(raw_pred, list):
            raw_pred = str(raw_pred[0]) if len(raw_pred) > 0 else ""
            
        # 清理诸如 [Utility:5] 的反思令牌
        pred_clean = re.sub(r'\[.*?\]', '', raw_pred).strip()
        
        # 提取原试卷答案
        answers = g_data.get('answers', g_data.get('answer', []))
        if isinstance(answers, str):
            answers = [answers]
            
        if not answers:
            continue
            
        best_em = max([exact_match_score(pred_clean, ans) for ans in answers])
        best_f1 = max([f1_score(pred_clean, ans) for ans in answers])
        
        em_total += best_em
        f1_total += best_f1
        valid_count += 1
        
    print(f"\n📊 评测完成！共有效评测 {valid_count} 条数据")
    if valid_count > 0:
        print(f"🎯 Exact Match (EM): {100 * em_total / valid_count:.2f}")
        print(f"📈 F1 Score: {100 * f1_total / valid_count:.2f}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("用法: python evaluate_em_f1.py <答题卡.json> <原试卷.jsonl>")
    else:
        main(sys.argv[1], sys.argv[2])