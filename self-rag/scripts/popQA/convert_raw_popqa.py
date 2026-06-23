import csv
import json
import ast

input_tsv = "/openbayes/home/self_rag_reproduce/self-rag/popqa_raw/test.tsv"
output_jsonl = "/openbayes/home/self_rag_reproduce/self-rag/eval_data_official/inputs/popqa_base.jsonl"

with open(input_tsv, "r", encoding="utf-8") as fin, open(output_jsonl, "w", encoding="utf-8") as fout:
    reader = csv.DictReader(fin, delimiter="\t")
    for row in reader:
        possible_answers = row["possible_answers"]
        try:
            answers = ast.literal_eval(possible_answers)
        except Exception:
            answers = [possible_answers]

        item = {
            "id": int(row["id"]) if row["id"] else row["id"],
            "question": row["question"],
            "answers": answers,
            "prop": row.get("prop", ""),
            "s_wiki_title": row.get("s_wiki_title", ""),
            "pop": int(row["s_pop"]) if row.get("s_pop") else None
        }
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"saved to {output_jsonl}")
