import json
import argparse


def collect_answers(answer_obj):
    answers = []

    value = answer_obj.get("Value")
    if value:
        answers.append(value)

    for key in ["Aliases", "NormalizedAliases", "HumanAnswers"]:
        vals = answer_obj.get(key, [])
        if isinstance(vals, str):
            vals = [vals]
        for v in vals:
            if v and v not in answers:
                answers.append(v)

    return answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        raw = json.load(f)

    with open(args.output, "w", encoding="utf-8") as fout:
        for item in raw["Data"]:
            example = {
                "question": item["Question"],
                "answers": collect_answers(item["Answer"]),
                "question_id": item["QuestionId"]
            }

            fout.write(json.dumps(example, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
