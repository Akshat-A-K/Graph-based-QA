"""show_comparisons.py
Print a sample of comparison-style HotpotQA questions.

Usage:
    python show_comparisons.py --num 20
"""
import os
import json
import argparse
import random


def parse_args():
    p = argparse.ArgumentParser(description="Show comparison questions from HotpotQA")
    p.add_argument("--dataset", default="hotpot_train_v1.1.json",
                   help="Path to HotpotQA JSON (relative to this script if not absolute)")
    p.add_argument("--num", type=int, default=20, help="How many comparison questions to show")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument("--show-context", action="store_true", help="Also print the first context title for each question")
    return p.parse_args()


def load_dataset(path):
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    data = load_dataset(args.dataset)
    comps = [item for item in data if item.get("type") == "comparison"]
    if not comps:
        print("No comparison questions found in dataset.")
        return

    random.seed(args.seed)
    random.shuffle(comps)
    take = comps[: args.num]

    for i, item in enumerate(take, start=1):
        qid = item.get("_id", "N/A")
        q = item.get("question", "")
        gold = item.get("answer", "")
        level = item.get("level", "")
        print(f"{i:03d}. id={qid}  [{level}]\n  Q: {q}\n  Gold: {gold}")
        # if args.show_context and item.get("context"):
        #     first_title = item["context"][0][0] if item["context"] else ""
        #     print(f"  Context title: {first_title}")
        print("-" * 60)


if __name__ == "__main__":
    main()
