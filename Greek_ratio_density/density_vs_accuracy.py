"""
Does lexical density predict whether the models got an item right?

Takes the density of each of the 44 sentences and compares it with how often
the models answered that sentence's items correctly, in the Pharasiot
condition, on sentiment, topic and NLI.

Reads   lexical_density_sentences.tsv
        gold_files/                    to link sentence ids to item ids
        outputs/                       the model answers
Writes  density_vs_accuracy.tsv

Run from the folder holding those:  python density_vs_accuracy.py
"""

import os
import csv
import json
from collections import defaultdict


DENSITY_FILE = "lexical_density_sentences.tsv"
GOLD_FOLDER = "../gold_files"
OUTPUTS_FOLDER = "../outputs"
OUTPUT_FILE = "density_vs_accuracy.tsv"

MODELS = ["krikri", "meltemi", "llama"]
SEEDS = [42, 7]

# task: (gold file, the column holding the sentence id)
TASKS = {
    "sentiment": ("gold_sentiment_classification.tsv", "sent_id"),
    "topic": ("gold_topic_classification.tsv", "sent_id"),
    "nli": ("gold_NLI.tsv", "context_sent_id"),
}

# the bands the sentences get grouped into
BANDS = [
    ("0.00 (nothing recoverable)", 0.00, 0.001),
    ("0.01 - 0.39 (mostly opaque)", 0.001, 0.40),
    ("0.40 - 0.69 (mixed)", 0.40, 0.70),
    ("0.70 - 1.00 (mostly Greek)", 0.70, 1.001),
]


def clean(text):
    """Strip whitespace and stray quotes."""
    return (text or "").strip().strip('"').strip()


def read_density():
    """Read the per-sentence density into  sent_id -> row."""
    density = {}

    with open(DENSITY_FILE, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["density"] == "":
                continue
            density[row["sent_id"]] = {
                "density": float(row["density"]),
                "n_counted": int(row["n_counted"]),
            }

    return density


def read_item_to_sentence(task):
    """Which sentence each item of a task belongs to:  item_id -> sent_id."""
    filename, id_column = TASKS[task]
    path = os.path.join(GOLD_FOLDER, filename)

    prefix = {"sentiment": "sen", "topic": "top", "nli": "nli"}[task]
    mapping = {}

    with open(path, encoding="utf-8-sig") as f:
        for number, row in enumerate(csv.DictReader(f, delimiter="\t"), start=1):
            item_id = "{}_{:02d}".format(prefix, number)
            mapping[item_id] = clean(row[id_column])

    return mapping


def read_correctness(task):
    """How often each item was answered correctly:  item_id -> (correct, tries)."""
    correct = defaultdict(int)
    tries = defaultdict(int)

    for model in MODELS:
        for seed in SEEDS:
            name = "{}_pharasiot_seed{}_{}.jsonl".format(task, seed, model)
            path = os.path.join(OUTPUTS_FOLDER, name)

            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)
                item_id = row["item_id"]
                tries[item_id] += 1
                if row["parsed_letter"] == row["correct_answer_position"]:
                    correct[item_id] += 1

    return {item: (correct[item], tries[item]) for item in tries}


def main():
    density = read_density()
    print("sentences with a density score: {}".format(len(density)))

    rows = []

    for task in TASKS:
        item_to_sentence = read_item_to_sentence(task)
        correctness = read_correctness(task)

        for item_id in sorted(correctness):
            sent_id = item_to_sentence.get(item_id)

            if sent_id not in density:
                print("  no density for {} ({})".format(item_id, sent_id))
                continue

            n_correct, n_tries = correctness[item_id]

            rows.append({
                "task": task,
                "item_id": item_id,
                "sent_id": sent_id,
                "density": density[sent_id]["density"],
                "n_counted": density[sent_id]["n_counted"],
                "n_correct": n_correct,
                "n_tries": n_tries,
                "accuracy": round(n_correct / n_tries, 4),
            })

    columns = ["task", "item_id", "sent_id", "density", "n_counted",
               "n_correct", "n_tries", "accuracy"]

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("wrote {}  ({} items)".format(OUTPUT_FILE, len(rows)))

    # --- accuracy by density band -----------------------------------------
    print()
    print("=" * 66)
    print("ACCURACY BY DENSITY BAND")
    print("=" * 66)
    print()
    print("  density = share of content words recoverable from Standard Greek")
    print("  accuracy = how often the models answered correctly, Pharasiot only")
    print()
    print("  {:30}{:>8}{:>10}{:>12}".format("band", "items", "answers", "accuracy"))
    print("  " + "-" * 60)

    for label, low, high in BANDS:
        picked = [r for r in rows if low <= r["density"] < high]

        if not picked:
            continue

        n_correct = sum(r["n_correct"] for r in picked)
        n_tries = sum(r["n_tries"] for r in picked)

        print("  {:30}{:>8}{:>10}{:>11.1f}%".format(
            label, len(picked), n_tries, n_correct / n_tries * 100))

    print()
    n_correct = sum(r["n_correct"] for r in rows)
    n_tries = sum(r["n_tries"] for r in rows)
    print("  {:30}{:>8}{:>10}{:>11.1f}%".format(
        "all items", len(rows), n_tries, n_correct / n_tries * 100))

    # --- the same, per task -----------------------------------------------
    print()
    print("=" * 66)
    print("THE SAME, TASK BY TASK")
    print("=" * 66)

    for task in TASKS:
        print()
        print("  " + task.upper())
        print("  {:30}{:>8}{:>10}{:>12}".format("band", "items", "answers", "accuracy"))
        print("  " + "-" * 60)

        for label, low, high in BANDS:
            picked = [r for r in rows
                      if r["task"] == task and low <= r["density"] < high]

            if not picked:
                continue

            n_correct = sum(r["n_correct"] for r in picked)
            n_tries = sum(r["n_tries"] for r in picked)

            print("  {:30}{:>8}{:>10}{:>11.1f}%".format(
                label, len(picked), n_tries, n_correct / n_tries * 100))

    # --- length check -----------------------------------------------------
    # the fully transparent sentences are all very short, so high density and
    # short length are mixed together. This repeats the comparison using only
    # sentences with at least five counted words.
    print()
    print("=" * 66)
    print("LONGER SENTENCES ONLY (at least 5 content words counted)")
    print("=" * 66)
    print()
    print("  If the pattern holds here too, sentence length is not the cause.")
    print()
    print("  {:30}{:>8}{:>10}{:>12}".format("band", "items", "answers", "accuracy"))
    print("  " + "-" * 60)

    longer = [r for r in rows if r["n_counted"] >= 5]

    for label, low, high in BANDS:
        picked = [r for r in longer if low <= r["density"] < high]

        if not picked:
            continue

        n_correct = sum(r["n_correct"] for r in picked)
        n_tries = sum(r["n_tries"] for r in picked)

        print("  {:30}{:>8}{:>10}{:>11.1f}%".format(
            label, len(picked), n_tries, n_correct / n_tries * 100))

    print()
    print("  {} of {} items come from a sentence with 5+ counted words".format(
        len(longer), len(rows)))

    # --- the extremes -----------------------------------------------------
    print()
    print("=" * 66)
    print("THE EXTREMES")
    print("=" * 66)

    for label, picked in [
        ("NOTHING RECOVERABLE (density 0.00)",
         [r for r in rows if r["density"] == 0.0]),
        ("EVERYTHING RECOVERABLE (density 1.00)",
         [r for r in rows if r["density"] == 1.0]),
    ]:
        print()
        print("  " + label)

        by_sentence = defaultdict(lambda: [0, 0, 0])
        for row in picked:
            entry = by_sentence[row["sent_id"]]
            entry[0] += row["n_correct"]
            entry[1] += row["n_tries"]
            entry[2] = row["n_counted"]

        for sent_id in sorted(by_sentence):
            n_correct, n_tries, n_counted = by_sentence[sent_id]
            print("    {:16} {:>3}/{:<3} = {:>5.1f}%   ({} content words)".format(
                sent_id, n_correct, n_tries, n_correct / n_tries * 100, n_counted))

        n_correct = sum(r["n_correct"] for r in picked)
        n_tries = sum(r["n_tries"] for r in picked)
        print("    {:16} {:>3}/{:<3} = {:>5.1f}%   OVERALL".format(
            "", n_correct, n_tries, n_correct / n_tries * 100))


main()
