"""
Functions for scoring the Pharasiot dialect evaluation study.

Every table_* function returns a list of dicts, one per row. The notebook
turns them into DataFrames; nothing here prints or formats.

Reads  outputs/   model answers, one .jsonl per task/condition/seed/model
       metadata/  prompts and answer keys, one .tsv per task/condition/seed
"""

import os
import json
import csv
from collections import Counter

from scipy.stats import binomtest


# --- settings -------------------------------------------------------------
# Everything about the study lives here; nothing below should need editing.

OUTPUTS_FOLDER = "../outputs"
METADATA_FOLDER = "../metadata"

MODELS = ["krikri", "meltemi", "llama"]
SEEDS = [42, 7]
LETTERS = ["α", "β", "γ", "δ"]
LEVELS = ["word", "sentence", "paragraph"]

# these two reuse the same labels across items, so they get a majority
# baseline and macro-F1; the others have different options every item
CLASSIFICATION_TASKS = ["sentiment", "topic"]

# models with a working Greek control, used for the pooled McNemar
GREEK_ADAPTED_MODELS = ["krikri", "meltemi"]

TASKS = [
    "lexical_translation",
    "sentiment",
    "topic",
    "nli",
    "factual",
    "context_tracking",
    "discourse",
]

# task: (items, options, conditions, linguistic level)
TASK_INFO = {
    "lexical_translation": (40, 4, ["pharasiot"], "word"),
    "sentiment": (44, 3, ["pharasiot", "greek"], "sentence"),
    "topic": (40, 4, ["pharasiot", "greek"], "sentence"),
    "nli": (40, 4, ["pharasiot", "greek"], "sentence"),
    "factual": (15, 4, ["pharasiot", "greek"], "paragraph"),
    "context_tracking": (15, 4, ["pharasiot", "greek"], "paragraph"),
    "discourse": (15, 4, ["pharasiot", "greek"], "paragraph"),
}


def n_items(task):
    """Number of items in the task."""
    return TASK_INFO[task][0]


def n_options(task):
    """Number of answer options the task shows."""
    return TASK_INFO[task][1]


def conditions(task):
    """Conditions the task was run in."""
    return TASK_INFO[task][2]


def level(task):
    """Linguistic level: word, sentence or paragraph."""
    return TASK_INFO[task][3]


def has_greek(task):
    """True if the task has a Standard Greek control."""
    return "greek" in conditions(task)


def tasks_with_greek():
    """Tasks that can be compared across conditions."""
    return [task for task in TASKS if has_greek(task)]


# --- loading --------------------------------------------------------------

def load_results(task, condition, seed, model):
    """One result file as  item_id -> (chosen letter, correct letter)."""
    name = "{}_{}_seed{}_{}.jsonl".format(task, condition, seed, model)
    answers = {}

    # one JSON object per line
    with open(os.path.join(OUTPUTS_FOLDER, name), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # skip blank lines at the end of the file
            if not line:
                continue
            row = json.loads(line)
            answers[row["item_id"]] = (row["parsed_letter"],
                                       row["correct_answer_position"])

    return answers


def load_metadata(task, condition, seed):
    """One metadata file as  item_id -> row."""
    name = "{}_{}_seed{}.tsv".format(task, condition, seed)
    items = {}

    with open(os.path.join(METADATA_FOLDER, name), encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            items[row["item_id"]] = row

    return items


def load_label_pairs(task, condition, seed, model):
    """Letters turned into label names: (gold labels, predicted labels)."""
    metadata = load_metadata(task, condition, seed)
    answers = load_results(task, condition, seed, model)

    gold = []
    predicted = []

    # sorted so the two lists line up the same way every run
    for item_id in sorted(answers):
        chosen, _ = answers[item_id]
        row = metadata[item_id]
        gold.append(row["correct_answer_text"])
        # the metadata says which label sits behind the chosen letter
        predicted.append(row.get("option_" + str(chosen), "UNPARSED"))

    return gold, predicted


# --- metrics --------------------------------------------------------------

def count_correct(answers):
    """How many items the model got right."""
    correct = 0

    # each value is (letter chosen, letter that was correct)
    for chosen, key in answers.values():
        if chosen == key:
            correct += 1

    return correct


def pooled_score(task, condition, model):
    """(correct, total) over both orderings."""
    correct = 0
    total = 0

    # add up the two option orderings
    for seed in SEEDS:
        answers = load_results(task, condition, seed, model)
        correct += count_correct(answers)
        total += len(answers)

    return correct, total


def chance_level(task):
    """Score expected from guessing at random."""
    return 1.0 / n_options(task)


def majority_baseline(task):
    """Score from always answering the task's most common label."""
    # any file will do, the gold labels are the same in all of them
    metadata = load_metadata(task, conditions(task)[0], SEEDS[0])

    labels = []
    for row in metadata.values():
        labels.append(row["correct_answer_text"])

    # the biggest class, as a share of all items
    counts = Counter(labels)
    return max(counts.values()) / len(labels)


def gold_label_counts(task):
    """How many items carry each label."""
    metadata = load_metadata(task, conditions(task)[0], SEEDS[0])

    labels = []
    for row in metadata.values():
        labels.append(row["correct_answer_text"])

    return dict(Counter(labels))


def precision_recall_f1(gold, predicted, label):
    """Precision, recall and F1 for one label."""
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # sort every item by what the model said and what was true
    for truth, guess in zip(gold, predicted):
        if guess == label and truth == label:
            true_positives += 1
        elif guess == label:
            false_positives += 1
        elif truth == label:
            false_negatives += 1

    # a label the model never used has no precision; one absent from the
    # gold has no recall. Both are reported as 0.
    if true_positives + false_positives == 0:
        precision = 0.0
    else:
        precision = true_positives / (true_positives + false_positives)

    if true_positives + false_negatives == 0:
        recall = 0.0
    else:
        recall = true_positives / (true_positives + false_negatives)

    # harmonic mean, so a good score on one cannot hide a bad one on the other
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def compare_conditions(task, model, models=None):
    """Sort the items into (both, greek only, pharasiot only, neither).

    Summed over both orderings, and over several models if given.
    """
    # one model unless a list was given
    if models is None:
        models = [model]

    both = 0
    greek_only = 0
    pharasiot_only = 0
    neither = 0

    for one_model in models:
        for seed in SEEDS:
            pharasiot = load_results(task, "pharasiot", seed, one_model)
            greek = load_results(task, "greek", seed, one_model)

            # the same item in the two conditions
            for item_id, (pharasiot_chosen, pharasiot_key) in pharasiot.items():
                greek_chosen, greek_key = greek[item_id]
                pharasiot_right = pharasiot_chosen == pharasiot_key
                greek_right = greek_chosen == greek_key

                # most specific case first, or the elifs below would catch it
                if pharasiot_right and greek_right:
                    both += 1
                elif greek_right:
                    greek_only += 1
                elif pharasiot_right:
                    pharasiot_only += 1
                else:
                    neither += 1

    return both, greek_only, pharasiot_only, neither


def mcnemar_p_value(greek_only, pharasiot_only):
    """Two-sided p for the discordant items splitting this unevenly.

    Under no dialect effect each disagreement is a coin flip.
    """
    disagreements = greek_only + pharasiot_only

    # nothing disagreed, so there is nothing to test
    if disagreements == 0:
        return 1.0

    # two-sided, so either side can be passed in
    smaller_side = min(greek_only, pharasiot_only)
    return binomtest(smaller_side, disagreements, 0.5,
                     alternative="two-sided").pvalue


def mcnemar_verdict(greek_only, pharasiot_only, p_value):
    """Significant, not significant, or too few disagreements to tell.

    Below six disagreements even a unanimous split cannot reach p < 0.05,
    so a null result there means the test had no power, not that the
    conditions are alike.
    """
    if greek_only + pharasiot_only < 6:
        return "underpowered"
    elif p_value < 0.05:
        return "significant"
    else:
        return "not significant"


# --- tables ---------------------------------------------------------------

def table_accuracy():
    """Accuracy per task, condition, ordering and model, against the baselines."""
    rows = []

    for task in TASKS:
        # only the classification tasks have a label to default to
        if task in CLASSIFICATION_TASKS:
            majority = majority_baseline(task)
        else:
            majority = None

        for condition in conditions(task):
            for model in MODELS:
                for seed in SEEDS:
                    answers = load_results(task, condition, seed, model)
                    correct = count_correct(answers)
                    total = len(answers)

                    rows.append({
                        "task": task,
                        "condition": condition,
                        "seed": seed,
                        "model": model,
                        "n_items": total,
                        "n_correct": correct,
                        "accuracy": correct / total,
                        "chance": chance_level(task),
                        "majority": majority,
                        "beats_chance": correct / total > chance_level(task),
                        "beats_majority": (majority is not None
                                           and correct / total > majority),
                    })

    return rows


def table_macro_f1():
    """Macro-F1 for the classification tasks."""
    rows = []

    for task in CLASSIFICATION_TASKS:
        for condition in conditions(task):
            for model in MODELS:
                for seed in SEEDS:
                    gold, predicted = load_label_pairs(task, condition, seed, model)

                    # one F1 per label, then average them with equal weight
                    scores = []
                    for label in sorted(set(gold)):
                        _, _, f1 = precision_recall_f1(gold, predicted, label)
                        scores.append(f1)

                    rows.append({
                        "task": task,
                        "condition": condition,
                        "seed": seed,
                        "model": model,
                        "macro_f1": sum(scores) / len(scores),
                    })

    return rows


def table_per_class():
    """Precision, recall and F1 for every label of the classification tasks."""
    rows = []

    for task in CLASSIFICATION_TASKS:
        for condition in conditions(task):
            for model in MODELS:
                for seed in SEEDS:
                    gold, predicted = load_label_pairs(task, condition, seed, model)

                    # labels taken from the gold, so one the model never used
                    # still gets a row
                    for label in sorted(set(gold)):
                        precision, recall, f1 = precision_recall_f1(
                            gold, predicted, label)

                        rows.append({
                            "task": task,
                            "condition": condition,
                            "seed": seed,
                            "model": model,
                            "label": label,
                            "n_gold": gold.count(label),
                            "n_predicted": predicted.count(label),
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                        })

    return rows


def table_position_bias():
    """Accuracy with the model's favourite letter removed from the key.

    Scores only the items where that letter was wrong, so points won by
    always pressing it do not count. Both orderings pooled.
    """
    rows = []

    for task in TASKS:
        letters = LETTERS[:n_options(task)]

        for condition in conditions(task):
            for model in MODELS:
                # start every letter at zero so an unused one still shows up
                counts = Counter()
                for letter in letters:
                    counts[letter] = 0

                filtered_total = 0
                filtered_correct = 0
                favourite = None

                # first pass: how often was each letter chosen
                for seed in SEEDS:
                    for chosen, _ in load_results(task, condition, seed, model).values():
                        if chosen in counts:
                            counts[chosen] += 1

                # most_common gives [(letter, count)], so [0][0] is the letter
                favourite = counts.most_common(1)[0][0]

                # second pass: score the items where the favourite was wrong
                for seed in SEEDS:
                    for chosen, key in load_results(task, condition, seed, model).values():
                        if key != favourite:
                            filtered_total += 1
                            if chosen == key:
                                filtered_correct += 1

                row = {
                    "task": task,
                    "condition": condition,
                    "model": model,
                    "favourite": favourite,
                    "times_chosen": counts[favourite],
                    "n_responses": sum(counts.values()),
                    "filtered_correct": filtered_correct,
                    "filtered_total": filtered_total,
                    "filtered_accuracy": filtered_correct / filtered_total,
                    "chance": chance_level(task),
                }
                # one column per letter, for reading the spread at a glance
                for letter in letters:
                    row["chose_" + letter] = counts[letter]

                rows.append(row)

    return rows


def table_binomial():
    """Is each score better than guessing? Both orderings pooled."""
    rows = []

    for task in TASKS:
        baseline = chance_level(task)

        for condition in conditions(task):
            for model in MODELS:
                correct, total = pooled_score(task, condition, model)
                # one-sided: we only care about scoring above chance
                p_value = binomtest(correct, total, baseline,
                                    alternative="greater").pvalue

                rows.append({
                    "task": task,
                    "condition": condition,
                    "model": model,
                    "n_correct": correct,
                    "n_items": total,
                    "accuracy": correct / total,
                    "chance": baseline,
                    "p_value": p_value,
                    "beats_chance": p_value < 0.05,
                })

    return rows


def table_mcnemar():
    """Is the dialect gap real? One row per model, plus the two pooled."""
    rows = []

    for task in tasks_with_greek():
        for model in MODELS:
            _, greek_only, pharasiot_only, _ = compare_conditions(task, model)
            p_value = mcnemar_p_value(greek_only, pharasiot_only)

            rows.append({
                "task": task,
                "model": model,
                "greek_only": greek_only,
                "pharasiot_only": pharasiot_only,
                "discordant": greek_only + pharasiot_only,
                "p_value": p_value,
                "verdict": mcnemar_verdict(greek_only, pharasiot_only, p_value),
            })

        # and once more with Llama left out: its answer distributions show it
        # is not attempting several tasks, so its results are not interpretable
        _, greek_only, pharasiot_only, _ = compare_conditions(
            task, None, models=GREEK_ADAPTED_MODELS)
        p_value = mcnemar_p_value(greek_only, pharasiot_only)

        rows.append({
            "task": task,
            "model": "krikri+meltemi",
            "greek_only": greek_only,
            "pharasiot_only": pharasiot_only,
            "discordant": greek_only + pharasiot_only,
            "p_value": p_value,
            "verdict": mcnemar_verdict(greek_only, pharasiot_only, p_value),
        })

    return rows


def table_dialect_cost():
    """Size of the gap, and what happened to each item in the two conditions."""
    rows = []

    for task in tasks_with_greek():
        for model in MODELS:
            greek_correct, total = pooled_score(task, "greek", model)
            pharasiot_correct, _ = pooled_score(task, "pharasiot", model)
            both, greek_only, pharasiot_only, neither = compare_conditions(task, model)

            greek_accuracy = greek_correct / total
            pharasiot_accuracy = pharasiot_correct / total

            # share of its own Greek score the model keeps in Pharasiot
            if greek_correct == 0:
                retention = 0.0
            else:
                retention = pharasiot_correct / greek_correct

            rows.append({
                "task": task,
                "model": model,
                "n_items": total,
                "greek_correct": greek_correct,
                "pharasiot_correct": pharasiot_correct,
                "greek_accuracy": greek_accuracy,
                "pharasiot_accuracy": pharasiot_accuracy,
                "gap_items": greek_correct - pharasiot_correct,
                "gap_points": (greek_accuracy - pharasiot_accuracy) * 100,
                "retention": retention,
                "both": both,
                "greek_only": greek_only,
                "pharasiot_only": pharasiot_only,
                "neither": neither,
            })

    return rows


def table_by_level():
    """Totals for word, sentence and paragraph level."""
    rows = []

    for this_level in LEVELS:
        # the tasks that sit at this level
        tasks_here = [task for task in TASKS if level(task) == this_level]

        for model in MODELS:
            pharasiot_correct = 0
            pharasiot_total = 0
            greek_correct = 0
            greek_total = 0

            for task in tasks_here:
                correct, total = pooled_score(task, "pharasiot", model)
                pharasiot_correct += correct
                pharasiot_total += total

                if has_greek(task):
                    correct, total = pooled_score(task, "greek", model)
                    greek_correct += correct
                    greek_total += total

            row = {
                "level": this_level,
                "model": model,
                "pharasiot_correct": pharasiot_correct,
                "pharasiot_total": pharasiot_total,
                "pharasiot_accuracy": pharasiot_correct / pharasiot_total,
            }

            # the word level has no Greek control, so there is no gap to report
            if greek_total == 0:
                row["greek_accuracy"] = None
                row["gap_points"] = None
            else:
                row["greek_correct"] = greek_correct
                row["greek_total"] = greek_total
                row["greek_accuracy"] = greek_correct / greek_total
                row["gap_points"] = (greek_correct / greek_total
                                     - pharasiot_correct / pharasiot_total) * 100

            rows.append(row)

    return rows


def table_overall():
    """One row per model, every task pooled.

    Retention counts only the tasks with a Greek control, so lexical
    translation does not distort it.
    """
    rows = []

    for model in MODELS:
        pharasiot_correct = 0
        pharasiot_total = 0
        greek_correct = 0
        greek_total = 0
        paired_pharasiot_correct = 0

        for task in TASKS:
            correct, total = pooled_score(task, "pharasiot", model)
            pharasiot_correct += correct
            pharasiot_total += total

            # kept separately so retention compares like with like
            if has_greek(task):
                paired_pharasiot_correct += correct
                correct, total = pooled_score(task, "greek", model)
                greek_correct += correct
                greek_total += total

        rows.append({
            "model": model,
            "pharasiot_correct": pharasiot_correct,
            "pharasiot_total": pharasiot_total,
            "pharasiot_accuracy": pharasiot_correct / pharasiot_total,
            "greek_correct": greek_correct,
            "greek_total": greek_total,
            "greek_accuracy": greek_correct / greek_total,
            "retention": paired_pharasiot_correct / greek_correct,
        })

    return rows


# --- report ---------------------------------------------------------------

def _banner(text, character="="):
    """Heading line, padded to 78 characters."""
    return [character * 78, text, character * 78]


def _as_percent(frame, columns):
    """Copy of the frame with the given columns shown as percentages."""
    frame = frame.copy()

    for column in columns:
        if column in frame:
            frame[column] = frame[column].map(
                lambda value: "" if value is None or value != value
                else "{:.1f}%".format(value * 100))

    return frame


def format_report(tables):
    """Every table as readable text, for saving alongside the .tsv files."""
    lines = []

    # step 1
    lines += _banner("STEP 1 - DESCRIPTIVE METRICS")
    lines.append("")

    accuracy = tables["accuracy"]
    for task in TASKS:
        rows = accuracy[accuracy.task == task]

        lines += _banner("TASK: " + task, "-")
        lines.append("items: {}   options: {}   chance: {:.1f}%".format(
            n_items(task), n_options(task), chance_level(task) * 100))

        # only the classification tasks have a majority baseline
        if task in CLASSIFICATION_TASKS:
            lines.append("majority-class baseline: {:.1f}%".format(
                majority_baseline(task) * 100))
            lines.append("gold label counts: " + str(gold_label_counts(task)))

        lines.append("")
        shown = _as_percent(rows, ["accuracy", "chance", "majority"])
        lines.append(shown[["model", "condition", "seed", "n_correct",
                            "n_items", "accuracy", "beats_chance"]
                           ].to_string(index=False))
        lines.append("")

        # macro-F1, classification tasks only
        if task in CLASSIFICATION_TASKS:
            lines.append("  MACRO-F1 (equal weight to every label)")
            lines.append("")
            macro = tables["macro_f1"]
            lines.append(macro[macro.task == task][
                ["model", "condition", "seed", "macro_f1", "accuracy", "gap"]
            ].to_string(index=False, float_format="%.3f"))
            lines.append("")

            lines.append("  PER LABEL")
            lines.append("")
            per_class = tables["per_class"]
            lines.append(per_class[per_class.task == task][
                ["model", "condition", "seed", "label", "n_gold",
                 "n_predicted", "precision", "recall", "f1"]
            ].to_string(index=False, float_format="%.2f"))
            lines.append("")

        lines.append("  POSITION-BIAS CHECK (both orderings pooled)")
        lines.append("  scored only on items where the favourite letter was wrong")
        lines.append("")
        bias = tables["position_bias"]
        letter_columns = ["chose_" + letter for letter in LETTERS[:n_options(task)]]
        shown = _as_percent(bias[bias.task == task],
                            ["filtered_accuracy", "chance"])
        shown[letter_columns] = shown[letter_columns].astype(int)
        lines.append(shown[["model", "condition"] + letter_columns
                           + ["favourite", "times_chosen", "filtered_correct",
                              "filtered_total", "filtered_accuracy", "survives"]
                           ].to_string(index=False))
        lines.append("")

    # step 2
    lines += _banner("STEP 2 - SIGNIFICANCE TESTS")
    lines.append("")
    lines.append("BINOMIAL - each score against chance, both orderings pooled")
    lines.append("")
    shown = _as_percent(tables["binomial"], ["accuracy", "chance"])
    lines.append(shown[["task", "condition", "model", "n_correct", "n_items",
                        "accuracy", "p_value", "beats_chance"]
                       ].to_string(index=False, float_format="%.4f"))
    lines.append("")

    lines.append("McNEMAR - Standard Greek against Pharasiot on the same items")
    lines.append("")
    lines.append("  greek_only    right in Greek, wrong in Pharasiot")
    lines.append("  pharasiot_only  the reverse")
    lines.append("  underpowered  fewer than six disagreements, so no split")
    lines.append("                could have reached significance")
    lines.append("")
    lines.append(tables["mcnemar"].to_string(index=False, float_format="%.4f"))
    lines.append("")
    lines.append("The krikri+meltemi rows leave Llama out: its answer")
    lines.append("distributions show it is not attempting several of the tasks.")
    lines.append("")
    lines.append("Many tests are run here. At the 0.05 level about one in twenty")
    lines.append("comes out significant by luck, so a couple of borderline")
    lines.append("results are expected.")
    lines.append("")

    # step 3
    lines += _banner("STEP 3 - DIALECT COST")
    lines.append("")
    lines.append("PER TASK - both orderings pooled")
    lines.append("")
    shown = _as_percent(tables["dialect_cost"],
                        ["greek_accuracy", "pharasiot_accuracy", "retention"])
    lines.append(shown[["task", "model", "greek_correct", "pharasiot_correct",
                        "n_items", "gap_items", "gap_points", "retention"]
                       ].to_string(index=False, float_format="%.1f"))
    lines.append("")

    lines.append("ITEM PROFILE - what happened to each item in the two conditions")
    lines.append("  neither = failed in Greek too, so not the dialect's doing")
    lines.append("")
    lines.append(tables["dialect_cost"][
        ["task", "model", "both", "greek_only", "pharasiot_only", "neither"]
    ].to_string(index=False))
    lines.append("")

    lines.append("BY LINGUISTIC LEVEL")
    lines.append("  word = lexical translation")
    lines.append("  sentence = sentiment, topic, NLI")
    lines.append("  paragraph = factual, context tracking, discourse")
    lines.append("")
    shown = _as_percent(tables["by_level"],
                        ["pharasiot_accuracy", "greek_accuracy"])
    lines.append(shown[["level", "model", "pharasiot_correct", "pharasiot_total",
                        "pharasiot_accuracy", "greek_accuracy", "gap_points"]
                       ].to_string(index=False, na_rep="", float_format="%.1f"))
    lines.append("")

    lines.append("OVERALL - every task pooled")
    lines.append("  retention counts only the six tasks with a Greek control")
    lines.append("")
    shown = _as_percent(tables["overall"],
                        ["pharasiot_accuracy", "greek_accuracy", "retention"])
    lines.append(shown[["model", "pharasiot_correct", "pharasiot_total",
                        "pharasiot_accuracy", "greek_correct", "greek_total",
                        "greek_accuracy", "retention"]].to_string(index=False))
    lines.append("")

    return "\n".join(lines)
