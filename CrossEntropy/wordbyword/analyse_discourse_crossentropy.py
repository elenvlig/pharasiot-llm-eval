"""
Every number in the "Cross-entropy and lexical density" and "Word-level loss"
paragraphs of the discourse continuation section.

It prints each figure with a label, so each one can be checked against the text.
Nothing is written; it only reads.

Reads
    word_level_final/words_{krikri,meltemi,llama}.jsonl
        the word-level cross-entropy run (the original run with the 17 re-run
        sequences merged in), one row per word
    outputs/discourse_{condition}_seed{seed}_{model}.jsonl
        the 12 answer files, for the lost / kept / other grouping
    discourse_final/gold_discourse_continuation.tsv
        to name each item by tale and sentence numbers, and to link disc_NN to para_NN
    lexical_density_discourse_options.tsv
        the density of each context and of each option

Run from the folder holding those four:

    python analyse_discourse_crossentropy.py

Definitions used, the same as in the text:

    option loss      the loss summed over the option's words divided by the number
                     of pieces those words take, with the three context sentences
                     read before it. Punctuation is not part of any word, so it is
                     not counted here; this is why the figure differs a little from
                     `loss_scored` in the seqs files.
    loss gap         Pharasiot minus Standard Greek, for the correct option
    lost             correct in Standard Greek and wrong in Pharasiot, at one
                     ordering or both
    kept             correct in both varieties at both orderings
    other            everything else: wrong in Standard Greek at one ordering or
                     both, without being lost to the dialect
    word loss        the loss summed over the pieces (tokens) of one word
    recoverable      a word native readers judged recoverable from its form alone
                     ("shared" = yes); "no" is not recoverable; "skip" is a function
                     word, which the density does not count
    repeated word    a word of an option whose form also occurs in the three context
                     sentences of the same passage, ignoring case and surrounding
                     punctuation
"""

import csv
import json
import os
from collections import defaultdict, Counter
from statistics import mean

WORDS_DIR = "word_level_final"
ANSWERS_DIR = "outputs"
GOLD_FILE = os.path.join("discourse_final", "gold_discourse_continuation.tsv")
DENSITY_FILE = "lexical_density_discourse_options.tsv"

MODELS = ("krikri", "meltemi", "llama")
SEEDS = (42, 7)
ITEMS = ["disc_%02d" % i for i in range(1, 16)]
OPTIONS = ("correct", "wrong_1", "wrong_2", "wrong_3")
WRONG = OPTIONS[1:]


# ---------------------------------------------------------------- reading

def read_gold():
    with open(GOLD_FILE, encoding="utf-8") as f:
        return {r["item_id"]: r for r in csv.DictReader(f, delimiter="\t")}


def read_density():
    table = {}
    with open(DENSITY_FILE, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            table[r["item_id"]] = {k: (float(v) if v != "" else None)
                                   for k, v in r.items() if k != "item_id"}
    return table


def read_answers():
    answers = {}
    for condition in ("pharasiot", "greek"):
        for seed in SEEDS:
            for model in MODELS:
                path = os.path.join(
                    ANSWERS_DIR,
                    "discourse_%s_seed%d_%s.jsonl" % (condition, seed, model))
                with open(path, encoding="utf-8") as f:
                    rows = [json.loads(l) for l in f if l.strip()]
                answers[(condition, seed, model)] = {r["item_id"]: r for r in rows}
    return answers


def read_words():
    rows = defaultdict(list)
    for model in MODELS:
        path = os.path.join(WORDS_DIR, "words_%s.jsonl" % model)
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows[model].append(json.loads(line))
    return rows


gold = read_gold()
density = read_density()
answers = read_answers()
words = read_words()

paragraph_of = {i: gold[i]["paragraph_id"] for i in ITEMS}


def name(item):
    """Fairytale1.8--11, from the first context sentence and the target sentence."""
    first = gold[item]["context_sent_ids"].split("|")[0].strip()
    last = gold[item]["target_sent_id"].split(".")[1]
    return first + "--" + last


def is_correct(condition, seed, model, item):
    row = answers[(condition, seed, model)][item]
    return row["parsed_letter"] == row["correct_answer_position"]


def group_of(model, item):
    """lost / kept / other, as defined at the top."""
    sg = [is_correct("greek", s, model, item) for s in SEEDS]
    ph = [is_correct("pharasiot", s, model, item) for s in SEEDS]
    if any(g and not p for g, p in zip(sg, ph)):
        return "lost"
    if all(sg) and all(ph):
        return "kept"
    return "other"


# ------------------------------------------------------------ option loss

option_loss = {}
for model in MODELS:
    totals = defaultdict(lambda: [0.0, 0])
    for r in words[model]:
        if r["kind"] != "option":
            continue
        key = (model, r["text_id"], r["option"], r["variety"])
        totals[key][0] += r["loss_sum"]
        totals[key][1] += r["n_tokens"]
    for key, (loss, pieces) in totals.items():
        option_loss[key] = loss / pieces

missing = [(m, i, o, v) for m in MODELS for i in ITEMS for o in OPTIONS
           for v in ("pharasiot", "greek") if (m, i, o, v) not in option_loss]
if missing:
    raise SystemExit("These option sequences are not in the word files: %s" % missing[:5])


def gap(model, item):
    return (option_loss[(model, item, "correct", "pharasiot")]
            - option_loss[(model, item, "correct", "greek")])


# -------------------------------------------------------- word selections

def option_words(model, variety="pharasiot"):
    return [r for r in words[model]
            if r["kind"] == "option" and r["variety"] == variety]


def context_words(model, item):
    return [r for r in words[model]
            if r["kind"] == "passage" and r["variety"] == "pharasiot"
            and r["text_id"] == paragraph_of[item] and r["segment"] == "context"]


def plain(word):
    return word.lower().strip("'’‘ʼ.,;:!?«»\"")


context_forms = {(m, i): {plain(r["word"]) for r in context_words(m, i)}
                 for m in MODELS for i in ITEMS}


def repeats_context(model, row):
    return plain(row["word"]) in context_forms[(model, row["text_id"])]


# ------------------------------------------------------------------ report

def heading(text):
    print()
    print(text)
    print("-" * len(text))


heading("Table: mean loss of the correct option, and the gap")
print("%-8s %6s %6s %7s  %-12s %-12s %-12s" %
      ("", "SG", "Ph", "All 15", "Lost", "Kept", "Other"))
for model in MODELS:
    sg = mean(option_loss[(model, i, "correct", "greek")] for i in ITEMS)
    ph = mean(option_loss[(model, i, "correct", "pharasiot")] for i in ITEMS)
    cells = []
    for g in ("lost", "kept", "other"):
        values = [gap(model, i) for i in ITEMS if group_of(model, i) == g]
        cells.append("%.2f (%d)" % (mean(values), len(values)) if values else "--- (0)")
    print("%-8s %6.2f %6.2f %7.2f  %-12s %-12s %-12s" %
          (model, sg, ph, mean(gap(model, i) for i in ITEMS), *cells))
print()
for model in MODELS:
    print("%-8s passages where the Pharasiot loss is the higher: %d of 15" %
          (model, sum(gap(model, i) > 0 for i in ITEMS)))

heading("Context density")
values = [density[i]["context"] for i in ITEMS]
print("range %.2f to %.2f, mean %.2f" % (min(values), max(values), mean(values)))
order = sorted(ITEMS, key=lambda i: (round(density[i]["context"], 3), ITEMS.index(i)))
groups = [order[:5], order[5:10], order[10:]]
for label, g in zip(("lowest five ", "middle five ", "highest five"), groups):
    print("%s mean %.2f  %s" % (label, mean(density[i]["context"] for i in g),
                                ", ".join(name(i) for i in g)))
print()
for condition in ("pharasiot", "greek"):
    print("%s correct responses, out of 10 per group:" % condition)
    for model in MODELS:
        print("   %-8s %s" % (model, [sum(is_correct(condition, s, model, i)
                                          for s in SEEDS for i in g) for g in groups]))
print()
for model in MODELS:
    by_group = [mean(gap(model, i) for i in g) for g in groups]
    low, rest = by_group[0], mean(gap(model, i) for i in groups[1] + groups[2])
    print("%-8s mean loss gap by group %s | lowest five %.2f against the other ten %.2f" %
          (model, ["%.2f" % v for v in by_group], low, rest))
print()
for model in MODELS:
    lost = [density[i]["context"] for i in ITEMS if group_of(model, i) == "lost"]
    kept = [density[i]["context"] for i in ITEMS if group_of(model, i) == "kept"]
    print("%-8s mean context density, lost %.2f against kept %.2f" %
          (model, mean(lost), mean(kept)))
print()
answered = {i: sum(is_correct("pharasiot", s, m, i) for s in SEEDS for m in MODELS)
            for i in ITEMS}
worst = [i for i in ITEMS if answered[i] <= 2]
print("answered correctly in at most two of the six Pharasiot responses: %s" %
      ", ".join(name(i) for i in worst))
print("   of these, in the middle five: %s" %
      ", ".join(name(i) for i in worst if i in groups[1]))

heading("Option density")
with_word = [i for i in ITEMS if density[i]["correct"] is not None]
higher = sum(density[i]["correct"] > mean(density[i][w] for w in WRONG) for i in with_word)
print("passages whose correct option holds a counted word: %d of 15" % len(with_word))
print("   of these, the correct option is denser than the wrong ones on average: %d" % higher)

heading("Table: words of the Pharasiot options")
print("%-8s %6s %6s %7s %7s  %-10s %-10s" %
      ("", "Recov", "Not", "pieces", "pieces", "Lost", "Kept"))
for model in MODELS:
    rows = option_words(model)
    yes = [r for r in rows if r["shared"] == "yes"]
    no = [r for r in rows if r["shared"] == "no"]
    correct_rows = [r for r in rows if r["option"] == "correct"]
    cells = []
    for g in ("lost", "kept"):
        chosen = [r for r in correct_rows if group_of(model, r["text_id"]) == g]
        cells.append("%.1f (%d)" % (mean([r["loss_sum"] for r in chosen]),
                                    len({r["text_id"] for r in chosen})))
    print("%-8s %6.1f %6.1f %7.1f %7.1f  %-10s %-10s" %
          (model, mean([r["loss_sum"] for r in yes]), mean([r["loss_sum"] for r in no]),
           mean([r["n_tokens"] for r in yes]), mean([r["n_tokens"] for r in no]), *cells))

heading("Word counts and the two kinds")
rows = option_words(MODELS[0])
counts = Counter(r["shared"] for r in rows)
print("%d words in %d Pharasiot options: %d recoverable, %d not recoverable, %d function words"
      % (len(rows), len({r["seq_id"] for r in rows}),
         counts["yes"], counts["no"], counts["skip"]))
print()
for model in MODELS:
    rows = option_words(model)
    yes = [r for r in rows if r["shared"] == "yes"]
    no = [r for r in rows if r["shared"] == "no"]
    skip = [r for r in rows if r["shared"] == "skip"]
    by_option = defaultdict(list)
    for r in rows:
        by_option[r["seq_id"]].append(r)
    both = higher_in = 0
    for option_rows in by_option.values():
        y = [r["loss_sum"] for r in option_rows if r["shared"] == "yes"]
        n = [r["loss_sum"] for r in option_rows if r["shared"] == "no"]
        if y and n:
            both += 1
            higher_in += mean(n) > mean(y)
    print("%-8s function words %.1f | piece loss, recoverable %.2f against not recoverable %.2f"
          % (model, mean([r["loss_sum"] for r in skip]),
             sum(r["loss_sum"] for r in yes) / sum(r["n_tokens"] for r in yes),
             sum(r["loss_sum"] for r in no) / sum(r["n_tokens"] for r in no)))
    print("         not recoverable words have the higher mean in %d of the %d options holding both"
          % (higher_in, both))

heading("Words the context has already shown")
for model in MODELS:
    rows = option_words(model)
    counted = [r for r in rows if r["shared"] in ("yes", "no")]
    repeated = [r for r in counted if repeats_context(model, r)]
    fresh = [r for r in counted if not repeats_context(model, r)]
    print("%-8s repeated %d, mean %.1f | not repeated %d, mean %.1f | repeated in correct options %d"
          % (model, len(repeated), mean([r["loss_sum"] for r in repeated]),
             len(fresh), mean([r["loss_sum"] for r in fresh]),
             sum(r["option"] == "correct" for r in repeated)))
print()
model = MODELS[0]
with_repeat = {r["text_id"] for r in option_words(model)
               if r["option"] == "correct" and r["shared"] in ("yes", "no")
               and repeats_context(model, r)}
wrong_repeats = [(name(r["text_id"]), r["option"], r["word"]) for r in option_words(model)
                 if r["option"] != "correct" and r["shared"] in ("yes", "no")
                 and repeats_context(model, r)]
print("passages whose correct option repeats a counted context word: %d of 15 (%s)"
      % (len(with_repeat), ", ".join(sorted(name(i) for i in with_repeat))))
print("wrong options that do: %d of 45 (%s)"
      % (len(wrong_repeats), "; ".join("%s %s %s" % w for w in wrong_repeats)))
print()
for model in MODELS:
    a = [is_correct("pharasiot", s, model, i) for s in SEEDS for i in ITEMS if i in with_repeat]
    b = [is_correct("pharasiot", s, model, i) for s in SEEDS for i in ITEMS if i not in with_repeat]
    print("%-8s Pharasiot correct, with a repeat %d/%d (%.0f%%), without %d/%d (%.0f%%)"
          % (model, sum(a), len(a), 100 * sum(a) / len(a),
             sum(b), len(b), 100 * sum(b) / len(b)))
print()
print("the river example, Fairytale5.76--79 (disc_10):")
for model in MODELS:
    in_context = [round(r["loss_sum"], 1) for r in context_words(model, "disc_10")
                  if plain(r["word"]) == "ποτάμι"]
    in_option = [round(r["loss_sum"], 1) for r in option_words(model)
                 if r["text_id"] == "disc_10" and r["option"] == "correct"
                 and plain(r["word"]) == "ποτάμι"]
    print("   %-8s context %s, correct option %s" % (model, in_context, in_option))

heading("Loss gap, passage by passage")
for model in MODELS:
    ranked = sorted(ITEMS, key=lambda i: -gap(model, i))
    print("%-8s largest  %s" % (model, ", ".join(
        "%s %.2f (%s)" % (name(i), gap(model, i), group_of(model, i)) for i in ranked[:3])))
    print("%-8s smallest %s" % ("", ", ".join(
        "%s %.2f (%s)" % (name(i), gap(model, i), group_of(model, i)) for i in ranked[-3:])))
    print("%-8s rank of Fairytale5.104--107: %d of 15" % ("", ranked.index("disc_12") + 1))
print()
print("the engagement example, Fairytale5.104--107 (disc_12), correct option:")
for model in MODELS:
    row = [(r["word"], r["shared"], round(r["loss_sum"], 1)) for r in option_words(model)
           if r["text_id"] == "disc_12" and r["option"] == "correct"]
    top = sorted(row, key=lambda t: -t[2])[:2]
    print("   %-8s highest two: %s" % (model, top))
