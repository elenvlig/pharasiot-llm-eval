#!/usr/bin/env python3
"""
generate_prompts.py — Phase 3, Script 1

Reads a gold TSV, applies a .txt prompt template, and writes model-ready prompts.

    python generate_prompts.py --task lexical_translation

Outputs (per task, both conditions):
    prompts/{task}_{condition}.jsonl     one prompt object per line
    metadata/{task}_{condition}.tsv      same info, Excel-inspectable
    logs/{task}_skipped.tsv              items skipped + why
    logs/{task}_greek_target_review.tsv  items where the Greek target form was guessed

Design constraints (from phase3_handoff.md):
  - templates live in .txt files, never hardcoded here
  - option order is seeded PER ITEM, and is IDENTICAL across the two conditions
    (so Pharasiot vs Greek is a properly paired comparison)
  - incomplete gold items are skipped and logged, never crash the run
  - prompts are model-agnostic: no chat template, no BOS/EOS, no system prompt.
    run_model.py applies each model's own chat template. This is what lets the
    same prompt file be sent to Krikri, Meltemi and Llama.

Author: Eleni Vligouridou
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import unicodedata
from pathlib import Path

# Greek option letters, in order.
LETTERS = ["α", "β", "γ", "δ", "ε", "στ"]

NLI_POOL = []  # populated from the gold file; see build_nli

# Template placeholder names for options, in order: {option_a}, {option_b}, ...
OPTION_KEYS = ["option_a", "option_b", "option_c", "option_d", "option_e", "option_f"]


def strip_accents(text):
    """Remove Greek diacritics so 'άνθρωπος' and 'ανθρωπος' compare equal."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def normalize(text):
    """Lowercase, strip accents and whitespace. Used for matching only."""
    return strip_accents(text.strip().lower())


def tokenize_greek(sentence):
    """Split a sentence into word tokens, dropping punctuation."""
    return re.findall(r"[^\W\d_]+", sentence, flags=re.UNICODE)


def find_target_in_greek(gloss, greek_sentence):
    """
    Locate the form of `gloss` as it actually appears in `greek_sentence`.

    The gold file stores the gloss in citation form, but the Greek sentence
    contains it inflected. Returns (surface_form, match_method).
    match_method is one of: exact, caseless, accentless, stem, fallback.
    """
    if not gloss or not greek_sentence:
        return gloss, "fallback"

    tokens = tokenize_greek(greek_sentence)
    gloss_words = tokenize_greek(gloss)

    # Multi-word glosses: don't try to be clever, use the gloss as written.
    if len(gloss_words) != 1:
        return gloss, "fallback"

    g = gloss_words[0]

    for token in tokens:
        if token == g:
            return token, "exact"
    for token in tokens:
        if token.lower() == g.lower():
            return token, "caseless"
    for token in tokens:
        if normalize(token) == normalize(g):
            return token, "accentless"

    # Inflectional variation: share a long enough stem.
    g_norm = normalize(g)
    stem_len = max(5, len(g_norm) - 3)
    best = None
    for token in tokens:
        t_norm = normalize(token)
        if len(t_norm) < stem_len:
            continue
        if t_norm[:stem_len] == g_norm[:stem_len]:
            if best is None or abs(len(t_norm) - len(g_norm)) < abs(len(normalize(best)) - len(g_norm)):
                best = token
    if best is not None:
        return best, "stem"

    return gloss, "fallback"


# --------------------------------------------------------------------------
# Task registry
# --------------------------------------------------------------------------
# Each task declares:
#   gold_file       filename inside --gold-dir
#   required        columns that must be non-empty or the item is skipped
#   n_options       number of answer choices
#   build           row -> (fields_by_condition, options, correct_text, gold_meta, notes)
#
# `fields_by_condition` maps "pharasiot"/"greek" to the non-option placeholders
# that the template needs for that condition.

def build_lexical_translation(row, condition_notes):
    """Build one lexical-translation item from a gold row. Pharasiot only."""
    fields = {
        "pharasiot": {
            "context": row["context_sentence_pharasiot"].strip(),
            "target_word": row["target_form_in_sentence"].strip(),
        }
    }
    options = [
        row["correct_greek_gloss"].strip(),
        row["distractor_1"].strip(),
        row["distractor_2"].strip(),
        row["distractor_3"].strip(),
    ]
    gold_meta = {
        "context_sent_id": row.get("context_sent_id", "").strip(),
        "pharasiot_lemma": row.get("pharasiot_lemma", "").strip(),
        "upos": row.get("upos", "").strip(),
    }
    return fields, options, options[0], gold_meta


SENTIMENT_LABELS = ["θετικό", "αρνητικό", "ουδέτερο"]

# The gold file stores English labels; the prompt shows Greek ones.
LABEL_MAP = {"positive": "θετικό", "negative": "αρνητικό", "neutral": "ουδέτερο"}


def build_sentiment(row, condition_notes):
    """Build one sentiment item from a gold row."""
    raw_label = row["sentiment_label"].strip().lower()
    if raw_label not in LABEL_MAP:
        raise ValueError(f"unknown label '{raw_label}' (expected {sorted(LABEL_MAP)})")

    fields = {
        "pharasiot": {"sentence": row["pharasiot_text"].strip()},
        "greek": {"sentence": row["greek_translation"].strip()},
    }
    gold_meta = {"context_sent_id": row.get("sent_id", "").strip()}
    return fields, list(SENTIMENT_LABELS), LABEL_MAP[raw_label], gold_meta


TOPIC_LABELS = {
    "social_interaction": "κοινωνική συναναστροφή",
    "work_daily_life":    "εργασία και καθημερινή ζωή",
    "violence_threat":     "βία ή απειλή",
    "travel_adventure":    "ταξίδι ή περιπέτεια",
}


def build_topic(row, condition_notes):
    """Build one topic-classification item from a gold row."""
    raw_label = row["topic_label"].strip().lower()
    if raw_label not in TOPIC_LABELS:
        raise ValueError(f"unknown topic '{raw_label}' (expected {sorted(TOPIC_LABELS)})")

    fields = {
        "pharasiot": {"sentence": row["pharasiot_text"].strip()},
        "greek": {"sentence": row["greek_translation"].strip()},
    }
    gold_meta = {"context_sent_id": row.get("sent_id", "").strip()}
    return fields, list(TOPIC_LABELS.values()), TOPIC_LABELS[raw_label], gold_meta


def build_nli(row, condition_notes):
    """
    Four Greek candidate sentences; the model picks the one entailed by the
    Pharasiot premise. Distractors are this row's contradiction and neutral
    hypotheses, plus one drawn from a different item.

    The fourth distractor is taken from another row's entailment hypothesis
    rather than from the narrative text, so all four options share the same
    register. A raw corpus sentence would be stylistically obvious and could be
    excluded without any reasoning, which would make the item easier than
    intended.
    """
    base = row["item_id"].strip()
    correct = row["entailment_greek"].strip()

    # Deterministic pick of an unrelated hypothesis, seeded by item_id.
    pool = [h for iid, h in NLI_POOL if iid != base and normalize(h) != normalize(correct)]
    rng = random.Random(f"nli-distractor::{base}")
    foreign = rng.choice(pool) if pool else ""

    options = [correct, row["contradiction_greek"].strip(),
               row["neutral_greek"].strip(), foreign]

    fields = {
        "pharasiot": {"premise": row["context_sentence_pharasiot"].strip()},
        "greek": {"premise": row["context_sentence_greek"].strip()},
    }
    gold_meta = {
        "context_sent_id": row.get("context_sent_id", "").strip(),
        "foreign_distractor": foreign,
    }
    return fields, options, correct, gold_meta


def build_factual(row, condition_notes):
    """
    Paragraph-level factual QA. The question and all four options are in
    Standard Greek in both conditions; only the paragraph changes. So the
    Pharasiot condition is cross-variety comprehension, as with NLI.
    """
    fields = {
        "pharasiot": {"paragraph": row["context_paragraph_pharasiot"].strip(),
                      "question": row["question_greek"].strip()},
        "greek": {"paragraph": row["context_paragraph_greek"].strip(),
                  "question": row["question_greek"].strip()},
    }
    options = [row["correct_answer_greek"].strip(), row["distractor_1"].strip(),
               row["distractor_2"].strip(), row["distractor_3"].strip()]
    gold_meta = {
        "paragraph_id": row.get("paragraph_id", "").strip(),
        "fairytale": row.get("fairytale", "").strip(),
    }
    return fields, options, options[0], gold_meta


def build_discourse(row, condition_notes):
    """
    Which sentence continues the passage? Unlike the other paragraph tasks,
    the candidate continuations exist in both varieties, so each condition is
    presented entirely in its own variety rather than cross-variety.
    The correct continuation is always index 0 before shuffling.
    """
    fields = {
        "pharasiot": {"paragraph": row["context_pharasiot"].strip()},
        "greek": {"paragraph": row["context_greek"].strip()},
    }
    options = {
        "pharasiot": [row["correct_continuation_pharasiot"].strip()] +
                     [row[f"distractor_{i}_pharasiot"].strip() for i in (1, 2, 3)],
        "greek": [row["correct_continuation_greek"].strip()] +
                 [row[f"distractor_{i}_greek"].strip() for i in (1, 2, 3)],
    }
    correct_text = {c: options[c][0] for c in options}
    gold_meta = {
        "paragraph_id": row.get("paragraph_id", "").strip(),
        "fairytale": row.get("fairytale", "").strip(),
        "target_sent_id": row.get("target_sent_id", "").strip(),
    }
    return fields, options, correct_text, gold_meta


TASKS = {
    "discourse": {
        "gold_file": "gold_discourse_continuation.tsv",
        "required": ["item_id", "context_pharasiot", "context_greek",
                     "correct_continuation_pharasiot", "correct_continuation_greek",
                     "distractor_1_pharasiot", "distractor_2_pharasiot", "distractor_3_pharasiot",
                     "distractor_1_greek", "distractor_2_greek", "distractor_3_greek"],
        "n_options": 4,
        "conditions": ["pharasiot", "greek"],
        "build": build_discourse,
    },
    "context_tracking": {
        "gold_file": "gold_contexttracking_prompts.tsv",
        "required": ["item_id", "context_paragraph_pharasiot", "context_paragraph_greek",
                     "question_greek", "correct_answer_greek",
                     "distractor_1", "distractor_2", "distractor_3"],
        "n_options": 4,
        "conditions": ["pharasiot", "greek"],
        "build": build_factual,   # identical column layout to the factual task
    },
    "factual": {
        "gold_file": "gold_factual_prompts.tsv",
        "required": ["item_id", "context_paragraph_pharasiot", "context_paragraph_greek",
                     "question_greek", "correct_answer_greek",
                     "distractor_1", "distractor_2", "distractor_3"],
        "n_options": 4,
        "conditions": ["pharasiot", "greek"],
        "build": build_factual,
    },
    "nli": {
        "gold_file": "gold_NLI.tsv",
        "required": ["item_id", "context_sentence_pharasiot", "context_sentence_greek",
                     "entailment_greek", "contradiction_greek", "neutral_greek"],
        "n_options": 4,
        "conditions": ["pharasiot", "greek"],
        "build": build_nli,
    },
    "topic": {
        "gold_file": "gold_topic_classification.tsv",
        "required": ["item_id", "pharasiot_text", "greek_translation", "topic_label"],
        "n_options": 4,
        "conditions": ["pharasiot", "greek"],
        "build": build_topic,
    },
    "sentiment": {
        "gold_file": "gold_sentiment_classification.tsv",
        "required": ["item_id", "pharasiot_text", "greek_translation", "sentiment_label"],
        "n_options": 3,
        "conditions": ["pharasiot", "greek"],
        "build": build_sentiment,
    },
    "lexical_translation": {
        "gold_file": "gold_lexical_translation.tsv",
        "required": [
            "item_id",
            "context_sentence_pharasiot",
            "target_form_in_sentence",
            "correct_greek_gloss",
            "distractor_1",
            "distractor_2",
            "distractor_3",
        ],
        "n_options": 4,
        # Pharasiot only. The Greek version of this task is degenerate: the Greek
        # sentence contains the gloss, so the answer sits inside the question.
        # Chance level (25%) is the baseline for this task instead.
        "conditions": ["pharasiot"],
        "build": build_lexical_translation,
    },
    # --- Step 2 and 3 tasks plug in here, same shape -----------------------
    # "sentiment": {...}, "topic": {...}, "nli": {...},
    # "factual": {...}, "context_tracking": {...}, "discourse_continuation": {...}
}



# --------------------------------------------------------------------------
# Core generation
# --------------------------------------------------------------------------

def load_template(templates_dir, task, condition, lang):
    """
    Resolve a template file. Tries the language-specific name first so an
    English-instruction control run can coexist with the Greek default:
        {task}_{condition}.{lang}.txt   ->  {task}_{condition}.txt
    """
    candidates = []
    if lang != "el":
        candidates.append(templates_dir / f"{task}_{condition}.{lang}.txt")
    candidates.append(templates_dir / f"{task}_{condition}.txt")

    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8").strip("\n"), path

    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"No template found for {task}/{condition}. Tried: {tried}")


def shuffle_options(options, seed, task, item_id):
    """
    Deterministic per-item shuffle. The seed deliberately excludes `condition`
    so option positions are identical in the Pharasiot and Greek prompts.
    """
    rng = random.Random(f"{seed}::{task}::{item_id}")
    shuffled = list(options)
    rng.shuffle(shuffled)
    return shuffled


def clean_cell(value):
    """
    Repair the artefacts that survive a trip through Word or Excel:
      - MICRO SIGN (µ) masquerading as Greek mu (μ)
      - non-breaking spaces
      - stray quote marks wrapping a cell
      - runs of whitespace
    These look identical on screen but tokenize differently, which would make
    the Greek condition look harder than it is.
    """
    if value is None:
        return ""
    v = value.replace("\u00b5", "\u03bc").replace("\u00a0", " ")
    # Latin letters that are visually identical to Greek ones, when they appear
    # inside a word that is otherwise Greek (a common copy-paste artefact).
    LOOKALIKE = {"o": "ο", "O": "Ο", "A": "Α", "B": "Β", "E": "Ε", "H": "Η",
                 "I": "Ι", "K": "Κ", "M": "Μ", "N": "Ν", "P": "Ρ", "T": "Τ",
                 "X": "Χ", "Y": "Υ", "Z": "Ζ", "v": "ν", "p": "ρ"}
    def _fix(word):
        if re.search(r"[\u0370-\u03ff\u1f00-\u1fff]", word):
            return "".join(LOOKALIKE.get(ch, ch) for ch in word)
        return word
    v = " ".join(_fix(w) for w in v.split(" "))
    v = re.sub(r'^"+|"+$', "", v.strip())
    return re.sub(r"\s+", " ", v).strip()


def read_gold(path, task):
    """
    Read a gold TSV by splitting on tabs directly rather than via csv, because
    a single stray quote in one cell makes the csv module swallow the rows that
    follow it. Also tolerates a header with more columns than the data rows.
    """
    lines = open(path, encoding="utf-8-sig").read().rstrip("\n").split("\n")
    if not lines:
        return []
    header = [clean_cell(h) for h in lines[0].rstrip("\r").split("\t")]

    rows = []
    for n, line in enumerate(lines[1:], start=1):
        fields = [clean_cell(f) for f in line.rstrip("\r").split("\t")]
        fields += [""] * (len(header) - len(fields))
        row = dict(zip(header, fields))
        # Some gold files identify items by sent_id, or not at all. Either is
        # fine; we just need a stable per-item key.
        if not row.get("item_id"):
            row["item_id"] = f"{task[:3]}_{n:02d}"
        rows.append(row)
    return rows


def generate(task, args, seed):
    spec = TASKS[task]
    conditions = spec["conditions"]
    gold_path = Path(args.gold_dir) / spec["gold_file"]
    if not gold_path.exists():
        sys.exit(f"ERROR: gold file not found: {gold_path}")

    rows = read_gold(gold_path, task)
    if task == "nli":
        global NLI_POOL
        NLI_POOL = [(r["item_id"].strip(), r.get("entailment_greek", "").strip())
                    for r in rows if r.get("entailment_greek", "").strip()]
    if not rows:
        sys.exit(f"ERROR: {gold_path} has no data rows.")

    # Fail early and legibly if the file simply doesn't have the columns we need.
    present = set(rows[0].keys()) | {"item_id"}
    absent = [c for c in spec["required"] if c not in present]
    if absent:
        sys.exit(
            f"ERROR: {gold_path} is missing these column(s): {', '.join(absent)}\n"
            f"       Columns actually found: {', '.join(sorted(present))}\n"
            f"       Either rename the columns in the TSV, or edit the 'required' list\n"
            f"       and the build_{task}() function in this script to match."
        )

    if args.limit:
        rows = rows[: args.limit]

    templates_dir = Path(args.templates_dir)
    templates = {}
    for condition in conditions:
        text, path = load_template(templates_dir, task, condition, args.instructions_lang)
        templates[condition] = text
        print(f"  template [{condition}]: {path}")

    prompts_dir = Path(args.output_dir)
    metadata_dir = Path(args.metadata_dir)
    logs_dir = Path(args.logs_dir)
    for d in (prompts_dir, metadata_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    built = {c: [] for c in conditions}
    skipped = []
    greek_notes = []

    def emit(item_id, fields, options, correct_text, gold_meta):
        """
        Shuffle one item's options and write it into every condition.

        `options` is normally a single list shared by both conditions. For
        discourse continuation the candidate sentences exist in both varieties,
        so it may instead be a dict {condition: [...]}. In that case the same
        permutation is applied to every condition's list, which keeps option
        positions aligned across conditions and preserves the paired design.
        """
        per_condition = isinstance(options, dict)
        reference = options[conditions[0]] if per_condition else options

        if len(set(normalize(o) for o in reference)) != len(reference):
            skipped.append({"item_id": item_id, "reason": "duplicate options after normalization"})
            return
        if len(reference) != spec["n_options"]:
            skipped.append({"item_id": item_id,
                            "reason": f"expected {spec['n_options']} options, got {len(reference)}"})
            return

        # Permute indices, not values, so parallel lists stay in step.
        order = list(range(len(reference)))
        random.Random(f"{seed}::{task}::{item_id}").shuffle(order)
        letters = LETTERS[: len(order)]

        # Locate the correct option by its text. It is NOT always the first
        # element: label-set tasks (sentiment, topic) build a fixed list of
        # labels and mark whichever one is gold.
        ref_correct = correct_text[conditions[0]] if isinstance(correct_text, dict) else correct_text
        correct_index = next(i for i, t in enumerate(reference)
                             if normalize(t) == normalize(ref_correct))
        correct_letter = letters[order.index(correct_index)]

        for condition in conditions:
            texts = [options[condition][i] for i in order] if per_condition \
                    else [reference[i] for i in order]
            placeholders = dict(fields[condition])
            for key, text in zip(OPTION_KEYS, texts):
                placeholders[key] = text
            try:
                prompt = templates[condition].format(**placeholders)
            except KeyError as e:
                sys.exit(f"ERROR: template {task}_{condition} uses unknown placeholder {e}")

            built[condition].append({
                "item_id": item_id,
                "task": task,
                "condition": condition,
                "prompt": prompt,
                "correct_answer_position": correct_letter,
                "correct_answer_text": correct_text[condition] if isinstance(correct_text, dict) else correct_text,
                "option_positions": dict(zip(letters, texts)),
                "gold_metadata": gold_meta,
            })

    for row in rows:
        item_id = (row.get("item_id") or "").strip()
        if not item_id:
            skipped.append({"item_id": "(blank)", "reason": "no item_id"})
            continue

        missing = [c for c in spec["required"] if not (row.get(c) or "").strip()]
        if missing:
            skipped.append({"item_id": item_id, "reason": "empty: " + ", ".join(missing)})
            continue

        try:
            result = spec["build"](row, greek_notes)
        except (ValueError, KeyError) as e:
            skipped.append({"item_id": item_id, "reason": f"bad data: {e}"})
            continue

        # Most tasks produce one item per gold row. NLI produces three, one per
        # hypothesis, so a builder may return a list of (item_id, ...) tuples.
        if isinstance(result, list):
            for sub in result:
                emit(*sub)
        else:
            emit(item_id, *result)

    # --- write outputs ----------------------------------------------------
    suffix = f"_seed{seed}"
    if args.instructions_lang != "el":
        suffix += f"_instr-{args.instructions_lang}"

    for condition in conditions:
        items = built[condition]

        jsonl_path = prompts_dir / f"{task}_{condition}{suffix}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for obj in items:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        tsv_path = metadata_dir / f"{task}_{condition}{suffix}.tsv"
        letters = LETTERS[: spec["n_options"]]
        header = (["item_id", "task", "condition", "correct_answer_position", "correct_answer_text"]
                  + [f"option_{l}" for l in letters]
                  + sorted({k for obj in items for k in obj["gold_metadata"]})
                  + ["prompt"])
        with open(tsv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            for obj in items:
                record = {
                    "item_id": obj["item_id"],
                    "task": obj["task"],
                    "condition": obj["condition"],
                    "correct_answer_position": obj["correct_answer_position"],
                    "correct_answer_text": obj["correct_answer_text"],
                    "prompt": obj["prompt"].replace("\t", " ").replace("\n", "\\n"),
                }
                for letter, text in obj["option_positions"].items():
                    record[f"option_{letter}"] = text
                record.update(obj["gold_metadata"])
                writer.writerow(record)

        print(f"  wrote {len(items):>3} prompts -> {jsonl_path}")
        print(f"                      -> {tsv_path}")

    if skipped:
        skip_path = logs_dir / f"{task}{suffix}_skipped.tsv"
        with open(skip_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["item_id", "reason"], delimiter="\t")
            writer.writeheader()
            writer.writerows(skipped)
        print(f"  SKIPPED {len(skipped)} item(s) -> {skip_path}")
        for s in skipped:
            print(f"    - {s['item_id']}: {s['reason']}")
    else:
        print("  skipped: none")

    if greek_notes:
        review_path = logs_dir / f"{task}_greek_target_review.tsv"
        with open(review_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["item_id", "correct_greek_gloss", "guessed_greek_target",
                            "match_method", "context_sentence_greek"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(greek_notes)
        print(f"  REVIEW {len(greek_notes)} Greek target form(s) -> {review_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation prompts from gold TSV files.")
    parser.add_argument("--task", required=True, choices=sorted(TASKS.keys()))
    parser.add_argument("--seed", type=int, nargs="+", default=[42],
                        help="One or more seeds, e.g. --seed 42 7. Each produces its own "
                             "set of files, named with the seed, so nothing is overwritten.")
    parser.add_argument("--gold-dir", default="./gold_files/")
    parser.add_argument("--templates-dir", default="./templates/")
    parser.add_argument("--output-dir", default="./prompts/")
    parser.add_argument("--metadata-dir", default="./metadata/")
    parser.add_argument("--logs-dir", default="./logs/")
    parser.add_argument("--instructions-lang", default="el",
                        help="Instruction language. 'el' is the main condition; "
                             "'en' loads {task}_{condition}.en.txt for a control run.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N gold rows.")
    args = parser.parse_args()

    for seed in args.seed:
        print(f"\nTask: {args.task}  |  seed: {seed}  |  instructions: {args.instructions_lang}")
        generate(args.task, args, seed)
    print()


if __name__ == "__main__":
    main()
