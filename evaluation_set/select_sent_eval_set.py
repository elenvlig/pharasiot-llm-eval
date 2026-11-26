"""
select_sent_eval_set.py
-----------------------
Selects a 40-sentence evaluation set from the Pharasiot treebank using
stratified greedy selection. The goal is a balanced sample that:
  * covers a range of sentence lengths (short/medium/long)
  * covers a range of syntactic complexity (simple/mid/complex)
  * includes both direct-speech and narrative registers
  * includes enough sentences with contact-induced phenomena
    (postposed adpositions, gender mismatches, borrowings, etc.)
  * avoids near-duplicates (sentences that share many content words)

The selection has three types of constraints:

  * HARD QUOTAS
    For length, complexity, and register, we set exact counts.
    A candidate sentence is only accepted if picking it wouldn't
    exceed any of these quotas.

  * SOFT TARGETS
    For contact-induced features (postposed adpositions, borrowings,
    etc.), we score each candidate based on how much it helps us reach
    the target. Higher score = closer to the desired distribution.

  * NEAR-DUPLICATE REJECTION
    If a candidate shares more than 50% of its content lemmas with an
    already-picked sentence, we reject it. This keeps the sample
    lexically diverse.

Usage:
  python3 select_sent_eval_set.py <sentences.tsv> <n_sentences>

Output:
  eval_set_40.tsv           (the 40 selected sentences with their features)
  translation_template.tsv  (a helper file for the user to add Greek translations)
"""

import sys
import csv
import random


# ---------------------------------------------------------------------------
# Constants and configuration
# ---------------------------------------------------------------------------

SEED = 7  # for reproducibility

# How many sentences we want in the eval set (overridable by CLI arg)
DEFAULT_TARGET_TOTAL = 40

# ---- HARD QUOTAS ----
# Length quotas: exact counts per length band
# These sum to 40 (12 + 16 + 12)
LENGTH_QUOTAS = {
    "short": 12,
    "medium": 16,
    "long": 12,
}

# Complexity quotas: exact counts per complexity band
# These also sum to 40 (12 + 14 + 14)
COMPLEXITY_QUOTAS = {
    "simple": 12,
    "mid": 14,
    "complex": 14,
}

# Register quotas: reflect the corpus proportions
# In the whole treebank about 34% of sentences are direct speech and 66%
# are narrative; we keep the same ratio in the eval set.
REGISTER_QUOTAS = {
    "direct": 14,
    "narrative": 26,
}

# ---- SOFT TARGETS ----
# These are goals we try to reach but don't strictly enforce. For each
# feature we know how many sentences in the treebank have it, and we
# target a proportional number in the eval set.
SOFT_TARGETS = {
    "postadp_gte_1": 6,        # at least 1 postposed adposition
    "relcl_gte_1": 3,          # at least 1 relative clause
    "gendermis_gte_1": 6,      # at least 1 gender-mismatch
    "postsubj_gte_1": 11,      # at least 1 postposed subject
    "postobj_gte_1": 19,       # at least 1 postposed object
    "pron3_gte_1": 12,         # at least 1 third-person pronoun
    "borrow_high": 21,         # >= 3 borrowings
    "borrow_low": 19,          # < 3 borrowings
    "neg_gte_1": 6,            # at least 1 negation
}

# Weight for the lexical-novelty bonus. A sentence gets a bonus for
# introducing lemmas that haven't appeared in already-picked sentences.
# This encourages lexical diversity beyond just avoiding duplicates.
NOVELTY_WEIGHT = 0.34

# Threshold for near-duplicate rejection.
# If two sentences share more than this fraction of content lemmas, we
# consider them near-duplicates.
NEAR_DUPLICATE_JACCARD = 0.5


# ---------------------------------------------------------------------------
# Helper: load sentences from the TSV
# ---------------------------------------------------------------------------

def load_sentences(sentences_path):
    """Load the sentences.tsv file into a list of dicts.

    Each dict has all the feature columns. Numeric columns get converted
    from strings to integers.
    """
    numeric_columns = ["len", "borrow", "adp", "propn", "pron", "pron3",
                       "neg", "postsubj", "postobj", "postadp", "gendermis",
                       "depth", "n_sub", "n_conj", "relcl", "cscore",
                       "is_direct"]

    sentences = []
    with open(sentences_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Convert numeric fields
            for column in numeric_columns:
                if column in row:
                    try:
                        row[column] = int(row[column])
                    except (ValueError, KeyError):
                        row[column] = 0
            sentences.append(row)

    print(f"  Loaded {len(sentences)} sentences from {sentences_path}")
    return sentences


# ---------------------------------------------------------------------------
# Helper: extract content lemmas from a sentence's raw text
# ---------------------------------------------------------------------------

def get_content_words(sentence_text):
    """Extract a set of content-ish tokens from raw text.

    We don't have lemma info in sentences.tsv (only features), so we
    approximate by taking the text, lowercasing, and splitting on
    non-alphabetic characters. This is a crude approximation of "content
    words" but it works well for detecting near-duplicates.
    """
    import re
    text = sentence_text.lower()
    # Match sequences of letters (both Latin and Greek)
    tokens = re.findall(r"[a-zα-ωϊϋΐΰάέήίόύώ]+", text)
    return set(tokens)


# ---------------------------------------------------------------------------
# Helper: compute Jaccard similarity between two sets
# ---------------------------------------------------------------------------

def jaccard_similarity(set_a, set_b):
    """How similar are two sets of words? Returns a number from 0 to 1.

    Jaccard = size of intersection / size of union.
    0 means no overlap, 1 means identical.
    """
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Helper: check if a candidate meets each hard quota
# ---------------------------------------------------------------------------

def can_still_add(candidate, picked_counts, quotas):
    """Check whether adding `candidate` would exceed a hard quota.

    picked_counts is a dict counting how many of each type we've picked
    so far, e.g. {"short": 3, "medium": 5, "long": 2}.

    Returns True if the candidate fits within the quotas, False if adding
    it would push us over.
    """
    for category, target in quotas.items():
        already_have = picked_counts.get(category, 0)
        if already_have >= target:
            # We're already at or over the target for this category
            # Check if the candidate falls into this category
            if categorize(candidate, quotas) == category:
                return False
    return True


def categorize(candidate, quotas):
    """Return the category name that `candidate` belongs to for a set of
    quotas.

    Different quotas have different categorization rules, so we look at
    the quota keys to figure out what to check.
    """
    # For length quotas, check the "band" column
    if "short" in quotas:
        return candidate.get("band", "medium")
    # For complexity quotas, check the "cband" column
    if "simple" in quotas:
        return candidate.get("cband", "mid")
    # For register quotas, check the "register" column
    if "direct" in quotas:
        return candidate.get("register", "narrative")
    return None


# ---------------------------------------------------------------------------
# Helper: score a candidate for the soft targets
# ---------------------------------------------------------------------------

def score_candidate(candidate, soft_picked_counts, seen_lemmas):
    """Give a candidate sentence a score based on how much it helps us
    reach the soft targets, plus a bonus for introducing new lemmas.

    Higher score = better candidate.
    """
    score = 0.0

    # --- Soft target bonuses ---
    # For each soft target we haven't reached yet, give a bonus if this
    # candidate has that feature.
    if candidate["postadp"] >= 1 and soft_picked_counts["postadp_gte_1"] < SOFT_TARGETS["postadp_gte_1"]:
        score += 1
    if candidate["relcl"] >= 1 and soft_picked_counts["relcl_gte_1"] < SOFT_TARGETS["relcl_gte_1"]:
        score += 1
    if candidate["gendermis"] >= 1 and soft_picked_counts["gendermis_gte_1"] < SOFT_TARGETS["gendermis_gte_1"]:
        score += 1
    if candidate["postsubj"] >= 1 and soft_picked_counts["postsubj_gte_1"] < SOFT_TARGETS["postsubj_gte_1"]:
        score += 1
    if candidate["postobj"] >= 1 and soft_picked_counts["postobj_gte_1"] < SOFT_TARGETS["postobj_gte_1"]:
        score += 1
    if candidate["pron3"] >= 1 and soft_picked_counts["pron3_gte_1"] < SOFT_TARGETS["pron3_gte_1"]:
        score += 1
    if candidate["neg"] >= 1 and soft_picked_counts["neg_gte_1"] < SOFT_TARGETS["neg_gte_1"]:
        score += 1

    # Borrowing bucket: high (>=3) or low (<3)
    if candidate["borrow"] >= 3:
        if soft_picked_counts["borrow_high"] < SOFT_TARGETS["borrow_high"]:
            score += 1
    else:
        if soft_picked_counts["borrow_low"] < SOFT_TARGETS["borrow_low"]:
            score += 1

    # --- Novelty bonus ---
    # Give extra credit for lemmas we haven't seen yet in picked sentences.
    candidate_lemmas = get_content_words(candidate["text"])
    new_lemmas = candidate_lemmas - seen_lemmas
    if candidate_lemmas:
        novelty_ratio = len(new_lemmas) / len(candidate_lemmas)
    else:
        novelty_ratio = 0
    score += NOVELTY_WEIGHT * novelty_ratio

    return score


# ---------------------------------------------------------------------------
# Helper: update the picked counts after adding a sentence
# ---------------------------------------------------------------------------

def update_soft_counts(candidate, soft_picked_counts):
    """After adding a candidate, bump the soft-target counts for any
    features it has."""
    if candidate["postadp"] >= 1:
        soft_picked_counts["postadp_gte_1"] += 1
    if candidate["relcl"] >= 1:
        soft_picked_counts["relcl_gte_1"] += 1
    if candidate["gendermis"] >= 1:
        soft_picked_counts["gendermis_gte_1"] += 1
    if candidate["postsubj"] >= 1:
        soft_picked_counts["postsubj_gte_1"] += 1
    if candidate["postobj"] >= 1:
        soft_picked_counts["postobj_gte_1"] += 1
    if candidate["pron3"] >= 1:
        soft_picked_counts["pron3_gte_1"] += 1
    if candidate["neg"] >= 1:
        soft_picked_counts["neg_gte_1"] += 1

    if candidate["borrow"] >= 3:
        soft_picked_counts["borrow_high"] += 1
    else:
        soft_picked_counts["borrow_low"] += 1


# ---------------------------------------------------------------------------
# The greedy selection loop
# ---------------------------------------------------------------------------

def select_sentences(sentences, target_total):
    """Run the stratified greedy selection.

    Returns a list of the chosen sentence dicts, in the order they were
    picked.
    """
    random.seed(SEED)

    # Skip title sentences (their sent_id ends with "Title")
    candidates = []
    for sentence in sentences:
        if sentence["sent_id"].endswith("Title"):
            continue
        candidates.append(sentence)
    print(f"  {len(candidates)} candidate sentences (after removing titles)")

    # Initialize counters
    length_picked = {"short": 0, "medium": 0, "long": 0}
    complexity_picked = {"simple": 0, "mid": 0, "complex": 0}
    register_picked = {"direct": 0, "narrative": 0}
    soft_picked_counts = {key: 0 for key in SOFT_TARGETS}
    seen_lemmas = set()

    picked = []

    # Keep picking until we hit the target total
    while len(picked) < target_total:
        # Score all remaining candidates
        best_candidate = None
        best_score = -1
        best_index = -1

        # Shuffle a bit so equal-scoring candidates don't always tie the same way
        shuffled_candidates = candidates[:]
        random.shuffle(shuffled_candidates)

        for i, candidate in enumerate(shuffled_candidates):
            # Skip candidates that would break a hard quota
            if not can_still_add(candidate, length_picked, LENGTH_QUOTAS):
                continue
            if not can_still_add(candidate, complexity_picked, COMPLEXITY_QUOTAS):
                continue
            if not can_still_add(candidate, register_picked, REGISTER_QUOTAS):
                continue

            # Reject near-duplicates
            candidate_lemmas = get_content_words(candidate["text"])
            is_duplicate = False
            for previously_picked in picked:
                previous_lemmas = get_content_words(previously_picked["text"])
                if jaccard_similarity(candidate_lemmas, previous_lemmas) > NEAR_DUPLICATE_JACCARD:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue

            # Score the candidate
            candidate_score = score_candidate(candidate, soft_picked_counts, seen_lemmas)

            if candidate_score > best_score:
                best_score = candidate_score
                best_candidate = candidate
                best_index = i

        # If we couldn't find any valid candidate, stop
        if best_candidate is None:
            print(f"  WARNING: could not find a valid candidate at step {len(picked) + 1}")
            print(f"           picked so far: {len(picked)} of {target_total}")
            break

        # Pick the best candidate
        picked.append(best_candidate)

        # Update all the counters
        length_band = best_candidate.get("band", "medium")
        length_picked[length_band] += 1

        complexity_band = best_candidate.get("cband", "mid")
        complexity_picked[complexity_band] += 1

        register = best_candidate.get("register", "narrative")
        register_picked[register] += 1

        update_soft_counts(best_candidate, soft_picked_counts)

        seen_lemmas |= get_content_words(best_candidate["text"])

        # Remove the picked candidate from the list so we don't pick it again
        candidates.remove(best_candidate)

    print(f"  Selected {len(picked)} sentences")

    # --- Print summary ---
    print(f"\n  Length distribution: {length_picked}")
    print(f"  Complexity distribution: {complexity_picked}")
    print(f"  Register distribution: {register_picked}")
    print(f"  Soft-target counts:")
    for key, count in soft_picked_counts.items():
        target = SOFT_TARGETS[key]
        marker = "OK" if count >= target else "under"
        print(f"    {key:20s}: {count}/{target} ({marker})")
    print(f"  Total unique content lemmas covered: {len(seen_lemmas)}")

    return picked


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

def write_eval_set(picked, output_path):
    """Write the picked sentences (with all features) to a TSV file."""
    if not picked:
        print("  No sentences to write.")
        return

    columns = list(picked[0].keys())

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(columns)
        for row in picked:
            values = [row.get(col, "") for col in columns]
            writer.writerow(values)

    print(f"  Wrote {output_path}")


def write_translation_template(picked, output_path):
    """Write a simpler file for the user to fill in Greek translations.

    This template has 4 columns:
      sent_id, pharasiot_text, english_text (empty), greek_translation_FILL_IN
    """
    columns = ["sent_id", "pharasiot_text", "english_text",
               "greek_translation_FILL_IN"]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(columns)
        for sentence in picked:
            row = [
                sentence["sent_id"],
                sentence["text"],
                "",  # English translation, to be added later
                "",  # Greek translation, for the user to fill in
            ]
            writer.writerow(row)

    print(f"  Wrote {output_path}")


def write_sentences_file(picked, output_path):
    """Write a simple sentences file (id + text only)."""
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["sent_id", "text"])
        for sentence in picked:
            writer.writerow([sentence["sent_id"], sentence["text"]])
    print(f"  Wrote {output_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 select_sent_eval_set.py <sentences.tsv> [n_sentences]")
        sys.exit(1)

    sentences_path = sys.argv[1]

    if len(sys.argv) >= 3:
        target_total = int(sys.argv[2])
    else:
        target_total = DEFAULT_TARGET_TOTAL

    print(f"Target: {target_total} sentences")

    print(f"Loading sentences from {sentences_path}...")
    all_sentences = load_sentences(sentences_path)

    print(f"Running stratified greedy selection...")
    picked = select_sentences(all_sentences, target_total)

    print(f"\nWriting output files...")
    write_eval_set(picked, "eval_set_40.tsv")
    write_translation_template(picked, "translation_template.tsv")
    write_sentences_file(picked, "sentences_eval.tsv")

    print("Done.")


if __name__ == "__main__":
    main()
