"""
03_select_paragraph_pool.py
---------------------------
Selects 15 paragraphs (4 consecutive sentences each) from the Pharasiot
treebank. These paragraphs will be used for the four paragraph-level tasks
in the evaluation (factual prompts, context tracking, discourse continuation,
paragraph coreference).

Selection logic:

  For each of the 6 fairytales (F1 through F6):

    (1) Find all 4-sentence windows made of consecutive sentence positions.
        (e.g. sentences at positions 5, 6, 7, 8 in a fairytale)

    (2) Filter out any window that:
        - contains a sentence already in the evaluation set (would leak
          the test set into the paragraph pool)
        - has a "bridge" 4th sentence that ends with a colon (like
          "And X said:") - these set up direct speech and are content-free
        - is too short overall (fewer than 20 tokens total)

    (3) Score each remaining window by content richness (length,
        3rd-person pronouns, proper nouns, syntactic complexity).

    (4) Greedily pick the highest-scoring windows for each fairytale,
        skipping ones that overlap with already-picked windows or that
        are near-duplicates of them.

  The number picked per fairytale is set by TARGET_PER_FAIRYTALE.
  F4 only has 11 sentences so we can only fit 1 paragraph from it;
  the others fit 2 or 3.

The seed is fixed so re-running gives the same paragraphs.

Usage:
  python3 03_select_paragraph_pool.py <sentences.tsv> <eval_set_40.tsv> <pharasiot.conllu>

Output:
  paragraph_pool_15.tsv                  (the 15 selected paragraphs)
  paragraph_translation_template.tsv     (for the user to add Greek translations)
"""

import sys
import csv
import re
import random


SEED = 7           # fixed seed for reproducibility
WINDOW_SIZE = 4    # each paragraph is 4 consecutive sentences

# Minimum total length of a window (in tokens). Very short windows lack
# substantive content for the paragraph-level tasks. This threshold was
# chosen after seeing that windows below ~20 tokens were fragmentary.
MIN_TOTAL_LENGTH = 20

# Near-duplicate rejection: if two windows share more than this fraction
# of content tokens, we treat them as near-duplicates and skip.
NEAR_DUPLICATE_JACCARD = 0.35

# How many paragraphs to pick per fairytale.
# F4 is small (only 11 sentences total, 1 of which is in the eval set)
# so we can only fit 1 non-overlapping 4-sentence window there.
# We redistribute the "missing" slot to F6 to keep the total at 15.
TARGET_PER_FAIRYTALE = {
    1: 3,
    2: 3,
    3: 2,
    4: 1,
    5: 3,
    6: 3,
}

# Parsing a sentence ID into (fairytale number, position)
def parse_sent_id(sent_id):
    """Turn a sentence ID like 'Fairytale1.7' into (1, 7).

    Title sentences (like 'Fairytale1.0Title') return (None, None) because
    we don't want to include titles in our paragraph windows.
    """
    match = re.match(r"Fairytale(\d+)\.(\d+)(Title)?$", sent_id)
    if match is None:
        return None, None
    if match.group(3):  # ends with "Title"
        return None, None

    fairytale_number = int(match.group(1))
    position = int(match.group(2))
    return fairytale_number, position


# Loading  sentences from the sentences.tsv file
def load_sentences(sentences_path):
    """Load sentences.tsv and return a list of dicts.

    Each dict has all the feature columns plus two extra fields we add:
      _fairytale : the fairytale number (an int)
      _position  : the sentence's position within the fairytale (an int)

    We skip title sentences and sentences from Fairytale8 (which only has
    3 sentences total, not enough for a 4-sentence window).
    """
    numeric_columns = ["len", "pron3", "propn", "cscore"]

    sentences = []
    with open(sentences_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            fairytale_num, position = parse_sent_id(row["sent_id"])

            # Skip titles and non-F1-F6 sentences
            if fairytale_num is None:
                continue
            if fairytale_num > 6:
                continue

            # Add the extra fields for convenience
            row["_fairytale"] = fairytale_num
            row["_position"] = position

            # Convert numeric fields from string to int
            for column in numeric_columns:
                try:
                    row[column] = int(row[column])
                except (ValueError, KeyError):
                    row[column] = 0

            sentences.append(row)

    return sentences


#Loading the set of sentence IDs in the evaluation set
def load_eval_ids(eval_set_path):
    """Return the set of sentence IDs that are in the evaluation set.

    We use this set to make sure we don't include any eval-set sentences
    in the paragraph pool (that would leak the test set).
    """
    eval_ids = set()
    with open(eval_set_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eval_ids.add(row["sent_id"])
    return eval_ids



# Loading  English glosses from the CoNLL-U file
def load_english_glosses(conllu_path):
    """Read the CoNLL-U file and pull out the '# text_en =' lines.

    Returns a dict mapping sentence ID to its English gloss. We use these
    English glosses in the translation template to help the user (and
    reviewers) understand the Pharasiot without having to know the dialect.
    """
    glosses = {}
    current_sent_id = None

    with open(conllu_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")

            if line.startswith("# sent_id"):
                # "# sent_id = Fairytale1.7"
                current_sent_id = line.split("=", 1)[1].strip()
            elif line.startswith("# text_en") and current_sent_id is not None:
                # "# text_en = Every day the hunters come..."
                gloss = line.split("=", 1)[1].strip()
                glosses[current_sent_id] = gloss

    return glosses


# Checking whether a sentence is a "bridge" to direct speech
def is_bridge_sentence(text):
    """A bridge sentence is one that sets up direct speech following it.

    In the Pharasiot treebank, bridges are marked by ending with a colon
    (possibly followed by a closing quote), for example:

      "Einanan tzai tou horou ta haivana:"     (And the village animals said:)
      "To lahtori ipen 'ti:"                   (The rooster said:)

    We reject windows whose last sentence is a bridge, because for the
    discourse-continuation task the model would have to predict a content-
    free "correct answer".
    """
    # Strip trailing whitespace and any closing quotation mark
    stripped = text.rstrip()
    if stripped.endswith("\u00bb"):  # "»" - right-pointing double angle quote
        stripped = stripped[:-1].rstrip()

    return stripped.endswith(":")


# ---------------------------------------------------------------------------
# Find all candidate windows for each fairytale
# ---------------------------------------------------------------------------

def find_candidate_windows(sentences, eval_ids):
    """Look at every possible 4-sentence window in each fairytale and
    return only the ones that pass all our filters.

    Returns a dict:  fairytale_number -> list of windows.
    Each window is a list of 4 sentence dicts.
    """
    # Group sentences by fairytale
    sentences_by_fairytale = {}
    for sentence in sentences:
        fairytale_num = sentence["_fairytale"]
        if fairytale_num not in sentences_by_fairytale:
            sentences_by_fairytale[fairytale_num] = []
        sentences_by_fairytale[fairytale_num].append(sentence)

    candidates = {}

    for fairytale_num, fairytale_sentences in sentences_by_fairytale.items():
        # Sort by position so we can look at consecutive windows
        fairytale_sentences.sort(key=lambda s: s["_position"])

        valid_windows = []

        # Try every possible starting position for a 4-sentence window
        for i in range(len(fairytale_sentences) - WINDOW_SIZE + 1):
            window = fairytale_sentences[i : i + WINDOW_SIZE]

            # --- Filter 1: sentences must be strictly consecutive ---
            # If the treebank is missing any sentence (say position 5 skipped),
            # we don't want to join sentence 4 to sentence 6.
            positions = [s["_position"] for s in window]
            first_position = positions[0]
            last_position = positions[-1]
            if last_position - first_position != WINDOW_SIZE - 1:
                continue

            # --- Filter 2: no sentence in the window can be in the eval set ---
            has_eval_sentence = False
            for s in window:
                if s["sent_id"] in eval_ids:
                    has_eval_sentence = True
                    break
            if has_eval_sentence:
                continue

            # --- Filter 3: the 4th sentence must not be a bridge ---
            if is_bridge_sentence(window[-1]["text"]):
                continue

            # --- Filter 4: total length must be at least MIN_TOTAL_LENGTH ---
            total_length = 0
            for s in window:
                total_length += s["len"]
            if total_length < MIN_TOTAL_LENGTH:
                continue

            # All filters passed - keep this window
            valid_windows.append(window)

        candidates[fairytale_num] = valid_windows

    return candidates


# Statifying content 
def score_window(window):
    """Give a window a richness score. Higher = better paragraph.

    The scoring combines four things:
      - total length (longer = more content to ask questions about)
      - 3rd-person pronouns (help coreference resolution items)
      - proper nouns (help entity-tracking items)
      - syntactic complexity (varied sentence structures)

    Weights were chosen so pronouns and proper nouns dominate a bit,
    but length still matters for filtering out very short windows.
    """
    total_length = 0
    total_pron3 = 0
    total_propn = 0
    total_complexity = 0

    for sentence in window:
        total_length += sentence["len"]
        total_pron3 += sentence["pron3"]
        total_propn += sentence["propn"]
        total_complexity += sentence["cscore"]

    # Weighted sum
    score = (0.3 * total_length
             + 3.0 * total_pron3
             + 2.0 * total_propn
             + 0.5 * total_complexity)
    return score


# ---------------------------------------------------------------------------
# Helpers for near-duplicate detection
# ---------------------------------------------------------------------------

def get_content_tokens(window):
    """Extract a set of alphabetic tokens from a window's combined text.

    We use this to compare two windows for near-duplication.
    """
    combined_text = ""
    for sentence in window:
        combined_text += sentence["text"].lower() + " "

    # Grab sequences of letters (both Latin and Greek)
    tokens = re.findall(r"[a-z\u03b1-\u03c9\u03ca\u03cb\u0390\u03b0\u03ac\u03ad\u03ae\u03af\u03cc\u03cd\u03ce]+", combined_text)
    return set(tokens)


def jaccard(set_a, set_b):
    """Return the Jaccard similarity between two token sets.

    Jaccard = size of intersection divided by size of union.
    0 means no overlap, 1 means identical.
    """
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Greedy selection of paragraphs
# ---------------------------------------------------------------------------

def select_paragraphs(candidates, targets):
    """Greedy pick: for each fairytale, pick the top-scoring windows that:
      - don't overlap with already-picked windows in the same fairytale
      - aren't near-duplicates of already-picked windows

    Returns a list of (fairytale_number, window) tuples in selection order.
    """
    rng = random.Random(SEED)
    selected = []

    # Process fairytales in numeric order
    for fairytale_num in sorted(targets.keys()):
        target_count = targets[fairytale_num]
        available_windows = candidates.get(fairytale_num, [])

        if not available_windows:
            print(f"  WARNING: no valid windows for Fairytale{fairytale_num}")
            continue

        # --- Score each window and sort by score (highest first) ---
        # We also include a small random number for tie-breaking, so equal
        # scores don't always come out in the same order.
        scored_windows = []
        for window in available_windows:
            score = score_window(window)
            tie_breaker = rng.random()
            scored_windows.append((score, tie_breaker, window))

        # Sort: highest score first, then random tie-breaker
        scored_windows.sort(key=lambda t: (-t[0], t[1]))

        # --- Greedy pick ---
        picked_windows_this_fairytale = []
        picked_positions = set()   # positions already used
        picked_token_sets = []     # content tokens of already-picked windows

        for score, tie_breaker, window in scored_windows:
            # Stop when we have enough paragraphs for this fairytale
            if len(picked_windows_this_fairytale) >= target_count:
                break

            # Check position overlap with already-picked windows
            window_positions = set()
            for s in window:
                window_positions.add((fairytale_num, s["_position"]))
            if window_positions & picked_positions:
                continue

            # Check near-duplicate content
            window_tokens = get_content_tokens(window)
            is_near_duplicate = False
            for previously_picked_tokens in picked_token_sets:
                similarity = jaccard(window_tokens, previously_picked_tokens)
                if similarity > NEAR_DUPLICATE_JACCARD:
                    is_near_duplicate = True
                    break
            if is_near_duplicate:
                continue

            # This window is OK to pick
            picked_windows_this_fairytale.append(window)
            picked_positions |= window_positions
            picked_token_sets.append(window_tokens)

        # Warn if we couldn't hit the target for this fairytale
        if len(picked_windows_this_fairytale) < target_count:
            print(f"  WARNING: Fairytale{fairytale_num} only yielded "
                  f"{len(picked_windows_this_fairytale)}/{target_count} paragraphs")

        # Add to the master list
        for window in picked_windows_this_fairytale:
            selected.append((fairytale_num, window))

    return selected


def write_paragraph_pool(selected, output_path):
    """Write paragraph_pool_15.tsv: the pool of selected paragraphs with
    metadata like their sentence IDs, combined text, and scoring info.
    """
    columns = ["paragraph_id", "fairytale", "sent_ids", "text",
               "total_length", "total_pron3", "total_propn",
               "total_complexity", "score"]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(columns)

        for i, (fairytale_num, window) in enumerate(selected, start=1):
            paragraph_id = f"para_{i:02d}"
            fairytale_name = f"Fairytale{fairytale_num}"

            # Join the 4 sentence IDs with " | " for readability
            sent_ids_joined = " | ".join(s["sent_id"] for s in window)

            # Join the 4 sentence texts with a single space
            combined_text = " ".join(s["text"] for s in window)

            # Sum the feature values
            total_length = sum(s["len"] for s in window)
            total_pron3 = sum(s["pron3"] for s in window)
            total_propn = sum(s["propn"] for s in window)
            total_complexity = sum(s["cscore"] for s in window)

            score = score_window(window)

            writer.writerow([paragraph_id, fairytale_name, sent_ids_joined,
                             combined_text, total_length, total_pron3,
                             total_propn, total_complexity, f"{score:.1f}"])


def write_translation_template(selected, glosses, output_path):
    """Write paragraph_translation_template.tsv: the file for the user to
    fill in Greek translations.

    Includes the Pharasiot text and the English glosses (concatenated per
    paragraph) so the user has both as reference.
    """
    columns = ["paragraph_id", "fairytale", "sent_ids", "pharasiot_text",
               "english_text", "greek_translation_FILL_IN"]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(columns)

        for i, (fairytale_num, window) in enumerate(selected, start=1):
            paragraph_id = f"para_{i:02d}"
            fairytale_name = f"Fairytale{fairytale_num}"

            sent_ids_joined = " | ".join(s["sent_id"] for s in window)
            pharasiot_text = " ".join(s["text"] for s in window)

            # Join the English glosses (or a placeholder if any is missing)
            english_parts = []
            for s in window:
                gloss = glosses.get(s["sent_id"], "[no gloss]")
                english_parts.append(gloss)
            english_text = " ".join(english_parts)

            writer.writerow([paragraph_id, fairytale_name, sent_ids_joined,
                             pharasiot_text, english_text, ""])
def main():
    if len(sys.argv) < 4:
        print("Usage: python3 03_select_paragraph_pool.py "
              "<sentences.tsv> <eval_set_40.tsv> <pharasiot.conllu>")
        sys.exit(1)

    sentences_path = sys.argv[1]
    eval_set_path = sys.argv[2]
    conllu_path = sys.argv[3]

    # Load everything we need
    print(f"Loading sentences from {sentences_path}...")
    sentences = load_sentences(sentences_path)
    print(f"  Loaded {len(sentences)} non-title sentences from F1-F6")

    print(f"Loading evaluation set from {eval_set_path}...")
    eval_ids = load_eval_ids(eval_set_path)
    print(f"  {len(eval_ids)} sentence IDs will be excluded")

    print(f"Loading English glosses from {conllu_path}...")
    glosses = load_english_glosses(conllu_path)
    print(f"  Loaded {len(glosses)} glosses")

    # Find candidate windows
    print(f"Finding candidate 4-sentence windows...")
    candidates = find_candidate_windows(sentences, eval_ids)
    for fairytale_num in sorted(candidates.keys()):
        print(f"  Fairytale{fairytale_num}: "
              f"{len(candidates[fairytale_num])} candidate windows")

    # Greedy selection
    print(f"Selecting paragraphs (targets: {TARGET_PER_FAIRYTALE})...")
    selected = select_paragraphs(candidates, TARGET_PER_FAIRYTALE)
    print(f"  Selected {len(selected)} paragraphs total")

    # Write outputs
    pool_path = "/mnt/user-data/outputs/paragraph_pool_15.tsv"
    print(f"Writing pool to {pool_path}...")
    write_paragraph_pool(selected, pool_path)

    template_path = "/mnt/user-data/outputs/paragraph_translation_template.tsv"
    print(f"Writing translation template to {template_path}...")
    write_translation_template(selected, glosses, template_path)

    # Final summary
    print()
    print("Selected paragraphs:")
    print(f"  {'id':<10} {'fairytale':<12} {'range':<30} "
          f"{'len':>4} {'pron3':>5} {'propn':>5} {'score':>7}")
    for i, (fairytale_num, window) in enumerate(selected, start=1):
        paragraph_id = f"para_{i:02d}"
        fairytale_name = f"Fairytale{fairytale_num}"
        range_str = f"{window[0]['sent_id']} .. {window[-1]['sent_id']}"

        total_length = sum(s["len"] for s in window)
        total_pron3 = sum(s["pron3"] for s in window)
        total_propn = sum(s["propn"] for s in window)
        score = score_window(window)

        print(f"  {paragraph_id:<10} {fairytale_name:<12} {range_str:<30} "
              f"{total_length:>4} {total_pron3:>5} {total_propn:>5} "
              f"{score:>7.1f}")

    print("Done.")


if __name__ == "__main__":
    main()
