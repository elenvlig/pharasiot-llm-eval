"""
extract_features.py
-------------------
Reads the Pharasiot treebank (a CoNLL-U file) and computes a set of
linguistic features for every sentence. These features are used later on
to select a balanced evaluation set (see select_sent_eval_set.py).

CoNLL-U is a standard file format for treebanks. Each sentence starts
with a few lines like `# sent_id = ...` and `# text = ...`, then has one
token per line with fields separated by tabs. Sentences are separated by
blank lines.

The features we compute per sentence:
  * length in tokens (excluding punctuation)
  * length band (short / medium / long)
  * number of dialect-specific words (lemmas the treebank writes with
    Latin letters, e.g. jαναβάρι, menjουλίσι, bά)
  * number of adpositions, proper nouns, pronouns, 3rd-person pronouns
  * number of negation markers
  * number of post-verbal subjects, objects, and adpositions
  * gender mismatch count (a Pharasiot-specific phenomenon)
  * syntactic complexity (parse-tree depth, subordinate clauses,
    coordinated clauses, relative clauses)
  * a combined complexity score and its band (simple / mid / complex)
  * register (direct speech vs. narrative) inferred from punctuation
  * the raw sentence text (Pharasiot)

Usage:
  python3 extract_features.py <pharasiot.conllu>

Output:
  sentences.tsv (in the current directory)
"""

import re
import sys


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sentences are grouped by length into three bands.
# These thresholds were chosen so each band contains roughly a third
# of the treebank.
SHORT_MAX_LENGTH = 9    # up to and including 9 tokens = "short"
MEDIUM_MAX_LENGTH = 15  # 10 to 15 tokens = "medium"; anything above = "long"

# Sentences are also grouped by syntactic complexity.
# The complexity score combines parse depth, subordinate clauses, and
# coordinated clauses. These thresholds group sentences into simple, mid,
# and complex bands.
SIMPLE_MAX_SCORE = 3    # 0 to 3 = "simple"
MID_MAX_SCORE = 6       # 4 to 6 = "mid"; 7 or more = "complex"


# ---------------------------------------------------------------------------
# Helper: read one sentence from the CoNLL-U file
# ---------------------------------------------------------------------------

def read_sentences_from_conllu(conllu_path):
    """Read the CoNLL-U file and yield one sentence at a time.

    Each yielded sentence is a dict with:
      sent_id : the sentence ID (e.g. "Fairytale1.7")
      text    : the raw Pharasiot text (from the "# text =" line)
      tokens  : a list of token dicts (one per word in the sentence)

    A token dict has these fields (only the ones we need):
      id      : integer position in the sentence (1-indexed)
      form    : the surface form (the word as it appears)
      lemma   : the base form of the word
      upos    : universal part-of-speech tag (e.g. NOUN, VERB, ADP)
      feats   : morphological features (e.g. Gender=Masc|Number=Sing)
      head    : the ID of this token's head in the parse tree (or 0 for root)
      deprel  : the dependency relation to the head (e.g. nsubj, obj)
    """
    with open(conllu_path, encoding="utf-8") as f:
        current_sent_id = None
        current_text = None
        current_tokens = []

        for line in f:
            # Remove trailing newlines and Windows-style \r
            line = line.rstrip("\r\n")

            # Comment lines start with "#" and give us metadata
            if line.startswith("#"):
                if line.startswith("# sent_id"):
                    # Format: "# sent_id = Fairytale1.7"
                    current_sent_id = line.split("=", 1)[1].strip()
                elif line.startswith("# text ="):
                    # Format: "# text = Some Pharasiot sentence."
                    current_text = line.split("=", 1)[1].strip()
                continue

            # A blank line marks the end of a sentence
            if line == "":
                if current_sent_id is not None:
                    yield {
                        "sent_id": current_sent_id,
                        "text": current_text or "",
                        "tokens": current_tokens,
                    }
                # Reset for the next sentence
                current_sent_id = None
                current_text = None
                current_tokens = []
                continue

            # Otherwise this line is a token. Split by tab.
            fields = line.split("\t")
            if len(fields) < 8:
                continue  # malformed line, skip it

            # We only care about main tokens, not multi-word ranges (e.g. "1-2")
            token_id = fields[0]
            if "-" in token_id or "." in token_id:
                continue

            # Build a small dict for this token
            token = {
                "id": int(token_id),
                "form": fields[1],
                "lemma": fields[2],
                "upos": fields[3],
                "feats": fields[5],
                "head": int(fields[6]) if fields[6] != "_" else 0,
                "deprel": fields[7],
            }
            current_tokens.append(token)

        # Don't forget the last sentence if the file doesn't end with a blank line
        if current_sent_id is not None:
            yield {
                "sent_id": current_sent_id,
                "text": current_text or "",
                "tokens": current_tokens,
            }


# ---------------------------------------------------------------------------
# Helper: get gender from a feats string
# ---------------------------------------------------------------------------

def get_gender(feats_string):
    """Extract the Gender value from a UD feats string, or None if absent.

    The feats field looks like "Case=Nom|Gender=Masc|Number=Sing".
    We split on "|" to get individual features, then look for one that
    starts with "Gender=".
    """
    if feats_string == "_" or not feats_string:
        return None

    parts = feats_string.split("|")
    for part in parts:
        if part.startswith("Gender="):
            return part.split("=", 1)[1]  # e.g. "Masc" or "Fem"
    return None


# ---------------------------------------------------------------------------
# Helper: compute parse-tree depth
# ---------------------------------------------------------------------------

def compute_tree_depth(tokens):
    """Return the maximum depth of the dependency parse tree.

    The root token has depth 1. Every other token's depth is one more
    than its head's depth. The overall tree depth is the maximum depth
    across all tokens.
    """
    if not tokens:
        return 0

    # Build a lookup from token ID to its head ID
    head_of = {}
    for token in tokens:
        head_of[token["id"]] = token["head"]

    # Depth of each token, computed by walking up to the root
    depths = {}
    max_depth = 0
    for token in tokens:
        current_id = token["id"]
        depth = 0
        seen = set()  # detect cycles just in case
        while current_id != 0 and current_id not in seen:
            seen.add(current_id)
            depth += 1
            current_id = head_of.get(current_id, 0)
        depths[token["id"]] = depth
        if depth > max_depth:
            max_depth = depth

    return max_depth


# ---------------------------------------------------------------------------
# Main feature extraction
# ---------------------------------------------------------------------------

def compute_features(sentence):
    """Compute all features for one sentence.

    Returns a dict with feature names as keys. See the module docstring
    for what each feature means.
    """
    tokens = sentence["tokens"]
    text = sentence["text"]

    # --- Basic length ---
    # Count only non-punctuation tokens as "words"
    word_count = 0
    for t in tokens:
        if t["upos"] != "PUNCT":
            word_count += 1

    # --- Length band ---
    if word_count <= SHORT_MAX_LENGTH:
        length_band = "short"
    elif word_count <= MEDIUM_MAX_LENGTH:
        length_band = "medium"
    else:
        length_band = "long"

    # --- POS-based counts ---
    # We tally the number of each interesting UPOS tag
    borrowing_count = 0
    adposition_count = 0
    propn_count = 0     # proper nouns
    pron_count = 0      # all pronouns
    pron3_count = 0     # 3rd-person pronouns specifically
    negation_count = 0

    # And the post-verbal counts
    postposed_subject_count = 0
    postposed_object_count = 0
    postposed_adposition_count = 0

    # Gender mismatch: when a modifier and its head have different genders
    gender_mismatch_count = 0

    # We'll also need to find each verb's position for post-verbal checks
    verb_positions = set()
    for t in tokens:
        if t["upos"] == "VERB":
            verb_positions.add(t["id"])

    # Build a quick lookup for token by ID (for gender-mismatch checks)
    token_by_id = {}
    for t in tokens:
        token_by_id[t["id"]] = t

    # Now walk through every token
    for t in tokens:
        lemma = t["lemma"].lower() if t["lemma"] != "_" else ""
        upos = t["upos"]
        tok_id = t["id"]
        head_id = t["head"]

        # Dialect-specific words: the treebank writes their lemmas with
        # Latin letters (e.g. jαναβάρι, menjουλίσι, bά), so count those
        if re.search(r"[A-Za-z]", lemma):
            borrowing_count += 1

        # Count by part of speech
        if upos == "ADP":
            adposition_count += 1
            # Check for postposed adposition (adp positioned after its head)
            if head_id > tok_id:
                postposed_adposition_count += 1

        elif upos == "PROPN":
            propn_count += 1

        elif upos == "PRON":
            pron_count += 1
            # 3rd-person pronoun? Check the feats string
            if "Person=3" in t["feats"]:
                pron3_count += 1

        # Negation is usually marked by the "Polarity=Neg" feature
        if "Polarity=Neg" in t["feats"]:
            negation_count += 1

        # Post-verbal subject: is this a subject whose verb head is before it?
        if t["deprel"] in ("nsubj", "csubj"):
            if head_id in verb_positions and head_id < tok_id:
                postposed_subject_count += 1

        # Post-verbal object: is this an object whose verb head is before it?
        if t["deprel"] in ("obj", "iobj", "ccomp"):
            if head_id in verb_positions and head_id < tok_id:
                postposed_object_count += 1

        # Gender mismatch: check if this token's gender disagrees with its head
        my_gender = get_gender(t["feats"])
        if my_gender is not None and head_id in token_by_id:
            head_token = token_by_id[head_id]
            head_gender = get_gender(head_token["feats"])
            if head_gender is not None and head_gender != my_gender:
                # Only flag if it's a modifier-head relationship
                if t["deprel"] in ("amod", "det", "nummod"):
                    gender_mismatch_count += 1

    # --- Syntactic complexity ---
    tree_depth = compute_tree_depth(tokens)

    # Count subordinate clauses (advcl, ccomp, xcomp, acl, csubj)
    subordinate_count = 0
    for t in tokens:
        if t["deprel"] in ("advcl", "ccomp", "xcomp", "acl", "csubj"):
            subordinate_count += 1

    # Count coordinated clauses (conj at the sentence level)
    coordination_count = 0
    for t in tokens:
        if t["deprel"] == "conj":
            coordination_count += 1

    # Count relative clauses (acl:relcl)
    relative_clause_count = 0
    for t in tokens:
        if t["deprel"] == "acl:relcl" or t["deprel"] == "acl":
            relative_clause_count += 1

    # Combined complexity score. Weights are simple ints and were chosen
    # so that no single feature dominates.
    complexity_score = tree_depth + subordinate_count + coordination_count

    # Complexity band
    if complexity_score <= SIMPLE_MAX_SCORE:
        complexity_band = "simple"
    elif complexity_score <= MID_MAX_SCORE:
        complexity_band = "mid"
    else:
        complexity_band = "complex"

    # --- Register (direct speech vs. narrative) ---
    # Sentences with quote marks or exclamation marks tend to be direct speech.
    # This is a rough heuristic but works well enough for our purposes.
    is_direct_speech = 0
    if any(ch in text for ch in ["«", "»", '"', "!"]):
        is_direct_speech = 1
    register = "direct" if is_direct_speech else "narrative"

    # --- Source of the sentence ---
    # e.g. "Fairytale1.7" -> the source is "Fairytale"
    sent_id = sentence["sent_id"]
    source = sent_id.rstrip("0123456789.").rstrip(".")
    # If the sentence ID has a title marker (like "1.0Title"), strip that
    if source.endswith("Title"):
        source = source[:-5]

    return {
        "sent_id": sent_id,
        "src": source,
        "len": word_count,
        "band": length_band,
        "borrow": borrowing_count,
        "adp": adposition_count,
        "propn": propn_count,
        "pron": pron_count,
        "pron3": pron3_count,
        "neg": negation_count,
        "postsubj": postposed_subject_count,
        "postobj": postposed_object_count,
        "postadp": postposed_adposition_count,
        "gendermis": gender_mismatch_count,
        "depth": tree_depth,
        "n_sub": subordinate_count,
        "n_conj": coordination_count,
        "relcl": relative_clause_count,
        "cscore": complexity_score,
        "register": register,
        "is_direct": is_direct_speech,
        "text": text,
        "cband": complexity_band,
    }


# ---------------------------------------------------------------------------
# Write results to TSV
# ---------------------------------------------------------------------------

def write_output(all_features, output_path):
    """Write the list of feature dicts to a TSV file."""
    # The column order for the output file
    columns = ["sent_id", "src", "len", "band", "borrow", "adp", "propn",
               "pron", "pron3", "neg", "postsubj", "postobj", "postadp",
               "gendermis", "depth", "n_sub", "n_conj", "relcl", "cscore",
               "register", "is_direct", "text", "cband"]

    with open(output_path, "w", encoding="utf-8") as f:
        # Write the header row
        f.write("\t".join(columns) + "\n")

        # Write one row per sentence
        for row in all_features:
            values = []
            for col in columns:
                values.append(str(row[col]))
            f.write("\t".join(values) + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 extract_features.py <pharasiot.conllu>")
        sys.exit(1)

    conllu_path = sys.argv[1]

    print(f"Reading treebank from {conllu_path}...")
    all_features = []
    for sentence in read_sentences_from_conllu(conllu_path):
        features = compute_features(sentence)
        all_features.append(features)

    print(f"  Processed {len(all_features)} sentences")

    output_path = "sentences.tsv"
    print(f"Writing features to {output_path}...")
    write_output(all_features, output_path)

    print("Done.")


if __name__ == "__main__":
    main()
