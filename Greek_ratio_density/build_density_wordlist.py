"""
Builds the word list for the lexical density annotation.

Reads the 44 evaluation sentences, pulls every word from the treebank, and
deduplicates by word form. You fill in the `shared` column:

    yes   recoverable from Standard Greek
    no    not recoverable
    skip  not part of the question (articles, conjunctions, punctuation...)

Reads   pharasiot.conllu
        crossentropy_sentences.tsv    the 44 sentence ids
Writes  lexical_density_words.tsv

Run from the folder holding both:  python build_density_wordlist.py
"""

import csv
from collections import Counter


CONLLU_FILE = "pharasiot.conllu"
SENTENCES_FILE = "crossentropy_sentences.tsv"
OUTPUT_FILE = "lexical_density_words.tsv"

# punctuation is not a word, so it is left out of the list entirely
DROP_TAGS = ["PUNCT"]

# tags that are almost always function words. The script suggests `skip` for
# these, but the suggestion is only a starting point - change any of them.
FUNCTION_TAGS = ["DET", "ADP", "CCONJ", "SCONJ", "AUX", "PART"]


def read_conllu(path):
    """Read the treebank into  sent_id -> list of (form, lemma, upos)."""
    sentences = {}
    sent_id = None
    words = []

    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\r\n")

        # a few lines in the treebank are missing the space after "=",
        # so split on the "=" rather than matching a fixed prefix
        if line.startswith("# sent_id"):
            sent_id = line.split("=", 1)[1].strip()
            words = []
            continue

        if line == "":
            if sent_id and words:
                sentences[sent_id] = words
            sent_id = None
            continue

        if line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) < 4:
            continue

        # skip multiword ranges like "1-2"
        if "-" in parts[0]:
            continue

        words.append((parts[1], parts[2], parts[3]))

    # the last sentence has no blank line after it
    if sent_id and words:
        sentences[sent_id] = words

    return sentences


def read_evaluation_sentences(path):
    """Read the 44 sentence ids and their two texts."""
    sentences = {}

    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sentences[row["text_id"]] = (row["pharasiot_text"], row["greek_text"])

    return sentences


def main():
    treebank = read_conllu(CONLLU_FILE)
    evaluation = read_evaluation_sentences(SENTENCES_FILE)

    print("treebank sentences: {}".format(len(treebank)))
    print("evaluation sentences: {}".format(len(evaluation)))

    missing = [s for s in evaluation if s not in treebank]
    if missing:
        print("\n  NOT FOUND IN THE TREEBANK: {}".format(missing))

    # collect every word form, with its most common lemma and tag
    forms = {}
    counts = Counter()

    for sent_id in evaluation:
        if sent_id not in treebank:
            continue

        for form, lemma, upos in treebank[sent_id]:
            if upos in DROP_TAGS:
                continue

            key = form.lower()
            counts[key] += 1

            if key not in forms:
                forms[key] = {
                    "word_form": form,
                    "lemma": lemma,
                    "upos": upos,
                    "example_pharasiot": evaluation[sent_id][0],
                    "example_greek": evaluation[sent_id][1],
                }

    print("\nunique word forms: {}".format(len(forms)))
    print("total word tokens: {}".format(sum(counts.values())))

    # write, most frequent first, so the function words cluster at the top
    columns = ["word_form", "lemma", "upos", "n", "shared",
               "example_pharasiot", "example_greek"]

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()

        for key, _ in counts.most_common():
            row = forms[key]

            # suggest skip for the tags that are usually function words
            if row["upos"] in FUNCTION_TAGS:
                suggestion = "skip"
            else:
                suggestion = ""

            writer.writerow({
                "word_form": row["word_form"],
                "lemma": row["lemma"],
                "upos": row["upos"],
                "n": counts[key],
                "shared": suggestion,
                "example_pharasiot": row["example_pharasiot"],
                "example_greek": row["example_greek"],
            })

    print("\nwrote {}".format(OUTPUT_FILE))

    # how much is left to do
    by_tag = Counter(forms[k]["upos"] for k in forms)
    to_annotate = sum(n for tag, n in by_tag.items() if tag not in FUNCTION_TAGS)

    print("\nwords by tag:")
    for tag, n in by_tag.most_common():
        note = "  (suggested skip)" if tag in FUNCTION_TAGS else ""
        print("  {:8} {:4}{}".format(tag, n, note))

    print("\n  pre-filled as skip: {}".format(len(forms) - to_annotate))
    print("  left for you:       {}".format(to_annotate))


main()
