
import csv
import os
import re

APOS = {"’": "'", "‘": "'", "ʼ": "'", "΄": "'", "´": "'", "`": "'", "́": "'"}
OUT_DIR = "sentence_word_run"


def fold(s):
    """Same length as s, with every apostrophe variant written as '."""
    return "".join(APOS.get(c, c) for c in s)


def read_conllu(path):
    sents, sid = {}, None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\r\n")
        if line.startswith("# sent_id"):
            sid = line.split("=", 1)[1].strip(); sents[sid] = {"text": "", "toks": []}
        elif sid and line.startswith("# text ="):
            sents[sid]["text"] = line.split("=", 1)[1].strip()
        elif sid and line and not line.startswith("#"):
            p = line.split("\t")
            if len(p) >= 4 and "-" not in p[0]:
                sents[sid]["toks"].append((p[1], p[2], p[3]))
    return sents


def read_marks(*paths):
    marks = {}
    for path in paths:
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                marks[r["word_form"].lower()] = r["shared"].strip()
    return marks


def locate(text, toks, offset=0):
    """Find each treebank token in text, left to right. Returns (spans, failures)."""
    t = fold(text)
    cursor, spans, failed = 0, [], []
    for form, lemma, upos in toks:
        i = t.find(fold(form), cursor)
        if i < 0:
            failed.append(form); spans.append(None); continue
        spans.append((i + offset, i + len(form) + offset, form, lemma, upos))
        cursor = i + len(form)
    return spans, failed


GREEK_WORD = re.compile(r"\w+(?:['’‘΄]\w+)*")


def plain_words(text, offset=0):
    """Runs of letters, for text with no treebank tokens."""
    return [(m.start() + offset, m.end() + offset, m.group(), "", "")
            for m in GREEK_WORD.finditer(text)]


treebank = read_conllu("pharasiot.conllu")
marks = read_marks("lexical_density_words_final.tsv",
                   "lexical_density_words_passages_and_options_marked.tsv")
items = list(csv.DictReader(open("crossentropy_sentences.tsv", encoding="utf-8-sig"),
                            delimiter="\t"))

seqs, spans_out, problems, notes = [], [], [], []


def add_spans(seq_id, variety, spans, sentence_id):
    for k, s in enumerate(spans):
        if s is None:
            continue
        start, end, form, lemma, upos = s
        if upos == "PUNCT":
            continue                      # punctuation is not a word
        spans_out.append({
            "seq_id": seq_id, "variety": variety, "word_index": k, "word": form,
            "lemma": lemma, "upos": upos, "char_start": start, "char_end": end,
            "segment": "sentence", "sentence_id": sentence_id,
            "shared": marks.get(form.lower(), "") if variety == "pharasiot" else "",
        })


for it in items:
    sid = it["text_id"]
    for variety in ("pharasiot", "greek"):
        text = it[variety + "_text"]
        seq_id = f"{sid}|{variety}"
        seqs.append({"seq_id": seq_id, "kind": "sentence", "text_id": sid, "option": "",
                     "variety": variety, "scored_from": 0, "text": text})
        if variety == "pharasiot":
            if sid in treebank:
                spans, failed = locate(text, treebank[sid]["toks"])
                if failed:
                    problems.append(f"{seq_id}: not found {failed}")
            else:
                # not in the treebank; fall back to letter runs, so the sentence is
                # still scored. Only the lemma and part of speech are lost; the
                # density mark attaches by word form as usual.
                notes.append(f"{sid} is not in the treebank; its words come from the text itself")
                spans = plain_words(text)
            add_spans(seq_id, variety, spans, sid)
        else:
            add_spans(seq_id, variety, plain_words(text), sid)

# ---- checks: every word must sit where it says ----
bad = 0
by_seq = {s["seq_id"]: s["text"] for s in seqs}
for r in spans_out:
    seen = fold(by_seq[r["seq_id"]][r["char_start"]:r["char_end"]])
    if seen != fold(r["word"]):
        bad += 1
        if bad <= 8:
            problems.append(f"span mismatch {r['seq_id']} {r['word']!r} vs {seen!r}")

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "crossentropy_words_sequences.tsv"), "w",
          encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["seq_id", "kind", "text_id", "option", "variety",
                                      "scored_from", "text"], delimiter="\t")
    w.writeheader(); w.writerows(seqs)
cols = ["seq_id", "variety", "word_index", "word", "lemma", "upos", "char_start",
        "char_end", "segment", "sentence_id", "shared"]
with open(os.path.join(OUT_DIR, "crossentropy_words_spans.tsv"), "w",
          encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
    w.writeheader(); w.writerows(spans_out)

ph = [r for r in spans_out if r["variety"] == "pharasiot"]
counted = sum(1 for r in ph if r["shared"] in ("yes", "no"))
print("sequences:", len(seqs), " words:", len(spans_out), " span mismatches:", bad)
print("Pharasiot words:", len(ph),
      "| recoverable:", sum(1 for r in ph if r["shared"] == "yes"),
      "| not recoverable:", sum(1 for r in ph if r["shared"] == "no"),
      "| function words:", sum(1 for r in ph if r["shared"] == "skip"),
      "| unjudged:", len(ph) - counted - sum(1 for r in ph if r["shared"] == "skip"))
for n in notes:
    print("note:", n)
for p in problems[:25]:
    print("problem:", p)
