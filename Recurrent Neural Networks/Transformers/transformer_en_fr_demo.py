import numpy as np
np.random.seed(42)
# ── Toy EN → FR corpus ──────────────────────────────────────────
raw_pairs = [
    (["I",   "love",  "you"],          ["je",   "t'aime"]),
    (["I",   "am",    "happy"],        ["je",   "suis",  "heureux"]),
    (["she", "is",    "kind"],         ["elle", "est",   "gentille"]),
    (["we",  "learn", "together"],     ["nous", "apprenons", "ensemble"]),
    (["he",  "reads", "books"],        ["il",   "lit",   "des", "livres"]),
    (["I",   "speak", "French"],       ["je",   "parle", "francais"]),
    (["you", "are",   "welcome"],      ["vous", "etes",  "bienvenus"]),
    (["she", "loves", "music"],        ["elle", "aime",  "la", "musique"]),
]


# vocabulary

def build_vocab(pairs, side, specials):
    words = set()
    for pair in pairs:
        for w in pair[side]:
            words.add(w)
    vocab = specials + sorted(words)
    return {w: i for i,w in enumerate(vocab)}, {i:w for i,w in enumerate(vocab)}

en_to_ix, ix_to_en = build_vocab(raw_pairs, 0, ["<PAD>", "<EOS>"])
fr_to_ix, ix_to_fr = build_vocab(raw_pairs, 1, ["<PAD>", "<SOS>", "<EOS>"])

EN_SIZE = len(en_to_ix)
FR_SIZE = len(fr_to_ix)

print("English words:", en_to_ix)
print("French words:", fr_to_ix)
print("EN vocab size:", EN_SIZE)
print("FR vocab size:", FR_SIZE)


# ── Data preparation ─────────────────────────────────────────────

def make_pair(en_words, fr_words):
    src = [en_to_ix[w] for w in en_words] + [en_to_ix["<EOS>"]]
    tgt_in  = [fr_to_ix["<SOS>"]] + [fr_to_ix[w] for w in fr_words]
    tgt_out = [fr_to_ix[w] for w in fr_words] + [fr_to_ix["<EOS>"]]
    return src, tgt_in, tgt_out
dataset = [make_pair(en, fr) for en, fr in raw_pairs]
# test
src, tgt_in, tgt_out = dataset[0]
print("src     :", src)
print("tgt_in  :", tgt_in)
print("tgt_out :", tgt_out)