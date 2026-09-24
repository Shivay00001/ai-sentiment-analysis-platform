"""Real VADER-style lexicon sentiment analyzer. Standard library only.

Honest approach: a curated valence lexicon plus the documented VADER
heuristics (negations, intensifiers, punctuation emphasis, ALL-CAPS emphasis,
"but"-clause reweighting, compound normalization). No ML model, no randomness:
identical input always gives identical output.
"""

import math
import re

# Curated valence lexicon: word -> valence in [-4, +4].
LEXICON = {
    # strongly positive
    "excellent": 3.1, "amazing": 2.9, "awesome": 2.8, "fantastic": 2.9,
    "wonderful": 2.7, "brilliant": 2.8, "outstanding": 2.9, "superb": 2.9,
    "perfect": 3.0, "love": 3.0, "loved": 3.0, "loves": 3.0, "adore": 2.9,
    "thrilled": 2.9, "delighted": 2.8, "ecstatic": 3.0, "phenomenal": 2.9,
    # positive
    "good": 1.9, "great": 2.6, "nice": 1.8, "happy": 2.2, "glad": 1.9,
    "pleased": 2.0, "satisfied": 2.0, "enjoy": 2.1, "enjoyed": 2.2,
    "impressive": 2.3, "smooth": 1.5, "fast": 1.2, "easy": 1.6, "helpful": 2.0,
    "reliable": 1.9, "solid": 1.4, "clean": 1.3, "beautiful": 2.4,
    "recommend": 2.2, "worth": 1.6, "win": 2.0, "won": 2.0, "success": 2.3,
    "successful": 2.4, "best": 2.8, "better": 1.8, "improved": 1.9,
    "improvement": 1.8, "efficient": 1.8, "accurate": 1.7, "thank": 1.8,
    "thanks": 1.9, "appreciate": 2.0, "like": 1.7, "liked": 1.8, "likes": 1.7,
    "fun": 2.0, "excited": 2.3, "hopeful": 1.8, "optimistic": 2.0,
    "confident": 1.9, "proud": 1.8, "kind": 1.7, "friendly": 2.0,
    "generous": 1.9, "brave": 1.6, "smart": 1.8, "clever": 1.7,
    # strongly negative
    "terrible": -3.0, "awful": -2.9, "horrible": -2.9, "dreadful": -2.8,
    "atrocious": -3.0, "hate": -2.9, "hated": -2.9, "hates": -2.9,
    "disgusting": -2.8, "appalling": -2.8, "worst": -3.0, "pathetic": -2.6,
    "furious": -2.8, "livid": -2.9, "devastated": -2.8,
    # negative
    "bad": -2.1, "poor": -2.0, "sad": -2.0, "angry": -2.3, "annoyed": -1.9,
    "frustrated": -2.1, "frustrating": -2.1, "disappointed": -2.2,
    "disappointing": -2.1, "disappointment": -2.2, "upset": -2.0,
    "worried": -1.8, "anxious": -1.9, "afraid": -1.9, "scared": -2.0,
    "slow": -1.4, "broken": -2.3, "buggy": -2.0, "bug": -1.6, "bugs": -1.8,
    "crash": -2.2, "crashed": -2.2, "crashes": -2.2, "error": -1.9,
    "errors": -2.0, "fail": -2.3, "failed": -2.4, "fails": -2.3,
    "failure": -2.4, "useless": -2.4, "worthless": -2.6, "waste": -2.2,
    "regret": -2.1, "sorry": -1.4, "problem": -1.7, "problems": -1.8,
    "issue": -1.5, "issues": -1.6, "difficult": -1.5, "hard": -1.2,
    "confusing": -1.7, "confused": -1.6, "wrong": -1.9, "incorrect": -1.8,
    "stupid": -2.4, "dumb": -2.1, "ugly": -2.0, "boring": -1.8,
    "mediocre": -1.6, "weak": -1.6, "worse": -2.3, "decline": -1.8,
    "loss": -2.0, "lost": -1.9, "lose": -1.9, "risk": -1.5, "risky": -1.7,
    "expensive": -1.5, "overpriced": -1.9, "cheap": -0.8, "rude": -2.0,
    "unfair": -1.9, "never": -1.0, "complaint": -1.8, "refund": -0.6,
    "cancel": -1.6, "cancelled": -1.7, "avoid": -1.7, "warning": -1.3,
    "terribly": -2.4, "horribly": -2.4, "miserable": -2.5, "painful": -2.2,
    "sucks": -2.2, "sucked": -2.2, "garbage": -2.5, "trash": -2.3,
    "nightmare": -2.6, "disaster": -2.7, "catastrophe": -2.8,
    # negation-adjacent / neutral-ish
    "ok": 0.8, "okay": 0.8, "fine": 0.9, "decent": 1.0, "average": 0.2,
    "meh": -0.5, "whatever": -0.4,
}

NEGATIONS = {
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nowhere",
    "hardly", "scarcely", "barely", "dont", "don't", "doesnt", "doesn't",
    "didnt", "didn't", "isnt", "isn't", "arent", "aren't", "wasnt", "wasn't",
    "werent", "weren't", "cant", "can't", "couldnt", "couldn't", "wont",
    "won't", "wouldnt", "wouldn't", "shouldnt", "shouldn't", "aint", "ain't",
}

BOOSTERS = {
    "very": 0.3, "really": 0.3, "extremely": 0.5, "incredibly": 0.5,
    "absolutely": 0.4, "totally": 0.3, "completely": 0.4, "utterly": 0.4,
    "highly": 0.3, "so": 0.25, "quite": 0.2, "pretty": 0.2, "fairly": 0.15,
    "deeply": 0.3, "truly": 0.25, "especially": 0.25, "remarkably": 0.35,
    "exceptionally": 0.4,
}
DAMPENERS = {
    "slightly": -0.3, "somewhat": -0.3, "kind": -0.2, "sort": -0.2,
    "a bit": -0.25, "little": -0.25, "barely": -0.4, "hardly": -0.4,
    "almost": -0.2, "marginally": -0.3,
}

WORD_RE = re.compile(r"[a-zA-Z']+|[!?]+")


def _is_negated(words, i):
    # look back up to 3 tokens for a negation
    for j in range(max(0, i - 3), i):
        if words[j] in NEGATIONS:
            return True
    return False


def analyze(text: str) -> dict:
    """Return {neg, neu, pos, compound, label} for the text."""
    if not text or not text.strip():
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0, "label": "neutral"}

    tokens = WORD_RE.findall(text)
    words = [t.lower() for t in tokens if t.isalpha() or "'" in t]

    sentiments = []
    for i, w in enumerate(words):
        if w in ("kind", "sort") and i + 1 < len(words) and words[i + 1] == "of":
            continue  # "kind of" / "sort of" dampener handled below
        valence = LEXICON.get(w, 0.0)
        if valence == 0.0:
            continue
        # booster / dampener from previous token
        if i > 0:
            prev = words[i - 1]
            if prev in BOOSTERS:
                valence += math.copysign(BOOSTERS[prev], valence)
            elif prev in DAMPENERS:
                valence += math.copysign(DAMPENERS[prev], valence)
            elif prev in ("kind", "sort") and i > 1 and words[i - 2] != "of":
                pass
            if i > 1 and words[i - 2] in ("kind", "sort") and words[i - 1] == "of":
                valence += math.copysign(0.25, valence) * -1  # "kind of good" dampens
        # ALL CAPS emphasis
        if tokens and i < len(tokens) and tokens[i].isupper() and len(tokens[i]) > 1:
            valence += math.copysign(0.5, valence)
        # negation flips
        if _is_negated(words, i):
            valence *= -0.74
        sentiments.append(valence)

    # "but" reweighting: sentiment after "but" counts 1.5x, before counts 0.5x
    lowered = text.lower()
    if " but " in f" {lowered} " and sentiments:
        # approximate: split token stream at first "but"
        try:
            but_idx = words.index("but")
            # map sentiment entries to word positions (only lexicon words counted)
            lex_positions = [i for i, w in enumerate(words) if LEXICON.get(w, 0.0) != 0.0]
            weighted = []
            for pos, s in zip(lex_positions, sentiments):
                weighted.append(s * (1.5 if pos > but_idx else 0.5))
            sentiments = weighted
        except ValueError:
            pass

    # punctuation emphasis
    punct_boost = min(text.count("!"), 4) * 0.3 - min(text.count("?") , 4) * 0.0
    if sentiments:
        sentiments[-1] += math.copysign(punct_boost, sentiments[-1]) if punct_boost else sentiments[-1]

    total = sum(sentiments)
    # normalize like VADER's compound
    compound = total / math.sqrt(total * total + 15) if sentiments else 0.0
    compound = max(-1.0, min(1.0, compound))

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    # pos/neu/neg proportions over lexicon hits
    pos_s = sum(s for s in sentiments if s > 0)
    neg_s = sum(-s for s in sentiments if s < 0)
    denom = pos_s + neg_s + 1e-9
    if not sentiments:
        pos, neg, neu = 0.0, 0.0, 1.0
    else:
        pos = round(pos_s / denom, 3)
        neg = round(neg_s / denom, 3)
        neu = round(max(0.0, 1.0 - pos - neg), 3)

    return {"neg": neg, "neu": neu, "pos": pos,
            "compound": round(compound, 4), "label": label}
