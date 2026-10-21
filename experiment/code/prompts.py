"""Prompt construction, example retrieval and SQL extraction (PLAN section 5). Stdlib only.

One function builds every prompt so that all models, sets and conditions share it:
    messages_common(schema, question, evidence=None, shots=())   -> /api/chat messages
    messages_xiyan_native(template, schema, question, evidence)  -> single user message
Retrieval:
    TfIdf(pool).top_k(question, k=3)      S-sim (5.3): lower-case [a-z0-9]+, IDF over the fold pool
    fixed_shots(pool, seed=0)             S-fixed (5.3): one instance from each of 3 templates
Extraction (5.5):
    extract_sql(text) -> (sql or "", rule)
"""
import math
import random
import re
from collections import Counter

SYSTEM_TEMPLATE = (
    "You are a SQLite expert. Given the database schema below, write one SQLite query that answers "
    "the user's question.\nReturn only the SQL query, with no explanation.\n\n{schema}"
)


def user_turn(question, evidence=None):
    hint_line = f"Hint: {evidence}\n" if evidence else ""
    return f"{hint_line}Question: {question}"


def messages_common(schema, question, evidence=None, shots=()):
    """Zero-shot when shots is empty; otherwise shots are (question, sql) pairs, least similar first."""
    msgs = [{"role": "system", "content": SYSTEM_TEMPLATE.format(schema=schema.strip())}]
    for q, sql in shots:
        msgs.append({"role": "user", "content": user_turn(q)})
        msgs.append({"role": "assistant", "content": sql})
    msgs.append({"role": "user", "content": user_turn(question, evidence)})
    return msgs


def messages_xiyan_native(template, schema, question, evidence=None):
    """XiYan nl2sqlite_template_cn, copied verbatim from the model card (PLAN 5.4)."""
    prompt = template.format(dialect="SQLite", db_schema=schema.strip(), question=question,
                             evidence=evidence or "")
    return [{"role": "user", "content": prompt}]


# ------------------------------------------------------------------ retrieval

TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text):
    return TOKEN.findall(text.lower())


class TfIdf:
    """TF-IDF cosine retrieval. tf = raw count; idf = ln(N / df) + 1 computed over the pool
    (terms absent from the pool get weight 0). Ties are broken by item id (ascending)."""

    def __init__(self, pool):
        self.pool = sorted(pool, key=lambda e: e["id"])
        n = len(self.pool)
        df = Counter()
        self.tfs = []
        for e in self.pool:
            tf = Counter(tokens(e["question"]))
            self.tfs.append(tf)
            df.update(tf.keys())
        self.idf = {t: math.log(n / d) + 1.0 for t, d in df.items()}
        self.vecs = [self._vec(tf) for tf in self.tfs]

    def _vec(self, tf):
        v = {t: c * self.idf[t] for t, c in tf.items() if t in self.idf}
        norm = math.sqrt(sum(x * x for x in v.values()))
        return {t: x / norm for t, x in v.items()} if norm else {}

    def top_k(self, question, k=3):
        q = self._vec(Counter(tokens(question)))
        scored = []
        for e, v in zip(self.pool, self.vecs):
            s = sum(w * v.get(t, 0.0) for t, w in q.items())
            scored.append((-s, e["id"], e, s))
        scored.sort(key=lambda x: (x[0], x[1]))
        best = [(x[2], x[3]) for x in scored[:k]]
        return list(reversed(best))  # increasing similarity: most similar sits next to the question


def fixed_shots(pool, seed=0):
    """One instance from each of three distinct templates of the fold pool, random.Random(seed)."""
    rng = random.Random(seed)
    templates = sorted({e["template_id"] for e in pool})
    chosen = rng.sample(templates, 3)
    out = []
    for t in chosen:
        cands = sorted([e for e in pool if e["template_id"] == t], key=lambda e: e["id"])
        out.append(rng.choice(cands))
    return out


# ------------------------------------------------------------------ extraction

THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
FENCE = re.compile(r"```(sqlite|sql)?[ \t]*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
START = re.compile(r"\b(SELECT|WITH)\b", re.IGNORECASE)


def _first_statement(text):
    """Text up to and including the first ';' outside quotes (or all of it)."""
    quote = None
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"', "`"):
            quote = ch
        elif ch == ";":
            return text[: i + 1]
    return text


def extract_sql(text):
    """PLAN 5.5 rules, in order. Returns (sql, rule); sql == "" means "no SQL"."""
    t = THINK.sub("", text or "")
    m = FENCE.search(t)
    if m:
        body = m.group(2).strip()
        if body:
            return body, "fence"
    # an unterminated fence (e.g. cut off by the token cap): drop the opening fence marker
    t2 = re.sub(r"```(sqlite|sql)?", "", t, flags=re.IGNORECASE)
    s = START.search(t2)
    if s:
        return _first_statement(t2[s.start():]).strip(), "keyword"
    return "", "none"
