# Curated (AI-drafted) set: drafting record (PLAN 3.3, D12)

Label: **curated (AI-drafted)**. Never "hand-written". Status on 2026-10-16: **draft, not frozen**. It is not hashed into `experiment/config/data.lock.json` until the team's spot-check (`experiment/curated_spotcheck.md`) is signed off.

## Who and what drafted it
- **Questions and gold A:** the doer agent (Claude Code, model `claude-opus-5-5`), 2026-10-16. The 80 questions were written first, as a file containing no SQL (`drafts/questions.jsonl`, sha256 `125886046b827493fb8aed64757bd0e8b4dd4ece7a702554565a5e19cb1922ae`). Gold A (`drafts/goldA.jsonl`) was written after that. The coverage grid has 10 categories with 8 questions each (PLAN 3.3 step 1). Values such as zone names and payment types were taken from a look at the database's value distributions. Template wording was avoided.
- **Gold B (independent):** a separate, non-interactive Claude Code session (`claude -p --tools "" --no-session-persistence`, CLI 2.1.301, model `claude-opus-5-5`, session id `c761078c-7bed-46cf-9527-948cb8279916`, 2026-10-16). It ran from an empty scratch directory and had no tools and no access to gold A or the project files. Its whole input is `drafts/goldB_prompt.txt` (sha256 `19c1cdbdf9f04a981ea9f1052c27a15e98ecd030045d4ba1624dda526957d8fc`): the schema, output conventions and the 95 questions (80 new plus 15 seed). Its raw response is kept in `drafts/goldB_session_response.json`, and the extracted queries are in `drafts/goldB.jsonl`.
- **Seed items:** the 15 exploratory items in `experiment/code/handwritten.jsonl`. Their existing SQL serves as gold A. They went through the same gold-B and automatic checks and are flagged `seed: true`.

## Checks (`experiment/code/curate_check.py`; output `curated_check.json`)
- Both gold queries are executed with `eval_exec.py`'s executor and normalisation, and an item is accepted only if the results are equal (ordered comparison when `ordered`).
- Each query must run in under 10 s, and the result must be non-empty and non-degenerate. A single 0 answer counts as degenerate for curated items.
- A rank-tie check at the `LIMIT` cut is applied to every top-k or arg-max item.
- Each item gets a structure tag: `template-like` if its masked SQL skeleton equals that of one of the 26 templates, else `novel`.

## Outcome
- **95 candidates** (80 new plus 15 seed). Gold A and gold B returned the same result on all 95.
- **92 accepted** (77 new plus 15 seed). Three were rejected as degenerate, because the answer is 0: `cur_066`, `cur_068`, `cur_077`.
- **Structure tags:** 2 template-like, 90 novel.

## Limitations the team should know
- Gold A and gold B come from the same model family in two separate sessions. The two queries are textually identical (up to whitespace and case) for 40 of 95 items. Their agreement shows the SQL is unambiguous to that model. It is weaker evidence than two independent human authors. The team spot-check of 10 items (D12, reduced from 20 at the team's request on 2026-10-16) is the human check.
- Gold B's prompt stated output conventions: column order as mentioned, names rather than IDs, a 0-100 percentage scale and no rounding. Model prompts do not state these conventions. They are reflected in the question wording where it matters.
