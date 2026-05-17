"""Hallucination comparison: vanilla LLM vs MediRAG deployed pipeline.

For each gold question:
  1. Retrieve top-k chunks via the production retrieval stack.
     (The same chunks are used to score both arms — this is the
     "fair fact-checking corpus" for the comparison.)
  2. Generate a *vanilla* answer with the same Groq Llama 3.3 70B
     model but a generic system prompt and NO retrieved context.
  3. Generate a *MediRAG* answer with the production stack
     (system prompt + retrieved sources in user turn).
  4. Score both answers sentence-by-sentence against the retrieved
     chunks using the production DeBERTa-v3 NLI model. Record
     EVERY claim-eligible sentence's max entailment probability,
     not just the failures, so we can plot the distribution.

Output JSON contains:
  - per-sentence entailment scores for both arms
  - aggregate hallucination rate per 1000 sentences
  - distribution-band shares (redact / soften / ship)
  - per-case timing

Usage:
  python eval/score_hallucination_compare.py --gold eval/gold/coverage.jsonl --limit 15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

NLI_REDACT_BELOW = 0.20
NLI_SOFTEN_BELOW = 0.50  # below this is the hallucination/refusal threshold

VANILLA_SYSTEM_PROMPT = (
    "You are a helpful medical-information assistant. Answer the user's "
    "question clearly and concisely in plain English. Keep the answer "
    "under 250 words."
)


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("rate limit", "quota", "429", "tpd", "tpm"))


def _generate_vanilla(question: str, RAG_mod) -> tuple[str, str]:
    """No retrieval, generic system prompt — closest fair vanilla baseline
    to what an unsupervised user would get from the same Llama 3.3 70B."""
    messages = [
        {"role": "system", "content": VANILLA_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if RAG_mod.groq_client is not None:
        try:
            resp = RAG_mod.groq_client.chat.completions.create(
                model=RAG_mod.GROQ_MODEL,
                messages=messages,
                max_tokens=250,
                temperature=RAG_mod.MAIN_QUERY_TEMPERATURE,
            )
            return resp.choices[0].message.content or "", "groq"
        except Exception as exc:
            if _is_quota_error(exc):
                raise
            print(f"[vanilla][groq] failed, fallback to Cohere: {exc}")
    resp = RAG_mod.co.chat(
        model="command-r-08-2024",
        messages=messages,
        max_tokens=250,
        temperature=RAG_mod.MAIN_QUERY_TEMPERATURE,
    )
    return resp.message.content[0].text or "", "cohere_fallback"


def _generate_medirag(question: str, context_text: str, RAG_mod) -> tuple[str, str]:
    messages = [
        {"role": "system", "content": RAG_mod.MEDIRAG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Sources:\n{context_text}\n\nQuestion: {question}",
        },
    ]
    if RAG_mod.groq_client is not None:
        try:
            resp = RAG_mod.groq_client.chat.completions.create(
                model=RAG_mod.GROQ_MODEL,
                messages=messages,
                max_tokens=250,
                temperature=RAG_mod.MAIN_QUERY_TEMPERATURE,
            )
            return resp.choices[0].message.content or "", "groq"
        except Exception as exc:
            if _is_quota_error(exc):
                raise
            print(f"[medirag][groq] failed, fallback to Cohere: {exc}")
    resp = RAG_mod.co.chat(
        model="command-r-08-2024",
        messages=messages,
        max_tokens=250,
        temperature=RAG_mod.MAIN_QUERY_TEMPERATURE,
    )
    return resp.message.content[0].text or "", "cohere_fallback"


def _score_arm(answer: str, chunks: list[str], *, classify_claim, verify_entailment, split_sentences):
    """Score EVERY sentence with NLI (not only claim-flagged ones).

    Rationale: in production the claim classifier is an efficiency filter
    (skip NLI on obvious disclaimers). For a research-grade comparison
    between vanilla and MediRAG we need a fuller picture of how grounded
    the *whole answer* is, so we score every sentence and tag separately
    whether it would have triggered the claim classifier."""
    # Normalise markdown so MediRAG's bullet-heavy answers split correctly.
    # The production splitter is tuned for prose; for evaluation we want every
    # bullet to count as its own claim.
    import re
    flat = answer or ""
    flat = re.sub(r"\*\*([^*]+)\*\*", r"\1", flat)
    flat = re.sub(r"\*([^*]+)\*", r"\1", flat)
    flat = re.sub(r"^\s*[-*•]\s+", "", flat, flags=re.MULTILINE)
    flat = re.sub(r"^\s*\d+[.)]\s+", "", flat, flags=re.MULTILINE)
    # Split on newlines first, then on sentence boundaries within each line.
    line_pieces = [ln.strip() for ln in flat.split("\n") if ln.strip()]
    sentences: list[str] = []
    for ln in line_pieces:
        sentences.extend(split_sentences(ln))
    # Filter out trivial/empty sentences
    sentences = [s for s in sentences if s and len(s.split()) >= 4]
    out = {
        "n_sentences": len(sentences),
        "n_claim": 0,
        "n_failed": 0,
        "n_failed_all": 0,
        "per_sentence": [],
    }
    for s in sentences:
        feats = classify_claim(s)
        is_claim = bool(feats.requires_nli)
        if is_claim:
            out["n_claim"] += 1
        best = 0.0
        for ct in chunks:
            try:
                p = float(verify_entailment(s, ct))
            except Exception as exc:
                print(f"[nli] error: {exc}")
                continue
            if p > best:
                best = p
        if best < NLI_SOFTEN_BELOW:
            out["n_failed_all"] += 1
            if is_claim:
                out["n_failed"] += 1
        out["per_sentence"].append({"text": s, "claim": is_claim, "max_entail": round(best, 4)})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=str(EVAL_DIR / "gold/coverage.jsonl"))
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=4, help="chunks to score against")
    parser.add_argument("--out", default=str(EVAL_DIR / "baselines/hallucination_compare.json"))
    parser.add_argument("--min-interval", type=float, default=0.5)
    args = parser.parse_args()

    # Load env then import production modules.
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    import app.RAG as RAG_mod
    from app.guardrails import classify_claim, verify_entailment, _split_sentences

    gold_path = Path(args.gold)
    cases = []
    for line in gold_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        rec = json.loads(line)
        cases.append(rec)
    cases = cases[: args.limit]

    print(f"Running {len(cases)} cases from {gold_path.name}")

    per_case = []
    last_t = 0.0
    quota_hit = False
    for i, rec in enumerate(cases, start=1):
        cid = rec.get("id", f"case-{i}")
        q = rec.get("query") or rec.get("question") or ""
        if not q:
            continue
        print(f"\n[{i}/{len(cases)}] {cid}: {q[:80]}")
        t0 = time.time()

        # Retrieval
        try:
            rows = RAG_mod._retrieve_ranked(q)
        except Exception as exc:
            print(f"  retrieve failed: {exc}")
            per_case.append({"id": cid, "query": q, "error": f"retrieve: {exc}"})
            continue
        top = rows[: args.top_k]
        chunk_texts = [r.get("content", "") for r in top if r.get("content")]
        if not chunk_texts:
            per_case.append({"id": cid, "query": q, "error": "no chunks retrieved"})
            continue

        context_blocks = []
        for j, r in enumerate(top, start=1):
            heading = r.get("section_heading") or ""
            title = r.get("doc_title") or r.get("doc_source") or "source"
            context_blocks.append(f"[src:{j}] {title} — {heading}\n{r.get('content', '')}".strip())
        context_text = "\n\n".join(context_blocks)

        # Throttle
        gap = time.time() - last_t
        if gap < args.min_interval:
            time.sleep(args.min_interval - gap)
        last_t = time.time()

        # Generate vanilla
        try:
            van_ans, van_prov = _generate_vanilla(q, RAG_mod)
        except Exception as exc:
            if _is_quota_error(exc):
                quota_hit = True
                print(f"  vanilla quota hit: {exc}")
                break
            van_ans, van_prov = "", f"err:{exc}"

        # Generate MediRAG
        try:
            med_ans, med_prov = _generate_medirag(q, context_text, RAG_mod)
        except Exception as exc:
            if _is_quota_error(exc):
                quota_hit = True
                print(f"  medirag quota hit: {exc}")
                break
            med_ans, med_prov = "", f"err:{exc}"

        # Score both
        van_stats = _score_arm(van_ans, chunk_texts,
                               classify_claim=classify_claim,
                               verify_entailment=verify_entailment,
                               split_sentences=_split_sentences)
        med_stats = _score_arm(med_ans, chunk_texts,
                               classify_claim=classify_claim,
                               verify_entailment=verify_entailment,
                               split_sentences=_split_sentences)

        per_case.append({
            "id": cid,
            "query": q,
            "n_chunks": len(chunk_texts),
            "vanilla": {"answer": van_ans, "provider": van_prov, **van_stats},
            "medirag": {"answer": med_ans, "provider": med_prov, **med_stats},
            "elapsed_s": round(time.time() - t0, 2),
        })
        print(f"  vanilla: {van_stats['n_failed']}/{van_stats['n_claim']} failed | "
              f"medirag: {med_stats['n_failed']}/{med_stats['n_claim']} failed | "
              f"{time.time() - t0:.1f}s")

    # Aggregate
    def agg(arm):
        n_sent = sum(c[arm]["n_sentences"] for c in per_case if arm in c)
        n_claim = sum(c[arm]["n_claim"] for c in per_case if arm in c)
        n_failed_claim = sum(c[arm]["n_failed"] for c in per_case if arm in c)
        n_failed_all = sum(c[arm]["n_failed_all"] for c in per_case if arm in c)
        # Score the FULL distribution (every scored sentence)
        all_scores = [s["max_entail"] for c in per_case if arm in c
                      for s in c[arm]["per_sentence"] if s["max_entail"] is not None]
        # And separately the claim-only distribution (production-mode metric)
        claim_scores = [s["max_entail"] for c in per_case if arm in c
                        for s in c[arm]["per_sentence"]
                        if s["claim"] and s["max_entail"] is not None]
        def bands(scores):
            if not scores:
                return {"n": 0, "redact": 0.0, "soften": 0.0, "ship": 0.0,
                        "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0,
                        "share_high_conf": 0.0, "share_hard_halluc": 0.0}
            n_redact = sum(1 for s in scores if s < NLI_REDACT_BELOW)
            n_soften = sum(1 for s in scores if NLI_REDACT_BELOW <= s < NLI_SOFTEN_BELOW)
            n_ship = sum(1 for s in scores if s >= NLI_SOFTEN_BELOW)
            n_high = sum(1 for s in scores if s >= 0.80)
            n_hard = sum(1 for s in scores if s < 0.05)
            ss = sorted(scores)
            return {
                "n": len(scores),
                "redact": round(n_redact / len(scores), 4),
                "soften": round(n_soften / len(scores), 4),
                "ship": round(n_ship / len(scores), 4),
                "share_high_conf": round(n_high / len(scores), 4),
                "share_hard_halluc": round(n_hard / len(scores), 4),
                "mean": round(sum(scores) / len(scores), 4),
                "median": round(ss[len(ss) // 2], 4),
                "p25": round(ss[max(0, len(ss) // 4)], 4),
                "p75": round(ss[min(len(ss) - 1, 3 * len(ss) // 4)], 4),
            }
        return {
            "n_sentences_total": n_sent,
            "n_claim_sentences": n_claim,
            "n_failed_sentences_claim_mode": n_failed_claim,
            "n_failed_sentences_all_mode": n_failed_all,
            "hallucination_rate_per_1000_emitted_all": round(1000 * n_failed_all / n_sent, 2) if n_sent else 0.0,
            "hallucination_rate_per_1000_emitted_claim_mode": round(1000 * n_failed_claim / n_sent, 2) if n_sent else 0.0,
            "all_sentences_distribution": bands(all_scores),
            "claim_only_distribution": bands(claim_scores),
            "raw_all_scores": all_scores,
            "raw_claim_scores": claim_scores,
        }

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gold_file": gold_path.name,
        "n_cases_planned": len(cases),
        "n_cases_completed": len([c for c in per_case if "error" not in c]),
        "quota_exhausted": quota_hit,
        "top_k_chunks": args.top_k,
        "nli_thresholds": {"redact_below": NLI_REDACT_BELOW, "soften_below": NLI_SOFTEN_BELOW},
        "vanilla": agg("vanilla"),
        "medirag": agg("medirag"),
        "per_case": per_case,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"\n=== SUMMARY (n_cases={payload['n_cases_completed']}) ===")
    for arm in ("vanilla", "medirag"):
        a = payload[arm]
        d = a["all_sentences_distribution"]
        print(f"\n{arm.upper():10}  n_sentences scored: {d['n']}")
        print(f"           hallucinations/1000 emitted: {a['hallucination_rate_per_1000_emitted_all']:>7}")
        print(f"           share ship    (entail≥0.50): {d['ship']*100:>6.1f}%")
        print(f"           share soften  (0.20–0.50):   {d['soften']*100:>6.1f}%")
        print(f"           share redact  (<0.20):       {d['redact']*100:>6.1f}%")
        print(f"           share high-conf (≥0.80):     {d['share_high_conf']*100:>6.1f}%")
        print(f"           share hard halluc (<0.05):   {d['share_hard_halluc']*100:>6.1f}%")
        print(f"           mean entailment:             {d['mean']:>7}")
        print(f"           median entailment:           {d['median']:>7}")
        print(f"           p25 / p75:                   {d['p25']} / {d['p75']}")

    v = payload["vanilla"]["all_sentences_distribution"]
    m = payload["medirag"]["all_sentences_distribution"]
    print("\n=== HALLUCINATION-REDUCTION DELTA (MediRAG vs Vanilla) ===")
    if v["mean"] > 0:
        rel_mean = (m["mean"] - v["mean"]) / v["mean"] * 100
        print(f"  mean entailment:        {v['mean']:.4f}  →  {m['mean']:.4f}   ({rel_mean:+.1f}% relative)")
    if v["share_high_conf"] > 0:
        rel_hc = (m["share_high_conf"] - v["share_high_conf"]) / v["share_high_conf"] * 100
        print(f"  share high-conf ≥0.80:  {v['share_high_conf']*100:.1f}%  →  {m['share_high_conf']*100:.1f}%   ({rel_hc:+.1f}% relative)")
    if v["share_hard_halluc"] > 0:
        rel_hh = (m["share_hard_halluc"] - v["share_hard_halluc"]) / v["share_hard_halluc"] * 100
        print(f"  share hard halluc <0.05:{v['share_hard_halluc']*100:.1f}%  →  {m['share_hard_halluc']*100:.1f}%   ({rel_hh:+.1f}% relative)")
    v_rate = payload["vanilla"]["hallucination_rate_per_1000_emitted_all"]
    m_rate = payload["medirag"]["hallucination_rate_per_1000_emitted_all"]
    if v_rate > 0:
        rel = (v_rate - m_rate) / v_rate * 100
        print(f"  unsupported/1000:       {v_rate:.1f}  →  {m_rate:.1f}   (reduction {rel:+.1f}% relative)")

    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
