"""Build docs/MediRAG_Paper_Plan.docx — the final research-paper plan.

Regenerate with:  .venv\\Scripts\\python.exe docs/build_paper_plan_docx.py

Source of truth is this script. Edit here, re-run, do not hand-edit the .docx.
"""
from __future__ import annotations

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Inches

ACCENT = RGBColor(0x0B, 0x5C, 0x8A)
MUTED = RGBColor(0x55, 0x5F, 0x6B)
DANGER = RGBColor(0xA3, 0x1D, 0x1D)
GOOD = RGBColor(0x1B, 0x6B, 0x3A)

doc = Document()

# ── base styles ──────────────────────────────────────────────────────────
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(10.5)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.15

for name, size, color in (("Heading 1", 18, ACCENT), ("Heading 2", 14, ACCENT),
                          ("Heading 3", 11.5, RGBColor(0x22, 0x27, 0x2C))):
    st = doc.styles[name]
    st.font.name = "Calibri"
    st.font.size = Pt(size)
    st.font.color.rgb = color
    st.font.bold = True

for s in doc.sections:
    s.left_margin = s.right_margin = Inches(0.85)
    s.top_margin = s.bottom_margin = Inches(0.8)


# ── helpers ──────────────────────────────────────────────────────────────
def para(text="", *, bold=False, italic=False, size=None, color=None,
         space_after=6, align=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    if color:
        r.font.color.rgb = color
    p.paragraph_format.space_after = Pt(space_after)
    if align:
        p.alignment = align
    return p


def rich(*parts, style=None, space_after=6):
    """rich(("plain ", {}), ("bold", {"bold": True})) -> one mixed-format paragraph."""
    p = doc.add_paragraph(style=style)
    for text, fmt in parts:
        r = p.add_run(text)
        r.bold = fmt.get("bold", False)
        r.italic = fmt.get("italic", False)
        if fmt.get("mono"):
            r.font.name = "Consolas"
            r.font.size = Pt(9.5)
        if fmt.get("color"):
            r.font.color.rgb = fmt["color"]
    p.paragraph_format.space_after = Pt(space_after)
    return p


def bullet(text, level=0, bold_prefix=None):
    style = "List Bullet" if level == 0 else f"List Bullet {level + 1}"
    try:
        p = doc.add_paragraph(style=style)
    except KeyError:
        p = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        p.add_run(bold_prefix).bold = True
    p.add_run(text)
    p.paragraph_format.space_after = Pt(3)
    return p


def numbered(text, bold_prefix=None):
    p = doc.add_paragraph(style="List Number")
    if bold_prefix:
        p.add_run(bold_prefix).bold = True
    p.add_run(text)
    p.paragraph_format.space_after = Pt(3)
    return p


def code(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = "Consolas"
    r.font.size = Pt(9)
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    _shade(p, "F4F6F8")
    return p


def _shade(paragraph, hexcolor):
    pPr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hexcolor)
    pPr.append(shd)


def callout(title, body, fill="FFF4E5"):
    p = doc.add_paragraph()
    r = p.add_run(title)
    r.bold = True
    r.font.size = Pt(10.5)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.left_indent = Inches(0.12)
    _shade(p, fill)
    p2 = doc.add_paragraph()
    p2.add_run(body).font.size = Pt(10)
    p2.paragraph_format.left_indent = Inches(0.12)
    p2.paragraph_format.space_after = Pt(10)
    _shade(p2, fill)


def table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        r = hdr[i].paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(9.5)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            bold = val.startswith("**") and val.endswith("**")
            r = p.add_run(val.strip("*"))
            r.bold = bold
            r.font.size = Pt(9.5)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def hr():
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:color"), "C9D3DC")
    bdr.append(bottom)
    pPr.append(bdr)
    p.paragraph_format.space_after = Pt(10)


# ═════════════════════════════════════════════════════════════════════════
# TITLE
# ═════════════════════════════════════════════════════════════════════════
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("MediRAG → Research Paper")
r.bold = True
r.font.size = Pt(26)
r.font.color.rgb = ACCENT
t.paragraph_format.space_after = Pt(2)

st = doc.add_paragraph()
st.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = st.add_run("Final plan: audit findings, the paper to write, and the way forward")
r.font.size = Pt(12.5)
r.font.color.rgb = MUTED
st.paragraph_format.space_after = Pt(2)

d = doc.add_paragraph()
d.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = d.add_run("Prepared 17 September 2026  ·  Supersedes docs/PAPER_PLAN.md (7 August 2026)")
r.font.size = Pt(9.5)
r.font.color.rgb = MUTED
d.paragraph_format.space_after = Pt(14)

hr()

# ═════════════════════════════════════════════════════════════════════════
doc.add_heading("Read this first — the one-page verdict", level=1)

para("You asked two questions: where is this project really, and should you write a "
     "peer-reviewed paper on it. Short answers:", space_after=8)

rich(("Where it is: ", {"bold": True}),
     ("the engineering is genuinely strong and unusually well documented. The empirical "
      "claims are not. Every headline number in the repository is either statistically "
      "null, judged circularly, or was produced by a model that no longer exists. None of "
      "that is fatal — but none of it can go in a paper as it stands.", {}))

rich(("Should you write a paper: ", {"bold": True}),
     ("yes — but not the paper you set out to write. ", {}),
     ("\"Our cascade reduces hallucination\" is dead", {"bold": True}),
     (" (Fisher exact p = 0.548). The paper that is actually here is stronger, more "
      "honest, and more novel than that one would have been.", {}))

callout(
    "The paper to write",
    "In cascaded consumer-health RAG, safety layers purchase reductions in unsupported "
    "clinical claims at a large, systematically under-reported cost in access — and we "
    "measure both axes against clinician-adjudicated ground truth.\n\n"
    "Working title: \"What does safety cost? Measuring the hallucination–access frontier "
    "in cascaded consumer-health retrieval-augmented generation\"",
    fill="E8F1F8")

para("Why this reframe is the right call: it turns your three biggest liabilities into "
     "your three main contributions.", space_after=6)

table(
    ["What looks like a problem", "What it becomes in this paper"],
    [["54.5% over-refusal rate", "**The measured cost axis — the entire point of the paper**"],
     ["Statistically null hallucination result (p = 0.548)",
      "An honest finding that motivates measuring a frontier instead of claiming a point"],
     ["A commit that relaxed a safety guard and opened a hole",
      "A documented case study of a live system trading safety for access, badly"]],
    widths=[2.6, 4.2])

para("Timeline: 18 weeks. Cost: zero. Primary venue: JMIR, with AMIA and ML4H in parallel.",
     bold=True, space_after=4)

doc.add_page_break()

# ═════════════════════════════════════════════════════════════════════════
doc.add_heading("1. What I actually ran, and what I found", level=1)

para("Everything below is measured on your machine on 17 September 2026 — not read from "
     "your documentation. Where your documentation and the live system disagree, the live "
     "system wins, and in two cases they disagree substantially.", italic=True, space_after=10)

doc.add_heading("1.1 What works", level=2)

table(
    ["Check", "Result"],
    [["Supabase corpus", "**LIVE — 1,009 documents / 3,816 chunks**"],
     ["ingest/manifest/", "**RECOVERED — 7 manifests, 224 source URLs**"],
     ["app/RAG.py imports, 16 routes mount", "Clean (~50 s cold start)"],
     ["Deterministic test suite (8 files)", "**57 pass / 1 fail / 17 skipped**"],
     ["Red-flag engine vs 131 gold rows",
      "**73/73 sensitivity · 58/58 specificity · 0 false positives · 45/45 rules covered**"],
     ["Cohere key + embed / rerank / command-r models", "All live"],
     ["Groq key", "Valid"]],
    widths=[2.9, 3.9])

para("The red-flag result is the one number in this project that reproduces cleanly today "
     "— offline, in about four seconds, with no API keys at all. That is a real asset.",
     space_after=10)

doc.add_heading("1.2 Your corpus is eight times bigger than your documentation says", level=2)

rich(("Both docs/DOCUMED.md and your corpus audit record 131 documents. The live database "
      "holds ", {}),
     ("1,009 documents and 3,816 chunks.", {"bold": True}),
     (" The ingestion that produced them was never written back into the documentation.", {}))

callout("A correction to an earlier draft of this plan",
        "An earlier version of this document said that corpus growth had fixed your "
        "retrieval-gradability problem, quoting a jump from 21.4% to about 51%. That was "
        "wrong. It compared the CURRENT gold files against the CURRENT corpus and set the "
        "result beside an April figure computed on a DIFFERENT gold set. The numbers were "
        "never comparable.\n\n"
        "Measured properly, on the same 518 April gold strings:\n"
        "    against the April corpus (131 docs)  ->  111/518 = 21.4%\n"
        "    against today's corpus (1,009 docs)  ->  115/518 = 22.2%\n\n"
        "Growing the corpus 7.7x bought +0.8 percentage points. Essentially nothing.",
        fill="FDECEC")

para("The reason is the corpus composition problem in §1.3 (F8): the 878 new documents are "
     "almost all MedlinePlus patient-education pages, while your gold rows cite specific "
     "sources — WHO IMAI, mhGAP, IMCI, named NHS topics, MoHP standard treatment protocols. "
     "Adding more of the wrong kind of document does not make the right ones appear.",
     space_after=8)

para("Where the apparent improvement actually came from", bold=True, space_after=4)
para("Your gold files were rewritten on 21 April 2026 by "
     "eval/scripts/apply_gold_rewrite_2026_04_21.py. Across 198 rows and 604 source entries:",
     space_after=4)
table(
    ["Transform", "Entries", "Share"],
    [["**D — dropped as an invalid label**", "**365**", "**60%**"],
     ["K — kept unchanged", "121", "20%"],
     ["I — queued as an ingest candidate", "88", "15%"],
     ["**S — substituted with a corpus-aligned title**", "**30**", "**5%**"]],
    widths=[3.4, 1.6, 1.6])
para("Dropping 60% of the entries — disproportionately the ones nothing in the corpus could "
     "match — raises the remaining percentage by definition. That is survivorship, not "
     "improvement. A further 57 of 228 gold rows (25%) had every source dropped and are now "
     "excluded from retrieval scoring entirely.", space_after=6)

para("Was the rewrite wrong? Mostly no — and this matters for how you defend it. Many drops "
     "are plainly correct; the script's own note on one row reads \"SOCRATES is a template "
     "tag, not a retrieval source\", which is a genuinely bad label. The problem is not the "
     "cleanup. The problem is comparing a number computed after it with one computed before "
     "it and calling the difference progress.", space_after=8)

callout("What this means for the paper",
        "Keep retrieval recall OFF the headline. Your original decision — faithfulness "
        "carries correctness, retrieval recall is a caveated coverage bound — was right.\n\n"
        "If you report retrieval recall at all, disclose three things: the gold was "
        "rewritten after the fact, 60% of source entries were dropped, and 25% of rows "
        "cannot be scored. A reviewer who finds apply_gold_rewrite_2026_04_21.py in your "
        "repository without those disclosures in the paper will distrust the entire "
        "evaluation. The 5% substitution rate is separately worth naming as a mild "
        "circularity: those labels were changed to match what the corpus contains.",
        fill="FFF4E5")

doc.add_heading("1.3 The problems that block a paper", level=2)

rich(("F3 — Your generator model no longer exists. ", {"bold": True, "color": DANGER}),
     ("app/RAG.py line 150 requests llama-3.3-70b-versatile. Groq returns "
      "404 model_not_found. It has been decommissioned. Groq now serves gpt-oss-120b, "
      "gpt-oss-20b and qwen3.8-27b instead. ", {}),
     ("Every number in your repository came from a model you can no longer call, so all "
      "results must be regenerated regardless of any other decision.", {"bold": True}))

rich(("F4 — There is a live safety hole in your scope guard. ", {"bold": True, "color": DANGER}),
     ("This is the most urgent finding in this document. Your policy filter classifies the "
      "bluntest and most dangerous form of diagnosis as safe:", {}))

code('\'safe\'        <- "You have type 2 diabetes."\n'
     '\'safe\'        <- "You have asthma."\n'
     '\'safe\'        <- "You likely have anemia."\n'
     '\'diagnostic\'  <- "This sounds like asthma, but see a doctor."\n'
     '\'diagnostic\'  <- "Most likely you have hypertension."')

para("Only hedged phrasings are caught. A flat assertion — \"You have X\" — walks straight "
     "through. Your own test (eval/test_scope_guard.py line 56) says this should be caught, "
     "so that test has been failing against committed code, and because CI runs that file, "
     "main has been red. Tracing it with git shows the cause: commit b13bd83 "
     "(\"hallucination-zero sprint … scope-guard relax\") loosened the guard to reduce "
     "over-refusal, and this hole was the price.", space_after=8)

callout("Why this matters twice over",
        "It is a real safety bug in a deployed system and should be fixed this week. It is "
        "also the single best piece of evidence for your paper's thesis: here is a live "
        "system trading safety away to buy back access, with the exact commit, date and "
        "consequence on record. Capture the broken behaviour as evidence before fixing it.",
        fill="FDECEC")

rich(("F5 — Your headline result is statistically null. ", {"bold": True, "color": DANGER}),
     ("Re-analysing eval/baselines/hallucination_compare.json:", {}))

code("vanilla : 17/18 claim sentences unsupported = 94.4%\n"
     "MediRAG : 10/12 claim sentences unsupported = 83.3%\n"
     "absolute reduction .................. 11.1 percentage points\n"
     "Fisher exact, two-sided ............. p = 0.548")

para("In plain terms: that result is indistinguishable from random chance. Three further "
     "problems compound it:", space_after=4)
bullet("The reported \"38% reduction\" (144.07 → 89.29 per 1000) divides claim-sentence "
       "failures by total sentences. The improvement comes from the system emitting six "
       "fewer claim sentences, not from better grounding.")
bullet("The run stopped at 10 of 20 planned cases — quota_exhausted: true.")
bullet("The judge is cross-encoder/nli-deberta-v3-base, which is the same model your "
       "guardrail uses to filter. You are grading the exam you wrote the answers to.")
doc.add_paragraph().paragraph_format.space_after = Pt(4)

rich(("F8 — Your corpus is not Nepal-grounded, and a reviewer will check. ", {"bold": True, "color": DANGER}),
     ("This is new and your August plan never caught it.", {}))

table(
    ["Source", "Documents", "Share"],
    [["**MedlinePlus (US consumer health)**", "**826**", "**82%**"],
     ["NHS", "73", "7%"],
     ["WHO", "45", "4%"],
     ["EDCD, MoHP, WHO Nepal, NPHL, BPKIHS, CSH, DDA, Bir", "57 total", "5.6%"],
     ["Testing.com", "9", "1%"]],
    widths=[3.6, 1.6, 1.6])

para("You cannot claim a \"Nepal-grounded corpus\" — that is one database query away from "
     "being disproved. Your Nepal contribution is real, but it lives in the routing layer: "
     "nepal_care_tiers.yaml (7 tiers), redflag_rules.yaml (45 rules), and the "
     "EDCD/MoHP/WHO-Nepal document subset. State the composition in Table 1 of the paper "
     "and frame it precisely. Never let a reviewer discover this for you.", space_after=8)

para("Also outstanding: over-refusal is 54.5% (F6); stages 3 and 5 are unimplemented, so "
     "88 of your 329 gold rows have no code to grade (F9); and five API keys sit in "
     "plaintext comments in .env — never committed, but rotate them anyway.", space_after=4)

doc.add_page_break()

# ═════════════════════════════════════════════════════════════════════════
doc.add_heading("2. The paper", level=1)

doc.add_heading("2.1 The novel measurement — why this is publishable", level=2)

para("Every comparable system — FActScore, SelfCheckGPT, ALCE, RARR, Self-RAG, CRAG — "
     "validates hallucination detection against human labels. None of them validates "
     "refusal against anything. Abstention gets reported as a raw rate, with no ground "
     "truth for whether any individual refusal was actually justified.", space_after=6)

callout("The gap you can fill",
        "With clinician annotators, you can produce a clinician-adjudicated over-refusal "
        "rate: for every refusal the system makes, did a qualified clinician judge that a "
        "safe and useful answer was in fact available? Nobody reports this. It is a "
        "genuinely novel axis, and it is the y-axis of your central figure.",
        fill="E8F1F8")

doc.add_heading("2.2 Contributions, in order of strength", level=2)
numbered("for a deployed system, threshold-swept, with your shipped operating point marked. "
         "This is the novel measurement.", bold_prefix="Clinician-adjudicated safety–access frontier ")
numbered("faithfulness filtering under token streaming, where you may never display text "
         "you later retract. Batch-mode prior work does not address this at all. "
         "(guardrails.py:681)", bold_prefix="Streaming-safe sentence-level guardrailing — ")
numbered("validated on a held-out set written by a clinician who has not seen your rules. "
         "The LLM is never in the emergency path — that is a real architectural claim.",
         bold_prefix="Deterministic pre-LLM emergency triage, ")
numbered("ALCE and Liu et al. measure attribution after the fact; you enforce it at "
         "generation time by restricting the entailment premise to cited chunks. "
         "(guardrails.py:279)", bold_prefix="Citation binding as enforcement, not measurement — ")
numbered("framed as a routing-layer contribution, not a corpus claim.",
         bold_prefix="Nepal care-tier routing and an adversarial safety benchmark, ")

doc.add_heading("2.3 What to cut, deliberately", level=2)
bullet("Do not build stages 3 and 5. Cut them from the architecture claim and drop those 88 "
       "gold rows. This is product work with zero paper value.", bold_prefix="")
bullet("Do not use MIRAGE, MedQA or MedMCQA. They are exam-style multiple choice; your "
       "system deliberately refuses diagnostic reasoning and would score near zero for "
       "entirely correct reasons. Say this explicitly — it makes a good limitations paragraph.")
bullet("Do not claim a Nepal-grounded corpus (see F8).")

doc.add_heading("2.4 Where to send it", level=2)
table(
    ["Venue", "Role", "Notes"],
    [["**JMIR / JMIR Med Inform**", "**Primary**",
      "Rolling submission, no deadline pressure. Deployment + evaluation is squarely their "
      "scope and the LMIC angle is valued. Article processing charge applies — check "
      "student and LMIC waivers."],
     ["AMIA Annual Symposium", "Parallel", "Clinical informatics systems papers. Roughly a March deadline."],
     ["ML4H", "Parallel", "Safety guardrails for medical LLMs. Roughly a September deadline."],
     ["arXiv", "Preprint", "Post when Phase 4 completes — timestamps the work, costs nothing."],
     ["EMNLP System Demonstrations", "Fallback", "Much lower bar. Insurance policy — you have a live demo."]],
    widths=[1.9, 1.0, 3.9])
para("Verify every deadline yourself; conference cycles shift.", italic=True, size=9.5)

doc.add_page_break()

# ═════════════════════════════════════════════════════════════════════════
doc.add_heading("3. Running this for free", level=1)

para("You said you do not want to spend money on APIs. You do not have to, and the free "
     "route is genuinely better for the paper than the paid one.", space_after=8)

doc.add_heading("3.1 What your machine can do", level=2)
para("Measured: Intel Xeon W-2145 at 3.7 GHz, 5 cores, 69 GB RAM, 410 GB free disk, "
     "AVX-512 — and no GPU (only a Microsoft Basic Display Adapter, so this is a "
     "virtualized host). That means CPU-only inference. It rules out large dense models. "
     "It does not block the plan.", space_after=8)

doc.add_heading("3.2 The fact the whole strategy turns on", level=2)
callout("gpt-oss-20b is both open-weight AND served on Groq's free tier",
        "The same model weights can run in two places: on Groq for speed, and on your own "
        "machine for unlimited, perfectly reproducible runs. It is a Mixture-of-Experts "
        "model, so it activates only about 3.6B parameters per token — roughly 8B-class "
        "speed at considerably better quality. About 12 GB on disk.",
        fill="E8F1F8")

table(
    ["", "Groq free tier", "Local gpt-oss-20b"],
    [["Speed", "Very fast", "~95 s per answer (estimate)"],
     ["Quota", "Daily token cap", "**Unlimited**"],
     ["Reproducible sampling", "Not guaranteed", "**Exact, seed-pinned**"],
     ["Survives the model being retired", "**No — this already happened to you**", "**Yes**"]],
    widths=[1.9, 2.4, 2.5])

rich(("The strategy: ", {"bold": True}),
     ("Groq for bulk throughput, local for determinism-critical runs, quota overflow, and "
      "the permanent reproducibility archive. Validate that the two agree on a shared "
      "sample, then use whichever is available. No single point of failure, no money.", {}))

callout("This also writes your best methods paragraph",
        "\"We use open-weight models pinned by hash, because our earlier results became "
        "unreproducible when a hosted model was decommissioned mid-project.\" That is not "
        "an excuse — it is exactly the reproducibility argument reviewers want to see, and "
        "you have lived proof of it.",
        fill="E8F1F8")

doc.add_heading("3.3 Model options", level=2)
table(
    ["Model", "Size", "Fits 69 GB", "Est. speed", "Role"],
    [["**gpt-oss-20b** (MoE)", "~12 GB", "Yes", "8–14 tok/s", "**Primary local backbone**"],
     ["qwen3-8b / llama-3.1-8b", "~5 GB", "Yes", "5–9 tok/s", "Fast secondary arm"],
     ["gpt-oss-120b (MoE)", "~63 GB", "Tight", "3–6 tok/s", "Optional subset check"],
     ["Any 70B dense model", "~40 GB", "Yes", "1–2 tok/s", "**Unusable — do not attempt**"]],
    widths=[1.7, 0.9, 0.9, 1.1, 2.2])
para("These speeds are estimates. They must be benchmarked in Phase 1 before the schedule "
     "depends on them.", italic=True, bold=True, size=9.5, space_after=8)

doc.add_heading("3.4 Time budget, and how to cut it", level=2)
para("About 5,500 generations are needed in total. At roughly 95 seconds each that is "
     "around 145 hours — six days of continuous compute, with no slack for reruns. Three "
     "fixes, apply all of them:", space_after=4)
numbered("Arms A1–A6 share the same retrieved context for a given question, so the "
         "expensive prompt-processing step can be reused across arms rather than repeated.",
         bold_prefix="Reuse the prompt cache. ")
numbered("Three seeds only for the headline arms, one for the intermediate arms. That cuts "
         "21 runs per question to 13 — roughly 2,600 generations, under three days.",
         bold_prefix="Seed asymmetrically. ")
numbered("Bulk on the Groq free tier, determinism-critical subset locally.",
         bold_prefix="Split by provider. ")

doc.add_heading("3.5 Removing every remaining paid dependency", level=2)
table(
    ["What you pay for now", "Free replacement"],
    [["Generation (Groq llama-3.3-70b — dead)", "Groq free gpt-oss-20b **plus the same weights locally**"],
     ["Reranking (Cohere rerank-v3.5, trial rate limits)", "BAAI/bge-reranker-v2-m3 cross-encoder, runs on CPU"],
     ["Embeddings (Cohere embed-english-v3.0)",
      "**Already computed and stored in Supabase — do not change the embedder.** Switching "
      "would force re-embedding all 3,816 chunks and invalidate your frozen corpus."],
     ["NLI guardrail", "Already local and small"],
     ["Independent judge (new requirement)", "MiniCheck or AlignScore — both run locally on CPU"]],
    widths=[2.7, 4.1])

para("Register these as backups but do not depend on any single one: Google AI Studio "
     "(Gemini) free tier, GitHub Models, Cerebras Cloud free tier, OpenRouter free model "
     "variants, HuggingFace Inference Providers credits, Mistral's free tier.", space_after=4)

doc.add_page_break()

# ═════════════════════════════════════════════════════════════════════════
doc.add_heading("4. The 18-week plan", level=1)

doc.add_heading("Phase 0 — Secure, freeze, instrument (Week 1)", level=2)

callout("Do this before anything else",
        "Back up the corpus today. Export all 1,009 documents and 3,816 chunks — including "
        "the embedding vectors — to local disk, with a checksum, plus a second copy off "
        "this machine. Commit the 224 recovered manifest URLs in the same commit.\n\n"
        "That Supabase project disappeared once already and came back. Until this export "
        "exists, every remaining week of this plan is sitting on a single point of failure.",
        fill="FDECEC")

bullet("Write eval/corpus_manifest.json recording document and chunk counts and the full "
       "source breakdown. Tag the repository paper-v1. Every number in the paper comes "
       "from this tag or later.", bold_prefix="Freeze and record. ")
bullet("against the live corpus, to get the citable figure rather than my approximation.",
       bold_prefix="Re-run the real gradability audit ")
bullet("First freeze the broken behaviour as evidence with the commit SHA. Then fix it. "
       "Then re-run the coverage scorer to measure what the fix costs on the access axis "
       "— that cost is itself a data point on your central figure. Get CI green (58/58).",
       bold_prefix="Record, then fix, the scope-guard hole (F4). ")
bullet("Add an EVAL_DETERMINISTIC switch forcing temperature 0 and a fixed seed for "
       "evaluation, while production stays at 0.15. Add seed controls to the three main "
       "scorers and report mean with 95% confidence intervals. Add a run_manifest block to "
       "every result file recording git SHA, model IDs and weight hashes, backend, every "
       "threshold, and corpus counts. Archive the pre-paper-v1 baselines as uncitable. "
       "Rotate the five exposed keys.", bold_prefix="Make runs reproducible. ")

doc.add_heading("Phase 1 — Stand up the free stack (Week 2)", level=2)
bullet("Install Ollama or llama.cpp, pull gpt-oss-20b, record the weight hash.")
bullet("Add an LLM_BACKEND switch (groq / local / cohere) to app/RAG.py — the Groq SDK "
       "accepts a base_url, so this is a config change, not a rewrite.")
bullet("Benchmark real throughput before committing the schedule to it. If local speed "
       "comes in under about 5 tokens/second, demote local to the determinism subset and "
       "move bulk generation to free cloud tiers. The plan survives either outcome.",
       bold_prefix="")
bullet("Swap Cohere rerank for bge-reranker-v2-m3 and verify retrieval quality does not "
       "regress on a held-out sample before adopting it.")
bullet("Validate that local and Groq agree on 50 shared prompts at temperature 0, and "
       "report that agreement rate in the paper's reproducibility section.")

doc.add_heading("Phase 2 — Question set and runner (Weeks 3–4)", level=2)
para("Build a stratified set of 200 questions:", space_after=4)
table(
    ["Stratum", "n", "Source"],
    [["Consumer symptom", "40", "coverage.jsonl + K-QA subset"],
     ["Condition education", "40", "condition.jsonl"],
     ["Lab explainer", "25", "results.jsonl"],
     ["Nepal care navigation", "30", "navigation_stage2.jsonl"],
     ["Adversarial (must-refuse)", "45", "must_refuse.jsonl, expanded"],
     ["Emergency red-flag", "20", "redflag.jsonl positives"]],
    widths=[2.2, 0.7, 3.9])
para("Anchor the external portion in K-QA (Manes et al. 2024) — 1,212 real patient "
     "questions with clinician-written must-have and nice-to-have statements. It measures "
     "hallucination and comprehensiveness together, which feeds your frontier framing "
     "directly.", space_after=6)
rich(("Build the runner to retrieve once per question and reuse the identical context "
      "across every arm. ", {"bold": True}),
     ("That is what makes the comparison fair, and it removes the \"your baseline was "
      "rigged\" objection. Build checkpoint and resume in from day one — quota exhaustion "
      "has already truncated one of your experiments.", {}))

doc.add_heading("Phase 3 — The evaluation that carries the paper (Weeks 5–9)", level=2)
para("This is the bottleneck. Budget most of your effort here.", bold=True, space_after=6)

doc.add_heading("3.1  The ablation ladder", level=3)
table(
    ["Arm", "What it adds"],
    [["A0", "LLM only, no retrieval — the upper bound on hallucination"],
     ["A1", "Naive RAG: dense only, no rerank, no gate, no guardrail"],
     ["A2", "+ hybrid retrieval (dense + BM25 + RRF) and reranking"],
     ["A3", "+ abstention gate"],
     ["A4", "+ claim classifier and NLI verifier"],
     ["A5", "+ citation binding"],
     ["A6", "+ fusion-drift detection — the full system"]],
    widths=[0.7, 6.1])

doc.add_heading("3.2  Human ground truth on both axes — do not compromise here", level=3)
bullet("400 sentences across arms A1, A4 and A6. For each, given the retrieved context: "
       "supported / unsupported / contradicted / not-a-claim, plus a harm rating of none, "
       "minor, or potentially harmful. A harm-weighted hallucination rate matters far more "
       "to a clinical venue than a raw count.", bold_prefix="Claim axis: ")
bullet("every refusal the system makes across the 200 questions, roughly 150 of them. For "
       "each, given the question and the retrieved context: was the refusal warranted, "
       "unnecessary, or borderline? This is the novel part.",
       bold_prefix="Access axis: ")
bullet("Two annotators, blind to which arm produced each item. Report Cohen's kappa on both "
       "label sets, with a third rater adjudicating disagreements. The annotation guideline "
       "ships as an appendix — reviewers ask for it.")

doc.add_heading("3.3  Kill the circular judge", level=3)
para("Add a judge that is not your guardrail model — MiniCheck (Tang et al. 2024) is "
     "purpose-built, small and runs locally on CPU; AlignScore is the alternative. Then "
     "report agreement across all three: your guardrail's NLI, the independent judge, and "
     "the human labels. This converts your NLI from \"the metric\" into \"a validated "
     "proxy\", which is a small contribution in its own right.", space_after=6)

doc.add_heading("3.4  Show it is not one model's quirk", level=3)
para("Replicate A1 versus A6 only, 200 questions, one seed, on a second model family "
     "(qwen3.8-27b on Groq, or qwen3-8b locally). About 400 extra generations for a "
     "generalization claim.", space_after=6)

doc.add_heading("3.5  The central figure", level=3)
para("Sweep your two key thresholds across a grid and plot harm-weighted hallucination rate "
     "against clinician-adjudicated answer rate. Mark your shipped operating point, and "
     "mark the before and after positions of the b13bd83 scope-guard relaxation as two "
     "points on the same curve. This single figure is the paper — it turns 54.5% "
     "over-refusal from an embarrassment into a stated design choice on a measured frontier.",
     space_after=6)

doc.add_heading("Phase 4 — The novel mechanisms (Weeks 10–11)", level=2)
bullet("Instrument /query/stream and report distributions, not averages: "
       "time-to-first-token, time-to-first-safe-sentence (your real UX metric), total "
       "latency, and per-sentence NLI overhead. Your code comment claims \"1–2 seconds\" — "
       "it has never been measured. Measure this on the Groq path, not locally, or CPU "
       "speed will confound the result.", bold_prefix="Streaming latency. ")
bullet("Build 80 controlled items: 40 compound claims where the conjunction is supported by "
       "a single chunk, and 40 where each half is supported by a different chunk but the "
       "conjunction is not. Have a clinician verify every label, then report precision, "
       "recall and F1. Without this, fusion drift is code, not a contribution.",
       bold_prefix="Fusion-drift diagnostic. ")
bullet("Measure citation precision, citation recall, and the mis-binding rate — sentences "
       "supported by some chunk but not by the one they cite. Mis-binding is exactly what "
       "citation binding exists to catch, so quantify how often it fires.",
       bold_prefix="Citation attribution. ")

doc.add_heading("Phase 5 — The triage layer, done properly (Weeks 12–13)", level=2)
callout("Your perfect score is a liability, not an asset",
        "73/73 and 58/58 with zero false positives will read to a reviewer as overfitting — "
        "and they would be right. Your own log records gold cases being added to cover "
        "rules, and all 45 rules have at least one hand-written positive example. As it "
        "stands, this measures \"the regular expressions match the strings I wrote for "
        "them\", not clinical sensitivity.\n\n"
        "The fix: commission 120 held-out cases from a clinician who has never seen "
        "redflag_rules.yaml — 60 positives and 60 hard negatives (deliberate near-misses). "
        "Report sensitivity, specificity and dangerous-direction error rate on that set, "
        "separately from your development set. Expect the numbers to drop. That is the "
        "point: a held-out 0.90 is worth far more than a development 1.00.",
        fill="FFF4E5")

doc.add_heading("Phase 6 — Clinician expert review (Weeks 12–14, parallel, optional)", level=2)
para("50 system responses rated by 3 Nepali clinicians on safety, factual accuracy, "
     "usefulness and care-tier appropriateness, plus free-text harm flags. No patient data. "
     "Optional by design — never let this gate submission.", space_after=6)

doc.add_heading("Phase 7 — Write and submit (Weeks 15–18)", level=2)
para("Draft in this order: Results → Methods → Related Work → Discussion → Introduction → "
     "Abstract. Never write the abstract first; you do not yet know what the numbers say.",
     space_after=6)
para("Five figures: the cascade architecture with the emergency path shown bypassing the "
     "LLM entirely; the frontier curve; the ablation waterfall; judge validation with "
     "kappa; and latency distributions including time-to-first-safe-sentence.", space_after=6)
rich(("One thing you must do in Related Work: ", {"bold": True}),
     ("explicitly differentiate from Self-RAG and CRAG. They train critique tokens into the "
      "generator. You apply an external, model-agnostic, post-hoc cascade that works with "
      "any backbone and under streaming. That distinction is your defence against \"this is "
      "just RAG plus NLI\".", {}))

doc.add_page_break()

# ═════════════════════════════════════════════════════════════════════════
doc.add_heading("5. Ethics approval and annotators — what to do and why", level=1)

doc.add_heading("5.1 The key insight: your paper is not blocked on ethics", level=2)
para("Three different activities get conflated here, and they have genuinely different "
     "requirements:", space_after=4)
table(
    ["Activity", "Are they research subjects?", "Likely requirement"],
    [["**Annotators labelling model outputs** (Phase 3.2 — your actual contribution)",
      "**No.** They are research personnel. You collect no data about them.",
      "**Probably none. Confirm, but do not wait.**"],
     ["Clinicians giving Likert opinions (Phase 6, optional)",
      "Borderline — their opinions are your data",
      "Often exempt or expedited. Ask in writing."],
     ["A user pilot (not in this plan)", "Yes — real users, real health questions",
      "Full review, consent, data protection plan"]],
    widths=[2.4, 2.2, 2.2])
para("So the core contribution is almost certainly unblocked, and Phase 6 is optional. "
     "Ethics delay cannot kill this paper. You still ask in Week 1, in writing, because "
     "journals require an ethics statement and \"we assumed it was exempt\" is not one.",
     bold=True, space_after=8)

doc.add_heading("5.2 NHRC — what it is, why it matters, what to do", level=2)
rich(("What it is: ", {"bold": True}),
     ("the Nepal Health Research Council is the national body that reviews health research "
      "in Nepal. Its Ethical Review Board issues approval or exemption letters.", {}))
rich(("Why it matters: ", {"bold": True}),
     ("JMIR, AMIA and every medical venue require an ethics statement naming the approving "
      "body or the exemption determination. Without one you risk desk rejection — and "
      "discovery after publication is far worse.", {}))
para("Do this in Week 1:", bold=True, space_after=4)
numbered("Most Nepali universities and medical colleges have an Institutional "
         "Review Committee, and NHRC often expects institutional review first. For low-risk "
         "work an IRC letter may be enough. Ask your supervisor which applies — this one "
         "question could save you two months.", bold_prefix="Check your own institution first. ")
numbered("via nhrc.gov.np, with a one-page description: title, "
         "objective, what data you collect, from whom, how it is stored, risks, and consent "
         "process.", bold_prefix="Email the NHRC ERB secretariat ")
numbered("(a) Does having trained annotators label "
         "AI-generated text against source documents constitute human-subjects research? "
         "(b) Does a clinician expert-review survey with no patient data require full "
         "review, expedited review, or is it exempt? (c) Would a future 20–30 user pilot "
         "require full review?", bold_prefix="Ask these three questions explicitly: ")
numbered("and file it. That letter is your ethics statement.",
         bold_prefix="Get the answer in writing ")
para("Expect 6 to 12 weeks. That is precisely why it goes in Week 1 — and why nothing is "
     "allowed to depend on it.", italic=True, space_after=8)

doc.add_heading("5.3 Annotators — who, how many, and the protocol that makes it credible", level=2)
bullet("MBBS interns, final-year MBBS students, or junior doctors. Not pre-clinical "
       "students — they need enough clinical judgment to rate potential harm.", bold_prefix="Who: ")
bullet("two independent annotators plus one senior adjudicator for disagreements. "
       "Two is the minimum that lets you report kappa, and kappa is what makes the labels "
       "credible.", bold_prefix="How many: ")
bullet("IOM Maharajgunj (TU), BPKIHS Dharan, KUSMS Dhulikhel, Patan Academy of "
       "Health Sciences, NAMS/Bir. Nepal Medical College and KIST as backups. Identify a "
       "backup institution now — stalled recruitment is the single most likely thing to "
       "wreck your schedule.", bold_prefix="Where: ")
bullet("be concrete, not vague. \"About 6 hours of structured labelling, remote, "
       "over two weeks, paid Rs X, with acknowledgement in a publication\" works far better "
       "than a general request. Route through a department contact or your supervisor "
       "rather than cold email.", bold_prefix="How to ask: ")
bullet("and say so in the paper. Reviewers increasingly check for annotator "
       "compensation, and unpaid medical-student labour is an ethics flag.",
       bold_prefix="Pay them ")

para("The protocol that makes the labels defensible:", bold=True, space_after=4)
numbered("Write the annotation guideline before recruiting. It ships as an appendix.")
numbered("Calibration round: every annotator labels the same 30 items. Compute kappa.")
numbered("If kappa is below 0.6, stop. Revise the guideline, discuss the disagreements, "
         "recalibrate. Do not proceed to the main set with poor agreement — you cannot fix "
         "it afterwards.")
numbered("Blind annotators to which arm produced each item. Shuffle and strip identifiers. "
         "This is non-negotiable and reviewers look for it.")
numbered("Adjudicate disagreements with the third rater, and report both raw and "
         "adjudicated numbers.")
para("Volume: roughly 550 items across two annotators is about 18 person-hours. Very "
     "manageable — budget accordingly.", italic=True, space_after=6)

doc.add_page_break()

# ═════════════════════════════════════════════════════════════════════════
doc.add_heading("6. How every change gets documented", level=1)

para("You need to be able to explain and defend every change in this project yourself — in "
     "a viva, to a supervisor, and to reviewers. Working code you cannot justify is not "
     "useful to you. So documentation is a deliverable here, not a by-product.", space_after=8)

para("A single running file, docs/PAPER_WORKLOG.md, updated as work happens rather than "
     "reconstructed later. Every entry answers three questions in plain language:",
     space_after=4)
bullet("which file, and what is different now.", bold_prefix="What changed — ")
bullet("the problem it solves.", bold_prefix="Why — ")
bullet("what this changes for you and for the paper, not what it changes in the code.",
       bold_prefix="Impact on the project — ")

para("Six rules:", bold=True, space_after=4)
numbered("Every entry has What, Why and Impact.")
numbered("Plain language. If a technical term is unavoidable, define it once.")
numbered("Trade-offs get written down, not silently resolved. When there is a real choice, "
         "record the options and why one was picked. These become Methods paragraphs and "
         "viva answers.")
numbered("Negative results and mistakes get logged too. \"We tried X, it was worse, here "
         "are the numbers\" is paper material and stops you repeating it.")
numbered("Every change is also summarised in plain terms in conversation as it is made.")
numbered("Nothing is changed silently. If a file is modified, it appears in the worklog.")

para("This is not busywork. docs/DOCUMED.md — 8,600 lines of dated provenance — is already "
     "one of this project's rarest assets, because it means any threshold in your system "
     "can be traced back to a reason. Most projects cannot do that. The worklog continues "
     "that discipline through the paper phase, and it feeds directly into your Methods, "
     "Limitations and reproducibility sections.", space_after=8)

hr()

doc.add_heading("7. Risks, and what happens if they land", level=1)
table(
    ["Risk", "What protects you"],
    [["**Supabase vanishes again** — it already did once",
      "The Phase 0 local export with embeddings, plus an off-machine copy. After that the "
      "corpus is never a single point of failure again."],
     ["Local inference turns out too slow",
      "Demote local to the determinism subset; bulk moves to free cloud tiers with "
      "checkpoint and resume. The plan survives either way."],
     ["Groq retires gpt-oss-20b as well",
      "Your local weights are hash-pinned and archived. This is exactly why the local path exists."],
     ["Free-tier quota runs out mid-experiment",
      "Checkpoint and resume from day one, multi-provider failover, and the local path has no quota."],
     ["Annotator recruitment stalls",
      "Start Week 1, identify a backup institution up front, and let kappa calibration catch "
      "quality problems early."],
     ["Ethics approval is slow", "The core contribution is not human-subjects work. Phase 6 is optional."],
     ["Held-out red-flag numbers drop sharply",
      "Expected and correct. Report development and held-out separately."],
     ["**The results come out weaker than you hoped**",
      "The frontier framing is built for exactly this. A rigorous modest result with a clean "
      "curve is publishable; an unsupported strong claim is not. Decide now that you will "
      "report what you find."],
     ["Reviewer says \"this is just RAG plus NLI\"",
      "Contributions 1 and 2: clinician-adjudicated refusal ground truth, and streaming-safe "
      "guardrailing, each with a dedicated evaluation."],
     ["Reviewer checks the corpus and finds it is 82% MedlinePlus",
      "Pre-empt it. State the composition in Table 1 and frame Nepal as a routing-layer "
      "contribution. Never let a reviewer discover this for you."],
     ["Scope creep back into product work",
      "Freeze at paper-v1. Product bugs go to a branch. Paper numbers come from tags."]],
    widths=[2.3, 4.5])

doc.add_page_break()

# ═════════════════════════════════════════════════════════════════════════
doc.add_heading("8. The way forward — what happens next", level=1)

doc.add_heading("What I need from you", level=2)
para("Already done — thank you: Supabase is restored and the manifest folder is recovered.",
     italic=True, color=GOOD, space_after=6)

para("Start this week, in parallel with the technical work:", bold=True, space_after=4)
numbered("ask which applies to you, an institutional "
         "review committee or NHRC directly. This one question can save two months.",
         bold_prefix="Ask your supervisor about ethics review — ")
numbered("with the three questions in section 5.2.",
         bold_prefix="Email NHRC ")
numbered("You need two plus an adjudicator. Sending those emails in "
         "Week 1 rather than Week 5 is the difference between the core contribution landing "
         "and not landing.", bold_prefix="Start annotator outreach. ")
numbered("the five sitting in .env comments.", bold_prefix="Rotate ")

doc.add_heading("What I will do, in order", level=2)
numbered("all 1,009 documents and 3,816 chunks including embeddings, to local "
         "disk plus an off-machine copy. Pure insurance — changes nothing about how your "
         "system behaves.", bold_prefix="Back up the corpus: ")
numbered("the 224 recovered source URLs. Without them the corpus can never be "
         "rebuilt if it is lost again. Also changes nothing about behaviour.",
         bold_prefix="Commit the manifests: ")
numbered("capture the broken behaviour as dated evidence first, then "
         "fix it, then measure what the fix costs in over-refusal. This is the first real "
         "behaviour change, and I will show you the before and after numbers.",
         bold_prefix="Fix the scope-guard safety hole: ")
numbered("57 of 58 tests pass now; the failing one is the scope-guard bug "
         "above, so fixing it should close both.", bold_prefix="Get CI green: ")
numbered("so every change from here carries its What, Why and Impact.",
         bold_prefix="Open docs/PAPER_WORKLOG.md ")

callout("The one thing worth doing immediately regardless",
        "Back up the corpus. That Supabase project disappeared once already. Steps 1 and 2 "
        "change nothing about how your system behaves — they only make sure that if it "
        "happens again, it costs you an afternoon instead of the project.",
        fill="FDECEC")

hr()
para("Full technical plan with file paths, line numbers and exact commands: "
     "~/.claude/plans/i-have-a-folder-virtual-kahan.md",
     italic=True, size=9, color=MUTED)
para("This document is generated by docs/build_paper_plan_docx.py. Edit that script and "
     "re-run it rather than editing the .docx by hand.",
     italic=True, size=9, color=MUTED)

import pathlib
out = pathlib.Path(__file__).resolve().parent / "MediRAG_Paper_Plan.docx"
doc.save(out)
print(f"written: {out}")
