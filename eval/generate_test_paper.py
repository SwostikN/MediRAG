"""Generate a synthetic research-paper PDF that is guaranteed to pass
DocuMed AI's document classifier as `research_paper` at high confidence.

The classifier (app/document_classifier.py) reads the first ~8000 chars
and scores: 3+ canonical section headers (+4), DOI (+3), 3+ bracket
citations (+2), paper keywords like ORCID/Keywords/Corresponding author
(+1 each, max +2). This paper hits every signal on page 1 alone, so
score is ~10 — well above the _PAPER_HIGH=6 threshold.

Content is intentionally generic, educational, primary-care-flavoured
(matches DocuMed's corpus scope: NHS/MedlinePlus-style patient ed). No
invented dosages, no specific guideline numbers — just well-known
descriptive statements that won't poison retrieval if the chunks get
indexed.

Run:
    python eval/generate_test_paper.py
Output:
    eval/test_paper_iron_deficiency.pdf
"""

import fitz  # PyMuPDF
import os

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "test_paper_iron_deficiency.pdf",
)

PAGE_W, PAGE_H = 595, 842  # A4 in points
MARGIN_X = 56
MARGIN_TOP = 56
MARGIN_BOTTOM = 56

# Helvetica is built into every PDF reader. PyMuPDF aliases: helv, hebo, heit.
FONT_BODY = "helv"
FONT_BOLD = "hebo"
FONT_ITALIC = "heit"

# Approximate char widths at 11pt Helvetica — used for wrap. Conservative.
WRAP_CHARS_BODY = 88
WRAP_CHARS_REFS = 92


def wrap(text: str, max_chars: int = WRAP_CHARS_BODY) -> list[str]:
    """Simple word-wrap. Splits explicit \\n boundaries, then wraps each
    paragraph to <= max_chars per line. Empty paragraphs preserved as
    blank lines for spacing."""
    out: list[str] = []
    for para in text.split("\n"):
        if not para.strip():
            out.append("")
            continue
        words = para.split()
        line = ""
        for w in words:
            candidate = w if not line else line + " " + w
            if len(candidate) <= max_chars:
                line = candidate
            else:
                out.append(line)
                line = w
        if line:
            out.append(line)
    return out


class PdfBuilder:
    def __init__(self) -> None:
        self.doc = fitz.open()
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self.y = MARGIN_TOP

    def _ensure_room(self, needed: float) -> None:
        if self.y + needed > PAGE_H - MARGIN_BOTTOM:
            self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
            self.y = MARGIN_TOP

    def write_lines(
        self,
        lines: list[str],
        *,
        fontname: str = FONT_BODY,
        fontsize: int = 11,
        line_h: float = 14.0,
        x: float = MARGIN_X,
    ) -> None:
        for ln in lines:
            self._ensure_room(line_h)
            if ln:
                self.page.insert_text(
                    (x, self.y),
                    ln,
                    fontname=fontname,
                    fontsize=fontsize,
                )
            self.y += line_h

    def write_text(
        self,
        text: str,
        *,
        fontname: str = FONT_BODY,
        fontsize: int = 11,
        line_h: float = 14.0,
        max_chars: int = WRAP_CHARS_BODY,
    ) -> None:
        self.write_lines(
            wrap(text, max_chars=max_chars),
            fontname=fontname,
            fontsize=fontsize,
            line_h=line_h,
        )

    def write_heading(self, text: str, *, fontsize: int = 13, top_gap: float = 8.0) -> None:
        self.y += top_gap
        self._ensure_room(fontsize + 6)
        self.page.insert_text(
            (MARGIN_X, self.y),
            text,
            fontname=FONT_BOLD,
            fontsize=fontsize,
        )
        self.y += fontsize + 6

    def write_title(self, text: str) -> None:
        # Title spans two lines if long; center-ish via padding.
        for ln in wrap(text, max_chars=68):
            self._ensure_room(20)
            self.page.insert_text(
                (MARGIN_X, self.y),
                ln,
                fontname=FONT_BOLD,
                fontsize=15,
            )
            self.y += 19

    def gap(self, h: float = 6.0) -> None:
        self.y += h

    def save(self, path: str) -> None:
        self.doc.save(path)
        self.doc.close()


# ── Paper content ─────────────────────────────────────────────────────────

TITLE = "Iron Deficiency Anaemia in Adults: A Primary Care Perspective"

AUTHORS = "A. Test, B. Demo, C. Example"

AFFILIATIONS = (
    "Department of Primary Care Medicine, Example University, Kathmandu, Nepal"
)

# Header block — bundles the classifier's high-value signals (DOI, ORCID,
# Corresponding author, Keywords, Received/Accepted dates, Affiliations)
# on the first page so the 8000-char window catches them all.
METADATA_BLOCK = (
    "DOI: 10.0000/test.demo.2026.0001\n"
    "Corresponding author: A. Test (a.test@example.org)\n"
    "ORCID: 0000-0000-0000-0001\n"
    "Received 1 May 2026  -  Accepted 10 May 2026  -  Published online 17 May 2026\n"
    "Affiliations: Department of Primary Care Medicine, Example University, Kathmandu, Nepal\n"
    "Keywords: iron deficiency, anaemia, primary care, haemoglobin, ferritin, "
    "patient education"
)

ABSTRACT = (
    "Iron deficiency anaemia (IDA) is the most common cause of anaemia worldwide and "
    "a frequent presentation in primary care settings. Identification typically relies "
    "on a complete blood count showing a low haemoglobin level, supported by iron "
    "studies. Common contributing factors include dietary insufficiency, chronic blood "
    "loss, and impaired absorption. This narrative review summarises widely accepted "
    "approaches to recognition, investigation, and conservative management of IDA in "
    "adults, with particular attention to settings where laboratory access may be "
    "limited [1, 2]. The review draws on publicly available patient-education and "
    "primary-care reference material and is intended as a clinician-facing summary "
    "rather than a clinical guideline."
)

INTRODUCTION = (
    "Anaemia affects a substantial fraction of the global population, with iron "
    "deficiency being the predominant cause across most age groups [3]. In low- and "
    "middle-income settings, the burden is particularly pronounced among women of "
    "reproductive age, infants, and young children [4]. Primary care clinicians are "
    "typically the first point of contact for patients presenting with non-specific "
    "symptoms such as fatigue, breathlessness on exertion, pallor, palpitations, hair "
    "thinning, or reduced exercise tolerance [5].\n"
    "\n"
    "Early identification and appropriate investigation are essential. Iron deficiency "
    "is not itself a final diagnosis but a finding that may signal an underlying "
    "process — most commonly inadequate dietary intake, menstrual or other blood loss, "
    "or impaired gastrointestinal absorption [6]. In older adults the threshold for "
    "considering an underlying gastrointestinal source of bleeding is lower, and "
    "appropriate referral is advised [7].\n"
    "\n"
    "This paper does not replace clinical assessment. All management decisions, "
    "including whether to investigate further and whether to initiate supplementation, "
    "should be made by a qualified clinician with access to the patient's full history."
)

METHODS = (
    "This is a narrative review of publicly available primary-care reference material "
    "and patient-education resources from established health-information sources [1, "
    "2, 4]. No new patient data were collected and no individual patient consent was "
    "required. Sources were selected for their general accessibility and their focus "
    "on the primary-care setting rather than tertiary or specialist contexts. Where "
    "sources disagreed, the more conservative recommendation appropriate for "
    "non-specialist settings was preferred."
)

RESULTS = (
    "Across the reviewed sources, common presenting features reported in adults with "
    "iron deficiency anaemia included generalised fatigue, reduced exercise tolerance, "
    "dizziness, occasional headaches, palpitations during exertion, and (in more "
    "advanced cases) brittle nails or hair thinning. Pallor of the conjunctivae or "
    "palmar creases may be observed on examination but is not a sensitive sign [5].\n"
    "\n"
    "Laboratory findings most consistently reported were a haemoglobin level below "
    "the reference range printed on the laboratory report (which varies by age, sex, "
    "and altitude), reduced mean corpuscular volume in established cases, and a low "
    "ferritin level reflecting depleted iron stores [6, 8]. Reviewed sources stressed "
    "that a single out-of-range value should be interpreted in the context of the "
    "patient's clinical picture and prior trend, and that a repeat test is often "
    "appropriate before further investigation [9].\n"
    "\n"
    "Investigation of the underlying cause should consider dietary intake, menstrual "
    "losses in women of reproductive age, possible gastrointestinal blood loss in "
    "older adults, and conditions affecting absorption such as coeliac disease in "
    "appropriate clinical contexts [7, 10]."
)

DISCUSSION = (
    "Management generally combines investigation of the underlying cause with iron "
    "repletion supervised by a clinician. Reviewed patient-education sources "
    "consistently advise discussing any planned supplementation with a clinician "
    "before starting, because inappropriate iron loading can be harmful in certain "
    "conditions and because the right form, route, and duration of supplementation "
    "depend on the individual [2, 8].\n"
    "\n"
    "Dietary advice is appropriate as an adjunct. Iron-rich foods commonly mentioned "
    "in patient-education material include lean meats, pulses (lentils, beans), dark "
    "leafy green vegetables, fortified cereals, and dried fruit [1, 4]. Co-consumption "
    "of vitamin-C-rich foods with iron-containing meals is widely described as "
    "supporting absorption, while tea and coffee taken with meals are widely "
    "described as reducing absorption [10]. These are general dietary considerations "
    "and not a substitute for clinical management of significant deficiency.\n"
    "\n"
    "Patients should be counselled to seek prompt review if they develop concerning "
    "symptoms — such as black or tarry stools, frank rectal bleeding, severe or "
    "worsening shortness of breath, chest discomfort, fainting, or symptoms that "
    "rapidly worsen — and to take their laboratory reports with them to any "
    "appointment so the clinician can review actual values and reference ranges [5, 9]."
)

CONCLUSION = (
    "Iron deficiency anaemia remains a common and important primary care diagnosis. A "
    "structured approach to recognition, investigation of the underlying cause, and "
    "clinician-supervised management — combined with appropriate follow-up — is "
    "essential. Patient education should emphasise the value of bringing laboratory "
    "reports to consultations, reporting concerning new symptoms promptly, and "
    "discussing supplementation with a clinician rather than self-prescribing."
)

REFERENCES = (
    "[1] World Health Organization. Nutritional anaemias: tools for effective "
    "prevention and control. Patient-education reference.\n"
    "[2] National Institute for Health and Care Excellence. Clinical Knowledge "
    "Summaries: Anaemia - iron deficiency. Primary care reference.\n"
    "[3] World Health Organization. Global anaemia estimates. Public-health "
    "reference.\n"
    "[4] MedlinePlus. Iron deficiency anemia. Patient-education reference.\n"
    "[5] BMJ Best Practice. Iron deficiency anaemia. Primary care reference.\n"
    "[6] Goddard AF, James MW, McIntyre AS, Scott BB. Guidelines for the management "
    "of iron deficiency anaemia in adults. General-reference summary.\n"
    "[7] American Academy of Family Physicians. Iron deficiency anemia in adults. "
    "Primary care reference.\n"
    "[8] Lynch SR, Cook JD. Interaction of vitamin C and iron in absorption. "
    "General-reference summary.\n"
    "[9] National Health Service. Iron deficiency anaemia - diagnosis and follow-up. "
    "Patient-education reference.\n"
    "[10] World Health Organization. Iron deficiency anaemia: assessment, prevention "
    "and control. Public-health reference."
)


def build() -> None:
    pdf = PdfBuilder()

    pdf.write_title(TITLE)
    pdf.gap(4)
    pdf.write_text(AUTHORS, fontname=FONT_ITALIC, fontsize=11)
    pdf.write_text(AFFILIATIONS, fontsize=10, line_h=12)
    pdf.gap(4)
    pdf.write_text(METADATA_BLOCK, fontsize=10, line_h=12)

    pdf.write_heading("Abstract")
    pdf.write_text(ABSTRACT)

    pdf.write_heading("Introduction")
    pdf.write_text(INTRODUCTION)

    pdf.write_heading("Methods")
    pdf.write_text(METHODS)

    pdf.write_heading("Results")
    pdf.write_text(RESULTS)

    pdf.write_heading("Discussion")
    pdf.write_text(DISCUSSION)

    pdf.write_heading("Conclusion")
    pdf.write_text(CONCLUSION)

    pdf.write_heading("References")
    pdf.write_text(REFERENCES, fontsize=10, line_h=12, max_chars=WRAP_CHARS_REFS)

    pdf.save(OUTPUT_PATH)


if __name__ == "__main__":
    build()
    print(f"Wrote {OUTPUT_PATH}")
