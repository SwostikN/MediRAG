"""Unit tests for app/document_classifier.is_medically_relevant.

Locks in behaviour for the upload-gate domain check after the
2026-04 fix that added a CS/engineering counter-signal.

Cases:
  1. A CS/engineering design document ABOUT a medical RAG system
     (trivially mentions "patient", "diagnosis", "NICE", etc.) must be
     REJECTED — this is the RAG_Filtering_Documentation.pdf regression.
  2. A real NICE/NHS-style clinical summary must be ACCEPTED.
  3. A plain non-medical document (cooking recipe, civil engineering
     abstract) must be REJECTED.

Run: pytest eval/test_document_classifier.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.document_classifier import is_medically_relevant  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures — short excerpts chosen to mirror real upload inputs.
# ---------------------------------------------------------------------------

CS_MEDICAL_RAG_DOC = """
RAG Filtering Documentation

Overview
--------
This document describes the retrieval pipeline for our medical
question-answering system. The system uses pgvector plus MedCPT
embeddings with Cohere rerank. Below we outline the domain filter
used to reject off-topic uploads.

Design
------
We maintain a list of medical anchor phrases and compute cosine
similarity between each candidate chunk embedding and the anchor
embeddings.

```python
MEDICAL_ANCHORS = [
    "symptoms diagnosis treatment disease medication",
    "clinical guidelines drug dosage patient care",
    "pharmacology therapeutic intervention adverse effects",
]

def is_medical(chunk: str) -> bool:
    emb = embed(chunk)
    return max(cosine(emb, a) for a in anchor_embeddings) > 0.35
```

References to NICE, NHS, WHO guidelines are common in the corpus
so the anchor set covers those acronyms. The reranker is a
cross-encoder fine-tuned on PubMed pairs.

Deployment
----------
The FastAPI backend runs on Docker; the vector store is Postgres
with the pgvector extension. We use Cohere's rerank-3 endpoint
for the final top-k selection. Chunk size is 512 tokens with a
64-token sliding window overlap.
"""


REAL_MEDICAL_PAPER = """
Abstract

Background: Type 2 diabetes mellitus is a major cause of morbidity
and mortality worldwide. The National Institute for Health and Care
Excellence (NICE) recommends metformin as first-line pharmacological
therapy for most patients.

Methods: We conducted a randomised controlled trial of 1,204 patients
with newly-diagnosed type 2 diabetes attending NHS primary care
clinics. Patients were allocated to metformin monotherapy or
metformin plus empagliflozin. The primary outcome was HbA1c at
12 months; secondary outcomes included incidence of hypoglycaemia,
weight change, and patient-reported symptoms.

Results: Mean HbA1c fell by 1.1% in the combination arm vs 0.7% in
the monotherapy arm (p<0.001). The combination therapy was well
tolerated; adverse effects were mild. No episodes of severe
hypoglycaemia were recorded. Prevalence of treatment discontinuation
was 4.2% in the combination arm. Cardiovascular mortality did not
differ between groups at 12 months.

Conclusions: Early combination therapy achieves greater glycaemic
control than metformin monotherapy in this cohort of patients with
newly-diagnosed diabetes, consistent with recent WHO guidance.
Clinicians should discuss the benefits and risks with patients
before initiating combination therapy.
"""


NON_MEDICAL_DOC = """
An Introduction to Reinforced Concrete Design

This paper reviews the flexural behaviour of reinforced concrete
beams under combined axial and bending loads. The finite element
model was validated against experimental data from the University
of Tokyo structural laboratory. Key findings include a 15% reduction
in peak bending moment when transverse reinforcement is increased
from 0.4% to 0.8% by volume.

We implement the model in ABAQUS using a plasticity-damage
formulation. The concrete compressive strength is 35 MPa and the
steel yield strength is 500 MPa. Results are compared against
Eurocode 2 and ACI 318-19 provisions.
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rejects_cs_document_about_medical_rag():
    """A software design doc that DISCUSSES medical topics but is itself
    a CS/engineering artefact must be rejected. This is the exact
    regression that motivated the 2026-04 fix."""
    assert is_medically_relevant(CS_MEDICAL_RAG_DOC) is False


def test_accepts_real_medical_paper():
    """A standard clinical-trial abstract referencing NICE/NHS/WHO and
    using plain clinical vocabulary must pass the domain gate."""
    assert is_medically_relevant(REAL_MEDICAL_PAPER) is True


def test_rejects_plain_non_medical_document():
    """A civil-engineering paper with no clinical vocabulary at all
    must be rejected — sanity check that the original density rule
    still works."""
    assert is_medically_relevant(NON_MEDICAL_DOC) is False


def test_rejects_empty_text():
    assert is_medically_relevant("") is False
    assert is_medically_relevant(None) is False  # type: ignore[arg-type]
