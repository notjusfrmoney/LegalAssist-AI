# LegalAssist AI

LegalAssist AI is a Python FastAPI backend for a mobile-first wage-rights assistant for India. The current prototype is intentionally narrow: it focuses on wage-related employment rights and it does not generate legal notices or documents.

## Scope

Supported topics in this phase:
- unpaid or delayed salary
- final wages after resignation or termination
- salary deductions
- overtime compensation, subject to the applicable worker and establishment rules

State-specific minimum-wage rates, legal notice drafting, formal complaint generation, wrongful termination advice outside the supported source pack, and guaranteed legal outcomes are not supported.

## Disclaimer

This project is an informational prototype and is not legal advice or legal representation. It provides general legal information based only on the retrieved official sources in the local knowledge base.

## Tech stack

- Python 3.11
- FastAPI
- Pydantic / pydantic-settings
- Official Groq Python SDK
- PyMuPDF
- Sentence Transformers
- FAISS
- Pytest

## Repository structure

```text
legalassist-ai/
├── backend/
│   ├── app/
│   ├── scripts/
│   ├── tests/
│   ├── .env.example
│   └── requirements.txt
├── knowledge_base/
│   ├── raw/
│   ├── extracted/
│   ├── processed/
│   ├── source_manifest.json
│   └── source_checksums.json
├── evaluation/
├── outputs/
├── .gitignore
├── README.md
└── COPILOT_BUILD_GUIDE.md
```

## Setup on Windows PowerShell

```powershell
cd C:\Users\91832\Desktop\kukuFM
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

## Configure environment secrets

Create a local backend `.env` file by copying the example and fill only the required values:

```powershell
Copy-Item backend\.env.example backend\.env
```

Set `GROQ_API_KEY` and `GROQ_MODEL` in `backend/.env`. Conversational generation also supports `GROQ_TEMPERATURE`, `MAX_COMPLETION_TOKENS`, and `GROQ_TIMEOUT_SECONDS`. Retrieval uses the configured local FAISS store and does not send its API key to the frontend. Do not commit `.env` files.

## PDFs and source manifest

Place downloaded legal PDFs in `knowledge_base/raw/` and keep the local manifest updated when you add or rename files. The project must only use source PDFs present in the local folder. Missing files should be recorded clearly in the manifest rather than invented.

## Run the backend

```powershell
cd backend
uvicorn app.main:app --reload
```

Then open:
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/docs

## Conversational RAG

`POST /api/v1/chat` accepts a sequence of user/assistant messages and optional
profile facts. For follow-ups, the backend combines the current message with the
most recent user question to create a retrieval query without a separate query
rewriting model call. It retrieves up to five chunks from the existing normalized
FAISS `IndexFlatIP`, adds their text and provenance to a grounded system prompt,
and requests a structured response from Groq. The response includes the natural
language answer, source citations with chunk IDs and page references, clarification
status, and an optional follow-up question. The legacy `citations` and
`needs_clarification` keys remain for frontend compatibility; `sources` and
`clarification_needed` are also returned.

The assistant is limited to Indian employment wage-rights topics in the local
source pack. Legal claims must be grounded in retrieved excerpts; requests for
unsupported sections are answered with an explicit evidence limitation. Sources
marked `needs_review` are not presented as verified. This is informational only,
not legal advice or legal representation. State-specific rules may be absent, and
Central Rules may not apply to every establishment. Complex or high-risk matters
should be taken to a qualified legal professional.

Run the automated suite without using Groq quota:

```powershell
python -m pytest backend\tests -q
```

For a small real Groq integration check (uses the configured API key and incurs
normal API usage), run:

```powershell
python backend\scripts\run_phase5_manual_test.py
```

Representative responses are saved under
`evaluation/phase5_conversation_outputs/`. They contain only the sample requests,
answers and retrieved source references; API credentials are never written there.

## Run tests

```powershell
cd backend
pytest
```

## Known gaps

- State-specific rules and notifications may be missing from the local source pack.
- Central Rules do not automatically apply to every establishment.
- The Groq-generated explanation depends on the quality and coverage of the local source pack and should be checked for high-stakes decisions.
- This is a prototype for general legal information and should not replace legal advice.

## Reproducing the knowledge base

The knowledge base should be rebuilt from local official source PDFs, preserving their provenance, page numbers and metadata. The project intentionally does not download legal PDFs at runtime; it only uses the supplied local files.
