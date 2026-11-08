# Evaluation notes

This project is in the early backend skeleton phase. The legal pipeline, retrieval index and evaluation harness are intentionally deferred until the source pack and project structure are in place.

Planned evaluation work includes:
- retrieval precision for the expected legal section or FAQ item
- grounding checks against the official local PDFs
- citation validation against page numbers and section metadata
- scope control to avoid unsupported legal topics
- safety behaviour for complex disputes and immediate danger scenarios

Known limitations: the local source pack currently contains only a subset of the official PDFs, so the system must explicitly note missing sources rather than assume the legal position is complete.
