# Conversation Corpus

The sanitized conversation corpus has been published as a standalone dataset:

**Repository:** [chatgpt-conversation-corpus](https://github.com/queelius/chatgpt-conversation-corpus)

The dataset contains 1,906 sanitized ChatGPT conversations (Dec 2022 -- Apr 2025) with typed PII placeholders and a complete redaction audit trail. Licensed under CC-BY-4.0.

## Reproducing Published Results Without Raw Data

All published figures and tables can be reproduced from the derived data already included in this compendium:

- **Journal paper (PLOS Complex Systems):** All figures use data in `../temporal/`
- **Conference paper (Complex Networks 2025):** Parameter validation uses data in `../ablation/`

The derived data captures the complete statistical summaries; the raw conversations are only needed to regenerate embeddings from scratch or to perform new analyses not covered in the papers.
