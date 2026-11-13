# Build status

## Implemented and verified locally

- Actual dataset downloads and provenance; 14,490 usable records and 14 quarantined conflicting source records.
- Separate HOT/THAR TF-IDF models, calibration, validation gates and held-out metrics.
- Agent tools, validated evidence, persistence, overrides, CSV workflows and three interface screens.
- **21 tests passed locally and on Chirag**, covering core behavior and Streamlit screen rendering. Frozen manifests verified on both machines.
- Dashboard screenshot, source report, presentation deck and confusion-matrix charts.
- Real captioned walkthrough recorded: `artifacts/demo/HinglishGuard_5min.webm`, verified duration **300.16 seconds**. No model responses or metrics were simulated.
- 120 conversation drafts and 100 spelling-pair drafts, with explicit review status.

## Chirag deployment

Connected using the existing Windows SSH config, which is separate from the sandbox's default config.

- Verified GPU: NVIDIA RTX 2000 Ada Generation, 16,380 MiB VRAM.
- Project: `/home/chirag/HinglishGuard`.
- Isolated environment: `.venv`, Python 3.13.11, reusing read-only existing scientific packages; the other environment was not changed.
- Both XLM-RoBERTa models trained for three epochs and completed held-out evaluation. Weights, tokenizers and metadata were copied to the local project.
- HOT transformer macro-F1: 0.7437 on 443 held-out records. THAR transformer macro-F1: 0.7613 on 1,730 held-out records.
- A dedicated Ollama service runs on `127.0.0.1:11434` with project-specific model storage. Its existing runtime binary is reused; the other project's service on port 11435 is unchanged.
- Final agent: qwen3:4b-instruct. The thinking tag failed protocol validation during live checks; it is not the deployed default.
- Five live smoke cases completed without operational errors, with actual context/spelling/policy tool calls. Results are in artifacts/live_smoke.json; they are not accuracy benchmarks.
- Remote dashboard: 127.0.0.1:8501, forwarded to laptop http://127.0.0.1:8502. The local dashboard remains available on port 8501.

## Human work still required

The sampled annotation-review sheet, conversation labels and spelling meaning-preservation checks are drafts for review. No human-reviewed challenge scores are claimed. Use the review fields and commands in README.md before treating these as academic evidence.

## Remaining submission work

Complete human annotation/meaning-preservation review before final challenge scoring. The authored report and slides distinguish executed benchmark results, live smoke evidence and unexecuted human-reviewed experiments. The supplied recording uses explanatory captions rather than human narration; a narration script is also provided.
