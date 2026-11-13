# HinglishGuard — implementation and experiment report

## Research question

Does selective spelling normalization and parent-comment investigation improve Hindi/Hinglish moderation at useful automatic coverage, relative to independently trained offensive-language and religious-hate classifiers?

## Method

HOT and THAR remain separate supervised tasks. Globally grouped near/exact duplicates share one split; conflicting labels within a task are quarantined. Original text and source labels are preserved. Word/character TF-IDF logistic regression provides reproducible baselines. Transformer training targets XLM-RoBERTa-base. Each model has validation-only temperature scaling and precision-constrained decision thresholds.

A local Qwen3 agent selects tools to inspect spelling, retrieve supplied context and query either classifier. Final evidence must match observed sources. Deterministic gates retain the authority to flag, clear or escalate a comment. Runtime/tool failures are reported as human review rather than invented successful investigations.

## Executed benchmark

Prepared records: 14,490. Quarantined source records: 14. HOT has 443 held-out examples; THAR has 1,730. These are project-specific grouped splits, not the original papers' split or directly comparable published results.

| Task | TF-IDF test macro-F1 | Test accuracy |
|---|---:|---:|
| HOT, three source classes | 0.7419 | 0.8126 |
| THAR, binary source classes | 0.7290 | 0.7306 |

| Task | XLM-RoBERTa test macro-F1 |
|---|---:|
| HOT, three source classes | 0.7437 |
| THAR, binary source classes | 0.7613 |

Both transformer models completed three epochs on Chirag's RTX 2000 Ada Generation GPU (16,380 MiB). The HOT result is close to the TF-IDF baseline; THAR improves on its baseline. No general claim that transformers always win is supported.

Keyword binary macro-F1: HOT 0.7842 and THAR 0.5988. HOT keyword scores collapse its two offensive categories and are therefore not directly comparable with the three-class result above. Its published lexicon may overlap held-out vocabulary.

## Live integration and outstanding evaluation

The existing Windows SSH configuration successfully connected to Chirag. A dedicated loopback Ollama service reuses the installed runtime without changing the other project's service. Live testing of the default Qwen3 thinking tag produced invalid protocol responses, which correctly escalated to review. The final implementation uses qwen3:4b-instruct with policy-required tool checks and validated final JSON. Its digest is recorded in agent_runtime.json.

Five live smoke investigations completed without operational errors. They exercised context, spelling and policy tool selection, flagged a targeted insult and escalated counterspeech, missing context and untrusted instructions. A neutral comment also escalated because the calibrated classifier gates were uncertain. These observations demonstrate integration, not moderation accuracy or a benchmark improvement.

The 120 conversation cases and 100 spelling pairs remain authored drafts pending manual review; no challenge scores are claimed. Full ablation and robustness commands are implemented and reject unreviewed challenge labels.

## Acceptance evidence

Automated tests verify data boundaries, duplicate/context isolation, original-text preservation, evidence validation, invalid tools/variants, timeouts, step limits, persistent overrides, CSV validation and rendered interface screens. See STATUS.md for current test results and reproduction commands in README.md.

## Limitations

Social-media source labels contain noise and inconsistent contextual interpretation. THAR's AntiReligion labels may not align perfectly with a deployment moderation policy. Romanized/Devanagari and dialect distributions differ across sources. Automatic-threshold validation precision is not a deployment guarantee. Authored challenges require independent review and do not establish production validity. Human feedback is logged without automatic retraining.

## Further evaluation

Freeze source checksums, split hash, model versions, calibration, thresholds and agent configuration. Review development challenge cases, tune only there, then run the frozen 80-case test partition. Report coverage/error jointly, per-category false positives, tool counts and p50/p95 latency. Report spelling consistency only after human confirmation that every perturbation preserves meaning.

## Follow-up experiments (14 Nov 2026)

Two experiments planned in the original report were executed on the same grouped split. All calibration and gate fitting used validation data only.

| Task | XLM-RoBERTa | TF-IDF + XLM-R | HingRoBERTa | TF-IDF + HingRoBERTa |
|---|---:|---:|---:|---:|
| HOT test macro-F1 | 0.744 | 0.773 | 0.789 | **0.806** |
| HOT hate-inducing F1 | 0.623 | 0.638 | 0.667 | **0.696** |
| THAR test macro-F1 | 0.761 | 0.774 | 0.779 | **0.781** |
| THAR validation auto-coverage | 15% | 35% | 38% | **43%** |

A Hinglish-pretrained encoder helps most on the small, three-class HOT task; averaging with TF-IDF adds a further gain because the two make different errors. On HOT the bootstrap intervals still overlap with XLM-RoBERTa (n=443), so the improvement is likely but not established. The agent's contribution remains unmeasured until the challenge sets are human-reviewed.
