# Changelog

Every change records **what** changed, **why**, the **files** involved and how it was **verified**.
Newest entries first. Things that were deliberately *not* changed are listed with their reasons.

---

## 2026-11-14 (later) — Validation-selected final systems, extra training data

**Final numbers (selected on validation from 12 pre-registered candidates; test scored once):** HOT 0.800 (95% CI 0.754–0.848), hate-inducing F1 0.681, 93% auto-decided; THAR 0.790 (0.767–0.808), 46% auto-decided. HOT = 3 HingRoBERTa seeds; THAR = HingRoBERTa + HingRoBERTa-mixed + MuRIL.

**Correction:** the earlier 0.806 (TF-IDF + HingRoBERTa) was the best of five configurations compared on the test set, so it is slightly optimistic; its validation score (0.771) would not have selected it. README now quotes 0.800.

**Extra data:** `scripts/intermediate_train.py` pre-trains HingRoBERTa on 42,523 comments from TRAC-1 Hindi (Apache-2.0) and MACD Hindi (no licence stated; research use, never redistributed). 2 comments matching HOT/THAR validation/test text were removed. THAR validation improved 0.768 → 0.785 for the single model, but no final ensemble selected it.

**New:** `scripts/select_ensemble.py`, `scripts/intermediate_train.py`; plot includes the final system.

---

## 2026-11-14 — Hinglish encoder, ensembles, publishing

**Summary:** ran the two planned model experiments, generalised ensembles, and prepared the repository for GitHub. 41/41 tests pass and `ruff check .` is clean.

**Experiments (validation for all tuning; each test set scored once):**

| Model | HOT test macro-F1 (95% CI) | THAR test macro-F1 (95% CI) |
|---|---|---|
| XLM-RoBERTa (existing) | 0.744 (0.691–0.791) | 0.761 (0.740–0.781) |
| TF-IDF + XLM-RoBERTa ensemble | 0.773 (0.722–0.820) | 0.774 (0.754–0.792) |
| HingRoBERTa (`l3cube-pune/hing-roberta`) | 0.789 (0.739–0.834) | 0.779 (0.757–0.797) |
| TF-IDF + HingRoBERTa ensemble | 0.806 (0.756–0.852) | 0.781 (0.760–0.799) |

HOT hate-inducing F1 0.623 → 0.696. Validation auto-decision coverage (≥90% precision gates): THAR 15% → 43%, HOT 78% → 89%. HingRoBERTa trained on the GPU host for 3 epochs (332 s THAR, 89 s HOT) with the unchanged split (hash `3a15976a…`). Existing models were not overwritten; TF-IDF and XLM-R were re-evaluated only to add bootstrap intervals (same point estimates; backups in `artifacts/_backup_eval/`).

**Code:**
- `models.py`: `Predictor` takes the model kind from metadata, so any folder name works; `calibrate_ensemble(..., output_name=)`; new `available_backends()`.
- `cli.py`: `calibrate-ensemble --members a,b --output-name x`; `review` / `evaluate-spelling` / `evaluate-challenges` accept any trained backend.
- `app.py`: sidebar and evaluation screen list every trained model automatically.
- `scripts/plot_results.py`: README results chart (light + dark).
- `tests/test_improvements.py`: named ensemble + backend discovery test.

**Publishing:** MIT license for the code; `data/challenges/spelling.json` git-ignored because it quotes dataset text; README rewritten around results without local paths; demo video and deck moved to a GitHub release.

**Not changed:** the agent was not re-benchmarked: its challenge sets still need human review.

---

## 2026-11-13 — Code review fixes

**Summary:** 40/40 tests pass (26 existing + 14 new), and `ruff check .` is clean. Trained model weights, metadata, split manifest and evaluation files were **not** modified. Nothing here was run on Chirag/GPU.

### Action required by you

| # | What | Why |
|---|---|---|
| 1 | **Regenerate the Discord bot token** in the Discord Developer Portal, then set it with `$env:DISCORD_TOKEN='…'`. After that, delete `config/discord.token`. | The token has been stored in plain text in the project folder. Anyone who received a copy of the folder (zip, USB, cloud sync) has it. |
| 2 | **Your bot no longer deletes messages based on the LLM community check alone.** To restore the old behaviour, add `"community_llm_can_delete": true` to `config/discord.local.json`. | See *Security & safety → Community-LLM deletions* below. Your local config has `community_profanity_rule: true`, so this affects your bot. |
| 3 | Trust the folder in git, then make the first commit: `git config --global --add safe.directory "D:/AGENTIC AND NLP PROJECT/HinglishGuard"`, then `git add -A` and `git commit -m "Baseline"`. | `git init` is done, but the folder is owned by the Windows account `CodexSandboxOffline`, so git refuses to work in it until you mark it trusted. Changing global git config is your decision, so it wasn't done for you. |
| 4 | From an **Administrator** PowerShell, inside the project folder: `Get-ChildItem -Force -Directory -Filter '.test-*' \| ForEach-Object { takeown /f $_.FullName /r /d y; icacls $_.FullName /grant "${env:USERNAME}:F" /t; Remove-Item $_.FullName -Recurse -Force }` | 10 leftover `.test-*` folders belong to `CodexSandboxOffline`. Your account cannot even read their permissions, so they could not be removed. |
| 5 | **Before the final challenge test run**, run `python -m hinglishguard.cli freeze` again. | The code hashes in `artifacts/frozen_run.json` no longer match, so `evaluate-challenges --partition test` will refuse to run, as intended. The manifest was already out of date before this change: `discord_bot.py`, `community_policy.py` and `discord.local.json` were added on Nov 4, after the Oct 31 freeze. Weight, metadata and split hashes are all unchanged. |

### Security & safety

**Token can come from an environment variable.** `hinglishguard/discord_bot.py` (`load_token`)
- *What:* `DISCORD_TOKEN` is read first; `config/discord.token` is only a fallback. The error messages now name the source that failed.
- *Why:* A plain-text secret inside a project folder gets copied whenever the folder does. Environment variables don't travel with the folder.
- *Verified:* `test_token_prefers_environment`.

**Packaging script excludes secrets.** `scripts/package_source.py`
- *What:* `config/discord.token`, `config/discord.local.json`, `.env`, `.git` and all `.test-*` folders are now excluded from `deployment.tar.gz`.
- *Why:* The previous exclusion list didn't mention the token file. If the script had been run after Nov 4, the token would have been inside the deployment tarball. The existing tarball was checked and **does not** contain it.

**Community-LLM deletions are now opt-in.** `discord_bot.py` (`deletion_allowed`, worker); `config/discord.example.json`
- *What:* A FLAG produced only by `community_policy.review`, a single qwen3:4b call, no longer deletes a message unless `community_llm_can_delete: true`. Instead it is audited as `HUMAN_REVIEW_community_llm_suggested_removal`, with the full result attached.
- *Why:* That code path labels itself `validated_on_benchmark: False`. Every other deletion path goes through validated thresholds with at least 90% precision. One small-model judgement shouldn't remove user messages unless the operator chooses that explicitly.
- *Verified:* `test_community_llm_flag_needs_opt_in_to_delete`.

**Mixed-script profanity entry removed.** `hinglishguard/community_policy.py`
- *What:* Removed `chutिया` (Latin `chut` + Devanagari `िया`).
- *Why:* That string never occurs in real text, so it could never match. `chutiya` and `चूतिया` were already in the list.
- *Verified:* `test_profanity_words_are_single_script` checks every entry is a single script.

**Gitignore updated.** `.gitignore`
- *What:* `.test-*/` replaces `.test-tmp*/`, and `.ruff_cache/` was added.
- *Why:* `.test-community-1`, `.test-contextual-1` and `.test-discord-1` didn't match the old pattern.
- *Verified:* `git check-ignore` confirms the token, local config, `artifacts/` and `.test-*` are all ignored.

### Agent & normalization

**Single settings module.** New `hinglishguard/settings.py`, used by `agent.py`, `community_policy.py` and `app.py`
- *What:* The model name and Ollama URL now live in one place. They can be overridden with `HG_AGENT_MODEL` / `HG_OLLAMA_HOST`, and the host is still restricted to loopback.
- *Why:* `qwen3:4b-instruct` and `http://127.0.0.1:11434` were hard-coded in 5 places. `community_policy.py` also skipped the loopback check entirely.

**Cue patterns defined once.** `hinglishguard/agent.py`
- *What:* `QUOTATION_CUE`, `POLICY_CUE` and `REFERENCE_CUE` are module constants built from one shared term list.
- *Why:* The quotation regex was written out twice and the copies could drift apart. The behaviour is identical: the policy cue still adds `theek nahi|galat hai|not okay`.

**One evidence-repair turn.** `hinglishguard/agent.py`
- *What:* If the final JSON quotes evidence that isn't an exact substring, the model gets one correction message and can answer again. A second failure still fails the investigation, as before. The result records `evidence_repairs`.
- *Why:* Small models often paraphrase a quote slightly. Previously that failed the whole investigation and sent it to human review, even when the decision itself was fine. Evidence still has to match exactly, so the check is just as strict.
- *Verified:* `test_one_evidence_repair_then_success`, `test_repeated_fabricated_evidence_still_fails`. The existing `test_missing_context_and_fabricated_evidence` still passes.

**Better spelling normalization.** `hinglishguard/normalization.py`
- *What:*
  - Look-alike characters *between letters* are mapped: `0→o 1→i 3→e 4→a 5→s 7→t @→a $→s`, so `ch0tiya` becomes `chotiya` and `m@darchod` becomes `madarchod`.
  - Four or more single spaced-out letters are joined, so `c h u t i y a` becomes `chutiya`.
  - New `OBFUSCATION_CUE` makes the agent's required `inspect_spelling` check fire for these patterns too.
  - Removed the no-op alias `bhosdike→bhosdike`.
  - `ambiguous` is now true for any edit other than collapsing repeated letters.
- *Why:* The tool previously handled only repeated letters plus 3 aliases. Hinglish abuse very commonly uses number/symbol substitution and spaced letters. Standalone numbers (`500`, `2024`) and `@mentions` are left unchanged on purpose, and the original text is still never modified.
- *Verified:* 4 new tests, plus the existing negation test. A real TF-IDF HOT model classifies `tu ch0tiya hai` as Abusive.

### Models & evaluation

**Bootstrap confidence intervals.** `hinglishguard/models.py` (`bootstrap_macro_f1`, `metrics`)
- *What:* Every validation and test report now includes `macro_f1_ci95`: 1,000 resamples, seed 42, percentile method. The Streamlit Evaluation page shows it when present.
- *Why:* Without an interval, 0.742 vs 0.744 looks like progress. These are the intervals for the existing test results, computed from the saved confusion matrices with no files rewritten:

  | Task | Model | Macro-F1 | 95% CI |
  |---|---|---|---|
  | HOT | TF-IDF | 0.742 | 0.687–0.790 |
  | HOT | XLM-R | 0.744 | 0.687–0.789 |
  | THAR | TF-IDF | 0.729 | 0.706–0.750 |
  | THAR | XLM-R | 0.761 | 0.738–0.781 |

  On HOT, the two models are statistically indistinguishable. On THAR, XLM-R is modestly better.
- *Verified:* `test_bootstrap_interval_contains_point_estimate`.

**Ensemble backend.** `models.py` (`calibrate_ensemble`, `Predictor` for `ensemble`, `batched_probabilities`, `is_truncated`); `cli.py` (`calibrate-ensemble`); `app.py` (sidebar option)
- *What:* The ensemble averages the already-calibrated TF-IDF and transformer probabilities and fits new decision gates on **validation data only**, without retraining. It refuses to run if the members disagree on classes or positive labels, or were trained on a different split.
- *Why:* THAR currently decides only about 15% of comments automatically. Combining two models with different errors is the cheapest thing to try. **It has not been run yet**, because it needs the transformer, which requires torch and the Chirag machine, so no improvement is claimed.
- *Verified:* `test_ensemble_calibration_and_prediction` (end to end with two TF-IDF members on CPU), `test_ensemble_rejects_mismatched_split`.

**Configurable transformer training.** `hinglishguard/transformer.py`, `cli.py`
- *What:* New options `--base-model`, `--seed`, `--epochs` and `--output-name`. All defaults are the previous values (xlm-roberta-base, 42, 3, `transformer`), and they are now recorded in `metadata.json`. `evaluate --backend` accepts any output name.
- *Why:* So you can try Hinglish-specific encoders (`l3cube-pune/hing-roberta`, `google/muril-base-cased`) and measure seed variance without overwriting the working model.
- *Verified:* Not run: training needs CUDA. The CLI wiring is covered by lint and import checks.

### App & storage

**SQL-side paging and filtering.** `hinglishguard/storage.py`, `app.py`
- *What:* `Store.results(limit=None, recommendation=None)` filters and limits in SQL. The review queue shows the newest 500 runs.
- *Why:* The app loaded every run ever saved on every interaction, so it got slower as history grew. Calling with no arguments behaves exactly as before.
- *Verified:* `test_results_limit_and_filter`.

**Cached JSON reads with explicit UTF-8.** `app.py`
- *What:* The audit file, `records.json` (~8 MB), evaluation reports and the keyword report are read through `st.cache_data`, keyed on file modification time. All reads use UTF-8.
- *Why:* `records.json` was re-parsed on every widget click. Two `read_text()` calls had no encoding, so Windows would use cp1252 and could crash on Hindi text.
- *Verified:* `test_application_screens_render` against the real artifacts.

### Discord bot reliability

`hinglishguard/discord_bot.py`

- **Bounded revision cache.** *What:* `RevisionCache` keeps the 10,000 most recently updated message IDs. *Why:* the plain dict grew forever in a long-running bot. *Verified:* `test_revision_cache_is_bounded`.
- **Audit writes off the event loop.** *What:* writes now run through `asyncio.to_thread`, protected by a lock. *Why:* synchronous SQLite commits blocked the Discord gateway loop.
- **Full error logging.** *What:* every `except Exception` now calls `LOG.exception(...)` as well as auditing. *Why:* failures were recorded only as a status string, with no traceback to debug from.

### Housekeeping

| What | Why |
|---|---|
| Deleted `artifacts/demo/page@6049c0bb….webm` | A byte-identical copy of `HinglishGuard_5min.webm`, verified with `cmp` (19 MB). |
| Deleted `hinglishguard.egg-info/` | A stale build byproduct. The venv uses a PEP 660 editable install (`__editable__…pth` + `dist-info`), so nothing reads this folder. |
| Removed 3 unused imports (`challenges.py`, `tests/test_core.py`) | Found by ruff. |
| Added `[tool.ruff]` (rules `E9`, `F`) and `ruff` to the `dev` extra | Catches real bugs (undefined names, unused imports) without forcing a style rewrite. |
| Added `.github/workflows/ci.yml` | Runs lint and tests on Python 3.11 and 3.13 on every push. Tests need no data, weights, GPU or Ollama. |
| `git init` (branch `main`, no commits) | The project had no version control at all. |
| README: new *Optional model experiments*, *Settings* and *Discord bot* sections; updated agent and test notes | Documents everything above. |
| `config/discord.example.json` lists every key with safe defaults | The bot had three undocumented keys. |

### Deliberately not changed

| Recommendation | Why not |
|---|---|
| Reformat the code with a formatter | It would rewrite every line, bury these fixes in noise and change every frozen code hash. Better done as its own commit once git is set up. |
| Rename `requirements.chirag.lock` / `run_chirag.sh` | These names appear in STATUS.md, README, the freeze manifest and on the remote machine. Renaming them would break recorded evidence for cosmetic gain. |
| Convert the 1.1 GB weights to fp16 / ONNX | That changes model hashes and needs re-evaluation on Chirag. torch isn't installed locally. |
| Delete `artifacts/deployment.tar.gz` | It may be what is deployed. It is now outdated, so regenerate it with `python scripts/package_source.py`. |
| Per-user rate limiting in the Discord bot | On reflection this is harmful: skipping investigations for fast posters would let spammers bypass moderation. The queue limit already sends overflow to human review. |
| Kill timed-out classifier threads | Python can't forcibly stop a thread. The real fix is process isolation, which is too large a change for now. This limitation was already documented in the README. |
| Retrain models, focal loss, augmenting the weak HOT class 1 | These need the GPU machine. The tools for it (`--base-model`, `--seed`, ensemble) are now in place. |
| Human annotation and meaning-preservation review | Must be done by a person. It's still the main gap for academic claims (see STATUS.md). |
| Devanagari↔Roman transliteration in the normalizer | Needs a proper transliteration library and review. Noted in the README as not implemented. |
