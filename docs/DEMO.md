# Five-minute demonstration script

Before recording: complete remote GPU/Ollama setup, select the available trained backend, verify offline inference, and retain a backup recording. Do not present recorded classifier-only runs as live agent reasoning.

- **0:00–0:40:** Explain the two independent NLP tasks and the investigation goal.
- **0:40–1:20:** Review a neutral Hinglish comment; show both classifier scores and the decision gates.
- **1:20–2:00:** Review a targeted insult, then a reviewed obfuscated variant; compare tool activity and preserved original text.
- **2:00–2:50:** Review counterspeech with a parent comment; show the retrieved evidence and any escalation.
- **2:50–3:30:** Submit an ambiguous reply with no parent; explain human review rather than invented context.
- **3:30–4:10:** Import examples/comments.csv, process a case, record a human override and export results.
- **4:10–5:00:** Show the separate held-out metrics and coverage/error trade-off. State which experiments ran and the remaining limitations.

Both GPU-trained classifiers and the live instruction-model agent are now deployed on Chirag. The dedicated dashboard is forwarded to localhost:8502. If the SSH tunnel is closed, demonstrate the visible unavailable outcome honestly rather than presenting it as a successful investigation.
