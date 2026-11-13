"""Build honest charts and a presentation from available held-out evaluations."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches,Pt

root=Path(__file__).resolve().parents[1]; output=root/'artifacts/presentation'; output.mkdir(parents=True,exist_ok=True)
reports={}
for task in ('hot','thar'):
    for backend in ('tfidf','transformer'):
        path=root/f'artifacts/{task}/{backend}/evaluation.json'
        if path.exists(): reports[(task,backend)]=json.loads(path.read_text())
for (task,backend),report in reports.items():
    fig,ax=plt.subplots(figsize=(6,5))
    matrix=report['confusion_matrix']; ax.imshow(matrix,cmap='Blues')
    for i,row in enumerate(matrix):
        for j,v in enumerate(row): ax.text(j,i,str(v),ha='center',va='center',color='black')
    ax.set(xticks=range(len(report['classes'])),yticks=range(len(report['classes'])),xticklabels=report['classes'],yticklabels=report['classes'],xlabel='Predicted source label',ylabel='True source label',title=f'{task.upper()} / {backend}: held-out confusion matrix')
    fig.tight_layout(); fig.savefig(output/f'{task}_{backend}_confusion.png',dpi=180); plt.close(fig)

slides=[
 ('HinglishGuard','Abuse and religious-hate detection with a bounded investigation agent\nCombined NLP + Agentic AI semester project'),
 ('Problem and research question','Hinglish mixes scripts, spelling variants and context.\nCan selective investigation improve moderation at useful automatic coverage?'),
 ('Independent datasets','HOT: non-offensive, hate-inducing and abusive.\nTHAR: AntiReligion / Non-AntiReligion.\nNever treat non-hate as proof of non-abuse.\nSource revisions and checksums recorded.'),
 ('NLP pipeline','Globally grouped duplicate-safe splits.\nWord + character TF-IDF logistic regression.\nXLM-RoBERTa fine-tuning configuration.\nValidation-only calibration and precision-constrained gates.'),
 ('Agent architecture','Local Qwen3 through Ollama.\nSelect spelling, context, policy and classifier tools.\nValidate evidence against actual observations.\nDeterministic gates apply the final recommendation.'),
 ('Reliability','Bounded tool calls, turns and time.\nNo arbitrary code or external moderation actions.\nMissing context, conflicting results and failures go to human review.\nHuman overrides preserve original predictions.'),
 ('Executed held-out results','\n'.join(f'{t.upper()} / {b}: macro-F1 {r["macro_f1"]:.4f}, n={r["n"]}' for (t,b),r in reports.items())+'\nScores use project-specific grouped splits.'),
 ('Challenge evaluation','120 authored conversation drafts: 40 development / 80 test.\n100 spelling-pair drafts.\nManual review required before scoring.\nCompare coverage, errors, false flags, tool use and latency.'),
 ('Demo and limitations','Show the original text, both classifier results and tool evidence.\nDemonstrate escalation and persistent human override.\nDo not present unexecuted agent or transformer experiments as completed.\nDataset labels and synthetic challenges limit deployment claims.'),
 ('Sources','HOT: sawan16/Hinglish-Offensive-Text-Classification\nMathur et al. 2018: aclanthology.org/W18-5118\nTHAR: github.com/aakash-dl/THAR\nXLM-RoBERTa: FacebookAI/xlm-roberta-base\nOllama: docs.ollama.com/capabilities/tool-calling')]
deck=Presentation(); deck.slide_width=Inches(13.333); deck.slide_height=Inches(7.5)
for title,body in slides:
    slide=deck.slides.add_slide(deck.slide_layouts[1]); slide.shapes.title.text=title
    slide.placeholders[1].text=body
    for paragraph in slide.placeholders[1].text_frame.paragraphs: paragraph.font.size=Pt(25)
for (task,backend),report in reports.items():
    slide=deck.slides.add_slide(deck.slide_layouts[5]); slide.shapes.title.text=f'{task.upper()} / {backend}'
    slide.shapes.add_picture(str(output/f'{task}_{backend}_confusion.png'),Inches(3.4),Inches(1.3),height=Inches(5.8))
deck.save(output/'HinglishGuard.pptx')
print(output/'HinglishGuard.pptx')
