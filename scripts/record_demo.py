"""Record a real, captioned five-minute walkthrough of the private deployed app.

Requires an active localhost:8502 SSH tunnel and Playwright Chromium.
No responses or scores are simulated. Captions exist only in the recording browser.
"""
from pathlib import Path
import time
from playwright.sync_api import sync_playwright

root=Path(__file__).resolve().parents[1]; output=root/'artifacts/demo'; output.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    context=browser.new_context(viewport={'width':1440,'height':1000},record_video_dir=str(output),record_video_size={'width':1440,'height':1000})
    page=context.new_page(); started=time.monotonic()
    page.goto('http://127.0.0.1:8502',wait_until='networkidle')
    def stage(title,body,seconds=12):
        page.evaluate('''([title,body])=>{let el=document.getElementById('demo-caption');if(!el){el=document.createElement('div');el.id='demo-caption';document.body.appendChild(el);}el.style.cssText='position:fixed;bottom:10px;left:310px;right:18px;z-index:999999;background:#14243a;color:white;padding:18px 22px;border-radius:10px;font:19px system-ui;box-shadow:0 4px 18px #0004;pointer-events:none';el.replaceChildren();let strong=document.createElement('strong');strong.textContent=title;let text=document.createElement('div');text.textContent=body;text.style.marginTop='7px';el.append(strong,text);}''',[title,body])
        print(title,flush=True); page.wait_for_timeout(seconds*1000)
    stage('HinglishGuard — live research prototype','Two independent NLP classifiers plus a local agent that investigates spelling and conversation context.',15)
    page.get_by_role('combobox').first.click(); page.get_by_role('option',name='transformer',exact=True).click()
    stage('GPU-trained classifiers; self-hosted agent','XLM-RoBERTa models were trained on Chirag. Qwen3 4B Instruct uses only approved tools. The interface is private over SSH.',12)
    def review(text,parent):
        page.get_by_label('Comment',exact=True).fill(text)
        page.get_by_label('Parent comment (optional)',exact=True).fill(parent)
        page.get_by_role('button',name='Investigate',exact=True).click()
        page.get_by_role('button',name='Download investigation JSON').wait_for(timeout=90000)
    review('Is policy se main sehmat nahi hoon.','Aap is proposal ke bare mein kya sochte hain?')
    stage('1. Neutral disagreement','Inspect both classifier scores. High-precision decision gates may still request review when evidence is uncertain.',18)
    page.get_by_label('Comment',exact=True).scroll_into_view_if_needed()
    review('Tu ch*tiya hai, chup baith.','Maine bas apni rai share ki thi.')
    stage('2. Disguised spelling','The original comment is preserved. The agent can inspect a reviewed spelling alternative and retrieve the supplied parent.',20)
    page.get_by_text('Evidence and tool timeline',exact=True).click()
    stage('Observable tool activity','The timeline contains actual tool calls and validated observations, rather than a fabricated explanation.',15)
    page.get_by_label('Comment',exact=True).scroll_into_view_if_needed()
    review('Kisi ko chutiya bolna theek nahi hai.','Kisi ne abusive comment bheja hai.')
    stage('3. Counterspeech','A gaali appearing in a condemnation is different from an attack. Conflicting classifier and contextual evidence is escalated.',20)
    page.get_by_label('Comment',exact=True).scroll_into_view_if_needed()
    review('Inko yahan se nikaal do.','')
    stage('4. Missing context','The referent is unclear. Human review preserves uncertainty; the system does not invent a parent conversation.',18)
    page.get_by_text('Review queue',exact=True).click()
    page.get_by_text('Open investigation',exact=True).wait_for()
    stage('5. Persistent review queue','Comments, model outputs, evidence and human decisions are stored separately. CSV imports are validated and deduplicated.',20)
    page.get_by_role('textbox',name='Review note').fill('Demo review: context required; preserve original model recommendation.')
    page.get_by_role('button',name='Save human decision',exact=True).click()
    stage('Human override audit trail','Saving a review appends a human decision without overwriting the original prediction or automatically retraining a model.',15)
    page.get_by_text('Evaluation',exact=True).click()
    page.get_by_text('HOT · offensive language',exact=True).wait_for()
    stage('6. Separate held-out evaluations','HOT preserves three source categories. THAR is a separate binary task. Their labels and headline scores are not merged.',20)
    page.get_by_text('THAR · religious hate',exact=True).scroll_into_view_if_needed()
    stage('Measured results, not promised accuracy','HOT transformer macro-F1: 0.744. THAR transformer macro-F1: 0.761. These are project-specific grouped test splits.',20)
    page.get_by_text('Dataset audit',exact=True).scroll_into_view_if_needed()
    stage('Reproducibility and limits','Source revisions, checksums, grouped splits, calibration and model versions are recorded. Authored challenges still need human review.',18)
    while time.monotonic()-started<300:
        remaining=300-(time.monotonic()-started)
        stage('What this demonstration establishes','Live classifier inference, real tool use, evidence validation and review workflows. It does not establish Instagram performance or human-reviewed challenge accuracy.',min(15,remaining))
    video=page.video
    context.close()
    video.save_as(str(output/'HinglishGuard_5min.webm'))
    browser.close()
print('Saved',output/'HinglishGuard_5min.webm',flush=True)
