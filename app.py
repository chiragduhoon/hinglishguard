from pathlib import Path
import json
import pandas as pd
import streamlit as st
from hinglishguard.runtime import investigator
from hinglishguard.models import available_backends
from hinglishguard.importing import parse_upload,csv_export
from hinglishguard.settings import AGENT_MODEL,OLLAMA_TAGS_URL

ROOT=Path(__file__).resolve().parent
QUEUE_PAGE_SIZE=500  # newest runs shown in the queue; older runs stay in SQLite and exports
st.set_page_config(page_title='HinglishGuard',page_icon='🛡️',layout='wide')
st.title('HinglishGuard')
st.caption('Hindi/Hinglish abuse and religious-hate investigation • Self-hosted • Research prototype')
# Every trained model folder appears automatically (e.g. hingroberta, ensemble-hing).
backends=available_backends(ROOT/'artifacts') or ['tfidf','transformer']
backend=st.sidebar.selectbox('Classifier',backends)
if st.sidebar.button('Check local agent connection'):
    try:
        import httpx
        response=httpx.get(OLLAMA_TAGS_URL,timeout=3); response.raise_for_status()
        models=[m['name'] for m in response.json().get('models',[])]
        if AGENT_MODEL in models: st.sidebar.success(f'Ollama and {AGENT_MODEL} are available')
        else: st.sidebar.warning(f'Ollama is running. Pull {AGENT_MODEL} before investigations.')
    except Exception:
        st.sidebar.warning('Ollama unavailable. Classifier findings remain available; investigations require human review.')

@st.cache_data(show_spinner=False)
def read_json(path,mtime_ns):
    # mtime_ns is part of the cache key so a rewritten file is reloaded; records.json is ~8 MB.
    return json.loads(Path(path).read_text(encoding='utf-8'))

def cached_json(path):
    return read_json(str(path),path.stat().st_mtime_ns)

@st.cache_resource
def load(backend,signatures,agent_model):
    return investigator(ROOT,backend,agent_model=agent_model)

resource_files=list((ROOT/'artifacts').glob(f'*/{backend}/metadata.json'))+[ROOT/'hinglishguard'/name for name in ('agent.py','runtime.py','models.py','normalization.py','settings.py')]
signatures=tuple((str(p),p.stat().st_mtime_ns) for p in resource_files)
engine=load(backend,signatures,AGENT_MODEL)
if len(engine.predictors)!=2:
    st.warning('Both trained models are required. Run the data preparation and training commands in README.md. Incomplete investigations go to human review.')
st.sidebar.caption('No issue detected is limited to the supported tasks. It does not certify that a comment is universally safe.')
page=st.sidebar.radio('Workspace',['Review a comment','Review queue','Evaluation'])

def show(result):
    st.subheader(result['recommendation'].replace('_',' ').title())
    st.write(result['brief_explanation'])
    if result.get('error_status'): st.warning('Degraded investigation: '+result['error_status'])
    cols=st.columns(2)
    for col,key,title in zip(cols,['abuse_result','religious_hate_result'],['Offensive language · HOT','Religious hate · THAR']):
        with col:
            st.markdown('**'+title+'**')
            finding=result.get(key)
            if finding:
                st.write(finding['label']); st.bar_chart(pd.DataFrame({'score':finding['scores']}))
                st.caption('Decision branch: '+finding['branch']+' · Scores are model estimates.')
            else: st.caption('Model unavailable')
    st.write('Reasons:',', '.join(result['reason_codes']))
    if result['keyword_matches']: st.write('Keyword signals:',', '.join(result['keyword_matches']))
    for variant in result['normalized_variants']:
        with st.expander('Spelling comparison'):
            st.write('Original:',result['original_text']); st.write('Alternative:',variant['candidate']); st.json(variant['edits'])
    with st.expander('Evidence and tool timeline'):
        st.json(result['evidence']); st.json(result['tool_events'])
    st.download_button('Download investigation JSON',json.dumps(result,ensure_ascii=False,indent=2),file_name='investigation.json',mime='application/json',key='download_'+result['run_id'])

if page=='Review a comment':
    with st.form('review'):
        text=st.text_area('Comment',max_chars=2000,placeholder='Paste a Hindi or Hinglish comment')
        parent=st.text_area('Parent comment (optional)',max_chars=2000)
        submit=st.form_submit_button('Investigate',type='primary')
    if submit:
        try:
            with st.spinner('Checking classifiers and investigating evidence…'):
                st.session_state['last']=engine.run({'text':text,'parent_text':parent or None})
        except Exception as exc: st.error(str(exc))
    if 'last' in st.session_state: show(st.session_state['last'])
elif page=='Review queue':
    upload=st.file_uploader('Import UTF-8 CSV: text required; id, parent_id, parent_text optional',type='csv')
    if upload and st.button('Validate and add to queue'):
        try:
            comments=parse_upload(upload.getvalue())
            for comment in comments: engine.store.add_comment(comment)
            st.session_state['batch']=[c.model_dump() for c in comments]; st.session_state['batch_cursor']=0
            st.success(f'{len(comments)} comments validated. Run or step through the batch below.')
        except Exception as exc: st.error(str(exc))
    batch=st.session_state.get('batch',[]); cursor=st.session_state.get('batch_cursor',0)
    if batch and cursor<len(batch):
        st.progress(cursor/len(batch),text=f'{cursor}/{len(batch)} processed')
        run_all=st.checkbox('Continue automatically between comments',value=False)
        paused=st.button('Pause after current comment')
        if paused: st.session_state['batch_running']=False
        if st.button('Start / resume batch') or st.session_state.get('batch_running',False):
            st.session_state['batch_running']=run_all
            with st.spinner(f'Investigating comment {cursor+1}…'):
                engine.run(batch[cursor])
            st.session_state['batch_cursor']=cursor+1
            if cursor+1==len(batch): st.session_state['batch_running']=False
            st.rerun()
    selected=st.selectbox('Filter recommendation',['ALL','FLAG','HUMAN_REVIEW','NO_ISSUE_DETECTED'])
    # Filter and limit in SQL so a long history does not load on every rerun.
    visible=engine.store.results(limit=QUEUE_PAGE_SIZE,recommendation=None if selected=='ALL' else selected)
    if len(visible)==QUEUE_PAGE_SIZE: st.caption(f'Showing the newest {QUEUE_PAGE_SIZE} investigations.')
    st.dataframe(pd.DataFrame([{'run_id':r['run_id'],'comment':r['original_text'],'recommendation':r['recommendation'],'human_decision':(r['human_override'] or {}).get('decision'),'latency_ms':r['latency_ms']} for r in visible]),hide_index=True)
    if visible:
        rid=st.selectbox('Open investigation',[r['run_id'] for r in visible]); result=next(r for r in visible if r['run_id']==rid)
        show(result)
        with st.form('override'):
            decision=st.selectbox('Human decision',['HUMAN_REVIEW','FLAG','NO_ISSUE_DETECTED']); note=st.text_input('Review note')
            if st.form_submit_button('Save human decision'): engine.store.override(rid,decision,note); st.rerun()
    st.download_button('Export visible CSV',csv_export(visible),'reviews.csv','text/csv')
    st.download_button('Export visible JSON',json.dumps(visible,ensure_ascii=False,indent=2),'reviews.json','application/json')
else:
    audit=ROOT/'data/processed/audit.json'
    if audit.exists():
        st.subheader('Dataset audit'); st.json(cached_json(audit))
        with st.expander('Browse dataset examples'):
            task=st.selectbox('Source dataset',['hot','thar'])
            split=st.selectbox('Partition',['train','validation','test'])
            records=cached_json(ROOT/'data/processed/records.json')
            preview=[{'text':r['original_text'],'source_label':r['label'],'split':r['split']} for r in records if r['task']==task and r['split']==split][:50]
            st.caption('First 50 matching records. Source labels may contain annotation errors; these are research data.')
            st.dataframe(pd.DataFrame(preview),hide_index=True)
    for task in ('hot','thar'):
        st.subheader('HOT · offensive language' if task=='hot' else 'THAR · religious hate')
        for kind in backends:
            report=ROOT/f'artifacts/{task}/{kind}/evaluation.json'
            if report.exists():
                value=cached_json(report); ci=value.get('macro_f1_ci95')
                st.write(kind,'macro-F1:',round(value['macro_f1'],3),f"(95% CI {ci[0]:.3f}–{ci[1]:.3f})" if ci else '')
                st.dataframe(pd.DataFrame(value['confusion_matrix'],index=value['classes'],columns=value['classes']))
                with st.expander(kind+' metrics'): st.json(value)
            else: st.caption(kind+': no held-out evaluation available')
    for report in sorted((ROOT/'artifacts').glob('challenge_*.json')):
        with st.expander(report.stem): st.json(json.loads(report.read_text(encoding='utf-8')))
    for report in sorted((ROOT/'artifacts').glob('spelling_*.json')):
        with st.expander(report.stem): st.json(json.loads(report.read_text(encoding='utf-8')))
    keyword_report=ROOT/'artifacts/keyword_evaluation.json'
    if keyword_report.exists():
        with st.expander('Keyword baselines (binary tasks)'): st.json(cached_json(keyword_report))
    st.caption('Challenge drafts require explicit human review before evaluation. Benchmark metrics and authored challenges are reported separately.')
