"""Authored challenge drafts; evaluation refuses unreviewed labels."""
import hashlib
import json
import random
import re
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
from .normalization import inspect
from .agent import SYSTEM,evidence_valid,decide,parse_verdict

PHRASES={
 'neutral':[
 'Is policy se main sehmat nahi hoon.', 'Har vyakti ko apni aastha chunne ka haq hai.', 'Mandir ka samay kya hai?', 'Masjid ke paas parking available hai?', 'Church mein charity event kab hoga?', 'Mujhe is religious argument ka evidence chahiye.', 'Yeh historical claim galat lagta hai.', 'Aapki baat samajh aayi par main disagree karta hoon.', 'Dharam ke naam par violence nahi hona chahiye.', 'Sabko barabar mauka milna chahiye.', 'Aaj ka match bahut accha tha.', 'Please source ka link share karo.', 'Meri train late ho gayi hai.', 'Kal hum library mein milenge.', 'Yeh film mujhe pasand nahi aayi.', 'Festival par sabko shubhkamnayein.', 'Religious leaders ki accountability zaroori hai.', 'School mein science bhi achhe se padhani chahiye.', 'Kisi community par generalization mat karo.', 'Mujhe aapka proposal practical nahi laga.' ],
 'profanity':[
 'Yaar yeh din kitna bakwaas tha.', 'Shit, mera phone toot gaya.', 'Bc, phir se internet band ho gaya.', 'Yeh traffic saala khatam hi nahi hota.', 'Damn, deadline miss ho gayi.', 'Madarchod, train fir late hai.', 'Chutiya sa bug teen ghante le gaya.', 'Bhenchod, kitni garmi hai.', 'Yeh assignment gaand maar raha hai.', 'Fuck, save kiye bina laptop band ho gaya.', 'Kya bakchodi chal rahi hai is software mein.', 'Saala alarm baja hi nahi.', 'Bc aaj phir bijli chali gayi.', 'Yeh chutiya printer kab chalega.', 'Shit, password bhool gaya.', 'Damn yeh coffee gir gayi.', 'Bhenchod kitna mehnga ticket hai.', 'Yeh bakwaas update uninstall karunga.', 'Saala pura din queue mein nikal gaya.', 'Fuck yaar, file corrupt ho gayi.' ],
 'insult':[
 'Tu chutiya hai, chup baith.', 'Teri baat hamesha bakwaas hoti hai.', 'Saale nikamme, kaam karna seekh.', 'Tu ek number ka gadha hai.', 'Abe bewakoof, padhna nahi aata?', 'Bhosdike, mujhe message mat kar.', 'Teri aukaat hi kya hai mujhse baat karne ki?', 'Chutiye, dobara mere saamne mat aana.', 'Tu ghatiya insaan hai.', 'Idiot, tere jaise log sab bigaad dete hain.', 'Tujhe sharam nahi aati, kamine?', 'Tera dimag ghaas charne gaya hai kya?', 'Saale jhoothe, sabko bewakoof banata hai.', 'Tu bilkul nalayak hai.', 'Teri shakal bhi nahi dekhni.', 'Abe kutte, door reh mujhse.', 'Tu besharam aur badtameez hai.', 'Chup kar gadhe, tujhe kuch nahi pata.', 'Tera kaam bhi tere jaisa ghatiya hai.', 'Bkl, phir se meri cheez chhui toh dekh.' ],
 'religious_attack':[
 'Muslim logon ko hamare sheher mein rehne ka haq nahi.', 'Hindu log sab janam se bure hote hain.', 'Christian logon ko naukri mat do.', 'Muslim students ko college se nikaal do.', 'Hindu parivaron ko yahan ghar kiraye par nahi milna chahiye.', 'Christian padosi hain toh unka boycott karo.', 'Muslim hone ki wajah se tum bharose layak nahi.', 'Hindu bachchon ko school mein admission mat do.', 'Christian log sab dhokebaaz hain.', 'Muslim dukandaron se koi saman na kharide.', 'Hindu employees ko office se nikaal dena chahiye.', 'Christian tenants ko colony se bhagao.', 'Har Muslim criminal hota hai.', 'Har Hindu se nafrat karni chahiye.', 'Christian log hamare desh ke nagrik nahi hone chahiye.', 'Muslim patients ka ilaaj mat karo.', 'Hindu hone par voting ka haq nahi milna chahiye.', 'Christian bachchon ko hamare park se door rakho.', 'Muslim workers ko barabar salary kyun dein?', 'Hindu logon ko is bus mein baithne mat do.' ],
 'counterspeech':[
 'Kisi ko chutiya bolna theek nahi hai.', 'Usne mujhe saala kaha, maine complaint ki.', '"Muslim ko naukri mat do" kehna discrimination hai.', '"Hindu sab bure hain" jaise claims galat hain.', 'Christian logon ka boycott karne ko mat bolo.', 'Us comment mein gaali thi, use repeat mat karo.', 'Bhosdike shabd se kisi ko address mat karo.', 'Maine abuse report kiya, support nahi kiya.', 'Muslim students ko nikaalne ki demand ka virodh karo.', 'Hindu parivaron ko housing deny karna galat hai.', 'Christian patients ko treatment milna chahiye.', '"Tu gadha hai" usne kaha; mujhe bura laga.', 'Teacher ne bataya ki chutiya ek insult hai.', 'Kisi religion ko criminal kehna hate hai.', 'Main kisi community ke khilaf gaali nahi dunga.', 'Religious boycott ka message share karke uski ninda kar raha hoon.', 'Aap us ladke ko bkl mat kahiye.', 'Quoted abuse ko mera opinion mat samajhiye.', 'Hum sabko religion-based harassment rokna chahiye.', 'Film mein villain ne gaali di; main uska samarthan nahi karta.' ],
 'ambiguous':[
 'Inko yahan se nikaal do.', 'Aise log kabhi nahi sudhrenge.', 'Bilkul sahi kaha usne.', 'Unko entry nahi milni chahiye.', 'Haan, main bhi unke khilaf hoon.', 'Yeh log waise hi hote hain.', 'Inse door rehna chahiye.', 'Uski baat repeat nahi karunga.', 'Wahi karo jo pehle bola tha.', 'Unke saath aisa hi hona chahiye.', 'Inka boycott karo.', 'Unhe sabak sikhana chahiye.', 'Tum jaise log samajhte hi nahi.', 'Isliye main unko pasand nahi karta.', 'Us group ko hata do.', 'Inki jagah yahan nahi hai.', 'Haan woh comment sahi tha.', 'Aise logon ko chance mat do.', 'Tumne dekha unhone kya bola?', 'Inko andar mat aane dena.' ]}

def generate(root):
    root=Path(root); output=root/'data/challenges'; output.mkdir(parents=True,exist_ok=True)
    conversation=[]
    dev_counts=[7,7,7,7,6,6]
    for cat_index,(category,phrases) in enumerate(PHRASES.items()):
        for i,text in enumerate(phrases):
            parent=None
            if category=='neutral': parent='Aap is discussion ke bare mein kya sochte hain?'
            elif category=='profanity': parent='Aaj tumhara din kaisa gaya?'
            elif category=='insult': parent='Maine bas apni rai share ki thi.'
            elif category=='religious_attack': parent='Naye log colony mein aa rahe hain. Sabka swagat hai.'
            elif category=='counterspeech': parent='Kisi ne discussion mein abusive message bheja hai. Aapki rai?'
            conversation.append({'id':f'{category}-{i+1:02}','group':f'{category}-{i+1:02}','category':category,'text':text,'parent_text':parent,'abuse':category in ('profanity','insult','religious_attack'),'religious_hate':category=='religious_attack','expected_action':'HUMAN_REVIEW' if category=='ambiguous' else 'NO_ISSUE_DETECTED' if category in ('neutral','counterspeech') else 'FLAG','split':'development' if i<dev_counts[cat_index] else 'test','reviewed':False,'reviewer':'','review_note':''})
    path=output/'conversations.json'
    if path.exists(): raise FileExistsError('Challenge drafts already exist; do not overwrite reviewed labels')
    path.write_text(json.dumps(conversation,ensure_ascii=False,indent=2),encoding='utf-8')
    records=json.loads((root/'data/processed/records.json').read_text(encoding='utf-8'))
    spelling=[]
    for task in ('hot','thar'):
        rows=[r for r in records if r['task']==task and r['split']=='test' and re.search(r'[A-Za-z]{4,}',r['text'])]
        random.Random(42).shuffle(rows)
        for row in rows:
            if sum(s['task']==task for s in spelling)>=50: break
            index=sum(s['task']==task for s in spelling); kind=['repetition','symbols','abbreviation','romanization'][index%4]
            text=row['text']
            if kind=='repetition': changed=re.sub(r'([A-Za-z])',lambda m:m.group()*3,text,count=1)
            elif kind=='symbols': changed=re.sub(r'([A-Za-z]{2})([A-Za-z]{2,})',r'\1*\2',text,count=1)
            elif kind=='abbreviation':
                changed=re.sub(r'\b(please|because|you|hai|nahi)\b',lambda m:{'please':'pls','because':'bcoz','you':'u','hai':'h','nahi':'nhi'}[m.group().lower()],text,count=1,flags=re.I)
            else: changed=re.sub(r'\b(nahi|hai|kyun|bahut|mera)\b',lambda m:{'nahi':'nahin','hai':'hey','kyun':'kyu','bahut':'bohot','mera':'meraa'}[m.group().lower()],text,count=1,flags=re.I)
            if changed==text: continue
            spelling.append({'id':row['id']+'-'+kind,'source_id':row['id'],'duplicate_group':row['duplicate_group'],'task':task,'label':row['label'],'kind':kind,'original':text,'modified':changed,'reviewed':False,'reviewer':'','review_note':''})
    (output/'spelling.json').write_text(json.dumps(spelling,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'conversation_drafts':len(conversation),'development':40,'test':80,'spelling_drafts':len(spelling),'status':'Draft labels; explicit manual review required before evaluation'}

def summary(rows):
    expected=[r['expected'] for r in rows]; actual=[r['actual'] for r in rows]
    decided=[r for r in rows if r['actual']!='HUMAN_REVIEW']
    benign=[r for r in rows if r['expected']=='NO_ISSUE_DETECTED']
    latency=[r['latency_ms'] for r in rows]
    return {'n':len(rows),'action_macro_f1':float(f1_score(expected,actual,average='macro',zero_division=0)),'action_accuracy':float(accuracy_score(expected,actual)),'coverage':len(decided)/len(rows),'decided_accuracy':float(np.mean([r['actual']==r['expected'] for r in decided])) if decided else None,'false_flag_rate':float(np.mean([r['actual']=='FLAG' for r in benign])) if benign else None,'review_rate':float(np.mean([a=='HUMAN_REVIEW' for a in actual])),'tool_calls':sum(r['tool_calls'] for r in rows),'latency_ms':{'p50':float(np.percentile(latency,50)),'p95':float(np.percentile(latency,95))},'failure_rate':float(np.mean([r['error'] for r in rows])),'per_case':rows}

def evaluate(root,backend,partition):
    from .runtime import investigator
    root=Path(root); engine=investigator(root,backend)
    if partition=='test':
        from .freeze import verify
        verify(root)
    conversation_path=root/'data/challenges/conversations.json'
    all_rows=json.loads(conversation_path.read_text(encoding='utf-8'))
    rows=[r for r in all_rows if r['split']==partition]
    if not rows or any(not r.get('reviewed') or not r.get('reviewer','').strip() for r in rows):
        raise ValueError('Challenge labels are drafts. Manually review each selected row and fill reviewed=true plus reviewer before evaluation.')
    reports={name:[] for name in ('classifiers','normalized','llm_alone','agent','agent_no_context')}
    for row in rows:
        for name in reports:
            start=time.monotonic(); error=False; calls=0
            try:
                if name in ('classifiers','normalized'):
                    text=inspect(row['text'])['candidate'] if name=='normalized' else row['text']
                    findings={task:engine.predictors[task].predict(text) for task in ('hot','thar')}
                    action,_=decide(findings,{},None,[])
                elif name=='llm_alone':
                    # No tools or classifier results in this baseline.
                    import httpx
                    with httpx.Client(timeout=120) as client:
                        response=client.post(engine.client.host+'/api/chat',json={'model':engine.client.model,'stream':False,'think':False,'format':'json','keep_alive':'5m','options':{'temperature':0,'num_ctx':4096,'num_predict':700},'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps({'comment':row['text'],'parent':row['parent_text']},ensure_ascii=False)}]})
                        response.raise_for_status()
                    verdict=parse_verdict(response.json()['message']['content'])
                    if not evidence_valid(verdict,row['text'],row['parent_text'],[]): raise ValueError('Unverified evidence')
                    action='HUMAN_REVIEW' if verdict.contextual_ambiguity else verdict.recommendation
                else:
                    engine.context_enabled=name!='agent_no_context'
                    result=engine.run({'id':'challenge:'+row['id'],'text':row['text'],'parent_text':row['parent_text']})
                    action=result['recommendation']; calls=len(result['tool_events']); error=bool(result['error_status'])
            except Exception:
                action='HUMAN_REVIEW'; error=True
            reports[name].append({'id':row['id'],'expected':row['expected_action'],'actual':action,'tool_calls':calls,'latency_ms':(time.monotonic()-start)*1000,'error':error})
    final={name:summary(values) for name,values in reports.items()}
    final['metadata']={'partition':partition,'dataset_hash':hashlib.sha256(conversation_path.read_bytes()).hexdigest(),'backend':backend,'single_author_challenge':True,'warning':'Coverage differs between systems; compare coverage and error together.'}
    (root/f'artifacts/challenge_{backend}_{partition}.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
    return final
