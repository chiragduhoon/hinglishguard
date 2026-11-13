import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix,f1_score
from .models import Predictor
from .data import read_csv
from .normalization import keyword_matches

def keywords(root):
    root=Path(root); processed=root/'data/processed'
    records=json.loads((processed/'records.json').read_text(encoding='utf-8'))
    mapping=json.loads((processed/'label_mapping.json').read_text(encoding='utf-8'))
    lexicon=[r[0].strip() for r in read_csv(root/'data/raw/Hinglish_Profanity_List.csv') if r and r[0].strip()]
    report={}
    for task in ('hot','thar'):
        words=lexicon
        if task=='thar':
            positive=Counter(); negative=Counter()
            for row in records:
                if row['task']==task and row['split']=='train':
                    tokens=set(re.findall(r'\w+',row['text'].lower()))
                    (positive if row['label'] in mapping[task]['positive_labels'] else negative).update(tokens)
            candidates=[(w,(n+1)/(negative[w]+1)) for w,n in positive.items() if n>=10 and len(w)>=3]
            words=[w for w,score in sorted(candidates,key=lambda pair:pair[1],reverse=True)[:200] if score>=3]
        rows=[r for r in records if r['task']==task and r['split']=='test']
        y=[int(r['label'] in mapping[task]['positive_labels']) for r in rows]
        pred=[int(bool(keyword_matches(r['text'],words))) for r in rows]
        report[task]={'n':len(rows),'binary_macro_f1':f1_score(y,pred,average='macro'),'report':classification_report(y,pred,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(y,pred).tolist(),'labels':['negative','positive'],'lexicon_size':len(words),'note':'Binary positive/not-positive baseline, not comparable directly with HOT three-class macro-F1. HOT published lexicon may overlap test vocabulary; THAR terms use training rows only.'}
    (root/'artifacts/keyword_evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

def spelling(root,backend):
    root=Path(root)
    rows=json.loads((root/'data/challenges/spelling.json').read_text(encoding='utf-8'))
    if len(rows)!=100 or any(not r.get('reviewed') or not r.get('reviewer','').strip() for r in rows):
        raise ValueError('All 100 spelling pairs need manual meaning-preservation review and a reviewer name')
    report={}
    for task in ('hot','thar'):
        model=Predictor(root/'artifacts',task,backend)
        selected=[r for r in rows if r['task']==task]
        original=[]; modified=[]; truth=[]
        for row in selected:
            original.append(model.meta['classes'][int(model.probabilities([row['original']])[0].argmax())])
            modified.append(model.meta['classes'][int(model.probabilities([row['modified']])[0].argmax())])
            truth.append(row['label'])
        report[task]={'n':len(selected),'original_macro_f1':f1_score(truth,original,average='macro'),'modified_macro_f1':f1_score(truth,modified,average='macro'),'prediction_consistency':float(np.mean(np.array(original)==np.array(modified))),'original_accuracy':float(np.mean(np.array(original)==np.array(truth))),'modified_accuracy':float(np.mean(np.array(modified)==np.array(truth)))}
    (root/f'artifacts/spelling_{backend}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
