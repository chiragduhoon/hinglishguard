import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from .schema import Finding
from .normalization import clean

def temperature(logits, labels):
    logits=np.asarray(logits)
    result=minimize_scalar(lambda t: -np.log(np.clip(softmax(logits/t,axis=1)[np.arange(len(labels)),labels],1e-12,1)).mean(),bounds=(.1,10),method='bounded')
    return float(result.x)

def thresholds(probability, truth):
    p=np.asarray(probability); y=np.asarray(truth,dtype=bool)
    best={'lower':None,'upper':None,'coverage':0.0,'min_precision':.9,'min_support':30}
    for lower in [None]+list(np.arange(.05,.5,.05)):
        for upper in [None]+list(np.arange(.55,1,.05)):
            neg=np.zeros(len(p),dtype=bool) if lower is None else p<=lower
            pos=np.zeros(len(p),dtype=bool) if upper is None else p>=upper
            if neg.any() and (neg.sum()<30 or (~y[neg]).mean()<.9): continue
            if pos.any() and (pos.sum()<30 or y[pos].mean()<.9): continue
            # An enabled branch must have support; zero-support branches stay disabled.
            if lower is not None and not neg.any(): continue
            if upper is not None and not pos.any(): continue
            coverage=float((neg|pos).mean())
            if coverage>best['coverage']:
                best.update(lower=None if lower is None else float(lower),upper=None if upper is None else float(upper),coverage=coverage)
    return best

def branch(score, gates):
    if gates['upper'] is not None and score>=gates['upper']: return 'positive'
    if gates['lower'] is not None and score<=gates['lower']: return 'negative'
    return 'review'

def bootstrap_macro_f1(y, pred, classes, rounds=1000, seed=42):
    """Percentile 95% interval for macro-F1 by resampling evaluation rows with replacement.

    Lets reports say whether two systems actually differ (e.g. 0.742 vs 0.744 on 443 rows).
    """
    y=np.asarray(y); pred=np.asarray(pred); rng=np.random.default_rng(seed)
    if len(y)==0: return None
    scores=[f1_score(y[i],pred[i],labels=classes,average='macro',zero_division=0) for i in (rng.integers(0,len(y),len(y)) for _ in range(rounds))]
    return [float(np.percentile(scores,2.5)),float(np.percentile(scores,97.5))]

def metrics(y, probabilities, classes):
    pred=np.asarray(classes)[np.argmax(probabilities,axis=1)]
    truth=np.array([[int(v==c) for c in classes] for v in y])
    return {'macro_f1':float(f1_score(y,pred,average='macro')),'macro_f1_ci95':bootstrap_macro_f1(y,pred,list(classes)),'report':classification_report(y,pred,labels=classes,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(y,pred,labels=classes).tolist(),'classes':list(classes),'brier_multiclass':float(np.square(probabilities-truth).sum(axis=1).mean())}

def logits_of(model, texts):
    logits=model.decision_function(texts)
    return np.column_stack([-logits/2,logits/2]) if logits.ndim==1 else logits

def train_tfidf(processed, artifacts, task):
    processed,artifacts=Path(processed),Path(artifacts)
    records=json.loads((processed/'records.json').read_text(encoding='utf-8'))
    mapping=json.loads((processed/'label_mapping.json').read_text(encoding='utf-8'))[task]
    splits={s:[r for r in records if r['task']==task and r['split']==s] for s in ('train','validation','test')}
    model=Pipeline([('features',FeatureUnion([('word',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=40000,sublinear_tf=True)),('char',TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=60000,sublinear_tf=True))])),('classifier',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42))])
    model.fit([r['text'] for r in splits['train']],[r['label'] for r in splits['train']])
    classes=list(model.classes_); val=splits['validation']
    logits=logits_of(model,[r['text'] for r in val])
    temp=temperature(logits,np.array([classes.index(r['label']) for r in val]))
    probs=softmax(logits/temp,axis=1)
    positive=[classes.index(c) for c in mapping['positive_labels']]
    gates=thresholds(probs[:,positive].sum(axis=1),[r['label'] in mapping['positive_labels'] for r in val])
    output=artifacts/task/'tfidf'; output.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,output/'model.joblib')
    meta={'task':task,'backend':'tfidf','labels':mapping['labels'],'positive_labels':mapping['positive_labels'],'classes':classes,'temperature':temp,'thresholds':gates,'version':hashlib.sha256((output/'model.joblib').read_bytes()).hexdigest()[:16],'split_hash':hashlib.sha256((processed/'records.json').read_bytes()).hexdigest()}
    (output/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    report={'validation':metrics([r['label'] for r in val],probs,classes),'thresholds':gates,'status':'validation only; run evaluate after freeze'}
    (output/'training_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

def available_backends(artifacts):
    """Every trained model folder (artifacts/<task>/<name>/metadata.json), tfidf and transformer first."""
    names={p.parent.name for p in Path(artifacts).glob('*/*/metadata.json')}
    return [n for n in ('tfidf','transformer') if n in names]+sorted(names-{'tfidf','transformer'})

class Predictor:
    def __init__(self, artifacts, task, backend='tfidf',device='cpu'):
        self.path=Path(artifacts)/task/backend
        self.meta=json.loads((self.path/'metadata.json').read_text(encoding='utf-8'))
        self.task=task; self.device=device
        # The folder name is free (e.g. 'hingroberta', 'ensemble-hing'); the model kind comes from its metadata.
        self.backend=self.meta.get('backend',backend)
        if self.backend=='ensemble':
            # Members keep their own calibration; the ensemble only averages calibrated probabilities.
            self.members=[Predictor(artifacts,task,member,device) for member in self.meta['members']]
        elif backend=='tfidf':
            self.model=joblib.load(self.path/'model.joblib')
        else:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            # 4.57.6 misidentifies some locally saved non-Mistral tokenizers.
            # Explicit False preserves the trained XLM-R tokenizer unchanged.
            self.tokenizer=AutoTokenizer.from_pretrained(self.path,local_files_only=True,fix_mistral_regex=False)
            self.model=AutoModelForSequenceClassification.from_pretrained(self.path,local_files_only=True).to(device).eval()

    def probabilities(self,texts):
        if self.backend=='ensemble':
            return np.mean([member.probabilities(texts) for member in self.members],axis=0)
        texts=[clean(t) for t in texts]
        if self.backend=='tfidf': logits=logits_of(self.model,texts)
        else:
            import torch
            inputs=self.tokenizer(texts,padding=True,truncation=True,max_length=192,return_tensors='pt').to(self.device)
            with torch.inference_mode(): logits=self.model(**inputs).logits.float().cpu().numpy()
        return softmax(logits/self.meta['temperature'],axis=1)

    def is_truncated(self,text):
        if self.backend=='tfidf': return False
        if self.backend=='ensemble': return any(member.is_truncated(text) for member in self.members)
        return len(self.tokenizer.encode(clean(text)))>192

    def predict(self,text):
        truncated=self.is_truncated(text)
        p=self.probabilities([text])[0]; classes=self.meta['classes']
        positive=float(sum(p[i] for i,c in enumerate(classes) if c in self.meta['positive_labels']))
        return Finding(task=self.task,label=self.meta['labels'][classes[int(p.argmax())]],scores={self.meta['labels'][c]:float(p[i]) for i,c in enumerate(classes)},positive_score=positive,branch=branch(positive,self.meta['thresholds']),version=self.meta['version'],truncated=truncated)

def batched_probabilities(predictor, texts, size=32):
    return np.concatenate([predictor.probabilities(texts[i:i+size]) for i in range(0,len(texts),size)]) if texts else np.empty((0,len(predictor.meta['classes'])))

def calibrate_ensemble(processed, artifacts, task, device='cpu', members=('tfidf','transformer'), output_name='ensemble'):
    """Average already-calibrated member probabilities and fit new gates on validation data only.

    Nothing is retrained. The test split is untouched until `evaluate --backend ensemble`.
    """
    processed,artifacts=Path(processed),Path(artifacts)
    predictors=[Predictor(artifacts,task,member,device) for member in members]
    split_hash=hashlib.sha256((processed/'records.json').read_bytes()).hexdigest()
    first=predictors[0].meta
    for p in predictors:
        if p.meta['classes']!=first['classes'] or p.meta['positive_labels']!=first['positive_labels']:
            raise ValueError(f'{p.backend} uses different classes or positive labels from {predictors[0].backend}')
        if p.meta['split_hash']!=split_hash:
            raise ValueError(f'{p.backend} was trained on a different split manifest')
    records=json.loads((processed/'records.json').read_text(encoding='utf-8'))
    val=[r for r in records if r['task']==task and r['split']=='validation']
    texts=[r['text'] for r in val]
    probs=np.mean([batched_probabilities(p,texts) for p in predictors],axis=0)
    classes=first['classes']
    positive=[classes.index(c) for c in first['positive_labels']]
    gates=thresholds(probs[:,positive].sum(axis=1),[r['label'] in first['positive_labels'] for r in val])
    output=artifacts/task/output_name; output.mkdir(parents=True,exist_ok=True)
    version=hashlib.sha256('|'.join(f"{p.backend}:{p.meta['version']}" for p in predictors).encode()).hexdigest()[:16]
    meta={'task':task,'backend':'ensemble','members':list(members),'member_versions':{p.backend:p.meta['version'] for p in predictors},'labels':first['labels'],'positive_labels':first['positive_labels'],'classes':classes,'temperature':1.0,'thresholds':gates,'version':version,'split_hash':split_hash}
    (output/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    report={'validation':metrics([r['label'] for r in val],probs,classes),'thresholds':gates,'status':'validation only; run evaluate --backend ensemble for held-out metrics'}
    (output/'training_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

def evaluate(processed, artifacts, task,backend='tfidf',device='cpu'):
    predictor=Predictor(artifacts,task,backend,device)
    records=json.loads((Path(processed)/'records.json').read_text(encoding='utf-8'))
    rows=[r for r in records if r['task']==task and r['split']=='test']
    if hashlib.sha256((Path(processed)/'records.json').read_bytes()).hexdigest()!=predictor.meta['split_hash']:
        raise ValueError('Split manifest changed since training')
    probs=[]; latency=[]
    for row in rows:
        start=time.monotonic(); probs.append(predictor.probabilities([row['text']])[0]); latency.append((time.monotonic()-start)*1000)
    report=metrics([r['label'] for r in rows],np.array(probs),predictor.meta['classes'])
    report.update(n=len(rows),latency_ms={'p50':float(np.percentile(latency,50)),'p95':float(np.percentile(latency,95))},model_version=predictor.meta['version'])
    (predictor.path/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
