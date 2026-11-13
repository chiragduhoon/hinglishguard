"""Audited source downloads, source-specific parsers and globally grouped splits."""
import csv
import hashlib
import io
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from difflib import SequenceMatcher
import httpx
from .normalization import clean

SOURCES={
 'hot':('sawan16/Hinglish-Offensive-Text-Classification','HOT_Dataset_modified.csv'),
 'lexicon':('sawan16/Hinglish-Offensive-Text-Classification','Hinglish_Profanity_List.csv'),
 'thar':('aakash-dl/THAR','THAR-Dataset.csv'),
}

def download(root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    manifest={}
    with httpx.Client(timeout=60,follow_redirects=True) as client:
        for name,(repo,filename) in SOURCES.items():
            response=client.get(f'https://api.github.com/repos/{repo}/commits/HEAD'); response.raise_for_status()
            revision=response.json()['sha']
            url=f'https://raw.githubusercontent.com/{repo}/{revision}/{filename}'
            response=client.get(url); response.raise_for_status()
            (root/filename).write_bytes(response.content)
            readme=client.get(f'https://raw.githubusercontent.com/{repo}/{revision}/README.md'); readme.raise_for_status()
            (root/f'{name}_README.md').write_text(readme.text,encoding='utf-8')
            manifest[name]={'repo':repo,'file':filename,'revision':revision,'sha256':hashlib.sha256(response.content).hexdigest(),'url':url,'terms_file':f'{name}_README.md'}
    (root/'sources.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest

def read_csv(path):
    # Do not infer a header: HOT has none and includes blank separator rows.
    raw=Path(path).read_bytes()
    try: text=raw.decode('utf-8-sig')
    except UnicodeDecodeError: text=raw.decode('cp1252')
    return list(csv.reader(io.StringIO(text,newline='')))

def parse_hot(path):
    result=[]
    for row in read_csv(path):
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row)<2 or row[0].strip() not in ('0','1','2'):
            raise ValueError(f'Unexpected HOT row shape/label: {row[:1]}')
        if row[1].strip():
            result.append({'task':'hot','text':row[1],'label':row[0].strip()})
    return result

def parse_thar(path, text_column, label_column):
    rows=read_csv(path)
    headers=rows[0]
    if text_column not in headers or label_column not in headers:
        raise ValueError(f'THAR columns are {headers!r}; specify verified text and binary-label columns')
    ti,li=headers.index(text_column),headers.index(label_column)
    return [{'task':'thar','text':r[ti],'label':r[li].strip()} for r in rows[1:] if len(r)>max(ti,li) and r[ti].strip()]

def fingerprint(text):
    value=clean(text).casefold()
    value=re.sub(r'([a-z])\1{2,}',r'\1\1',value)
    return re.sub(r'[^\w\s<>]','',value)

def grouped_records(records):
    """Global exact groups plus >=.94 near duplicates using a length/token index."""
    parents=list(range(len(records)))
    def find(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]]; i=parents[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parents[max(a,b)]=min(a,b)
    keys=[fingerprint(r['text']) for r in records]
    exact={}; inverted=defaultdict(list)
    for i,key in enumerate(keys):
        if key in exact:
            union(i,exact[key]); continue
        exact[key]=i
        tokens=set(key.split()); candidates=Counter()
        for token in tokens:
            if len(token)>2 and token not in ('user','url','hai','the','aur'):
                for j in inverted[token]: candidates[j]+=1
        for j,shared in candidates.items():
            if abs(len(key)-len(keys[j]))<=max(3,int(max(len(key),len(keys[j]))*.07)):
                if shared>=max(1,min(len(tokens),len(set(keys[j].split())))//2) and SequenceMatcher(None,key,keys[j]).ratio()>=.94:
                    union(i,j)
        for token in tokens:
            if len(token)>2: inverted[token].append(i)
    groups=defaultdict(list)
    for i,r in enumerate(records): groups[find(i)].append(r)
    return list(groups.values())

def prepare(raw, output, mapping):
    raw,output=Path(raw),Path(output); output.mkdir(parents=True,exist_ok=True)
    records=parse_hot(raw/SOURCES['hot'][1])+parse_thar(raw/SOURCES['thar'][1],mapping['thar']['text_column'],mapping['thar']['label_column'])
    for r in records:
        task=mapping[r['task']]
        if r['label'] not in task['labels']:
            raise ValueError(f"Unmapped {r['task']} label {r['label']!r}")
        r['original_text']=r['text']; r['text']=clean(r['text'])
    groups=grouped_records(records)
    quarantine=[]; accepted=[]
    for group in groups:
        conflicting=any(len({r['label'] for r in group if r['task']==task})>1 for task in ('hot','thar'))
        if conflicting: quarantine.extend(group)
        else: accepted.append(group)
    # Greedy stratification over all task/label dimensions; group assignment is global.
    totals=Counter((r['task'],r['label']) for g in accepted for r in g)
    counts={s:Counter() for s in ('train','validation','test')}
    fractions={'train':.7,'validation':.15,'test':.15}
    random.Random(42).shuffle(accepted)
    accepted.sort(key=len,reverse=True)
    processed=[]
    for group in accepted:
        gc=Counter((r['task'],r['label']) for r in group)
        split=min(fractions,key=lambda s:sum(((counts[s][k]+n)/(totals[k]*fractions[s])) for k,n in gc.items()))
        counts[split].update(gc)
        gid=hashlib.sha256('|'.join(sorted({fingerprint(r['text']) for r in group})).encode()).hexdigest()[:20]
        seen=set()
        for r in group:
            key=(r['task'],r['text'],r['label'])
            if key in seen: continue
            seen.add(key)
            processed.append(dict(r,id=hashlib.sha256(repr(key).encode()).hexdigest()[:20],duplicate_group=gid,split=split))
    for name,rows in [('records',processed),('quarantine',quarantine)]:
        (output/f'{name}.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'label_mapping.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    audit={'raw_records':len(records),'prepared_records':len(processed),'quarantined':len(quarantine),'groups':len(accepted),'counts':dict(Counter(f"{r['task']}:{r['split']}:{r['label']}" for r in processed))}
    (output/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    # Annotation review sheet, never claim that machine-generated sampling is human review.
    review=[]
    for task in ('hot','thar'):
        for label in mapping[task]['labels']:
            subset=[r for r in processed if r['task']==task and r['label']==label]
            random.Random(42).shuffle(subset)
            review.extend(dict(r,reviewed=False,review_note='') for r in subset[:50])
    (output/'annotation_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    return audit
