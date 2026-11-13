import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone

def tracked(root):
    root=Path(root)
    paths=list((root/'hinglishguard').glob('*.py'))+list((root/'config').glob('*.json'))
    paths+=[root/'pyproject.toml']+list(root.glob('requirements*.lock'))
    paths+=list((root/'artifacts').glob('*/*/metadata.json'))
    paths+=list((root/'artifacts').glob('*/*/model.joblib'))
    paths+=list((root/'artifacts').glob('*/*/model.safetensors'))
    paths+=[root/'data/processed/records.json',root/'data/processed/label_mapping.json']
    paths+=[root/'artifacts/agent_runtime.json']
    values={}
    for p in paths:
        if p.exists():
            digest=hashlib.sha256()
            with p.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
            values[str(p.relative_to(root)).replace('\\','/')]=digest.hexdigest()
    return values

def freeze(root):
    root=Path(root)
    target=root/'artifacts/frozen_run.json'
    value={'created_at':datetime.now(timezone.utc).isoformat(),'files':tracked(root),'note':'Freezes code, mappings, split manifest, model weights and metadata. Test challenge labels remain separately hashed in the evaluation output.'}
    target.write_text(json.dumps(value,indent=2),encoding='utf-8')
    return value

def verify(root):
    target=Path(root)/'artifacts/frozen_run.json'
    if not target.exists(): raise ValueError('Run freeze after development tuning and before final challenge evaluation')
    value=json.loads(target.read_text(encoding='utf-8'))
    if tracked(root)!=value['files']: raise ValueError('Frozen code/model/split configuration changed; do not tune against test results')
    return value
