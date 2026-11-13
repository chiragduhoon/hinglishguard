import os
from pathlib import Path
from .storage import Store
from .models import Predictor
from .agent import Investigator,OllamaClient
from .data import read_csv

def investigator(root,backend='tfidf',context_enabled=True,agent_model='qwen3:4b-instruct'):
    root=Path(root); artifacts=root/'artifacts'; artifacts.mkdir(exist_ok=True)
    predictors={}
    for task in ('hot','thar'):
        if (artifacts/task/backend/'metadata.json').exists():
            predictors[task]=Predictor(artifacts,task,backend,os.environ.get('HG_CLASSIFIER_DEVICE','cpu'))
    lexicon_file=root/'data/raw/Hinglish_Profanity_List.csv'
    lexicon=[r[0] for r in read_csv(lexicon_file) if r and r[0].strip()] if lexicon_file.exists() else []
    return Investigator(Store(artifacts/'hinglishguard.sqlite'),predictors,client=OllamaClient(model=agent_model),lexicon=lexicon,context_enabled=context_enabled)
