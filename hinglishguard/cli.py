import argparse
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description='HinglishGuard reproducible data, training and investigation commands')
    parser.add_argument('--root',default='.',help='Project root containing data/ and artifacts/')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('download'); sub.add_parser('prepare'); sub.add_parser('doctor')
    sub.add_parser('freeze')
    BACKEND_HELP='tfidf, transformer, or any trained model/ensemble folder name under artifacts/<task>/'
    p=sub.add_parser('train'); p.add_argument('--task',choices=['hot','thar'],required=True)
    p.add_argument('--backend',choices=['tfidf','transformer'],default='tfidf'); p.add_argument('--device',default='cpu')
    p.add_argument('--batch-size',type=int,choices=[4,8],default=8)
    p.add_argument('--base-model',default=None,help='Transformer only, e.g. l3cube-pune/hing-roberta (default: xlm-roberta-base)')
    p.add_argument('--seed',type=int,default=42); p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--output-name',default='transformer',help='Transformer only: artifacts/<task>/<output-name>, so a new base model does not overwrite the existing one')
    p=sub.add_parser('evaluate'); p.add_argument('--task',choices=['hot','thar'],required=True)
    p.add_argument('--backend',default='tfidf',help='tfidf, transformer, ensemble, or a transformer --output-name'); p.add_argument('--device',default='cpu')
    p=sub.add_parser('calibrate-ensemble',help='Average tfidf+transformer probabilities and fit validation gates (no retraining)')
    p.add_argument('--task',choices=['hot','thar'],required=True); p.add_argument('--device',default='cpu')
    p.add_argument('--members',default='tfidf,transformer',help='Comma-separated trained backends to average, e.g. tfidf,hingroberta')
    p.add_argument('--output-name',default='ensemble',help='artifacts/<task>/<output-name>')
    p=sub.add_parser('review'); p.add_argument('text'); p.add_argument('--parent'); p.add_argument('--backend',help=BACKEND_HELP,default='tfidf')
    sub.add_parser('challenge-drafts')
    sub.add_parser('evaluate-keywords')
    p=sub.add_parser('evaluate-spelling'); p.add_argument('--backend',help=BACKEND_HELP,default='tfidf')
    p=sub.add_parser('evaluate-challenges'); p.add_argument('--backend',help=BACKEND_HELP,default='tfidf'); p.add_argument('--partition',choices=['development','test'],default='test')
    args=parser.parse_args(); root=Path(args.root).resolve(); raw=root/'data/raw'; processed=root/'data/processed'; artifacts=root/'artifacts'
    if args.command=='download':
        from .data import download
        result=download(raw)
    elif args.command=='prepare':
        from .data import prepare
        result=prepare(raw,processed,json.loads((root/'config/labels.json').read_text(encoding='utf-8')))
    elif args.command=='train':
        if args.backend=='tfidf':
            from .models import train_tfidf
            result=train_tfidf(processed,artifacts,args.task)
        else:
            from .transformer import train,DEFAULT_BASE_MODEL
            result=train(processed,artifacts,args.task,args.batch_size,base_model=args.base_model or DEFAULT_BASE_MODEL,seed=args.seed,epochs=args.epochs,output_name=args.output_name)
    elif args.command=='evaluate':
        from .models import evaluate
        result=evaluate(processed,artifacts,args.task,args.backend,args.device)
    elif args.command=='calibrate-ensemble':
        from .models import calibrate_ensemble
        result=calibrate_ensemble(processed,artifacts,args.task,args.device,members=tuple(m.strip() for m in args.members.split(',') if m.strip()),output_name=args.output_name)
    elif args.command=='review':
        from .runtime import investigator
        result=investigator(root,args.backend).run({'text':args.text,'parent_text':args.parent})
    elif args.command=='challenge-drafts':
        from .challenges import generate
        result=generate(root)
    elif args.command=='evaluate-challenges':
        from .challenges import evaluate
        result=evaluate(root,args.backend,args.partition)
    elif args.command=='evaluate-spelling':
        from .evaluation import spelling
        result=spelling(root,args.backend)
    elif args.command=='evaluate-keywords':
        from .evaluation import keywords
        result=keywords(root)
    elif args.command=='freeze':
        from .freeze import freeze
        result=freeze(root)
    else:
        import importlib.util,platform,shutil
        result={'python':platform.python_version(),'ollama_executable':shutil.which('ollama'),'packages':{p:bool(importlib.util.find_spec(p)) for p in ['torch','transformers','streamlit','pytest']}}
        try:
            import torch
            result.update(cuda=torch.cuda.is_available(),gpus=[{'name':torch.cuda.get_device_name(i),'vram_gb':round(torch.cuda.get_device_properties(i).total_memory/2**30,1)} for i in range(torch.cuda.device_count())])
        except Exception as e: result['gpu_error']=str(e)
    print(json.dumps(result,ensure_ascii=True,indent=2))

if __name__=='__main__': main()
