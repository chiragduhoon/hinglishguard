"""GPU training entrypoint, independent for each source task."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .models import temperature, thresholds, metrics

DEFAULT_BASE_MODEL='FacebookAI/xlm-roberta-base'
# Hinglish/Indic alternatives worth comparing: 'l3cube-pune/hing-roberta', 'google/muril-base-cased'.

def train(processed, artifacts, task, batch_size=8, base_model=DEFAULT_BASE_MODEL, seed=42, epochs=3, output_name='transformer'):
    """output_name lets a new base model train beside the existing one instead of overwriting it."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError('Transformer training requires the Chirag CUDA GPU; TF-IDF remains available on CPU')
    set_seed(seed)
    processed=Path(processed); output=Path(artifacts)/task/output_name; output.mkdir(parents=True,exist_ok=True)
    records=json.loads((processed/'records.json').read_text(encoding='utf-8'))
    mapping=json.loads((processed/'label_mapping.json').read_text(encoding='utf-8'))[task]
    classes=sorted(mapping['labels']); idmap={c:i for i,c in enumerate(classes)}
    splits={s:[r for r in records if r['task']==task and r['split']==s] for s in ('train','validation')}
    tokenizer=AutoTokenizer.from_pretrained(base_model)
    class Dataset(torch.utils.data.Dataset):
        def __init__(self,rows): self.rows=rows
        def __len__(self): return len(self.rows)
        def __getitem__(self,i):
            row=self.rows[i]
            return dict(tokenizer(row['text'],truncation=True,max_length=192),labels=idmap[row['label']])
    counts=np.bincount([idmap[r['label']] for r in splits['train']],minlength=len(classes))
    weights=torch.tensor(counts.sum()/(len(classes)*counts),dtype=torch.float32)
    class WeightedTrainer(Trainer):
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            labels=inputs.pop('labels'); outputs=model(**inputs)
            loss=torch.nn.functional.cross_entropy(outputs.logits,labels,weight=weights.to(outputs.logits.device,dtype=outputs.logits.dtype))
            return (loss,outputs) if return_outputs else loss
    model=AutoModelForSequenceClassification.from_pretrained(base_model,num_labels=len(classes),id2label={i:mapping['labels'][c] for i,c in enumerate(classes)},label2id={mapping['labels'][c]:i for i,c in enumerate(classes)})
    def compute(p):
        from sklearn.metrics import f1_score
        return {'macro_f1':f1_score(p.label_ids,p.predictions.argmax(axis=1),average='macro')}
    args=TrainingArguments(output_dir=str(output/'checkpoints'),learning_rate=2e-5,per_device_train_batch_size=batch_size,per_device_eval_batch_size=batch_size,gradient_accumulation_steps=16//batch_size,num_train_epochs=epochs,weight_decay=.01,eval_strategy='epoch',save_strategy='epoch',save_total_limit=1,load_best_model_at_end=True,metric_for_best_model='macro_f1',greater_is_better=True,fp16=True,seed=seed,report_to='none')
    trainer=WeightedTrainer(model=model,args=args,train_dataset=Dataset(splits['train']),eval_dataset=Dataset(splits['validation']),data_collator=DataCollatorWithPadding(tokenizer),compute_metrics=compute)
    trainer.train()
    prediction=trainer.predict(Dataset(splits['validation']))
    temp=temperature(prediction.predictions,prediction.label_ids)
    probs=softmax(prediction.predictions/temp,axis=1)
    gates=thresholds(probs[:,[idmap[c] for c in mapping['positive_labels']]].sum(axis=1),[r['label'] in mapping['positive_labels'] for r in splits['validation']])
    trainer.save_model(str(output)); tokenizer.save_pretrained(output)
    version=hashlib.sha256((output/'model.safetensors').read_bytes()).hexdigest()[:16]
    meta={'task':task,'backend':'transformer','classes':classes,'labels':mapping['labels'],'positive_labels':mapping['positive_labels'],'temperature':temp,'thresholds':gates,'version':version,'base_model':base_model,'seed':seed,'epochs':epochs,'split_hash':hashlib.sha256((processed/'records.json').read_bytes()).hexdigest()}
    (output/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    report={'validation':metrics([r['label'] for r in splits['validation']],probs,classes),'thresholds':gates}
    (output/'training_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
