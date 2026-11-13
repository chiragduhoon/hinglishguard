import csv
import io
from .schema import Comment

def parse_upload(data):
    if len(data)>2_000_000: raise ValueError('CSV exceeds 2 MB')
    reader=csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
    if not reader.fieldnames or 'text' not in reader.fieldnames: raise ValueError('CSV must have a text column')
    comments=[]
    for i,row in enumerate(reader):
        if i>=100: raise ValueError('Maximum 100 rows per upload')
        payload={k:row[k] for k in ('id','text','parent_id','parent_text') if row.get(k)}
        comments.append(Comment.model_validate(payload))
    return comments

def csv_export(results):
    output=io.StringIO(newline='')
    fields=['run_id','comment_id','original_text','recommendation','brief_explanation','human_decision','latency_ms','error_status']
    writer=csv.DictWriter(output,fieldnames=fields); writer.writeheader()
    for r in results:
        row={k:r.get(k,'') for k in fields}; row['human_decision']=(r.get('human_override') or {}).get('decision','')
        # Neutralize spreadsheet formulas without changing stored or JSON-exported text.
        for k,v in row.items():
            if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')): row[k]="'"+v
        writer.writerow(row)
    return output.getvalue()
