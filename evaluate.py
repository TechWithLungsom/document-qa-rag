import json
from pathlib import Path
from rag import ingest,Index,answer
CASES=[('How many days of annual leave?', 'hr_handbook.txt',1),('Who approves leave?', 'hr_handbook.txt',1),('How often can employees work remotely?', 'hr_handbook.txt',2),('How long is probation?', 'hr_handbook.txt',2),('How long is parental leave?', 'hr_handbook.txt',3),('What is the training reimbursement cap?', 'hr_handbook.txt',3),('What is the minimum password length?', 'it_security.txt',1),('When must security incidents be reported?', 'it_security.txt',2),('When do backups run?', 'it_security.txt',2),('What is the recovery time objective?', 'it_security.txt',3),('How long are customer records retained?', 'it_security.txt',3),('What is the office cafeteria menu?',None,None)]
def run():
    output=[]
    for size in [160,500]:
        chunks,stats=ingest('data',size,30); index=Index(chunks)
        for k in [3,5]:
            for q,source,page in CASES:
                relevant={c['id'] for c in chunks if c['source']==source and c['page']==page}
                hits=index.search(q,k); found={c['id'] for c in hits}; tp=len(found & relevant)
                output.append(dict(size=size,k=k,question=q,expected_source=source,expected_page=page,relevant_ids=sorted(relevant),retrieved=hits,precision=tp/len(hits) if hits else 0,recall=tp/len(relevant) if relevant else None,context_relevant=bool(tp),answer=answer(q,hits),answer_support='Extractive passages only; LLM support not evaluated'))
    Path('results/evaluation.json').write_text(json.dumps(output,indent=2))
    q='What is the leave policy?'
    Path('results/filter.json').write_text(json.dumps(dict(query=q,unfiltered=index.search(q,5),hr_only=index.search(q,5,'hr')),indent=2))
    print('Wrote 48 observed evaluation runs and metadata comparison.')
if __name__=='__main__': run()
