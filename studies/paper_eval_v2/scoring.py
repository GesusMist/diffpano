"""Freeze source-text-only scoring descriptions before any new common-v2 scores."""
import copy
import hashlib
import re
from .common import *
POLICY=dict(policy_id='development_short_scoring_v1',
    rule='Per prompt band, retain first sentence (boundary .!? followed by whitespace); at most first50 whitespace-delimited words. No generated images or metric scores consulted.',
    long_text='CLIP BPE still uses historical75-payload-token chunking without silent tokenizer truncation',
    generation_prompts='Unchanged; scoring descriptions are separate records with source hashes')


def prepare(run):
    cohorts=read(Path(run)/'inventory/cohorts.json')
    rows={}
    for prompt in cohorts['prompts']:
        lines=[' '.join(re.split(r'(?<=[.!?])\s+',line.strip(),maxsplit=1)[0].split()[:50]) for line in prompt['effective_lines']]
        raw=('\n'.join(lines)+'\n').encode()
        rows[prompt['prompt_id']]=dict(effective_lines=lines,effective_sha256=hashlib.sha256(raw).hexdigest(),
            policy_id=POLICY['policy_id'],policy_sha256=digest(POLICY),source_prompt_sha256=prompt['effective_sha256'])
    record=dict(policy=POLICY,records=rows)
    freeze(Path(run)/'scoring_prompts.json',record)
    return record


def evaluation_records(run):
    inventory=read(Path(run)/'inventory/records.json')
    rows=inventory['records'] if isinstance(inventory,dict) else inventory
    descriptions=read(Path(run)/'scoring_prompts.json')['records']
    rows=copy.deepcopy(rows)
    confirmation=Path(run)/'generation/inventory_records.json'
    if confirmation.exists():
        rows.extend(copy.deepcopy(read(confirmation)))
        if len({row['case_id'] for row in rows})!=len(rows):raise ValueError('Duplicate development/confirmation case identity')
    for row in rows:
        if row['prompt_id'] in descriptions:row['scoring_prompt']=descriptions[row['prompt_id']]
    return rows
