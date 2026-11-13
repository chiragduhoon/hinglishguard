"""Tests for the November 2026 fixes; see CHANGELOG.md for the reason behind each one."""
import json
import unicodedata
import pytest
from hinglishguard.normalization import inspect, OBFUSCATION_CUE
from hinglishguard.storage import Store
from hinglishguard.community_policy import WORDS, matches
from hinglishguard.discord_bot import deletion_allowed, RevisionCache, load_token
from hinglishguard.models import bootstrap_macro_f1
from test_core import engine, verdict


# --- Spelling normalization -------------------------------------------------

def test_character_substitutions_inside_words_only():
    result=inspect('tu ch0tiya hai aur m@darchod bhi')
    assert result['candidate']=='tu chotiya hai aur madarchod bhi'
    assert {e['kind'] for e in result['edits']}=={'character_substitution'}
    assert result['ambiguous'] is True
    # Plain numbers, prices and mentions are untouched.
    for text in ['price 500 rupees','call @rahul at 10','2024 mein','normal words 123']:
        assert inspect(text)['candidate']==text

def test_spaced_letters_are_joined():
    result=inspect('tu c h u t i y a hai')
    assert result['candidate']=='tu chutiya hai'
    assert result['edits'][0]['kind']=='spaced_letters'
    assert inspect('a b c')['candidate']=='a b c'  # three letters are too short to be an obfuscation

def test_obfuscation_cue_requires_spelling_check():
    for text in ['ch*tiya','chutiyaaa','ch0tiya','m@darchod','c h u t i y a']:
        assert OBFUSCATION_CUE.search(text), text
    for text in ['Is policy se main sehmat nahi hoon.','price 500','@rahul hello']:
        assert not OBFUSCATION_CUE.search(text), text

def test_spelling_check_becomes_required(tmp_path):
    result=engine(tmp_path,[verdict('ch0tiya')]*4).run({'text':'ch0tiya'})
    assert 'required_checks_incomplete' in result['reason_codes']


# --- Agent evidence repair --------------------------------------------------

def test_one_evidence_repair_then_success(tmp_path):
    result=engine(tmp_path,[verdict('invented'),verdict('hello')]).run({'text':'hello'})
    assert result['recommendation']=='NO_ISSUE_DETECTED'
    assert result['evidence_repairs']==1 and result['error_status'] is None

def test_repeated_fabricated_evidence_still_fails(tmp_path):
    result=engine(tmp_path,[verdict('invented')]*4).run({'text':'hello'})
    assert result['recommendation']=='HUMAN_REVIEW'
    assert 'not present' in result['error_status']


# --- Community rule / Discord ----------------------------------------------

def test_profanity_words_are_single_script():
    for word in WORDS:
        scripts={unicodedata.name(c).split()[0] for c in word}
        assert len(scripts)==1, ascii(word)
    assert matches('tu chutiya hai')==['chutiya']

def test_community_llm_flag_needs_opt_in_to_delete():
    base=dict(recommendation='FLAG',error_status=None,evidence=[{'quote':'x'}],run_id='a')
    community={**base,'community_policy':{'category':'profanity'}}
    assert deletion_allowed(base)
    assert not deletion_allowed(community)
    assert deletion_allowed(community,allow_community_llm=True)

def test_revision_cache_is_bounded():
    cache=RevisionCache(limit=3)
    for i in range(5): cache[i]=str(i)
    assert list(cache)==[2,3,4]
    cache[2]='new'  # updating refreshes recency
    cache[5]='5'
    assert list(cache)==[4,2,5]

def test_token_prefers_environment(tmp_path,monkeypatch):
    (tmp_path/'config').mkdir(); (tmp_path/'config/discord.token').write_text('file-token')
    monkeypatch.setenv('DISCORD_TOKEN','env-token')
    assert load_token(tmp_path)=='env-token'
    monkeypatch.delenv('DISCORD_TOKEN')
    assert load_token(tmp_path)=='file-token'
    (tmp_path/'config/discord.token').unlink()
    with pytest.raises(SystemExit): load_token(tmp_path)


# --- Storage paging ---------------------------------------------------------

def test_results_limit_and_filter(tmp_path):
    e=engine(tmp_path,[verdict('hello')]*3)
    for _ in range(3): e.run({'text':'hello'})
    store=Store(tmp_path/'test.sqlite')
    assert len(store.results())==3
    assert len(store.results(limit=2))==2
    assert len(store.results(recommendation='NO_ISSUE_DETECTED'))==3
    assert store.results(recommendation='FLAG')==[]


# --- Metrics and ensemble ---------------------------------------------------

def test_bootstrap_interval_contains_point_estimate():
    y=['0','1','2']*30; pred=['0','1','2']*25+['1','2','0']*5
    low,high=bootstrap_macro_f1(y,pred,['0','1','2'],rounds=200)
    assert 0<=low<=0.83<=high<=1

def _fixture(tmp_path):
    words={'0':'accha din mausam','1':'nafrat wale log','2':'gandi gaali bakwas'}
    records=[]
    for label,phrase in words.items():
        for i in range(60):
            split='train' if i<36 else 'validation' if i<48 else 'test'
            records.append({'task':'hot','text':f'{phrase} sample{i%7} extra{i%3}','label':label,'split':split})
    processed=tmp_path/'processed'; processed.mkdir()
    (processed/'records.json').write_text(json.dumps(records),encoding='utf-8')
    (processed/'label_mapping.json').write_text(json.dumps({'hot':{'labels':{'0':'normal','1':'hate','2':'abusive'},'positive_labels':['1','2']}}),encoding='utf-8')
    return processed,tmp_path/'artifacts'

def test_ensemble_calibration_and_prediction(tmp_path):
    from hinglishguard.models import train_tfidf,calibrate_ensemble,evaluate,Predictor
    processed,artifacts=_fixture(tmp_path)
    train_tfidf(processed,artifacts,'hot')
    # Two tfidf members keep the test CPU-only; production uses tfidf + transformer.
    report=calibrate_ensemble(processed,artifacts,'hot',members=('tfidf','tfidf'))
    assert report['validation']['macro_f1_ci95'] is not None
    meta=json.loads((artifacts/'hot/ensemble/metadata.json').read_text())
    assert meta['members']==['tfidf','tfidf'] and meta['temperature']==1.0
    finding=Predictor(artifacts,'hot','ensemble').predict('gandi gaali bakwas')
    assert finding.label=='abusive' and finding.version==meta['version']
    held_out=evaluate(processed,artifacts,'hot','ensemble')
    assert held_out['n']==36 and 'macro_f1_ci95' in held_out

def test_ensemble_rejects_mismatched_split(tmp_path):
    from hinglishguard.models import train_tfidf,calibrate_ensemble
    processed,artifacts=_fixture(tmp_path)
    train_tfidf(processed,artifacts,'hot')
    (processed/'records.json').write_text((processed/'records.json').read_text()+' ',encoding='utf-8')
    with pytest.raises(ValueError): calibrate_ensemble(processed,artifacts,'hot',members=('tfidf','tfidf'))


def test_named_ensemble_and_backend_discovery(tmp_path):
    """Ensembles can be saved under any name and are loaded by their metadata, not their folder name."""
    from hinglishguard.models import train_tfidf,calibrate_ensemble,evaluate,Predictor,available_backends
    processed,artifacts=_fixture(tmp_path)
    train_tfidf(processed,artifacts,'hot')
    calibrate_ensemble(processed,artifacts,'hot',members=('tfidf','tfidf'),output_name='ensemble-custom')
    assert (artifacts/'hot/ensemble-custom/metadata.json').exists() and not (artifacts/'hot/ensemble').exists()
    predictor=Predictor(artifacts,'hot','ensemble-custom')
    assert predictor.backend=='ensemble' and len(predictor.members)==2
    assert evaluate(processed,artifacts,'hot','ensemble-custom')['n']>0
    assert available_backends(artifacts)==['tfidf','ensemble-custom']
