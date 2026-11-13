from hinglishguard.discord_bot import deletion_allowed, revision


def test_delete_requires_complete_flag():
    result = dict(recommendation='FLAG', error_status=None, evidence=[{'quote': 'x'}], run_id='a')
    assert deletion_allowed(result)
    for key, value in [('recommendation', 'HUMAN_REVIEW'), ('error_status', 'timeout'), ('evidence', []), ('run_id', '')]:
        assert not deletion_allowed({**result, key: value})


def test_context_changes_revision():
    assert revision('same', 'parent one') != revision('same', 'parent two')
    assert revision('same', None) == revision('same', None)
