from hinglishguard.community_policy import matches, review, safe_warning, DEFAULT_WARNING


def test_whole_words():
    assert matches('you are a fucking asshole') == ['fucking', 'asshole']
    assert matches('tu chutiya hai') == ['chutiya']
    assert not matches('class assignment and butterfly')


def test_failed_investigation_cannot_be_overridden():
    result = {'error_status': 'timeout', 'reason_codes': ['timeout']}
    assert review('fuck you', None, result) is result
    result = {'error_status': None, 'reason_codes': ['contextual_ambiguity']}
    assert review('fuck you', None, result) is result


def test_warning_mentions_and_profanity_blocked():
    assert safe_warning('@everyone stop this now') == DEFAULT_WARNING
    assert safe_warning('Stop fucking doing that') == DEFAULT_WARNING
