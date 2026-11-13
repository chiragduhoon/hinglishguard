import re
import unicodedata

# Explicitly reviewed examples. Never infer arbitrary abbreviations or identities.
ALIASES = {'ch*tiya':'chutiya', 'chutiyaa':'chutiya', 'bkl':'behen ke lode'}

# Look-alike characters used to disguise words. They are only replaced when the
# character sits *between letters* (e.g. "ch0tiya", "m@darchod"), so ordinary
# numbers, prices, mentions and e-mail-style tokens are left alone.
SUBSTITUTIONS = {'0':'o', '1':'i', '3':'e', '4':'a', '5':'s', '7':'t', '@':'a', '$':'s'}
_SUBSTITUTION = re.compile(r'(?<=[A-Za-z])([013457@$]+)(?=[A-Za-z])')
# Four or more single letters separated by single spaces/dots/dashes: "c h u t i y a".
_SPACED = re.compile(r'(?<![A-Za-z])((?:[A-Za-z][ ._-]){3,}[A-Za-z])(?![A-Za-z])')
_REPEAT = re.compile(r'([A-Za-z])\1{2,}', re.I)

# Text that should make the agent inspect spelling before giving a verdict.
OBFUSCATION_CUE = re.compile(
    r'(?<=[A-Za-z])[*#._-](?=[A-Za-z])'      # ch*tiya, b.c
    r'|([A-Za-z])\1{2,}'                      # chutiyaaa
    r'|(?<=[A-Za-z])[013457@$]+(?=[A-Za-z])'  # ch0tiya, m@darchod
    r'|(?<![A-Za-z])(?:[A-Za-z][ ._-]){3,}[A-Za-z](?![A-Za-z])',  # c h u t i y a
    re.I)


def clean(text):
    text = unicodedata.normalize('NFC', str(text))
    text = re.sub(r'https?://\S+', '<URL>', text)
    text = re.sub(r'(?<!\w)@[\w]+', '<USER>', text)
    return re.sub(r'\s+', ' ', text).strip()


def inspect(text):
    """Return an uncertain spelling candidate plus every edit; the original is never modified."""
    edits=[]

    def spaced(m):
        joined=re.sub(r'[ ._-]','',m.group(1))
        edits.append({'before':m.group(1), 'after':joined, 'kind':'spaced_letters'})
        return joined

    def substitute(m):
        replacement=''.join(SUBSTITUTIONS[c] for c in m.group(1))
        edits.append({'before':m.group(1), 'after':replacement, 'kind':'character_substitution'})
        return replacement

    def repeat(m):
        replacement=m.group(1)*2
        edits.append({'before':m.group(), 'after':replacement, 'kind':'repetition'})
        return replacement

    candidate=_SPACED.sub(spaced, text)
    candidate=_SUBSTITUTION.sub(substitute, candidate)
    candidate=_REPEAT.sub(repeat, candidate)
    for source,target in ALIASES.items():
        pattern=r'(?<!\w)'+re.escape(source)+r'(?!\w)'
        if re.search(pattern,candidate,re.I):
            edits.append({'before':source,'after':target,'kind':'reviewed_alias'})
            candidate=re.sub(pattern,lambda _:target,candidate,flags=re.I)
    # Anything beyond collapsing repeated letters is an interpretation, not a certainty.
    return {'original':text,'candidate':candidate,'edits':edits,'ambiguous':any(e['kind']!='repetition' for e in edits)}


def keyword_matches(text, words):
    return sorted({w for w in words if w.strip() and re.search(r'(?<!\w)'+re.escape(w.strip())+r'(?!\w)',text,re.I)})
