"""
inventories.py — the closed label inventories the fact blocks code into.

Their order is fixed: a label's position is its code, in every cache and every
checkpoint. Re-exported from `blocks`, where they used to live.
"""

#: spaCy's ~45 dependency labels collapsed to a coarse, high-frequency set.
#: Coarse is the default: each class is seen far more often, which matters at
#: d_model=128 and is far more legible in a probing evaluation.
#: Switch with frozen.dep_granularity='fine' — a one-line config change.
DEP_COARSE_MAP = {
    # subject-ish
    'nsubj': 'subject', 'nsubjpass': 'subject', 'csubj': 'subject',
    'csubjpass': 'subject', 'expl': 'subject',
    # object-ish
    'dobj': 'object', 'obj': 'object', 'iobj': 'object', 'dative': 'object',
    'pobj': 'object', 'attr': 'object', 'oprd': 'object',
    # modifiers
    'amod': 'modifier', 'advmod': 'modifier', 'nmod': 'modifier',
    'npadvmod': 'modifier', 'acomp': 'modifier', 'appos': 'modifier',
    'det': 'modifier', 'poss': 'modifier', 'nummod': 'modifier',
    'quantmod': 'modifier', 'neg': 'modifier', 'predet': 'modifier',
    'compound': 'modifier', 'prep': 'modifier', 'prt': 'modifier',
    'case': 'modifier',
    # clausal
    'ccomp': 'clausal', 'xcomp': 'clausal', 'advcl': 'clausal',
    'relcl': 'clausal', 'acl': 'clausal', 'pcomp': 'clausal',
    'conj': 'clausal', 'parataxis': 'clausal',
    # function/structural
    'aux': 'function', 'auxpass': 'function', 'cc': 'function',
    'mark': 'function', 'punct': 'function', 'dep': 'function',
    'intj': 'function', 'meta': 'function',
    # head
    'ROOT': 'root',
}

DEP_COARSE_CLASSES = ('subject', 'object', 'modifier', 'clausal',
                      'function', 'root', 'other', 'none')

#: Universal POS tags — already coarse, used as-is.
POS_CLASSES = ('NOUN', 'PROPN', 'VERB', 'AUX', 'ADJ', 'ADV', 'PRON', 'DET',
               'ADP', 'NUM', 'CONJ', 'CCONJ', 'SCONJ', 'PART', 'INTJ',
               'PUNCT', 'SYM', 'X', 'none')

#: spaCy's entity labels kept as entities (vocab.morpholex.KEEP_ENTITIES), then 'none'.
ENTITY_CLASSES = ('ORG', 'PERSON', 'GPE', 'LOC', 'FAC', 'NORP', 'PRODUCT',
                  'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'none')

#: Every morphological feature bundle en_core_web_sm 3.8 can assign to an
#: alphabetic word — read from its attribute_ruler table (177 patterns, 77
#: bundles, 70 once punctuation is excluded). English morph is set entirely by
#: that table from the tagger's fine-grained tag, so the inventory is CLOSED:
#: no hashing, no collisions. A bundle outside it (another spaCy version) maps
#: to 'other' rather than silently aliasing a real class.
MORPH_BUNDLES = (
    'Aspect=Perf|Tense=Past|VerbForm=Part',
    'Aspect=Prog|Tense=Pres|VerbForm=Part',
    'Case=Acc|Gender=Fem|Number=Sing|Person=3|PronType=Prs',
    'Case=Acc|Gender=Fem|Number=Sing|Person=3|PronType=Prs|Reflex=Yes',
    'Case=Acc|Gender=Masc|Number=Sing|Person=3|PronType=Prs',
    'Case=Acc|Gender=Masc|Number=Sing|Person=3|PronType=Prs|Reflex=Yes',
    'Case=Acc|Gender=Neut|Number=Sing|Person=3|PronType=Prs',
    'Case=Acc|Gender=Neut|Number=Sing|Person=3|PronType=Prs|Reflex=Yes',
    'Case=Acc|Number=Plur|Person=1|PronType=Prs',
    'Case=Acc|Number=Plur|Person=1|PronType=Prs|Reflex=Yes',
    'Case=Acc|Number=Plur|Person=3|PronType=Prs',
    'Case=Acc|Number=Plur|Person=3|PronType=Prs|Reflex=Yes',
    'Case=Acc|Number=Sing|Person=1|PronType=Prs',
    'Case=Acc|Number=Sing|Person=1|PronType=Prs|Reflex=Yes',
    'Case=Acc|Number=Sing|Person=3|PronType=Prs|Reflex=Yes',
    'Case=Acc|Person=2|PronType=Prs',
    'Case=Acc|Person=2|PronType=Prs|Reflex=Yes',
    'Case=Nom|Gender=Fem|Number=Sing|Person=3|PronType=Prs',
    'Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs',
    'Case=Nom|Gender=Neut|Number=Sing|Person=3|PronType=Prs',
    'Case=Nom|Number=Plur|Person=1|PronType=Prs',
    'Case=Nom|Number=Plur|Person=3|PronType=Prs',
    'Case=Nom|Number=Sing|Person=1|PronType=Prs',
    'Case=Nom|Person=2|PronType=Prs',
    'ConjType=Cmp',
    'Definite=Def|PronType=Art',
    'Definite=Ind|PronType=Art',
    'Degree=Cmp',
    'Degree=Pos',
    'Degree=Sup',
    'Foreign=Yes',
    'Gender=Fem|Number=Sing|Person=3|Poss=Yes|PronType=Prs',
    'Gender=Masc|Number=Sing|Person=3|Poss=Yes|PronType=Prs',
    'Gender=Neut|Number=Sing|Person=3|Poss=Yes|PronType=Prs',
    'Gender=Neut|Number=Sing|Person=3|PronType=Prs',
    'Hyph=Yes',
    'Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin',
    'Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin',
    'Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin',
    'Mood=Ind|Tense=Past|VerbForm=Fin',
    'Mood=Ind|Tense=Pres|VerbForm=Fin',
    'NumType=Card',
    'NumType=Mult',
    'NumType=Ord',
    'Number=Plur',
    'Number=Plur|Person=1|Poss=Yes|PronType=Prs',
    'Number=Plur|Person=2|Poss=Yes|PronType=Prs',
    'Number=Plur|Person=3|Poss=Yes|PronType=Prs',
    'Number=Plur|PronType=Dem',
    'Number=Sing',
    'Number=Sing|Person=1|Poss=Yes|PronType=Prs',
    'Number=Sing|Person=3|Tense=Pres|VerbForm=Fin',
    'Number=Sing|PronType=Dem',
    'Number=Sing|PronType=Ind',
    'Person=2|Poss=Yes|PronType=Prs',
    'Person=2|PronType=Prs',
    'Polarity=Neg',
    'Poss=Yes',
    'Poss=Yes|PronType=Prs',
    'PronType=Dem',
    'PronType=Prs',
    'PronType=Rel',
    'Tense=Past|VerbForm=Fin',
    'Tense=Past|VerbForm=Part',
    'Tense=Pres|VerbForm=Fin',
    'VerbForm=Fin',
    'VerbForm=Ger',
    'VerbForm=Inf',
    'VerbType=Mod',
    '_',
)
