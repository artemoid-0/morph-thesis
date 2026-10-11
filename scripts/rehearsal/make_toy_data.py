#!/usr/bin/env python3
"""
Toy data for a rehearsal: a WikiText-shaped corpus, a ByteLevel BPE tokenizer
trained on it, and minimal pairs in BLiMP's jsonl format.

The corpus is templated English chosen to exercise every fact: derivations
(reconfiguration, unhappiness, governmental), inflections, pronouns and
determiners in both numbers, names, and WikiText headings. Scale is tiny
(~35k tokens): a rehearsal tests the MACHINERY, never an effect.

    python scripts/rehearsal/make_toy_data.py /tmp/rehearsal
"""
import os
import sys
import json
import random


def main(work):
    random.seed(7)
    names = ['Anna', 'Robert Boulter', 'Katherine', 'the committee', 'Paris', 'Mr. Jones']
    subj = ['The researchers', 'These models', 'They', 'The children', 'She', 'He',
            'These sketches', 'The governments', 'Their teachers', 'This reconfiguration',
            'The unhappiness of the workers', 'Most readers']
    verb = ['were reading older books', 'had reconfigured the system',
            'ran quickly towards the river', 'is reading the report',
            'saw the cuts on his hands', 'were drawn by them', 'read these papers carefully',
            'seems unbelievable', 'has been rebuilt twice', 'announced the governmental reforms',
            'walks to the station every morning', 'taught the youngest students']
    tail = ['in 1998', ', although critics disagreed', 'before the results were published',
            'after the war ended', 'near the old mill', 'while they toured Europe', '', '', '']

    def para(n):
        out = []
        for _ in range(n):
            s = f'{random.choice(subj)} {random.choice(verb)} {random.choice(tail)}'
            s = s.replace(' ,', ',').strip() + '.'
            if random.random() < 0.3:
                s = f'{random.choice(names)} said that ' + s[0].lower() + s[1:]
            out.append(s)
        return ' '.join(out)

    wt = os.path.join(work, 'wikitext')
    os.makedirs(wt, exist_ok=True)
    for split, n in (('train', 900), ('validation', 150)):
        lines = []
        for i in range(n):
            if i % 40 == 0:
                lines.append(f' = {random.choice(["History", "Reception", "Early life"])} = \n')
            lines.append(' ' + para(random.randint(2, 5)) + ' \n')
            lines.append('\n')
        open(os.path.join(wt, f'{split}.txt'), 'w').write(''.join(lines))

    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(open(os.path.join(wt, 'train.txt')), trainers.BpeTrainer(
        vocab_size=600, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(os.path.join(work, 'bpe_toy.json'))

    bl = os.path.join(work, 'blimp')
    os.makedirs(bl, exist_ok=True)

    def write(name, term, pairs):
        with open(os.path.join(bl, name + '.jsonl'), 'w') as f:
            for g, b in pairs:
                f.write(json.dumps({'sentence_good': g, 'sentence_bad': b, 'field': 'morphology',
                                    'linguistics_term': term}) + '\n')
    nouns = [('model', 'models'), ('sketch', 'sketches'), ('teacher', 'teachers'),
             ('government', 'governments'), ('book', 'books')]
    verbs = ['were reading older books', 'had reconfigured the system',
             'ran quickly towards the river']
    dna = []
    for _ in range(60):
        sg, pl = random.choice(nouns)
        v = random.choice(verbs)
        dna.append((f'These {pl} {v}.', f'These {sg} {v}.') if random.random() < .5
                   else (f'This {sg} {v}.', f'This {pl} {v}.'))
    write('determiner_noun_agreement_1', 'determiner_noun_agreement', dna)
    write('regular_plural_subject_verb_agreement_1', 'subject_verb_agreement',
          [(f'The {pl} were reading.', f'The {pl} was reading.') for _, pl in nouns] * 8)
    write('anaphor_gender_agreement', 'anaphor_agreement',
          [(f'{n} taught herself.', f'{n} taught himself.') for n in ('Anna', 'Katherine')] * 20)
    print(f'toy data in {work}: wikitext/, bpe_toy.json ({tok.get_vocab_size()} tokens), blimp/')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/tmp/rehearsal')
