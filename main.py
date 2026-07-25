"""
main.py

Entry point for the morpheme embedding thesis project.

Three pipelines are compared:
  BPE          — standard GPT-2 style byte-pair encoding (baseline)
  morph        — morpheme-based with Morfessor fallback (Pipeline 2)
  det          — morpheme-based, fully deterministic (Pipeline 3)

Workflow:
  1. python main.py build          — build all vocabularies (run once)
  2. python main.py test           — compare all three pipelines on test sentences
  3. python main.py test bpe det   — compare specific pipelines only

Files:
  common.py                  shared utilities (MorphoLex, NER, vocab builder)
  pipeline_bpe.py            Pipeline 1 — BPE baseline
  pipeline_morpheme.py       Pipeline 2 — morpheme + Morfessor
  pipeline_deterministic.py  Pipeline 3 — morpheme, deterministic
  build_vocabs.py            vocabulary construction (run once)
  test_sentences.py          structured comparison runner
  data/                      MorphoLEX_en.xlsx, CSVs, tokenizer
"""

import sys
import subprocess


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1]

    if command == 'build':
        steps = sys.argv[2:] or ['entities', 'vocab', 'bpe']
        subprocess.run([sys.executable, 'build_vocabs.py'] + steps)

    elif command == 'test':
        pipelines = sys.argv[2:] or ['bpe', 'morph', 'det']
        subprocess.run([sys.executable, 'test_sentences.py'] + pipelines)

    else:
        print(f'Unknown command: {command}')
        print('Usage: python main.py [build|test] [options]')


if __name__ == '__main__':
    main()