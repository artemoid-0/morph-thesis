"""
test_sentences.py

Runs all three pipelines on structured test sentences and prints
side-by-side comparison output.

Usage:
    python test_sentences.py              # run all three pipelines
    python test_sentences.py bpe          # BPE only
    python test_sentences.py morph det    # pipelines 2 and 3 only
"""

import sys
import pandas as pd

MORPHOLEX_PATH = 'data/MorphoLEX_en.xlsx'
VOCAB_CSV      = 'data/our_final_vocab.csv'
BPE_PATH       = 'data/bpe_tokenizer.json'

SENTENCES = {
    "morphologically_rich": [
        "The unbreakable determination of the misunderstood scientist was unprecedented.",
        "She systematically recontextualized the overrepresented underperforming subgroups.",
        "The antidisestablishmentarianism movement was counterproductive and hypersensitive.",
        "Decriminalization of previously criminalized substances remains controversial.",
        "The interconnectedness of globalization and deindustrialization is undeniable.",
    ],
    "inflected_forms": [
        "The cats are running quickly while the dogs were barking loudly.",
        "She has been studying harder than her classmates and performed better.",
        "The happiest children played more carefully with the oldest toys.",
        "Running quickly through darkened hallways, he tripped and fell badly.",
        "The brightest stars were hidden behind thickening clouds moving eastward.",
    ],
    "named_entities_single": [
        "London is the capital of England and a major financial hub.",
        "France and Germany signed a new agreement in Brussels last Monday.",
        "Australia recorded its hottest summer on record according to scientists.",
        "NASA launched another rocket from Florida toward the International Space Station.",
        "Shakespeare wrote his plays in London during the Elizabethan era.",
    ],
    "named_entities_multiword": [
        "The New York Times reported that the European Union passed new legislation.",
        "Joe Biden met with the Prime Minister of the United Kingdom in London.",
        "Goldman Sachs announced record profits as Wall Street celebrated the news.",
        "The San Francisco Giants won the World Series against the New York Yankees.",
        "North Korea and South Korea held talks at the demilitarized zone.",
    ],
    "function_words": [
        "Although she tried hard, neither the first nor the second attempt succeeded.",
        "Both the cat and the dog were sitting on the mat under the table.",
        "Whether or not they agree, we must proceed because time is running out.",
        "He said that unless we act, neither outcome would be satisfactory.",
        "Between the mountains and the sea, there was nothing but open grassland.",
    ],
    "oov_and_rare": [
        "The blockchain cryptocurrency market experienced significant volatility yesterday.",
        "Her Instagram feed was full of photoshopped selfies and hashtags.",
        "The YouTuber livestreamed his speedrun of the latest videogame release.",
        "They used a microservices architecture with containerized deployments.",
        "The mRNA vaccine technology represents a breakthrough in immunology.",
    ],
    "unknown_proper_nouns": [
        "Zuckerberg testified before Congress about data privacy concerns.",
        "Schwarzenegger won the governorship of California in a recall election.",
        "Dostoevsky wrote Crime and Punishment while deeply in debt.",
        "Mbappé scored a hat-trick in the final against Argentina.",
        "Wittgenstein spent his later years reconsidering his earlier philosophy.",
    ],
    "compound_words": [
        "The football match was broadcast on television to a worldwide audience.",
        "The software update included a new dashboard and a revamped homepage.",
        "Sunlight filtered through the bedroom window onto the hardwood floor.",
        "She downloaded the spreadsheet from the network drive and opened it.",
        "The blackbird sat on the fence post near the greenhouse.",
    ],
    "ambiguous_words": [
        "He sat by the bank of the river near the old bank building.",
        "The light was too light to cast a shadow in the morning light.",
        "She left the left luggage at the station before she left.",
        "The bat flew past the bat that the player dropped on the ground.",
        "They saw her duck as the duck swam past the old well.",
    ],
    "morphologically_simple": [
        "The cat sat on the mat and ate a rat.",
        "He ran and fell but got up and ran again.",
        "She said yes and he said no and they sat in silence.",
        "It was cold and wet and dark and they were tired.",
        "The sun set and the moon rose and the stars came out.",
    ],
    "mixed_realistic": [
        "Apple unveiled its new iPhone at a keynote event in San Francisco on Tuesday.",
        "The unvaccinated population in rural Mississippi faced unprecedented healthcare challenges.",
        "Goldman Sachs economists predicted that the Federal Reserve would raise interest rates.",
        "Elon Musk tweeted about the decentralization of cryptocurrency markets last Wednesday.",
        "The misunderstood artist from Brooklyn recontextualized traditional Japanese woodblock printing.",
        "Microsoft rebranded its Azure cloud services division after acquiring the startup.",
        "The disconnected communities along the Amazon river lack access to educational resources.",
        "Both the prosecution and the defense presented counterarguments to the jury in Chicago.",
        "Researchers at MIT demonstrated a breakthrough in superconducting quantum computing chips.",
        "The unhappy shareholders voted to restructure the underperforming telecommunications company.",
    ],
}


def fmt_ours(units):
    return ' | '.join(
        f'{u["token"]}[{u["category"][0]}]' for u in units
    )

def fmt_bpe(units):
    return ' | '.join(u['token'] for u in units)


def run(pipelines):
    total = sum(len(v) for v in SENTENCES.values())
    print(f'Testing {total} sentences across {len(SENTENCES)} categories\n')
    print('=' * 80)

    for category, sentences in SENTENCES.items():
        print(f'\n### {category.upper().replace("_", " ")} ###\n')
        for sent in sentences:
            print(f'Input: {sent}')
            for name, pipe in pipelines.items():
                units = pipe.encode(sent)
                if name == 'bpe':
                    print(f'  BPE:  {fmt_bpe(units)}')
                else:
                    print(f'  {name}: {fmt_ours(units)}')
            print()


if __name__ == '__main__':
    requested = set(sys.argv[1:]) or {'bpe', 'morph', 'det'}
    pipelines = {}

    vocab_df = pd.read_csv(VOCAB_CSV)

    if 'bpe' in requested:
        from pipeline_bpe import BPEPipeline
        print('Loading BPE pipeline...')
        pipelines['bpe'] = BPEPipeline(tokenizer_path=BPE_PATH)

    if 'morph' in requested:
        from pipeline_morpheme import MorphemePipeline
        print('Loading morpheme pipeline (with Morfessor)...')
        pipelines['morph'] = MorphemePipeline(vocab_df, MORPHOLEX_PATH)

    if 'det' in requested:
        from pipeline_deterministic import DeterministicPipeline
        print('Loading deterministic pipeline (no Morfessor)...')
        pipelines['det'] = DeterministicPipeline(vocab_df, MORPHOLEX_PATH)

    print()
    run(pipelines)