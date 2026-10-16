# Effects, paired by seed

Each effect is **gain = baseline BPB - cell BPB at the same seed** (positive: the cell helps), over the seeds the cell shares with its OWN baseline (the same label with the features replaced by `baseline`): per-seed gains, mean +/- standard error, and how many seeds agree in sign. Cells are never compared by their means: seeds differ by more than most effects, and a cell lacking an unlucky baseline seed would look better by exactly that luck.

Within-pipeline comparisons (Q2) and cross-pipeline ones (Q1) are never mixed in a single table.

## bpe/bpe_17k

baseline `bpe_baseline_bpe_17k`, seeds [0, 1, 2, 3, 4, 5, 6, 7]: 1.3477, 1.3482, 1.3573, 1.3478, 1.3575, 1.3473, 1.3494, 1.3483

| cell | paired seeds | gain per seed | mean gain +/- se | % | agree |
|---|---|---|---|---|---|
| bpe_dep-LEARN_bpe_17k | 0 | +0.0387 | +0.0387 (n=1: no claim) | +2.87% | - |
| bpe_ectx_bpe_17k | 0,1 | +0.0131 +0.0134 | +0.0132 +/- 0.0002 | +0.98% | 2/2 |
| bpe_mconst-LEARN-CAUSAL-ADD_bpe_17k | 0,1,2,3 | +0.0094 +0.0098 +0.0170 +0.0107 | +0.0117 +/- 0.0018 | +0.87% | 4/4 |
| bpe_mconst-LEARN-CAUSAL_bpe_17k | 0,1,2,3 | +0.0076 +0.0101 +0.0185 +0.0082 | +0.0111 +/- 0.0025 | +0.82% | 4/4 |
| bpe_mconst16-LEARN-CAUSAL_bpe_17k | 0,1 | +0.0092 +0.0053 | +0.0073 +/- 0.0019 | +0.54% | 2/2 |
| bpe_mconst32-LEARN-CAUSAL_bpe_17k | 0,1 | +0.0114 +0.0113 | +0.0114 +/- 0.0001 | +0.84% | 2/2 |
| bpe_mconst64-LEARN-CAUSAL_bpe_17k | 0,1 | +0.0112 +0.0088 | +0.0100 +/- 0.0012 | +0.74% | 2/2 |
| bpe_mfeatlex-LEARN-CAUSAL_bpe_17k | 0,1,2,3 | +0.0002 +0.0050 +0.0173 +0.0034 | +0.0065 +/- 0.0037 | +0.48% | 4/4 |
| bpe_mflat-LEARN-CAUSAL-ADD_bpe_17k | 0,1,2,3 | +0.0062 +0.0107 +0.0182 +0.0101 | +0.0113 +/- 0.0025 | +0.84% | 4/4 |
| bpe_mflat-LEARN-CAUSAL_bpe_17k | 0,1 | +0.0069 +0.0114 | +0.0092 +/- 0.0023 | +0.68% | 2/2 |
| bpe_mfreq-LEARN-CAUSAL_bpe_17k | 0,1 | +0.0080 +0.0110 | +0.0095 +/- 0.0015 | +0.70% | 2/2 |
| bpe_mhash+mstruct-LEARN-CAUSAL_bpe_17k | 0,1 | +0.0043 +0.0097 | +0.0070 +/- 0.0027 | +0.52% | 2/2 |
| bpe_mhash+mstruct-LEARN_bpe_17k | 0,1 | +0.0683 +0.0667 | +0.0675 +/- 0.0008 | +5.01% | 2/2 |
| bpe_mhash-LEARN-CAUSAL_bpe_17k | 0,1 | -0.0017 +0.0077 | +0.0030 +/- 0.0047 | +0.22% | 1/2 |
| bpe_mhash-LEARN-k1024_bpe_17k | 0 | +0.0352 | +0.0352 (n=1: no claim) | +2.61% | - |
| bpe_mhash-LEARN-k16365_bpe_17k | 0 | +0.0602 | +0.0602 (n=1: no claim) | +4.47% | - |
| bpe_mhash-LEARN-k256_bpe_17k | 0 | +0.0303 | +0.0303 (n=1: no claim) | +2.25% | - |
| bpe_mhash-LEARN-k4096_bpe_17k | 0 | +0.0471 | +0.0471 (n=1: no claim) | +3.49% | - |
| bpe_mhash-LEARN-k64_bpe_17k | 0 | +0.0226 | +0.0226 (n=1: no claim) | +1.68% | - |
| bpe_mhash-LEARN-k8192_bpe_17k | 0 | +0.0511 | +0.0511 (n=1: no claim) | +3.79% | - |
| bpe_mhash-LEARN_bpe_17k | 0,1 | +0.0602 +0.0601 | +0.0602 +/- 0.0000 | +4.46% | 2/2 |
| bpe_mlemma+mstruct-LEARN-CAUSAL_bpe_17k | 0,1 | +0.0073 +0.0077 | +0.0075 +/- 0.0002 | +0.56% | 2/2 |
| bpe_mlemma+mstruct-LEARN_bpe_17k | 0,1,2 | +0.0687 +0.0650 +0.0737 | +0.0691 +/- 0.0025 | +5.12% | 3/3 |
| bpe_mlemma-LEARN_bpe_17k | 0,1 | +0.0582 +0.0568 | +0.0575 +/- 0.0007 | +4.26% | 2/2 |
| bpe_mprefix+mstruct-LEARN-CAUSAL_bpe_17k | 0 | +0.0060 | +0.0060 (n=1: no claim) | +0.44% | - |
| bpe_mrole-LEARN_bpe_17k | 0 | -0.0015 | -0.0015 (n=1: no claim) | -0.11% | - |
| bpe_mroot+mstruct+mrole-LEARN_bpe_17k | 0,1,2 | +0.0550 +0.0646 +0.0631 | +0.0609 +/- 0.0030 | +4.51% | 3/3 |
| bpe_mroot+mstruct+mrole-SHUF-LEARN_bpe_17k | 0 | +0.0163 | +0.0163 (n=1: no claim) | +1.21% | - |
| bpe_mroot+mstruct-LEARN_bpe_17k | 0 | +0.0569 | +0.0569 (n=1: no claim) | +4.22% | - |
| bpe_mroot-LEARN_bpe_17k | 0 | +0.0401 | +0.0401 (n=1: no claim) | +2.98% | - |
| bpe_mroot-SHUF-LEARN_bpe_17k | 0 | -0.0042 | -0.0042 (n=1: no claim) | -0.31% | - |
| bpe_mroot_bpe_17k | 0,1 | +0.0181 +0.0203 | +0.0192 +/- 0.0011 | +1.42% | 2/2 |
| bpe_mstruct+mfeatlex-LEARN-CAUSAL_bpe_17k | 0,1 | +0.0094 +0.0096 | +0.0095 +/- 0.0001 | +0.70% | 2/2 |
| bpe_mstruct-LEARN-CAUSAL-ADD_bpe_17k | 0,1,2,3 | +0.0046 +0.0110 +0.0150 +0.0094 | +0.0100 +/- 0.0021 | +0.74% | 4/4 |
| bpe_mstruct-LEARN-CAUSAL-Smorphology_bpe_17k | 0,1,2 | +0.0050 +0.0108 +0.0165 | +0.0108 +/- 0.0033 | +0.80% | 3/3 |
| bpe_mstruct-LEARN-CAUSAL-Snames_bpe_17k | 0,1,2 | +0.0049 +0.0096 +0.0184 | +0.0110 +/- 0.0040 | +0.81% | 3/3 |
| bpe_mstruct-LEARN-CAUSAL_bpe_17k | 0,1,2,3 | +0.0073 +0.0111 +0.0176 +0.0098 | +0.0114 +/- 0.0022 | +0.85% | 4/4 |
| bpe_mstruct-SHUF_bpe_17k | 0 | +0.0037 | +0.0037 (n=1: no claim) | +0.27% | - |
| bpe_mstruct_bpe_17k | 0 | +0.0252 | +0.0252 (n=1: no claim) | +1.87% | - |
| bpe_mstructshuf-LEARN-CAUSAL_bpe_17k | 0,1,2,3 | +0.0081 +0.0129 +0.0204 +0.0090 | +0.0126 +/- 0.0028 | +0.93% | 4/4 |
| bpe_mzero-LEARN-CAUSAL-ADD_bpe_17k | 0,1,2,3 | +0.0041 +0.0015 +0.0070 -0.0015 | +0.0028 +/- 0.0018 | +0.20% | 3/4 |
| bpe_mzero-LEARN-CAUSAL_bpe_17k | 0,1,2,3 | +0.0083 +0.0071 +0.0210 +0.0058 | +0.0105 +/- 0.0035 | +0.78% | 4/4 |
| bpe_mzeronopos-LEARN-CAUSAL_bpe_17k | 0,1,2,3 | +0.0002 +0.0020 +0.0124 -0.0054 | +0.0023 +/- 0.0037 | +0.17% | 3/4 |
| bpe_mzeroposonly-LEARN-CAUSAL_bpe_17k | 0,1,2,3 | +0.0044 +0.0043 +0.0083 +0.0004 | +0.0043 +/- 0.0016 | +0.32% | 4/4 |
| bpe_pos-LEARN-CAUSAL-OFF0_bpe_17k | 0,1,2,3 | +0.0097 +0.0106 +0.0203 +0.0116 | +0.0130 +/- 0.0025 | +0.97% | 4/4 |
| bpe_pos-LEARN-CAUSAL-OFF5000_bpe_17k | 0,1,2,3 | +0.0108 +0.0084 +0.0167 +0.0085 | +0.0111 +/- 0.0019 | +0.82% | 4/4 |
| bpe_pos-LEARN-CAUSAL_bpe_17k | 0,1,2,3 | +0.0072 +0.0109 +0.0198 +0.0113 | +0.0123 +/- 0.0027 | +0.91% | 4/4 |
| bpe_poslex-LEARN-CAUSAL_bpe_17k | 0,1 | +0.0098 +0.0064 | +0.0081 +/- 0.0017 | +0.60% | 2/2 |

baseline `bpe_baseline_bpe_17k_st10k`, seeds [0, 1, 2, 3]: 1.5266, 1.5298, 1.5289, 1.5281

| cell | paired seeds | gain per seed | mean gain +/- se | % | agree |
|---|---|---|---|---|---|
| bpe_mconst-LEARN-CAUSAL_bpe_17k_st10k | 0,1,2,3 | +0.0003 +0.0045 +0.0043 +0.0050 | +0.0035 +/- 0.0011 | +0.23% | 4/4 |
| bpe_mfeatlex-LEARN-CAUSAL_bpe_17k_st10k | 0,1,2,3 | +0.0087 +0.0130 +0.0116 +0.0077 | +0.0102 +/- 0.0012 | +0.67% | 4/4 |
| bpe_mfreq-LEARN-CAUSAL_bpe_17k_st10k | 0,1,2,3 | +0.0064 +0.0071 +0.0098 +0.0093 | +0.0082 +/- 0.0008 | +0.53% | 4/4 |
| bpe_mstruct-LEARN-CAUSAL_bpe_17k_st10k | 0,1,2,3 | +0.0015 +0.0039 +0.0071 +0.0046 | +0.0043 +/- 0.0011 | +0.28% | 4/4 |
| bpe_pos-LEARN-CAUSAL-OFF0_bpe_17k_st10k | 0,1,2,3 | +0.0032 +0.0060 +0.0071 +0.0056 | +0.0055 +/- 0.0008 | +0.36% | 4/4 |
| bpe_pos-LEARN-CAUSAL-OFF2500_bpe_17k_st10k | 0,1,2,3 | +0.0010 +0.0053 +0.0038 +0.0026 | +0.0031 +/- 0.0009 | +0.21% | 4/4 |
| bpe_pos-LEARN-CAUSAL-OFF7500_bpe_17k_st10k | 0,1,2,3 | -0.0089 -0.0050 -0.0051 -0.0066 | -0.0064 +/- 0.0009 | -0.42% | 4/4 |
| bpe_pos-LEARN-CAUSAL-ON5000_bpe_17k_st10k | 0,1,2,3 | +0.0037 +0.0061 +0.0088 +0.0073 | +0.0065 +/- 0.0011 | +0.42% | 4/4 |
| bpe_pos-LEARN-CAUSAL_bpe_17k_st10k | 0,1,2,3 | +0.0143 +0.0177 +0.0172 +0.0169 | +0.0166 +/- 0.0008 | +1.08% | 4/4 |
| bpe_poslex-LEARN-CAUSAL_bpe_17k_st10k | 0,1,2,3 | +0.0098 +0.0149 +0.0151 +0.0138 | +0.0134 +/- 0.0012 | +0.88% | 4/4 |

baseline `bpe_baseline_bpe_17k_st20k`, seeds [0, 1, 2, 3]: 1.4325, 1.4342, 1.4351, 1.4335

| cell | paired seeds | gain per seed | mean gain +/- se | % | agree |
|---|---|---|---|---|---|
| bpe_mconst-LEARN-CAUSAL_bpe_17k_st20k | 0,1,2,3 | +0.0051 +0.0093 +0.0082 +0.0086 | +0.0078 +/- 0.0009 | +0.54% | 4/4 |
| bpe_mfeatlex-LEARN-CAUSAL_bpe_17k_st20k | 0,1,2,3 | +0.0065 +0.0113 +0.0135 +0.0087 | +0.0100 +/- 0.0015 | +0.70% | 4/4 |
| bpe_mfreq-LEARN-CAUSAL_bpe_17k_st20k | 0,1,2,3 | +0.0069 +0.0092 +0.0103 +0.0104 | +0.0092 +/- 0.0008 | +0.64% | 4/4 |
| bpe_mstruct-LEARN-CAUSAL_bpe_17k_st20k | 0,1,2,3 | +0.0046 +0.0076 +0.0090 +0.0068 | +0.0070 +/- 0.0009 | +0.49% | 4/4 |
| bpe_pos-LEARN-CAUSAL-DROP5000_bpe_17k_st20k | 0,1,2,3 | +0.0047 +0.0079 +0.0075 +0.0027 | +0.0057 +/- 0.0012 | +0.40% | 4/4 |
| bpe_pos-LEARN-CAUSAL-OFF0_bpe_17k_st20k | 0,1,2,3 | +0.0070 +0.0069 +0.0114 +0.0092 | +0.0086 +/- 0.0011 | +0.60% | 4/4 |
| bpe_pos-LEARN-CAUSAL-OFF5000_bpe_17k_st20k | 0,1,2,3 | +0.0026 +0.0058 +0.0067 +0.0027 | +0.0045 +/- 0.0011 | +0.31% | 4/4 |
| bpe_pos-LEARN-CAUSAL-ON5000_bpe_17k_st20k | 0,1,2,3 | +0.0109 +0.0115 +0.0165 +0.0151 | +0.0135 +/- 0.0014 | +0.94% | 4/4 |
| bpe_pos-LEARN-CAUSAL-TGT0.1_bpe_17k_st20k | 0,1,2,3 | +0.0014 +0.0026 +0.0001 -0.0001 | +0.0010 +/- 0.0006 | +0.07% | 3/4 |
| bpe_pos-LEARN-CAUSAL_bpe_17k_st20k | 0,1,2,3 | +0.0131 +0.0155 +0.0176 +0.0149 | +0.0153 +/- 0.0009 | +1.07% | 4/4 |
| bpe_pos-SHUF-LEARN-CAUSAL-TGT0.1_bpe_17k_st20k | 0,1,2,3 | -0.0004 +0.0011 +0.0012 +0.0006 | +0.0006 +/- 0.0004 | +0.04% | 3/4 |
| bpe_poslex-LEARN-CAUSAL_bpe_17k_st20k | 0,1,2,3 | +0.0101 +0.0129 +0.0147 +0.0133 | +0.0128 +/- 0.0010 | +0.89% | 4/4 |

baseline `bpe_baseline_bpe_17k_st20k_rope`, seeds [0, 1, 2, 3]: 1.3908, 1.3909, 1.3898, 1.3866

| cell | paired seeds | gain per seed | mean gain +/- se | % | agree |
|---|---|---|---|---|---|
| bpe_mconst-LEARN-CAUSAL_bpe_17k_st20k_rope | 0,1,2,3 | +0.0073 +0.0056 +0.0056 +0.0044 | +0.0057 +/- 0.0006 | +0.41% | 4/4 |
| bpe_mflat-LEARN-ADD_bpe_17k_st20k_rope | 0 | +0.0059 | +0.0059 (n=1: no claim) | +0.42% | - |
| bpe_mflat-LEARN-CAUSAL-ADD_bpe_17k_st20k_rope | 0,1,2 | +0.0072 +0.0086 +0.0058 | +0.0072 +/- 0.0008 | +0.52% | 3/3 |
| bpe_mflat-LEARN-CAUSAL_bpe_17k_st20k_rope | 0,1,2,3 | +0.0076 +0.0060 +0.0056 +0.0065 | +0.0064 +/- 0.0004 | +0.46% | 4/4 |
| bpe_mflat-LEARN_bpe_17k_st20k_rope | 0 | +0.0068 | +0.0068 (n=1: no claim) | +0.49% | - |
| bpe_mzero-LEARN-CAUSAL-ADD_bpe_17k_st20k_rope | 0,1,2 | -0.0021 +0.0041 +0.0015 | +0.0012 +/- 0.0018 | +0.08% | 2/3 |
| bpe_mzero-LEARN-CAUSAL_bpe_17k_st20k_rope | 0,1,2,3 | -0.0002 +0.0055 +0.0012 -0.0013 | +0.0013 +/- 0.0015 | +0.09% | 2/4 |

baseline `bpe_baseline_bpe_17k_st20k_w256`, seeds [0, 1, 2, 3, 4]: 1.2808, 1.2977, 1.2740, 1.2803, 1.2757

| cell | paired seeds | gain per seed | mean gain +/- se | % | agree |
|---|---|---|---|---|---|
| bpe_mconst-LEARN-CAUSAL_bpe_17k_st20k_w256 | 0,1,2,3,4 | +0.0057 +0.0200 -0.0033 +0.0028 +0.0022 | +0.0055 +/- 0.0039 | +0.43% | 4/5 |
| bpe_pos-LEARN-CAUSAL_bpe_17k_st20k_w256 | 0,1,2,3 | +0.0035 +0.0198 -0.0040 +0.0027 | +0.0055 +/- 0.0051 | +0.43% | 3/4 |

*No run of `bpe_baseline_bpe_17k_st10k_w256` exists, so these are not comparable:* `bpe_mconst-LEARN-CAUSAL-x2_bpe_17k_st10k_w256`, `bpe_mconst-LEARN-CAUSAL_bpe_17k_st10k_w256`, `bpe_pos-LEARN-CAUSAL-x2_bpe_17k_st10k_w256`, `bpe_pos-LEARN-CAUSAL_bpe_17k_st10k_w256`

*No run of `bpe_baseline_bpe_17k_st10k_w64` exists, so these are not comparable:* `bpe_mconst-LEARN-CAUSAL_bpe_17k_st10k_w64`, `bpe_pos-LEARN-CAUSAL_bpe_17k_st10k_w64`

*No run of `bpe_baseline_bpe_17k_st5k` exists, so these are not comparable:* `bpe_mconst-LEARN-CAUSAL_bpe_17k_st5k`, `bpe_pos-LEARN-CAUSAL_bpe_17k_st5k`

## bpe/bpe_uk_17k

baseline `bpe_baseline_bpe_uk_17k_ukwiki0.002_st10k`, seeds [0, 1, 2, 3]: 1.2718, 1.2738, 1.2764, 1.2741

| cell | paired seeds | gain per seed | mean gain +/- se | % | agree |
|---|---|---|---|---|---|
| bpe_mconst-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.002_st10k | 0,1,2,3 | +0.0057 +0.0102 +0.0105 +0.0134 | +0.0099 +/- 0.0016 | +0.78% | 4/4 |
| bpe_ucase-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.002_st10k | 0,1,2,3 | +0.0161 +0.0175 +0.0200 +0.0166 | +0.0175 +/- 0.0009 | +1.38% | 4/4 |
| bpe_ucaselex-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.002_st10k | 0,1,2,3 | +0.0137 +0.0148 +0.0162 +0.0151 | +0.0149 +/- 0.0005 | +1.17% | 4/4 |

baseline `bpe_baseline_bpe_uk_17k_ukwiki0.01_st10k`, seeds [0, 1, 2, 3]: 1.1118, 1.1104, 1.1158, 1.1170

| cell | paired seeds | gain per seed | mean gain +/- se | % | agree |
|---|---|---|---|---|---|
| bpe_mconst-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.01_st10k | 0,1,2,3 | +0.0081 +0.0038 +0.0115 +0.0106 | +0.0085 +/- 0.0017 | +0.77% | 4/4 |
| bpe_ucase-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.01_st10k | 0,1,2,3 | +0.0115 +0.0074 +0.0142 +0.0135 | +0.0117 +/- 0.0015 | +1.05% | 4/4 |
| bpe_ucaselex-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.01_st10k | 0,1,2,3 | +0.0093 +0.0054 +0.0119 +0.0129 | +0.0099 +/- 0.0017 | +0.89% | 4/4 |

baseline `bpe_baseline_bpe_uk_17k_ukwiki0.2`, seeds [0, 1, 2, 3]: 0.9475, 0.9483, 0.9462, 0.9459

| cell | paired seeds | gain per seed | mean gain +/- se | % | agree |
|---|---|---|---|---|---|
| bpe_mconst-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.2 | 0,1,2,3 | +0.0057 +0.0019 +0.0024 +0.0011 | +0.0028 +/- 0.0010 | +0.29% | 4/4 |
| bpe_mzero+ucase-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.2 | 0,1,2,3 | +0.0025 +0.0066 -0.0005 +0.0042 | +0.0032 +/- 0.0015 | +0.34% | 3/4 |
| bpe_ucase-LEARN-CAUSAL-ADD_bpe_uk_17k_ukwiki0.2 | 0,1 | +0.0021 +0.0033 | +0.0027 +/- 0.0006 | +0.28% | 2/2 |
| bpe_ucase-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.2 | 0,1,2,3 | +0.0027 -0.0007 +0.0026 +0.0011 | +0.0014 +/- 0.0008 | +0.15% | 3/4 |
| bpe_ucaselex-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.2 | 0,1,2,3 | +0.0018 -0.0013 +0.0031 -0.0003 | +0.0008 +/- 0.0010 | +0.09% | 2/4 |

*No run of `bpe_baseline_bpe_uk_17k_ukwiki0.05_st10k` exists, so these are not comparable:* `bpe_mconst-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.05_st10k`, `bpe_ucase-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.05_st10k`, `bpe_ucaselex-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.05_st10k`, `bpe_ucaseprep-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.05_st10k`

*No run of `bpe_baseline_bpe_uk_17k_ukwiki0.05_st10k_w256` exists, so these are not comparable:* `bpe_mconst-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.05_st10k_w256`

*No run of `bpe_baseline_bpe_uk_17k_ukwiki0.05_st10k_w512` exists, so these are not comparable:* `bpe_mconst-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.05_st10k_w512`

*No run of `bpe_baseline_bpe_uk_17k_ukwiki0.05_st20k` exists, so these are not comparable:* `bpe_mconst-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.05_st20k`, `bpe_ucase-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.05_st20k`

*No run of `bpe_baseline_bpe_uk_17k_ukwiki0.05_st20k_w256` exists, so these are not comparable:* `bpe_mconst-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.05_st20k_w256`

*No run of `bpe_baseline_bpe_uk_17k_ukwiki0.05_st20k_w512` exists, so these are not comparable:* `bpe_mconst-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.05_st20k_w512`

*No run of `bpe_baseline_bpe_uk_17k_ukwiki0.2_st10k` exists, so these are not comparable:* `bpe_ucase-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.2_st10k`

*No run of `bpe_baseline_bpe_uk_17k_ukwiki0.2_st20k` exists, so these are not comparable:* `bpe_ucase-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.2_st20k`

*No run of `bpe_baseline_bpe_uk_17k_ukwiki0.2_st5k` exists, so these are not comparable:* `bpe_mconst-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.2_st5k`, `bpe_ucase-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.2_st5k`

## bpe/bpe_uk_32k

*No run of `bpe_baseline_bpe_uk_32k_ukwiki0.05_st10k` exists, so these are not comparable:* `bpe_mconst-LEARN-CAUSAL_bpe_uk_32k_ukwiki0.05_st10k`, `bpe_ucase-LEARN-CAUSAL_bpe_uk_32k_ukwiki0.05_st10k`, `bpe_ucaselex-LEARN-CAUSAL_bpe_uk_32k_ukwiki0.05_st10k`

## bpe/bpe_uk_4k

*No run of `bpe_baseline_bpe_uk_4k_ukwiki0.05_st10k` exists, so these are not comparable:* `bpe_mconst-LEARN-CAUSAL_bpe_uk_4k_ukwiki0.05_st10k`, `bpe_ucase-LEARN-CAUSAL_bpe_uk_4k_ukwiki0.05_st10k`, `bpe_ucaselex-LEARN-CAUSAL_bpe_uk_4k_ukwiki0.05_st10k`

## bpe/bpe_uk_8k

*No run of `bpe_baseline_bpe_uk_8k_ukwiki0.05_st10k` exists, so these are not comparable:* `bpe_mconst-LEARN-CAUSAL_bpe_uk_8k_ukwiki0.05_st10k`, `bpe_ucase-LEARN-CAUSAL_bpe_uk_8k_ukwiki0.05_st10k`, `bpe_ucaselex-LEARN-CAUSAL_bpe_uk_8k_ukwiki0.05_st10k`

## bpe/bpe_ukf_17k

*No run of `bpe_baseline_bpe_ukf_17k_ukfiction0.1_st10k` exists, so these are not comparable:* `bpe_mconst-LEARN-CAUSAL_bpe_ukf_17k_ukfiction0.1_st10k`

## bpe/bpeend_uk_17k

*No run of `bpe_baseline_bpeend_uk_17k_ukwiki0.05_st10k` exists, so these are not comparable:* `bpe_mconst-LEARN-CAUSAL_bpeend_uk_17k_ukwiki0.05_st10k`, `bpe_ucase-LEARN-CAUSAL_bpeend_uk_17k_ukwiki0.05_st10k`, `bpe_ucaselex-LEARN-CAUSAL_bpeend_uk_17k_ukwiki0.05_st10k`, `bpe_ucaseprep-LEARN-CAUSAL_bpeend_uk_17k_ukwiki0.05_st10k`

*No run of `bpe_baseline_bpeend_uk_17k_ukwiki0.05_st10k_w256` exists, so these are not comparable:* `bpe_mconst-LEARN-CAUSAL_bpeend_uk_17k_ukwiki0.05_st10k_w256`

*No run of `bpe_baseline_bpeend_uk_17k_ukwiki0.05_st10k_w512` exists, so these are not comparable:* `bpe_mconst-LEARN-CAUSAL_bpeend_uk_17k_ukwiki0.05_st10k_w512`

*No run of `bpe_baseline_bpeend_uk_17k_ukwiki0.05_st20k` exists, so these are not comparable:* `bpe_mconst-LEARN-CAUSAL_bpeend_uk_17k_ukwiki0.05_st20k`, `bpe_ucase-LEARN-CAUSAL-OFF0_bpeend_uk_17k_ukwiki0.05_st20k`, `bpe_ucase-LEARN-CAUSAL-OFF5000_bpeend_uk_17k_ukwiki0.05_st20k`, `bpe_ucase-LEARN-CAUSAL_bpeend_uk_17k_ukwiki0.05_st20k`

*No run of `bpe_baseline_bpeend_uk_17k_ukwiki0.05_st20k_w256` exists, so these are not comparable:* `bpe_mconst-LEARN-CAUSAL_bpeend_uk_17k_ukwiki0.05_st20k_w256`

*No run of `bpe_baseline_bpeend_uk_17k_ukwiki0.05_st20k_w512` exists, so these are not comparable:* `bpe_mconst-LEARN-CAUSAL_bpeend_uk_17k_ukwiki0.05_st20k_w512`

*No run of `bpe_baseline_bpeend_uk_17k_ukwiki0.2` exists, so these are not comparable:* `bpe_mconst-LEARN-CAUSAL_bpeend_uk_17k_ukwiki0.2`, `bpe_ucase-LEARN-CAUSAL_bpeend_uk_17k_ukwiki0.2`, `bpe_ucaselex-LEARN-CAUSAL_bpeend_uk_17k_ukwiki0.2`, `bpe_ucaseprep-LEARN-CAUSAL_bpeend_uk_17k_ukwiki0.2`

## bpe/bpeend_ukf_17k

*No run of `bpe_baseline_bpeend_ukf_17k_ukfiction0.1_st10k` exists, so these are not comparable:* `bpe_mconst-LEARN-CAUSAL_bpeend_ukf_17k_ukfiction0.1_st10k`

## bpe/unigram_uk_17k

*No run of `bpe_baseline_unigram_uk_17k_ukwiki0.05_st10k` exists, so these are not comparable:* `bpe_mconst-LEARN-CAUSAL_unigram_uk_17k_ukwiki0.05_st10k`, `bpe_ucase-LEARN-CAUSAL_unigram_uk_17k_ukwiki0.05_st10k`, `bpe_ucaselex-LEARN-CAUSAL_unigram_uk_17k_ukwiki0.05_st10k`

## det/pdiwc

baseline `det_baseline_Spdiwc`, seeds [0, 1]: 1.5246, 1.5258

| cell | paired seeds | gain per seed | mean gain +/- se | % | agree |
|---|---|---|---|---|---|
| det_dep-LEARN_Spdiwc | 0 | +0.0530 | +0.0530 (n=1: no claim) | +3.48% | - |
| det_dep-S0.22_Spdiwc | 0 | +0.0422 | +0.0422 (n=1: no claim) | +2.77% | - |
| det_dep-SHUF_Spdiwc | 0 | -0.0261 | -0.0261 (n=1: no claim) | -1.71% | - |
| det_dep_Spdiwc | 0 | +0.0390 | +0.0390 (n=1: no claim) | +2.56% | - |

## morph/pdiwc

baseline `morph_baseline_Spdiwc`, seeds [0, 1]: 1.5292, 1.5297

| cell | paired seeds | gain per seed | mean gain +/- se | % | agree |
|---|---|---|---|---|---|
| morph_dep-LEARN_Spdiwc | 0 | +0.0571 | +0.0571 (n=1: no claim) | +3.73% | - |
| morph_dep-S0.22_Spdiwc | 0 | +0.0408 | +0.0408 (n=1: no claim) | +2.67% | - |
| morph_dep-SHUF_Spdiwc | 0 | -0.0268 | -0.0268 (n=1: no claim) | -1.75% | - |
| morph_dep_Spdiwc | 0 | +0.0379 | +0.0379 (n=1: no claim) | +2.48% | - |

---

*One paired seed carries no claim. Two agreeing seeds are not a noise estimate: read the standard error and the sign count together.*
