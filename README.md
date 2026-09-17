## Introduction
The 3D neighborhood test systematically identifies neighborhoods within a protein that have significant enrichments of case missense variants over control missense variants. For more information, see our manuscript: https://www.medrxiv.org/content/10.64898/2026.05.29.26354366v1. To reproduce results from the manuscript, first complete complete the setup for this software and then go to https://tinyurl.com/462tz36j for the input data and scripts.

## Installation & Setup

### Prerequisites
After cloning this repo, we recommend the following set of commands: 
```
conda create -n sir-env python=3.8
conda activate sir-env
conda install -c conda-forge pymol-open-source
pip install -r requirements_no_pymol.txt
```

### Reference Data Setup
We recommend the following directory structure: 
```
working-directory/
├── structure-informed-rvas/          # this repo
├── sir-reference-data/
│   ├── all_missense_variants_gr38.h5
│   ├── common_variants_uniprot.tsv
│   ├── lcr_positions_uniprot.tsv
│   ├── protein_sequence_guide.tsv
│   ├── pae_files/
│   ├── pdb_files/
│   └── pdb_pae_file_pos_guide.tsv
└── input/
    ├── SCHEMA_tutorial.tsv.gz
    └── filters/
        └── am_scan_99.tsv
```
To set this up:

1. Download sir-data.tar.gz [here](https://www.dropbox.com/scl/fi/83t5e8u33refxp1rin2sv/sir-data.tar.gz?rlkey=ect07qn4dmp6ewhnrp7jyymej&st=u6elrcjj&dl=0) or `wget -O sir-data.tar.gz "https://www.dropbox.com/scl/fi/83t5e8u33refxp1rin2sv/sir-data.tar.gz?rlkey=ect07qn4dmp6ewhnrp7jyymej&st=u6elrcjj&dl=1"`
2. Move sir-data.tar.gz to your working directory
3. Extract it: `tar -xzf sir-data.tar.gz`
4. (Optional) Remove the archive: `rm sir-data.tar.gz`

All commands in this tutorial should be run from the working directory.

## Basic 3DNT 

```
python structure-informed-rvas/run.py \
  --rvas-data-to-map [FOLDER/PATH/TO/DATA] \
  --reference-dir sir-reference-data/ \
  --results-dir [EXAMPLE/RESULTS/FOLDER] \
  --run-3dnt
```

For variant data formatting, see the section below, **Formatting requirements for --rvas-data-to-map**.

The above commands will result in the creation of two files: 
`p_values.h5`: results for real and simulated data that will be required to compute FDR and FWER
`all_proteins.fdr.tsv`: all neighborhood results, including the UniProt ID, central amino acid residue position, associated p-value, FDR and FWERs, number of case and control variants within the neighborhood, and the regularized case/control ratio within the neighborhood (for visualization).

Additional flags allow for several kinds of customization: e.g., to change the default radius of the neighborhood, the maximum allowable allele count, etc. To see these, run `python structure-informed-rvas/run.py -h`.

## Example

For the tutorial, we will use schizophrenia (SCZ) rare variant data from the SCHEMA consortium. The tutorial input file `input/SCHEMA_tutorial.tsv.gz` contains ~14,000 missense variants across 36 genes identified either as significant at FDR < 0.05 in the SCHEMA flagship analysis or with a missense-based burden P<0.001. It is included in the reference.tar.gz archive described above.

After extracting the archive, your working directory should look like:
```
working-directory/
├── structure-informed-rvas/
├── sir-reference-data/
└── input/
    ├── SCHEMA_tutorial.tsv.gz
    └── filters/
        └── am_scan_99.tsv
```

A `results` directory will also be created when the 3DNT is run.

### Running the 3DNT - Basic Version

The following command maps the variant data to proteins and runs the 3DNT with FDR correction across all 36 proteins:

```
python structure-informed-rvas/run.py \
  --rvas-data-to-map input/SCHEMA_tutorial.tsv.gz \
  --reference-dir sir-reference-data/ \
  --results-dir results_schema \
  --run-3dnt \
  --fdr-file schema_tutorial.fdr.tsv
```

This creates two files in `results_schema/`:
- `p_values.h5`: all per-neighborhood p-values and null distributions required for FDR computation
- `schema_tutorial.fdr.tsv`: all neighborhoods with their p-value, FDR, FWER, case/control counts, and ratio

The top hit is GRIA3 (P42263, neighborhood centered at aa 738) with p=0.000028 and FDR=0.19. No neighborhoods are significant at FDR < 0.05 without additional filtering.

### Using a variant-level filter: --df-filter

The sensitivity of the 3DNT can be improved by restricting the FDR computation to positions predicted to be functionally important, for example using AlphaMissense pathogenicity scores. The `--df-filter` flag accepts a TSV with `uniprot_id` and `aa_pos` columns specifying which positions to include in the FDR computation. Positions not in the filter are excluded from the FDR calculation and the FDR file.

```
python structure-informed-rvas/run.py \
  --rvas-data-to-map input/SCHEMA_tutorial.tsv.gz \
  --reference-dir sir-reference-data/ \
  --results-dir results_schema \
  --run-3dnt \
  --df-filter input/am_scan_99.tsv \
  --fdr-file schema_tutorial_am.fdr.tsv
```

With this filter, 3 neighborhoods are significant at FDR < 0.05:

| Gene | UniProt | aa center | p-value | FDR | FWER |
|------|---------|-----------|---------|-----|------|
| GRIA3 | P42263 | 738 | 0.000028 | 0.040 | 0.039 |
| SETD1A | O15047 | 195 | 0.000033 | 0.040 | 0.044 |
| ATP2B2 | Q01814 | 455 | 0.000049 | 0.040 | 0.058 |

The `--df-filter` flag can also be used to restrict the analysis to a specific set of proteins (by providing only `uniprot_id` with no `aa_pos` column), which is useful when running FDR correction on a pre-selected gene list.

### Separating the scan and FDR steps

For large studies it is common to split the input data by chromosome and run the scan in parallel, then combine the results and compute FDR once across all chromosomes. This requires three steps.

**Step 1: Run the scan for each chromosome (parallelizable)**

```bash
for CHROM in {1..22} X; do
    python structure-informed-rvas/run.py \
      --rvas-data-to-map input/data_chr${CHROM}.tsv.gz \
      --reference-dir sir-reference-data/ \
      --results-dir results/ \
      --pval-file p_values_chr${CHROM}.h5 \
      --run-3dnt --no-fdr
done
```

**Step 2: Combine per-chromosome p-value files**

```bash
python structure-informed-rvas/run.py \
  --combine-pval-files "p_values_chr*.h5" \
  --results-dir results/ \
  --pval-file p_values.h5
```

**Step 3: Compute FDR across all chromosomes**

```bash
python structure-informed-rvas/run.py \
  --reference-dir sir-reference-data/ \
  --results-dir results/ \
  --fdr-only \
  --fdr-file all_proteins.fdr.tsv
```

An optional `--df-filter` can be passed in Step 3 to restrict FDR to a subset of positions (e.g. AlphaMissense-filtered). FDR must be computed on the combined file to ensure the null distribution is pooled across all proteins.

### Formatting requirements for --rvas-data-to-map

Required: DNA coordinates of variant data that has not previously been mapped to UniProt proteins.

Both compressed and non-compressed file types and most standard delimiters will work with the code, with `.tsv.gz` or `.tsv.bgz` recommended.

In order to map variants to UniProt canonical proteins, the input data for each variant must contain information on chromosome, locus, reference allele, and alternate allele. The following formats for this data will work when calling run.py without any additional arguments:

Single column named `Variant ID`:
- chr:pos:ref:alt form (example: `chr1:925963:G:A`)
- chr-pos-ref-alt form (example: `chr1-925963-G-A`)

Two columns named `locus` and `alleles`:
- `locus` is a string (example: `chr1:925963`)
- `alleles` is a string
- `["ref", "alt"]` form using single-capitalized-letter amino acid codes (example: `["G","A"]`)
- `[ref, alt]` form using single-capitalized-letter amino acid codes (example: `[G,A]` or `[G, A]`)

Four columns named `chr`, `pos`, `ref`, and `alt`:
- `chr` is a string beginning with "chr" (example: `chr1`)
- `pos` is an integer (example: `925963`)
- `ref` is a single-capitalized-letter nucleotide (example: `G`)
- `alt` is a single-capitalized-letter nucleotide (example: `A`)

The single column formatting may be used with a column name other than `Variant ID` if the `--variant-id-col` argument is supplied while calling run.py.

Additionally, allele counts for cases and controls of each variant is required. These must be in integer formats, under columns named `ac_case` or `case` and `ac_control` or `control`. If alternate column names are used, this can be accounted for through the `--ac-case-col` and `--ac-control-col` arguments used while calling run.py.

### Just the mapping: --save-df-rvas and --get-nbhd

The residues and variants in a given neighborhood centered at a specific amino acid of a protein can be found using the `--get-nbhd` flag. The example below finds variants in the neighborhood centered at amino acid 738 in GRIA3 (P42263):

```
python structure-informed-rvas/run.py \
  --rvas-data-to-map input/SCHEMA_tutorial.tsv.gz \
  --reference-dir sir-reference-data/ \
  --get-nbhd \
  --uniprot-id P42263 \
  --aa-pos 738
```

You can also map variants to protein coordinates and save the result without running the 3DNT:

```
python structure-informed-rvas/run.py \
  --rvas-data-to-map input/SCHEMA_tutorial.tsv.gz \
  --reference-dir sir-reference-data/ \
  --save-df-rvas mapped_variants.tsv
```

### Visualization

Visualization requires PyMOL (see Installation). To visualize results for a single protein, use the `visualize` subcommand of `visualize_and_interpret.py`:

```
python structure-informed-rvas/visualize_and_interpret.py visualize \
  --uniprot P42263 \
  --results-dir results_schema/ \
  --reference-dir sir-reference-data/
```

This produces three PSE files under `results_schema/pymol_visualizations/`:
- `GRIA3_P42263_gray.pse`: plain grey cartoon of the protein structure
- `GRIA3_P42263_mut.pse`: structure with mutations shown as spheres (blue = control-only, red = case-only, purple = both)
- `GRIA3_P42263_ratio.pse`: structure colored by case/control neighborhood ratio (green = low, red = high)

Note: proteins whose structure spans multiple AlphaFold fragment files are skipped with a message.

### Functional Site Annotation

`visualize_and_interpret.py` can also annotate significant neighborhoods with functional site information drawn from UniProt. It does this by fetching annotated family members from UniProt, aligning them to the query protein, and transferring binding site, active site, and other functional site annotations via the alignment.

To annotate a single protein, use the `annotate` subcommand:

```
python structure-informed-rvas/visualize_and_interpret.py annotate Q09470 results_epi25/annotations/Q09470_features.tsv
```

This writes a TSV with one row per functional feature (feature type, positions, description, alignment confidence). Intermediate alignment results are cached so that re-runs across multiple proteins in the same family are fast.

To cross-reference a specific neighborhood with the resulting feature table, use `nbhd-features`:

```
python structure-informed-rvas/visualize_and_interpret.py nbhd-features \
  --features results_epi25/annotations/Q09470_features.tsv \
  --reference-dir sir-reference-data/ \
  --results-dir results_epi25/ \
  --uniprot Q09470 \
  --aa-pos 378
```

This writes two files to `results_epi25/neighborhoods/`:
- `KCNA1_Q09470_378_nbhd.tsv`: per-residue case and control allele counts within the neighborhood
- `KCNA1_Q09470_378_nbhd_features.tsv`: overlap of each functional feature with the neighborhood, sorted by fraction of case-only mutations

### Full Pipeline (run_all)

The `run_all` subcommand runs visualization, annotation, and neighborhood feature tables for every significant protein in an FDR results file in a single command:

```
python structure-informed-rvas/visualize_and_interpret.py run_all \
  --fdr-file results_schema/schema_tutorial_am.fdr.tsv \
  --results-dir results_schema/ \
  --reference-dir sir-reference-data/ \
  --significance-column fdr \
  --significance-cutoff 0.05
```

For each significant protein this produces the PSE files described above, plus `_nbhd.tsv` and `_nbhd_features.tsv` files in `results_schema/neighborhoods/`. Use `--significance-column fwer` to filter by FWER instead of FDR. Use `--skip-visualization` to run only annotation and neighborhood steps without PyMOL.

## 3DNT with a mutation-rate null

The basic 3DNT compares case to control variants within a neighborhood. For de novo
data there are no controls, and stand-ins such as untransmitted parental singletons are
a poor proxy for mutational opportunity: they have already been filtered by selection
and their trinucleotide spectrum differs from the de novo spectrum. This mode replaces
the control variants with a mutation-rate model.

For a neighborhood `N(i)` with observed de novo count `X_i` and summed mutation rate
`M_i`, in a gene with totals `N_g` and `M_g` over structure-covered residues:

| | test | null |
|---|---|---|
| **Test B** (`binomial`) | `X_i \| N_g ~ Binomial(N_g, M_i / M_g)` | enrichment relative to the rest of the same gene |
| **Test A** (`poisson`) | `X_i ~ Poisson(lambda_hat * M_i)` | enrichment relative to the mutation rate |

Both are one-sided (upper tail).

**Test B is the test of interest** — a within-gene comparison of a neighborhood against
the rest of the gene is what the 3DNT has always been, with rate-weighted rest-of-gene
replacing the control variants. **Test A guards against the way Test B fails under
regional constraint.** When a region of a gene is depleted for variation because
mutations there are incompatible with life, and because these are *de novo* mutations
so selection acts within a single generation, that region yields fewer observed de novos
among living probands than its mutation rate predicts. Test B conditions on `N_g` and
distributes it by mutation rate, so the de novos that do exist are pushed into the
unconstrained regions, and a neighborhood there looks enriched without being special.
Such a neighborhood sits at `X_i ~ lambda_hat * M_i`, so Test A does not call it.

Each test is corrected separately by the usual empirical FDR/FWER machinery, and a
neighborhood passes when it passes both: `fdr_max = max(fdr_poisson, fdr_binomial)`,
likewise `fwer_max`. On simulated data each test rejects at 4% under the true null,
Test B inflates to 17% under regional constraint while Test A stays at 4%, Test A
inflates to 80% under uniform gene-level enrichment while Test B stays at 4%, and
requiring both still recovers an injected cluster with at least 98% power.

### Setting up the rate reference

Mutational opportunity must be the summed rate over **all possible missense SNVs** at a
residue. A cohort de novo file cannot supply that, because it lists only variants
someone observed — for the ASD counts file that is 1.8M of 72.7M possible missense
variants, about 1% of each gene's real opportunity and selected on having been observed.
Using it skews observed/expected by up to 19.7x across mutation-rate deciles, against
1.37x for the full universe.

So build the per-residue table once, from the reference enumeration joined to the rate
file:

```
python structure-informed-rvas/precompute_mu_per_residue.py \
  --reference-dir sir-reference-data/
```

This writes `sir-reference-data/mu_per_residue.parquet` (11.1M residues, 19,576 genes).

### Running it

```
python structure-informed-rvas/run.py \
  --rvas-data-to-map input/ASD_de_novos.tsv.gz \
  --reference-dir sir-reference-data/ \
  --mu-file \
  --results-dir results_asd \
  --run-3dnt \
  --n-sims 1000 \
  --seed 1 \
  --fdr-file ASD_mutation_rate.fdr.tsv
```

`--mu-file` with no argument uses `<reference-dir>/roulette_missenses_filtered.parquet`.
The input file supplies only the observed de novo counts, in `ac_case` (or via
`--ac-case-col`); no control column is needed.

`--mu-col NAME` is the alternative: a rate column already present in the input file,
for comparison against a rate reference. It is only valid if that file enumerates every
possible missense variant including those with no observed de novos, and it warns
accordingly. `--mu-col` and `--mu-file` are mutually exclusive.

### Where lambda_hat is estimated, and why it matters

`lambda_hat = sum(x) / sum(mu)` converts relative rates into expected counts. It is
estimated **after all variant-level filters and before any gene-level selection**.
Gene-level selection is selection on the outcome. Measured on the ASD data, the
variant-level filters (common variants, LCR, max-AC) move `lambda_hat` by under 0.2%,
while applying the `--min-denovo` gene filter first inflates it by **1.43x**, because
that filter keeps only genes whose de novo count came out high.

`--rate-calibration-genes` restricts the calibration set; it defaults to the whole input
file. `--rate-calibration fixed:<v>` supplies `lambda_hat` directly, and `none` sets it
to 1 for a model that already gives absolute expected counts.

### Other flags

- `--min-denovo` (default 5) minimum de novos per gene, applied as strictly greater
  than, matching `scan_test._filter_proteins_by_allele_count`; so the default requires
  at least 6. Note the existing gene filter is internally inconsistent — it gates on
  `> 5` in one place and `< 5` in another — and this mode mirrors the binding one.
- `--max-residues` (default 10000) skips very long proteins; the distance matrix scales
  as `n_res^2`, so titin alone would need 9.4 GB.
- `--min-mu-coverage` (default 0, i.e. report only) skips genes whose rate coverage is
  below a threshold.
- `--simulate-null-from-mu` replaces observed counts with draws from the rate model and
  runs the whole pipeline on them, as a calibration check.
- `--seed` makes a run reproducible. It applies to the standard 3DNT too. If not given,
  a seed is generated, logged, and stored in the p-value file attributes.
- `--ignore-ac` is rejected in this mode: collapsing de novo counts to 0/1 discards
  recurrence, which is signal here.

### Output

Results are written per test into the `poisson` and `binomial` HDF5 groups of the
p-value file, and merged into one TSV with `p_`/`fdr_`/`fwer_` columns for each test,
plus `fdr_max`, `fwer_max`, per-gene `n_denovo_gene`, `mu_gene`, and `mu_coverage`.

### Splitting the scan across jobs

`--combine-pval-files` descends into the per-test groups, so the three-step
parallel workflow above works in this mode too. One extra requirement: `lambda_hat`
is estimated from whatever is in front of it, so a per-chromosome job would calibrate
on its own chromosome. Compute it once exome-wide and pass it to every chunk:

```
# from a full-input run's log, or the lambda_hat attribute of its p-value file
--rate-calibration fixed:0.0045
```

Combining warns if the chunks disagree on `lambda_hat`.

### Limitations

- **The Roulette rate file covers autosomes only.** No chrX rates are available in it,
  so 1,510 genes (2.83M possible missense variants, 421,858 residues) cannot be tested;
  they are skipped with an explicit message. Supply a rate file including chrX to
  include them.
- Autosomal rate coverage is 82–96% per chromosome (median ~92%, lowest on chr19), and
  the gaps cluster spatially within low-coverage genes. This costs power rather than
  biasing the tests: a variant's de novo count and its rate are dropped together, so
  `x_j` and `m_j` stay on the same variant set at every residue and each test remains
  exact over a narrower region. A neighborhood with no rate data has `M_i = 0` and
  necessarily `X_i = 0`, so it returns p = 1. Per-gene coverage is reported as
  `mu_coverage` (median 0.986 genome-wide; 0.981 across the ASD analysis genes).
- **Test A's standalone FDR/FWER are not calibrated when `--min-denovo` is in use, by
  design.** That filter keeps genes whose de novo count came out high, and Test A's null
  is deliberately left as the plain rate model rather than being conditioned on the
  filter -- the filter is there to spend compute where there is power, not as part of the
  inferential design. So the observed data sits above Test A's null: on the
  simulated-null ASD run the selected genes had a median observed/expected gene total of
  1.50, and Test A's per-gene FWER came out at 14.3% against a nominal 5%.

  Read Test A as a direction-of-effect guard, not as significance. Two things follow.
  Quote `fdr_binomial` / `fwer_binomial` and the combined `fdr_max` / `fwer_max`, never
  `fdr_poisson` on its own. And since Test A is the guard against regional constraint, a
  too-liberal Test A under-flags, so "passes Test B but not Test A" coming out empty is
  weak evidence rather than a demonstration that no hit is a constraint artifact.

  Test B is unaffected (4.3% on the same run) because conditioning on `N_g` cancels the
  selection exactly, and `fdr_max < q` requires `fdr_binomial < q`, so the reported set
  is a subset of the Test B set and inherits its validity. Running without `--min-denovo`
  removes the issue at the cost of testing every gene.
- FDR control over the *intersection* of two rejection sets is not guaranteed by the two
  marginal FDRs as a theorem. It is conservative in practice, since an intersection can
  only remove discoveries, and the simulations above measure it directly.
