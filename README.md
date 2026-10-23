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

### Calling the tool from Python

`run.py` exposes `build_parser()`, `validate_args(args)` and `main(argv)`, so an
analysis can be driven in process rather than through a subprocess:

```python
import run
run.main([
    '--rvas-data-to-map', 'input.tsv.gz',
    '--reference-dir', 'sir-reference-data',
    '--results-dir', 'results/',
    '--run-3dnt',
])
```

`main` returns 0 and raises `ValueError` on an unusable argument combination, which is
what the command line reports too. Everything the CLI does is reachable this way; the
two paths produce identical output.

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

`visualize_and_interpret.py` is a subcommand tool, so plain `-h` lists only the
subcommand names. Use `--help-all` to see every subcommand's options at once, or
`visualize_and_interpret.py <subcommand> -h` for one.

This produces three PSE files under `results_schema/pymol_visualizations/`:
- `GRIA3_P42263_gray.pse`: plain grey cartoon of the protein structure
- `GRIA3_P42263_mut.pse`: structure with mutations shown as spheres (blue = control-only, red = case-only, purple = both)
- `GRIA3_P42263_ratio.pse`: structure colored by case/control neighborhood ratio (green = low, red = high)

For a mutation-rate run the last two differ; see Visualizing mutation-rate results.

Note: proteins whose structure spans multiple AlphaFold fragment files are skipped with a message.

### Functional Site Annotation

`visualize_and_interpret.py` can also annotate significant neighborhoods with functional site information drawn from UniProt. It does this by fetching annotated family members from UniProt, aligning them to the query protein, and transferring binding site, active site, and other functional site annotations via the alignment.

To annotate a single protein, use the `annotate` subcommand:

```
python structure-informed-rvas/visualize_and_interpret.py annotate Q09470 results_epi25/annotations/Q09470_features.tsv
```

The family is capped at the 50 best-annotated members (`--max-family-members`). UniProt
returns up to 500 for a large family and each one costs a full-length pairwise
alignment, so the cap is most of the difference between a fast run and a slow one; the
members it drops are the ones with least to transfer.

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

### Annotation test

Permutation test of annotation sets (UniProt features, domains, clusters) for case/control burden. The annotation file is either:
- long: one row per annotation interval (`uniprot_id, concept_type, concept_value, start, end`) or per annotated residue (`uniprot_id, aa_pos, <columns>`). `--annotation-id` lists the columns joined into the annotation id (default `concept_type,concept_value`).
- wide (`--annotation-format wide`): one row per residue, one column per annotation. `--annotation-id` lists the columns used as ids (default all).

```
# long, binding/active sites expanded to their 3D neighborhood
python run.py --pre-mapped-rvas rvas.mapped.parquet --reference-dir sir-reference-data/ \
  --annotation-file annotations.parquet --annotation-expand 'Binding site,Active site' \
  --neighborhood-radius 15 --pae-cutoff 15 --results-dir results/

# wide, testing only the cluster_id column
python run.py --pre-mapped-rvas rvas.mapped.parquet --reference-dir sir-reference-data/ \
  --annotation-file clusters.parquet --annotation-format wide --annotation-id cluster_id \
  --results-dir results/
```

`--annotation-expand` takes id prefixes or `all`; without it nothing is expanded. Output is `results/annotation_test/results.tsv.gz` with `p_value`, `local_fdr`/`local_fwer`, `global_fdr`/`global_fwer` and `bh_fdr`. `--df-filter` (columns `uniprot_id`, optionally `aa_pos`, `aa_ref`, `aa_alt`) restricts the RVAS data to those proteins/positions/variants before the test, which also changes the null. A crashed run resumes from `per_protein/` shards if rerun with the same `--results-dir` and `--n-sims`.

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

### Annotating components

A `--conditional` run writes `<stem>.components.tsv`, and `run_all --components` makes
that the unit of analysis instead of a single neighborhood:

```
python structure-informed-rvas/visualize_and_interpret.py run_all \
  --components results_asd/asd.fdr.components.tsv \
  --pval-file results_asd/asd.pvals.h5 \
  --results-dir results_asd/ \
  --reference-dir sir-reference-data/
```

This is usually what you want. A component is the whole set of overlapping neighborhoods
reporting one signal, and it is what the conditional analysis actually masked; annotating
only the best-scoring window describes a part of the finding and calls it the finding.
`--fdr-file` is not needed -- the components file already holds the called components,
so `--significance-column` and `--all-signals` do not apply.

The residue set comes from the file rather than being recomputed, so **no structures are
needed** for the tables (PyMOL is still needed for the PSEs). That also guarantees the
region annotated is exactly the region conditioned on.

Per component you get, in `neighborhoods/`:

```
{gene}_{uniprot}_c{id}_{lead}_component.tsv            per-residue counts
{gene}_{uniprot}_c{id}_{lead}_component_features.tsv   features x component
```

and in `pymol_visualizations/`, `{gene}_{uniprot}_c{id}_{lead}_component.pse` -- the
component highlighted on the greyed-out chain with its mutations as spheres. The
whole-chain PSEs are still written once per protein.

Both the `_mut.pse` and the `_component.pse` also get a **toggleable component object**
drawn as a transparent orange sphere union, named `component_c{id}_{lead}`. Hide it with
`disable component_c1_842` and bring it back with `enable`; a protein with several
components gets one object each, so they can be shown independently. The camera is
unchanged by it, so `_mut.pse` still opens framed on the whole protein.

That envelope is **approximate**. One sphere per member centre is what a component
literally is, but 100 overlapping transparent spheres render as an opaque blob, so the
centres are thinned to roughly half a radius apart -- typically 13-27 spheres out of
45-104, covering 91-100% of the component's residues. The authoritative residue set is
`residues_pymol`, and the `_component.pse` colours exactly those residues; the spheres
are for seeing at a glance where the signal sits. `COMPONENT_SPHERE_SEPARATION`,
`_ALPHA` and `_COLOR` in `pymol_code.py` tune it.

**The proportions are relative to the region.** `prop_denovo_in_feature` and
`prop_mu_in_feature` are shares of the region being annotated, so the same feature scores
differently against a component than against one neighborhood, and the two are not
directly comparable. A component usually spans more non-domain sequence than a single
window does, which lowers a domain's share of the opportunity and so sharpens the
contrast: on the ASD data CDK13's ATP site goes from 2.2x enriched per-neighborhood to
6.1x for the protein kinase domain per-component, with the Mg2+ site at 31x. Read these
as "within this region", not as an absolute effect size.

### Visualizing mutation-rate results

Every subcommand above also accepts a mutation-rate p-value file and FDR table, and
adapts to them automatically -- the layouts are detected from the files themselves, so
no extra flag is needed to switch modes:

```
python structure-informed-rvas/visualize_and_interpret.py run_all \
  --fdr-file results_asd/asd.fdr.tsv \
  --pval-file results_asd/asd.pvals.h5 \
  --results-dir results_asd/ \
  --reference-dir sir-reference-data/
```

Four things differ from the case/control output.

**Which test.** A mutation-rate run holds two sets of results, one per null. `--test`
chooses which one drives the output, and defaults to `binomial` (Test B, the within-gene
comparison and the test of interest). `--test poisson` draws Test A instead; its
standalone significance is not calibrated (see Limitations), so use it as a
direction-of-effect check rather than as a second opinion on which hits are real.

**Significance column.** `--significance-column` defaults to `fdr`, as it does for a
case/control run -- but under this null `fdr` is the max over both tests, so a
neighborhood is only shown if it passed *both*. Hits are ranked by the selected test's
p-value. Pass `--significance-column fdr_binomial` to rank on Test B alone.

**No control class.** There is no control cohort, so `_mut.pse` shows de novo positions
in one colour (darker where a residue carries more than one) rather than the
blue/red/purple case/control scheme, and `_ratio.pse` is coloured by observed/expected
against the selected null on a diverging scale: white at obs/exp = 1, red above, blue
below, clipped at four-fold so one extreme neighborhood cannot flatten the rest.

**Feature tables.** `_nbhd.tsv` carries `mu` (the residue's mutational opportunity)
instead of `ac_control`, and `_nbhd_features.tsv` replaces `prop_case_only_in_feature`
with two columns:

| column | meaning |
| --- | --- |
| `prop_denovo_in_feature` | share of the neighborhood's de novos inside the feature |
| `prop_mu_in_feature` | share of the neighborhood's mutational opportunity inside the feature |

The second is the null for the first. A feature that simply covers more of the
neighborhood, or covers a mutational hotspot, scores highly on `prop_denovo_in_feature`
on its own; it is the gap between the two columns that says whether de novos are
concentrated there beyond what the rate predicts.

To visualize every independent signal from a `--conditional` run rather than one
neighborhood per gene, point `--fdr-file` at the `.conditional.tsv` and pass
`--all-signals`:

```
python structure-informed-rvas/visualize_and_interpret.py run_all \
  --fdr-file results_asd/asd.fdr.conditional.tsv \
  --pval-file results_asd/asd.pvals.h5 \
  --results-dir results_asd/ \
  --reference-dir sir-reference-data/ \
  --all-signals
```

Under `--neighborhood-radius multiple-small`/`multiple-big` each hit records the radius
the scan selected for it, and that radius is used when rebuilding the neighborhood;
`--radius` is only the fallback for results that record none.

### Direction: enrichment only

The tool looks for clusters of **case** variants. A neighborhood carrying fewer case
variants than expected gets `p = 1` and is never called.

What this does *not* do is compute one-sided p-values. The cells that survive keep the
two-sided value `fisher_exact` returns, so enriched p-values are identical to a
two-sided run -- the two modes stay directly comparable, and the retained values are
conservative relative to a true one-sided test. Because significance here is empirical,
with observed and null p-values passing through the same lookup, that monotone
difference cancels and the discovery set is what a one-sided test would give.

For the standard case/control 3DNT this is applied where Fisher's exact p-values are
built: the p-value lookup is a grid indexed by (cases in neighborhood, controls in
neighborhood), and cells with odds ratio < 1 keep `p = 1` without `fisher_exact` ever
being called for them. Putting it there matters for more than speed -- that same grid is
indexed into by the observed data and by every `--n-sims` null column, so the empirical
null is transformed identically and FDR/FWER stay calibrated for free. Applied anywhere
downstream it would have to be applied to the nulls too, and forgetting would leave the
null pool full of small depleted p-values the observed data can no longer produce: an
anti-conservative FDR, and a silent one.

`--include-depletion` restores the two-sided test for the case/control scan. The
mutation-rate null only ever tests enrichment, so the flag has no effect there and
there is no way to test depletion with it.

The case/control scan analyzes genes with at least `--min-alleles` (default 5) case and
control alleles each; the mutation-rate null uses `--min-denovo` the same way. Both are
inclusive, so the default admits a gene with exactly 5.

**This changes results.** Dropping a direction halves the multiple-testing burden, so
re-running existing analyses gives more power, not the same answer. On a four-gene
ClinVar run, FDR < 0.05 went from 1,315 neighborhoods to 793 and FWER < 0.05 from 526 to
481 -- but every one of the 481 is enriched, against 454 of the 526. P-values for
enriched neighborhoods are bit-for-bit unchanged; only the correction moves.

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
neighborhood passes when it passes both, which is what the reported `fdr` holds:
`fdr = max(fdr_poisson, fdr_binomial)`, likewise `fwer`. Those are the same column
names the case/control scan writes, so nothing downstream has to know which null
produced a table; the per-test `fdr_poisson` / `fdr_binomial` stay alongside. On simulated data each test rejects at 4% under the true null,
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

This writes `sir-reference-data/mu_per_residue_gnomad.parquet` (11.1M residues,
19,576 genes, chrX included). Point `--mu-file` at the Roulette parquet and `--out`
at `mu_per_residue.parquet` to build the Roulette-derived table instead.

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

`--mu-file` with no argument uses `<reference-dir>/gnomad_mu_snp_by_variant.parquet`,
and `--mu-residue-file` defaults to the matching
`<reference-dir>/mu_per_residue_gnomad.parquet`. The input file supplies only the
observed de novo counts, in `ac_case` (or via `--ac-case-col`); no control column is
needed.

#### Using Roulette instead

Roulette rates are still shipped in the reference directory and can be used by naming
both files explicitly:

```
--mu-file <reference-dir>/roulette_missenses_filtered.parquet \
--mu-residue-file <reference-dir>/mu_per_residue.parquet
```

**Roulette has no chrX.** The file covers autosomes only, so every chrX gene gets a
total mutation rate of zero and is skipped with a message. In the ASD analysis that
costs 1,510 genes (2.83M possible missense variants, 421,858 residues) — including
hits such as ALG13 and NAA10, which are only testable under gnomAD. Use Roulette for
comparison against the default, not as the primary rate model.

The two rate models are on different scales (per-residue `mu` has a median near 0.3
under Roulette and around 2e-08 under gnomAD), so `lambda_hat` is not comparable
between them and a `--rate-calibration fixed:` value from one must never be reused
with the other.

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

- `--min-denovo` (default 5) minimum de novos per gene, inclusive: the default admits a
  gene with exactly 5. `--min-alleles` is the same filter for the case/control scan.
  Measured on the ASD analysis, "at least 5" tests 1,267 genes against 805 for "more
  than 5". Nothing is lost at FWER — all 20 genes and all 227 neighborhoods survive, and
  one more gene is gained — and at FDR it goes from 87 genes to 91, the nine that drop
  out all moving from just under the line to just over it.
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
plus the combined `fdr`, `fwer`, and per-gene `n_denovo_gene`, `mu_gene`, `mu_coverage`.

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

- **Rate coverage is not complete, and what is missing depends on the rate file.** The
  default gnomAD file covers chrX; Roulette does not (see "Using Roulette instead"
  above), and against a Roulette-derived table all 1,510 chrX genes are skipped with an
  explicit message. The coverage figures below were measured on Roulette.
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
  Quote `fdr_binomial` / `fwer_binomial` and the combined `fdr` / `fwer`, never
  `fdr_poisson` on its own. And since Test A is the guard against regional constraint, a
  too-liberal Test A under-flags, so "passes Test B but not Test A" coming out empty is
  weak evidence rather than a demonstration that no hit is a constraint artifact.

  Test B is unaffected (4.3% on the same run) because conditioning on `N_g` cancels the
  selection exactly, and `fdr < q` requires `fdr_binomial < q`, so the reported set
  is a subset of the Test B set and inherits its validity. Running without `--min-denovo`
  removes the issue at the cost of testing every gene.
- FDR control over the *intersection* of two rejection sets is not guaranteed by the two
  marginal FDRs as a theorem. It is conservative in practice, since an intersection can
  only remove discoveries, and the simulations above measure it directly.

## Stratified 3DNT with exact CMH

The default 3D neighborhood test uses Fisher's exact test after pooling all
case and control allele counts for a protein.  For datasets assembled from
multiple cohorts, ancestry groups, sequencing batches, or other finite strata,
3DNT can instead use an exact Cochran-Mantel-Haenszel (CMH) test.

For protein `g`, neighborhood `r`, and stratum `k`, define:

- `N_gk`: total case + control allele count in the protein;
- `A_gk`: total case allele count in the protein;
- `M_rgk`: total allele count inside the neighborhood;
- `a_rgk`: case allele count inside the neighborhood.

Under the stratified null,

```text
a_rgk | N_gk, A_gk, M_rgk ~ Hypergeom(N_gk, A_gk, M_rgk)
```

and the exact CMH statistic is `S_rg = sum_k a_rgk`.  Its null distribution is
computed by convolving the stratum-specific hypergeometric distributions.  With
one stratum, this reduces to the ordinary two-sided Fisher exact test.

### Input format

CMH input is long-form: a variant may appear once in every stratum in which it
has a nonzero case or control count.  For example:

```text
Variant ID          stratum          ac_case    ac_control
chr1-100-A-G        EUR_batch1       2          0
chr1-100-A-G        AFR_batch2       0          1
chr1-250-C-T        EUR_batch1       1          0
```

There must be one count vector per `Variant ID x stratum`.  The column name is
specified by `--stratum-col`.

The allele-count filter is applied to each variant's **total AC across all
strata**.  For example, with `--ac-filter 6`, a variant with AC 4 in one stratum
and AC 4 in another is removed because its total AC is 8.

### Run CMH 3DNT

```bash
python structure-informed-rvas/run.py \
  --rvas-data-to-map input/stratified_data.tsv.gz \
  --reference-dir sir-reference-data/ \
  --results-dir results_cmh/ \
  --run-3dnt \
  --test-method cmh \
  --stratum-col stratum \
  --neighborhood-radius 15 \
  --n-sims 1000 \
  --fdr-file all_proteins.cmh.fdr.tsv
```

The ordinary Fisher test remains the default, so existing commands without
`--test-method cmh` are unchanged.

### Null simulations

The empirical FDR/FWER null also preserves strata.  For each protein and
stratum, the simulation keeps fixed:

1. total allele count at every residue
2. total case allele count in that protein-stratum

Case labels are redistributed across residues drawn from a multivariate
hypergeometric distribution.  This is the resampling analogue of the margins conditioned
on by the exact CMH test.

CMH writes the same core HDF5 datasets as the ordinary 3DNT, so the existing
`empirical_fdr.py`, p-value-file combination workflow, and visualization code
can be reused unchanged.

### Current limitations

The initial CMH implementation supports one numeric neighborhood radius.  The
`multiple-small` and `multiple-big` harmonic-mean radius modes are not yet
implemented for CMH.  `--ignore-ac` is also disabled for CMH because collapsing
a variant to a single presence/absence observation is ambiguous when the same
variant occurs in multiple strata.

The stratum should define a group within which case/control labels are
exchangeable under the null.  Examples include ancestry x cohort or ancestry x
sequencing batch.  CMH conditions on the protein-wide case burden separately in
each stratum; it does not estimate a parametric covariate effect.

## Conditional analysis

A gene can carry more than one 3D signal, but a single strong neighborhood makes that
hard to see: every neighborhood overlapping it inherits its variants, so the second,
third and fourth best hits in a gene are usually the same cluster viewed through
slightly shifted windows. Conditional analysis answers the question directly. Call the
signal, mask it out, and scan what is left.

**This runs by default** with `--run-3dnt`, for both nulls. Pass `--no-conditional` to
skip it. It also does not apply, silently, with `--fdr-only`, with `--remove-nbhd`, or
without `--run-3dnt`.

```
python structure-informed-rvas/run.py \
  --rvas-data-to-map input/SCHEMA_tutorial.tsv.gz \
  --reference-dir sir-reference-data/ \
  --results-dir results/ \
  --run-3dnt
```

### Components: the unit that gets conditioned on

The thing masked is a *component*, not a single neighborhood. One real signal is
reported by every window that overlaps it -- in the ASD analysis the 20 genes with an
FWER-significant neighborhood carry 746 FDR-significant neighborhoods between them --
so masking only the best window leaves the rest of the same signal in place for the next
round to re-find.

Two neighborhoods join the same component when **they share at least one case-mutated
residue**. The rule is about the mutations, not the geometry: plain residue overlap
connects any two centres within roughly twice the radius whether or not a variant sits in
the overlap, which chains distinct sites together across a domain. Requiring a shared
mutated residue means the two windows are reporting the same variants, which is what
makes them the same finding. Membership is transitive, so a run of windows walking along
one cluster of de novos is one component.

**Enriched neighborhoods only.** Components describe clusters of case variants, so
neighborhoods carrying fewer case variants than expected are excluded and the count is
logged. Since the scan is enrichment-only by default (see Direction below) this is a no-op on
fresh runs; it matters when reading results files produced with `--include-depletion` or
before that default existed. Enrichment is judged the way the test judges it -- observed
against expected at fixed margins, i.e. odds ratio >= 1 -- not by the reported `ratio`
column, which is regularized with pseudocounts and a global case/control scaling and so
disagrees near 1.

**Historic note.** The standard case/control scan is two-sided, so its
significant set mixes neighborhoods carrying more case variants than expected with ones
carrying fewer. A component is a cluster of case variants, so depleted neighborhoods are
excluded -- from the gene gate as well as from membership, since otherwise a gene could
qualify on a depleted neighborhood and then be conditioned on a cluster it had nothing to
do with. The count excluded is logged. On the ClinVar results this matters a lot:
clustering everything gave 1,440 "components" of which 81% were lone neighborhoods
holding no case variant at all, against 211 real ones. Clustering depletion would need a
rule based on the controls rather than the cases, which is not implemented. The
mutation-rate scan tests enrichment only, so the filter is a no-op there.

Two different gates:

| gate | flag | default | what it decides |
| --- | --- | --- | --- |
| gene | `--conditional-gene-criterion` / `--conditional-gene-cutoff` | `fwer` < 0.05 | whether a gene is considered at all |
| member | `--conditional-criterion` / `--conditional-cutoff` | `fdr` < `--fdr-cutoff` | which neighborhoods make up its components |

A gene therefore qualifies on strong evidence but conditions on everything that evidence
is spread across. FWER significance implies FDR significance at the same cutoff, so the
gate never admits a gene whose qualifying neighborhood is then left out of its component.

Round 0 is the ordinary scan. Then each round masks each active gene's **top component**
-- the one holding its best neighborhood -- and re-scans. Genes without a hit are never
re-scanned, so a round costs one scan per gene still carrying signal rather than a full
exome scan. The gene gate is re-applied each round, so a gene continues only on fresh
FWER-strength evidence. The loop stops when no gene has a further component, or at
`--conditional-max-rounds` (default 5) -- in practice it usually stops much earlier.

### Expect round 1 to be mostly empty, and why

Conditioning on a whole component removes a great deal more than one neighborhood did.
Measured on the ASD analysis it takes out a median of **13% of a gene's residues but 73%
of its de novos**; three genes are left with none at all and seven more with one or two.

Those genes are dropped before the re-scan rather than re-scanned to produce a table of
p = 1. Test B conditions on the gene's remaining de novo total, so with one or two left
no neighborhood can reach significance and the empty result is arithmetic, not evidence
of absence. The bar is `--min-denovo`, the same filter a gene had to clear to enter the
analysis; the log names the genes and their remaining counts.

So an empty round 1 is the normal outcome and means "all of this gene's signal was in
that component", not "the conditional analysis failed".

### What gets masked

The residues of the called component are zeroed in place: their case and control
allele counts, and -- under the mutation-rate null -- their mutational opportunity `m`
along with them.

Masking `m` is not optional there. Zeroing the counts alone would leave both nulls still
expecting de novos at residues whose variants had just been removed. Every neighborhood
overlapping the conditioned region would then look depleted in proportion to how much of
it was masked, and the neighborhoods immediately beside it -- exactly where a second
independent signal would sit -- would lose the power to show anything. In simulation,
observed/expected in the partly-masked neighborhoods falls to the surviving fraction of
the neighborhood (0.76 against a surviving fraction of 0.73) instead of staying at 1.0,
and power to recover a real adjacent cluster drops roughly threefold. See
`tests/test_conditional.py`.

A masked residue is never dropped from the results, only zeroed. Positions are recovered
from row order throughout the codebase, so the per-protein p-value table always keeps
one row per residue in residue order.

### Which positions are still tested

After masking, a centre is re-tested only if at least `--conditional-min-retained`
(default 0.5) of its neighborhood's residues survive. Below that, what remains is a
sliver of the original neighborhood and a small p-value there reflects a handful of rim
variants rather than residual structure. Positions failing the rule, and the masked
residues themselves, get p = 1 in both the observed and the null columns. The rule reads
only the structure and the mask, never the counts, so observed and null are neutralized
identically and the empirical null stays calibrated.

Under `--neighborhood-radius multiple-small` / `multiple-big` the rule is applied per
radius before the harmonic-mean combination, so a centre that keeps enough of its small
neighborhoods still contributes them even if its large ones are mostly masked.

### Multiple testing

Each round writes a p-value file that is the previous round's file with only the
re-scanned genes overwritten. The empirical null pool therefore still spans every gene
in the analysis -- unmasked nulls for the genes that were not touched, freshly drawn
masked nulls for the genes that were -- so FDR and FWER in round *k* are exome-wide and
on the same footing as in round 0.

What this does *not* correct for is the sequential testing itself. Each round is
controlled at the nominal level and the rounds are not jointly corrected, which matches
the usual convention in conditional association analysis. Read a later round as "there
is additional signal in this gene" rather than as an independently calibrated discovery.

### Output

```
all_proteins.fdr.tsv               round 0, identical to a plain run
all_proteins.pvals.h5
all_proteins.fdr.round1.tsv        full exome-wide table for each later round
all_proteins.pvals.round1.h5
all_proteins.fdr.components.tsv    every component found, all rounds
all_proteins.fdr.conditional.tsv   one row per independent signal, all rounds
```

Every p-value file records how it was made in its HDF5 attributes: `test_type`
(`case_control` or `mutation_rate`), `n_sims`, `seed`, and the direction settings
(`include_depletion` and `min_alleles`, or `direction`, `min_denovo`, `lambda_hat` and
`rate_calibration`). A file written before these existed has no `test_type`, which the
readers treat as case/control.

`components.tsv` carries, per component: `residues_pymol` (the union of its member
neighborhoods, in PyMOL selection notation), `centers_pymol`, `centers_detail`
(`pos:radius` per member, so a viewer can redraw the component as the union of spheres
it actually is, correct even under `multiple-*` where each centre picks its own radius),
`case_pymol` and
`case_detail` (`pos:count`, so recurrence survives), counts of case residues and alleles,
the lead neighborhood and its statistics, and `seq_span_aa`. Case/control runs add
`control_pymol` and control counts; mutation-rate runs add `mu_component` instead, since
there is no control cohort.

`seq_span_aa` is the span along the *sequence*, not a spatial diameter. A large value is
not a defect -- it means the component is assembled in 3D from parts of the chain that
are far apart in sequence, which is exactly what 3DNT exists to find.

The conditional table adds `round`, `conditioned_on` (the positions masked before that
round) and `n_conditioned_residues` to the usual result columns, plus the component the
row stands for: `component_id`, `n_centers`, `n_residues`, `n_case_alleles` and, from
round 1 on, `n_residues_overlapping_mask`. A gene with three rows carries three
independent 3D signals.

The FDR and FWER columns hold the component's best over its own members, not the value
at the lead position -- under the mutation-rate null those columns are maxima over both
tests, so the best-scoring member need not be the one that cleared the cutoff. Both
columns are always written, whichever `--conditional-criterion` was used.

### Other flags

| flag | meaning |
| --- | --- |
| `--conditional-max-rounds` | stop after this many rounds beyond round 0 (default 5) |
| `--conditional-criterion` | `fdr` (default) or `fwer`; both nulls name these columns the same way, and under a mutation-rate null each is the max over both tests, so passing requires passing both |
| `--conditional-cutoff` | threshold for that criterion; defaults to `--fdr-cutoff` |
| `--conditional-radius` | radius of the removed neighborhood for `--remove-nbhd`. `best` (default) takes it from `--neighborhood-radius`; the mask predates the scan, so no per-position radius exists yet, and under a multi-radius run `best` falls back to 15 with a warning -- set this flag explicitly there |
| `--conditional-min-retained` | fraction of a neighborhood that must survive for its centre to be re-tested (default 0.5) |

### Conditioning by hand: --remove-nbhd

`--remove-nbhd` runs a single conditional round on neighborhoods you name, in one
protein:

```
python structure-informed-rvas/run.py \
  --rvas-data-to-map input/SCHEMA_tutorial.tsv.gz \
  --reference-dir sir-reference-data/ \
  --results-dir results/ \
  --run-3dnt \
  --uniprot-id P42263 \
  --remove-nbhd 738
```

It masks the same way `--conditional` does, and now also works with the mutation-rate
null. `--uniprot-id` must name exactly one protein: the positions are residue numbers in
that protein, and applying them to a list of genes would silently remove unrelated
residues from each of the others. Note that the resulting p-value file holds only that
protein, so its FDR and FWER are computed against a one-protein null and are not
comparable to an exome-wide run -- `--conditional` is what keeps the multiple-testing
correction intact across rounds.
