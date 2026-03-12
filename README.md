## Introduction
The 3D neighborhood test systematically identifies neighborhoods within a protein that have significant enrichments of case missense variants over control missense variants.

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
│   ├── no_pae_uniprot.tsv
│   ├── gene_to_uniprot_id.tsv
│   ├── pae_files/
│   ├── pdb_files/
│   └── pdb_pae_file_pos_guide.tsv
└── input/
    ├── SCHEMA_tutorial.tsv.gz
    └── filters/
        └── am_scan_99.tsv
```
To set this up:

1. Download reference.tar.gz [here](https://www.dropbox.com/scl/fi/4cfclarvxrujqtfmu3l69/reference.tar.gz?rlkey=indsp0h06yv465b4faxjg95kx&dl=0)
2. Move reference.tar.gz to your working directory
3. Extract it: `tar -xzf reference.tar.gz`
4. (Optional) Remove the archive: `rm reference.tar.gz`

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
`p_values.h5`: all information relative to neighborhoods that will be required to run the FDR computation
`all_proteins.fdr.tsv`: all neighborhood results, including the UniProt ID, central amino acid residue position, associated p-value and FDR score, number of case and control variants within the neighborhood, and the case/control ratio within the neighborhood.

Additional flags allow for several kinds of customization: e.g., to change the default radius of the neighborhood, the maximum allowable allele count, etc. To see these, run `python structure-informed-rvas/run.py -h`.

## Example

For the tutorial, we will use schizophrenia (SCZ) rare variant data from the SCHEMA consortium. The tutorial input file `input/SCHEMA_tutorial.tsv.gz` contains ~14,000 missense variants across 36 genes identified as significant in the SCHEMA flagship analysis. It is included in the reference.tar.gz archive described above.

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

The sensitivity of the 3DNT can be improved by restricting the FDR computation to positions predicted to be functionally important, for example using AlphaMissense pathogenicity scores. The `--df-filter` flag accepts a TSV with `uniprot_id` and `aa_pos` columns specifying which positions to include in the FDR computation. Positions not in the filter are excluded from the FDR calculation but the scan test still runs on all variants.

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

The scan and FDR steps can be run separately, which is useful when running the scan on input files split by chromosome and then performing FDR correction across all chromosomes simultaneously:

```
python structure-informed-rvas/run.py \
  --rvas-data-to-map input/SCHEMA_tutorial.tsv.gz \
  --reference-dir sir-reference-data/ \
  --results-dir results_schema \
  --run-3dnt \
  --no-fdr
```

```
python structure-informed-rvas/run.py \
  --rvas-data-to-map input/SCHEMA_tutorial.tsv.gz \
  --reference-dir sir-reference-data/ \
  --results-dir results_schema \
  --run-3dnt \
  --fdr-only \
  --df-filter input/am_scan_99.tsv \
  --fdr-file schema_tutorial_am.fdr.tsv
```

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
