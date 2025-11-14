## Introduction
The 3D neighborhood test systematically identifies neighborhoods within a protein that have significant enrichments of case missense variants over control missense variants.

## Installation & Setup

### Prerequisites
Python 3.7 or higher is required. Install the package dependencies:
```bash
pip install -r requirements.txt
```

### Reference Data Setup
The reference data required to run the scan test can be downloaded [here](https://www.dropbox.com/scl/fi/iczfletneh6ev6r2jhsng/reference.tar.gz?rlkey=8dptjly3d3w6i1jl0r85lqrjs&st=68nrl442&dl=0). This data should be saved to a `sir-reference-data` directory.

## General Usage 

```
python run.py \
  --rvas-data-to-map [FOLDER/PATH/TO/DATA] \
  --reference-dir [FOLDER/PATH/TO/SIR-REFERENCE-DIR] \
  --results-dir [EXAMPLE/RESULTS/FOLDER] \
  --3dnt
```

For variant data formatting, see the section below, **Formatting requirements for --rvas-data-to-map**.

The above commands will result in the creation of two files: 
`p_values.h5`: all information relative to neighborhoods that will be required to run the FDR computation
`all_proteins.fdr.tsv`: all neighborhood results, including the UniProt ID, central amino acid residue position, associated p-value and FDR score, number of case and control variants within the neighborhood, and the case/control ratio within the neighborhood.

For information additional options -- e.g., change the default radius of the neighborhood, the maximum allowable allele count, etc. -- run `python run.py -h`

## Example

For the tutorial, we will use the following data: `input/Epi25_tutorial.tsv.bgz`. This file can be downloaded [here](https://www.dropbox.com/scl/fi/7onm5wfosd0g5z4k319n8/input.tar.gz?rlkey=rhbsoybl8y6r5uu37stcudat7&st=tom7nt38&dl=0)
This data originates from https://epi25.broadinstitute.org/ and is publicly available. After downloading, data cleaning steps were applied, including filtering on allele number, restricting allele count, restricting to certain chromosomes and proteins, and manipulating the formatting to match the 3DNT desired input format. 

Ensure all of these exist prior to running the tutorial:
Folder path to EPI25: `input/Epi25_tutorial.tsv.bgz`
Folder path to PDB Files: `sir-reference-data/pdb_files`
Folder path to PAE Files: `sir-reference-data/pae_files`
Folder path to variant mapping file: `sir-reference-data/all_missense_variants_gr38.h5`
Folder path to File Guide: `sir-reference-data/pdb_pae_file_pos_guide.tsv`
Folder path to Gene/Protein Guide: `sir-reference-data/gene_to_uniprot_id.tsv`

The results directory will be created during the process if it does not already exist. 

### Running the 3DNT - Basic Version
We will use the `--rvas-data-to-map` flag with the path to the EPI25 data. 
`variant-id-col`, `ac-control-col`, and `ac-case-col` all already match the assumed formatting (list formatting here) so we can leave out these flags.
We would like to run the 3D neighborhood test, so we will include the --3dnt flag.

Let us focus this 3DNT to only the top 8 EPI25 genes; which correspond to the following UniProt canonical IDs: 
GABRB3: P28472
GABRG2: P18507
SLC6A1: P30531
SCN1A: P35498
SLC2A1: P11166
INHBA: P08476
KCNA1: Q09470
CAPZB: P47756

The data included in this tutorial is already filtered to only variants on the above proteins, so we can run the following command to map our variant data to proteins, and run the 3DNT and FDR calculation.

```
python run.py \
  --rvas-data-to-map input/Epi25_tutorial.tsv.bgz \
  --reference-dir sir-reference-data/ \
  --results-dir results_epi25 \
  --3dnt \
  --fdr-file epi25_eight_proteins.fdr.tsv
```

If the original data included variants on proteins that we are not interested in, the following command would use only the subset of proteins that we are interested in. 

```
python run.py \
  --rvas-data-to-map input/Epi25_tutorial.tsv.bgz \
  --reference-dir sir-reference-data/ \
  --results-dir results_epi25 \
  --3dnt \
  --uniprot-id P28472,P18507,P30531,P35498,P11166,P08476,Q09470,P47756 \
  --fdr-file epi25_eight_proteins.fdr.tsv
```

A file named `epi25_eight_proteins.fdr.tsv` will be created and saved to the main directory, which contains all neighborhoods computed and their corresponding FDR score. Additionally, the `p_values.h5` file will be saved to the `results_epi25` directory, which contains all information from the 3DNT that is required to create the `all_proteins.fdr.tsv`. This is done so to enable the 3DNT and FDR steps to have the option to be done separately, for example to perform the 3DNT on input files split by chromosome and then perform FDR correction across all chromosomes simultaneously. Example commands for how to do this can be found here:

```
python run.py \
  --rvas-data-to-map input/Epi25_tutorial.tsv.bgz \
  --reference-dir sir-reference-data/ \
  --results-dir results_epi25 \
  --3dnt \
  --no-fdr
```

```
python run.py \
  --rvas-data-to-map input/Epi25_tutorial.tsv.bgz \
  --reference-dir sir-reference-data/ \
  --results-dir results_epi25 \
  --3dnt \
  --fdr-only \
  --fdr-file epi25_eight_proteins.fdr.tsv
```

### Formatting requirements for --rvas-data-to-map

Required: DNA coordinates of variant data that has not previously been mapped to UniProt proteins. 

Both compressed and non-compressed file types and most standard delimiters will work with the code, with `.tsv.bgz` recommended. 

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
- `ref` is a single-capitalized-letter string of an amino acid code (example: `G`)
- `alt` is a single-capitalized-letter string of an amino acid code (example: `A`)

The single column formatting may be used with a column name other than `Variant ID` if the `--variant-id-col` argument is supplied while calling run.py. 

Additionally, allele counts for cases and controls of each variant is required. These must be in integer formats, under columns named `ac_case` or `case` and `ac_control` or `control`. If alternate column names are used, this can be accounted for through the `--ac-case-col` and `--ac-control-col` arguments used while calling run.py.

### Just the mapping: --save-df-rvas and --get-nbhd
The residues and variants in a given neighborhood centered at a specific amino acid of a protein can be found using the `--get-nbhd` flag. The example below can be used to find information on the neighborhood centered at amino acid 378 in Q09470:

```
python run.py \
  --rvas-data-to-map input/Epi25_tutorial.tsv.bgz \
  --reference-dir sir-reference-data/ \
  --get-nbhd \
  --uniprot-id Q09470 \
  --aa-pos 378
```

You can also just map the data from --rvas-data-to-map to the uniprot ID and amino acid positions and save the result, with or without performing the 3DNT, with or without filtering with --uniprot-id:

```
python run.py \
  --rvas-data-to-map input/Epi25_tutorial.tsv.bgz \
  --reference-dir sir-reference-data/ \
  --save-df-rvas
```

### Visualization Example

In the resulting `all_proteins_epi25_no_pae.fdr.tsv` file, we can see that the most significant neighborhoods are found in protein Q09470. Tools for visualizing these results are available, and can be accessed through the following command:

```
python run.py \
  --rvas-data-to-map input/Epi25_tutorial.tsv.bgz \
  --reference-dir sir-reference-data/ \
  --results-dir results_epi25/ \
  --visualization \
  --uniprot-id Q09470
```

This creates a variety of PSE and PNG files, which can be found under the `pymol_visualizations` subdirectory of the given results directory. General images of the protein structure can be seen in `AF-Q09470-F1-model_v4_gray.pse` and `AF-Q09470-F1-model_v4.png`. `AF-Q09470-F1-model_v4_mut.png` and `AF-Q09470-F1-model_v4_mut.pse` show the individual case and control mutations on the protein structure (cases highlighted in red, controls in blue, and overlaps in purple). Lastly, `AF-Q09470-F1-model_v4_ratio.pse` and `AF-Q09470-F1-model_v4_ratio.png` show the ratio of cases to controls in neighborhoods, with red areas demonstrating a higher concentration of case variants. 

In `AF-Q09470-F1-model_v4_ratio.pse`, there is one region that is colored red. This region contains all five residues that are the central residues of the top resulting nieghborhoods from the 3DNT. 
