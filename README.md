## Introduction
The 3D neighborhood test systematically identifies neighborhoods within a protein that have significant enrichments of case missense variants over control missense variants.

## Installation & Setup
Coming soon: pip installation

The reference data required to run the 3DNT can be downloaded [here](https://www.dropbox.com/scl/fi/iczfletneh6ev6r2jhsng/reference.tar.gz?rlkey=8dptjly3d3w6i1jl0r85lqrjs&st=68nrl442&dl=0). This data should be saved to a `sir-reference-data` directory that will be used as an argument when running the 3DNT. Included in this download are PAE and PDB files for all human proteins, in subdirectories `pae_files` and `pdb_files`, respectively. Additionally, there is a mapping file `all_missense_variants_gr38.h5` that is used to map DNA coordinates of variants in gr38 to UniProt canonical proteins. Lastly, `gene_to_uniprot.tsv` can be used to map gene names to UniProt proteins, and `pdb_pae_file_pos_guide.tsv` describes all PAE and PDB files, along with the protein and amino acid residues covered by each file. 

The variant level data required to run the example in this tutorial can be found [here](https://www.dropbox.com/scl/fi/7onm5wfosd0g5z4k319n8/input.tar.gz?rlkey=rhbsoybl8y6r5uu37stcudat7&st=tom7nt38&dl=0). The final path for this data will be used as an argument when running the 3DNT tutorial commands. 

## General Usage 

```python run.py --rvas-data-to-map [FOLDER/PATH/TO/DATA] --reference-dir [FOLDER/PATH/TO/SIR-REFERENCE-DIR] --results-dir [EXAMPLE/RESULTS/FOLDER] --3dnt```

For variant data formatting, see `formatting_requirements.txt`. 

The above commands will result in the creation of two files: 
`p_values.h5`: all information relative to neighborhoods that will be required to run the FDR computation
`all_proteins.fdr.tsv`: all neighborhood results, including the UniProt ID, central amino acid residue position, associated p-value and FDR score, number of case and control variants within the neighborhood, and the case/control ratio within the neighborhood.

For information additional options -- e.g., change the default radius of the neighborhood, the maximum allowable allele count, etc. -- run `python run.py -h`

## Example

For the tutorial, we will use the following data: `'input/Epi25_tutorial.tsv.bgz'`. 
This data originates from https://epi25.broadinstitute.org/ and is publicly available. After downloading, data cleaning steps were applied, including filtering on allele number, restricting allele count, restricting to certain chromosomes and proteins, and manipulating the formatting to match the 3DNT desired input format. 

Ensure all of these exist prior to running the tutorial:
Folder path to EPI25: `'input/Epi25_tutorial.tsv.bgz'`
Folder path to PDB Files: `'sir-reference-data/pdb_files'`
Folder path to PAE Files: `'sir-reference-data/pae_files'`
Folder path to variant mapping file: `'sir-reference-data/all_missense_variants_gr38.h5'`
Folder path to File Guide: `'sir-reference-data/pdb_pae_file_pos_guide.tsv'`
Folder path to Gene/Protein Guide: `'sir-reference-data/gene_to_uniprot_id.tsv'`

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

```python run.py --rvas-data-to-map 'input/Epi25_tutorial.tsv.bgz' --reference-dir 'sir-reference-data' --results-dir 'results_epi25' --3dnt --fdr-file 'all_proteins_epi25.fdr.tsv'```

If the original data included variants on proteins that we are not interested in, the following command would use only the subset of proteins that we are interested in. 

```python run.py --rvas-data-to-map 'input/Epi25_tutorial.tsv.bgz' --reference-dir 'sir-reference-data' --results-dir 'results_epi25' --3dnt --uniprot-id 'P28472,P18507,P30531,P35498,P11166,P08476,Q09470,P47756' --fdr-file 'all_proteins_epi25.fdr.tsv'```

A file named `'all_proteins_epi25.fdr.tsv'` will be created and saved to the main directory, which contains all neighborhoods computed and their corresponding FDR score. Additionally, the `'p_values.h5'` file will be saved to the `'results_epi25'` directory, which contains all information from the 3DNT that is required to create the `'all_proteins.fdr.tsv'`. This is done so to enable the 3DNT and FDR steps to have the option to be done separately. Example commands for how to do this can be found here:

```python run.py --rvas-data-to-map 'input/Epi25_tutorial.tsv.bgz' --reference-dir 'sir-reference-data' --results-dir 'results_epi25' --3dnt --no-fdr```

```python run.py --rvas-data-to-map 'input/Epi25_tutorial.tsv.bgz' --reference-dir 'sir-reference-data' --results-dir 'results_epi25' --3dnt --fdr-only --fdr-file 'all_proteins_epi25.fdr.tsv'```

### Running the 3DNT - Additional Options
It may be useful to run the 3DNT without any PAE filtering at all. We can do this by setting the `'pae-cutoff'` to 0. We can also change the names of our results directory and result files to distinguish these results from our earlier results.  

```python run.py --rvas-data-to-map 'input/Epi25_tutorial.tsv.bgz' --pae-cutoff 0 --reference-dir 'sir-reference-data' --results-dir 'results_epi25_no_pae' --3dnt --fdr-file 'all_proteins_epi25_no_pae.fdr.tsv'```

After running the 3DNT, the residues and variants in a given neighborhood centered at a specific amino acid of a protein can be found using the `--get-nbhd` flag. The example below can be used to find information on the neighborhood centered at amino acid 378 in Q09470:

```python run.py --rvas-data-to-map 'input/Epi25_tutorial.tsv.bgz' --pae-cutoff 0 --reference-dir 'sir-reference-data' --results-dir 'results_epi25_no_pae' --get-nbhd --uniprot-id 'Q09470' --aa-pos 378```

Additional parameter options not shown in this tutorial can be found by running 'python run.py -h'.

### Visualization Example

In the resulting `'all_proteins_epi25_no_pae.fdr.tsv'` file, we can see that the most significant neighborhoods are found in protein Q09470. Tools for visualizing these results are available, and can be accessed through the following command:

```python run.py --rvas-data-to-map 'input/Epi25_tutorial.tsv.bgz' --reference-dir 'sir-reference-data' --results-dir 'results_epi25_no_pae' --visualization --uniprot-id 'Q09470'```

This creates a variety of PSE and PNG files, which can be found under the `'pymol_visualizations'` subdirectory of the given results directory. General images of the protein structure can be seen in `'AF-Q09470-F1-model_v4_gray.pse'` and `'AF-Q09470-F1-model_v4.png'`. `'AF-Q09470-F1-model_v4_mut.png'` and `'AF-Q09470-F1-model_v4_mut.pse'` show the individual case and control mutations on the protein structure (cases highlighted in red, controls in blue, and overlaps in purple). Lastly, `'AF-Q09470-F1-model_v4_ratio.pse'` and `'AF-Q09470-F1-model_v4_ratio.png'` show the ratio of cases to controls in neighborhoods, with red areas demonstrating a higher concentration of case variants. 

In `'AF-Q09470-F1-model_v4_ratio.pse'`, there is one region that is colored red. This region contains all five residues that are the central residues of the top resulting nieghborhoods from the 3DNT. 
