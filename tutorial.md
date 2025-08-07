## Introduction
The Structure Informed RVAS test implements a systematic scan of predicted protein structures to identify regions (neighborhoods) within that protein that have significant enrichments of case missense variants over control missense variants.

## Repository Structure Overview

All reference data required to run the scan test on new data can be found in the 'reference' directory, which should be downloaded upon installation of the package. This directory contains all PAE and PDB files, in subdirectories 'pae_files' and 'pdb_files', respectively. Additionally, there is a mapping file 'all_missense_variants_gr38.h5' that is used to map DNA coordinates of variants in gr38 to UniProt canonical proteins. Lastly, 'gene_to_uniprot.tsv' can be used to map gene names to UniProt proteins, and 'pdb_pae_file_pos_guide.tsv' describes all PAE and PDB files available in their respective folders, along with the protein and amino acid residues covered by each file. 

The 'input' directory contains all data needed to run the examples included in this tutorial, and should also be downloaded upon intallation of the package. 

## Installation & Setup
TBD - should be "pip install ..."
might need to do something with requirements.txt ... 
Data needed (schema data already formatted, PAE files, PDB files, etc.) will either be downloaded during installation or when running the software 

### GENERAL EXPLANATION
Basic Usage 

Version 1: using variant data that has not been mapped to UniProt

python run.py --rvas-data-to-map [FOLDER/PATH/TO/DATA] --reference-dir [FOLDER/PATH/TO/...?????] --results-dir [EXAMPLE/RESULTS/FOLDER] --scan-test 

Version 2: using variant data that has been mapped to UniProt

python run.py --rvas-data-mapped [FOLDER/PATH/TO/DATA] --reference-dir [FOLDER/PATH/TO/...?????] --results-dir [EXAMPLE/RESULTS/FOLDER] --scan-test 

For formatting requirements, see 'formatting_requirements.txt'. 

The above commands use the following default settings:
neighborhood-radius = 15.0 angstroms
pae-cutoff = 15.0
n-sims = 1000
genome-build = 'hg38' (*** currently the only genome-build supported, but future versions will support hg37)
which-proteins = 'all'
ac-filter = 5
fdr-cutoff = 0.05
ignore-ac = False

The above commands will result in the creation of two files: 

'p_values.h5': all information relative to neighborhoods that will be required to run the FDR computation
'all_proteins.fdr.tsv': all neighborhood results, including the UniProt ID, central amino acid residue position, associated p-value and FDR score, number of case and control variants within the neighborhood, and the case/control ratio within the neighborhood

### EXAMPLE

For the tutorial, we will use the following data: 'input/Epi25_tutorial.tsv.bgz'. 
This data originates from https://epi25.broadinstitute.org/ and is publicly available. After downloading, data cleaning steps were applied, including filtering on allele number, restricting allele count, restricting to certain chromosomes and proteins, and manipulating the formatting to match the scan test desired input format. 

Ensure all of these exist prior to running the tutorial:
Folder path to EPI25: 'input/Epi25_tutorial.tsv.bgz'
Folder path to PDB Files: 'reference/pdb_files'
Folder path to PAE Files: 'reference/pae_files'
Folder path to variant mapping file: 'reference/all_missense_variants_gr38.h5'
Folder path to File Guide: 'reference/pdb_pae_file_pos_guide.tsv'
Folder path to Gene/Protein Guide: 'reference/gene_to_uniprot_id.tsv'

The results directory will be created during the process if it does not already exist. 

#### Running the Scan Test - Basic Version
EPI25 data is not yet mapped to proteins so we will use the --rvas-data-to-map flag with the path to the EPI25 data. 
variant-id-col, ac-control-col, and ac-case-col all already match the assumed formatting (list formatting here) so we can leave out these flags.
We would like to run a scan test, so we will include the --scan-test flag.

Let us focus this scan test to only the top 8 EPI25 genes; which correspond to the following UniProt canonical IDs: 
GABRB3: P28472
GABRG2: P18507
SLC6A1: P30531
SCN1A: P35498
SLC2A1: P11166
INHBA: P08476
KCNA1: Q09470
CAPZB: P47756

The following command will map our variant data to proteins and then run the scan test, including the FDR calculation, for only the subset of proteins that we are interested in. 

python run.py --rvas-data-to-map 'input/Epi25_tutorial.tsv.bgz' --reference-dir 'reference' --results-dir 'results_epi25' --scan-test --which-proteins 'P28472,P18507,P30531,P35498,P11166,P08476,Q09470,P47756' --fdr-file 'all_proteins_epi25.fdr.tsv'

A file named 'all_proteins_epi25.fdr.tsv' will be created and saved to the main directory, which contains all neighborhoods computed and their corresponding FDR score. Additionally, the 'p_values.h5' file will be saved to the 'results_epi25' directory, which contains all information from the scan test that is required to create the 'all_proteins.fdr.tsv'. This is done so to enable the scan test and FDR steps to have the option to be done separately. Example commands for how to do this can be found here:

python run.py --rvas-data-to-map 'input/Epi25_tutorial.tsv.bgz' --reference-dir 'reference' --results-dir 'results_epi25' --scan-test --no-fdr --which-proteins 'P28472,P18507,P30531,P35498,P11166,P08476,Q09470,P47756'

python run.py --rvas-data-to-map 'input/Epi25_tutorial.tsv.bgz' --reference-dir 'reference' --results-dir 'results_epi25' --scan-test --fdr-only --which-proteins 'P28472,P18507,P30531,P35498,P11166,P08476,Q09470,P47756' --fdr-file 'all_proteins_epi25.fdr.tsv'

#### Running the Scan Test - Additional Options
It may be useful to run the scan test without any PAE filtering at all. We can do this by setting the 'pae-cutoff' to 0. We can also change the names of our results directory and result files to distinguish these results from our earlier results.  

python run.py --rvas-data-to-map 'input/Epi25_tutorial.tsv.bgz' --pae-cutoff 0 --reference-dir 'reference' --results-dir 'results_epi25_no_pae' --scan-test --which-proteins 'P28472,P18507,P30531,P35498,P11166,P08476,Q09470,P47756' --fdr-file 'all_proteins_epi25_no_pae.fdr.tsv'

Additional parameter options not shown in this tutorial can be found by running 'python run.py -h'.

#### Visualization Example

In the resulting 'all_proteins_epi25_no_pae.fdr.tsv' file, we can see that the most significant neighborhoods are found in protein Q09470. Tools for visualizing these results are available, and can be accessed through the following command:

python run.py --rvas-data-to-map 'input/Epi25_tutorial.tsv.bgz' --reference-dir 'reference' --results-dir 'results_epi25_no_pae' --visualization --uniprot-id 'Q09470'

This creates a variety of PSE and PNG files, which can be found under the 'pymol_visualizations' subdirectory of the given results directory. General images of the protein structure can be seen in 'AF-Q09470-F1-model_v4_gray.pse' and 'AF-Q09470-F1-model_v4.png'. 'AF-Q09470-F1-model_v4_mut.png' and 'AF-Q09470-F1-model_v4_mut.pse' show the individual case and control mutations on the protein structure (cases highlighted in red, controls in blue, and overlaps in purple). Lastly, 'AF-Q09470-F1-model_v4_ratio.pse' and 'AF-Q09470-F1-model_v4_ratio.png' show the ratio of cases to controls in neighborhoods, with red areas demonstrating a higher concentration of case variants. 

In 'AF-Q09470-F1-model_v4_ratio.pse', there is one region that is colored red. This region contains all five residues that are the central residues of the top resulting nieghborhoods from the scan test. 
