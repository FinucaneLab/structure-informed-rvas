## Inputs

You must have a reference directory with a subdirectory called `pdb_files` that contains all pdb files for your analysis. If you have not yet mapped your genetic data to protein coordinates, then your reference directory must have the mapping file [TODO: put the exact filename]. You can copy the reference directory from here. You then need to input the location of the reference directory with the `--reference-directory` flag.

If you have not yet mapped your rvas data to protein coordinates, use the `--rvas-data-to-map` argument to pass in a .tsv. The .tsv must have columns `chr`, `pos`, `ref`, `alt`. It must also have columns with allele count in cases and allele count in controls. The names of those columns can be `ac_case` and `ac_control`, or you can pass the names in using the flags `--ac-case-col` and `--ac-control-col`. The code will use the mapping file in the reference directory to do the mapping, and will use the pdb files already in the `pdb_files` directory.

If you have already mapped your rvas data to protein coordinates, then you can input your data using `--rvas-data-mapped`. Your dataframe must be a tsv that has columns `uniprot_id`, `aa_pos`, `aa_ref`, `aa_alt`, and either `ac_case` and `ac_control` or the names indicated by the flags `--ac-case-col` and `--ac-control-col`. You may either have an additional column `pdb_filename`, or you can use the `--pdb-filename` flag if there is only one pdb file and you prefer to input it in that way. [TODO right now I am assuming only one pdb file per protein in the scan test so there's no `aa_pos_file` column to deal with]

You can specify which test to run by choosing among the `--scan-test`, `--annotation-test`, and `--clinvar-test` flags. Right now, only the scan test works, but that will change soon!

The `--results-dir` flag points to the directory to store the results. Soon we will also have pymol sessions as part of the output.

## Example

We should put a tutorial here with data that can be downloaded easily. In the meantime, here is an example that works:

```
(venv) finucane@wm1f4-dfc sir_asd % ls
input		pdb_files	results		run.sh
(venv) finucane@wm1f4-dfc sir_asd % ls pdb_files/fold_dync1h1_model_0.pdb 
pdb_files/fold_dync1h1_model_0.pdb
(venv) finucane@wm1f4-dfc sir_asd % head input/DYNC1H1_ASD_missense_2025-03-31.txt 
Gene	Uniprot_ID	Variant_GRCh38	ENSP	aa_ref	aa_pos	aa_alt	dn_proband	t_proband	u_proband	isOS
DYNC1H1	Q14204	chr14:101964747:C:T	ENSP00000348965	S	19	L	1	0	0	FALSE
DYNC1H1	Q14204	chr14:101964753:T:G	ENSP00000348965	V	21	G	0	1	0	FALSE
DYNC1H1	Q14204	chr14:101964783:A:G	ENSP00000348965	Q	31	R	0	1	0	FALSE
DYNC1H1	Q14204	chr14:101964864:A:C	ENSP00000348965	E	58	A	0	1	0	FALSE
DYNC1H1	Q14204	chr14:101964870:G:A	ENSP00000348965	S	60	N	0	0	1	FALSE
DYNC1H1	Q14204	chr14:101964871:C:G	ENSP00000348965	S	60	R	0	0	1	FALSE
DYNC1H1	Q14204	chr14:101964872:G:A	ENSP00000348965	A	61	T	0	0	1	FALSE
DYNC1H1	Q14204	chr14:101964887:C:T	ENSP00000348965	R	66	C	0	1	0	FALSE
DYNC1H1	Q14204	chr14:101964888:G:A	ENSP00000348965	R	66	H	0	0	1	FALSE
(venv) finucane@wm1f4-dfc sir_asd % cat run.sh 
#!/bin/bash

python ../structure-informed-rvas/run.py \
	--rvas-data-mapped input/DYNC1H1_ASD_missense_2025-03-31.txt \
	--ac-case-col dn_proband \
	--ac-control-col u_proband \
	--pdb-file fold_dync1h1_model_0.pdb \
	--reference-dir ./ \
	--results-dir results/ \
	--scan-test

```