import glob
from pymol import cmd
import pandas as pd
import os
import ast
import gzip
from Bio.PDB import PDBParser
from Bio.PDB import StructureBuilder, PDBIO, Model, Chain

import re
from utils import read_p_values, read_original_mutation_data, read_p_values_quantitative
import h5py


def write_full_pdb(full_pdb, output_path, pdb_overlap = 1200):
    '''
    Write a list of Residue objects to a PDB file as a new structure.

    Parameters:
    - full_pdb (list): a list contains all residue objects from all pdb files of a uniprot id 
    - output_path (str): Path to save the output PDB file.
    - pdb_overlap (int): Number of overlapping residues between different pdb files of a uniprot id.
    '''

    try:
        builder = StructureBuilder.StructureBuilder()
        builder.init_structure("new_structure")
        builder.init_model(0)
        builder.init_chain('A')  

        model = builder.get_structure()[0]
        chain = model['A']

        for residue in full_pdb:
            if residue.id[1] == pdb_overlap + 1:
                print(residue)
            chain.add(residue.copy())  

        io = PDBIO()
        io.set_structure(builder.get_structure())
        io.save(output_path)
    except Exception as e:
        print(f"[ERROR] Failed to write PDB file: {e}")


def get_one_pdb(info_tsv, uniprot_id, reference_directory, pdb_overlap = 1200):
    '''
    Reconstructs a full protein PDB file from multiple PDB files that cover the protein's sequence.

    Parameters:
    - info_tsv (str): Name of the annotation file (TSV).
    - uniprot_id (str): UniProt ID of the protein to reconstruct.
    - reference_directory (str): Directory containing annotation and PDB files.
    - pdb_overlap (int): Number of overlapping residues between different pdb files of a uniprot id.
    '''

    info_tsv_p = os.path.join(reference_directory, info_tsv)
    try:
        info_df = pd.read_csv(info_tsv_p, sep='\t')
        if not os.path.isfile(info_tsv_p):
            print(f"[WARNING] Info TSV not found: {info_tsv_p}")
            return
    
        info_df['pos_covered'] = info_df['pos_covered'].apply(ast.literal_eval)
        pdbs = [{'filename': item, 'index': int(re.findall(r'\d+', item.split('-')[2])[0])}
                     for item in glob.glob(f'{reference_directory}/*{uniprot_id}*.gz')]
        full_pdb = []
        pdbs.sort(key=lambda pdb: pdb['index'])

        for pdb in pdbs:
            path = os.path.join(reference_directory, pdb['filename'])
            print('Reading pdb:', path)

            try:
                with gzip.open(path, "rt") as handle:
                    structure = PDBParser(QUIET=True).get_structure("protein", handle)
                residues = [res for model in structure for chain in model for res in chain]
                
                if pdb['index'] == 1:
                    full_pdb.extend(residues)
                    current_res_id = full_pdb[-1].id[1]

                else:
                    new_residue = residues[pdb_overlap:]
                    for i, res in enumerate(new_residue):
                        res_id = list(res.id)
                        res_id[1] = current_res_id + 1
                        res.id = tuple(res_id)
                        current_res_id += 1
                        full_pdb.append(res)

            except Exception as e:
                print(f"[ERROR] Failed to parse {p}: {e}")

        output_path = os.path.join(reference_directory, 'pdb_files', uniprot_id + '.pdb')
        write_full_pdb(full_pdb, output_path, pdb_overlap)

    except Exception as e:
        print(f"[ERROR] in get_one_pdb(): {e}")

def pymol_rvas(uniprot_id, reference_directory, results_directory):
    # make a pymol session with case and control mutations
    # output a gif and a .pse file
    '''
    Create PyMOL visualizations for RVAS results. For each PDB file:
    - Produces a basic image and PSE file.
    - Highlights mutations for control-only (blue), case-only (red), both (purple).
    - Outputs a second image and PSE file with mutations shown and colored.
    '''

    try:
        print(f"Creating PyMOL RVAS visualizations for {uniprot_id}")
        
        # Create pymol_visualizations subdirectory
        pymol_dir = os.path.join(results_directory, 'pymol_visualizations')
        os.makedirs(pymol_dir, exist_ok=True)
        
        # Read data from h5 file instead of individual tsv files
        df_results_p = os.path.join(results_directory, 'p_values.h5')
        if not os.path.isfile(df_results_p):
            print(f"[WARNING] Results file not found: {df_results_p}")
            return

        with h5py.File(df_results_p, 'r') as fid:
            df_rvas = read_original_mutation_data(fid, uniprot_id)
            # df_rvas now has the original per-residue mutation data in ac_case and ac_control columns
        
        print(f"  Loaded scan test data: {len(df_rvas)} amino acid positions")

        # Get PDB filename information
        info_tsv = 'pdb_pae_file_pos_guide.tsv'
        info = os.path.join(reference_directory, info_tsv)
        if not os.path.isfile(info):
            print(f"[WARNING] Info TSV not found: {info}")
            return
        info_df = pd.read_csv(info, sep='\t')
        
        tmp_info = info_df[info_df['uniprot_id'] == uniprot_id]
        df_rvas = get_pdb_filename(df_rvas, tmp_info)
        
        if not 'aa_pos_file' in df_rvas:
            df_rvas['aa_pos_file'] = df_rvas['aa_pos']
        pdbs = set(df_rvas['pdb_filename'].tolist())
        
        # cmd.set('ribbon_as_cylinders')
        # cmd.set("ribbon_radius", 0.5) 

        for item in pdbs:
            p = os.path.join(reference_directory, f'pdb_files/{item}')
            if not os.path.isfile(p):
                print(f"[ERROR] PDB file not found: {p}")
                continue
            print(f'Processing PDB file: {p}')

            cmd.load(p, "structure")
            pdb_filename = item.split('.')[0]

            # create sherif style image
            cmd.set('ambient', 0.5)
            cmd.set('ray_shadows', 0)
            cmd.set('ray_trace_mode', 1)
            cmd.set('ray_trace_gain', 0.05)
            cmd.bg_color("white")
            cmd.ray(2400, 1800)
            cmd.set("ray_opaque_background", 1)
            cmd.png(f"{pymol_dir}/{pdb_filename}.png")

            cmd.reset()
            cmd.color("grey")
            gray_pse = os.path.join(pymol_dir, f"{pdb_filename}_gray.pse")
            cmd.save(gray_pse)
            print(f"  Saved gray PSE file: {gray_pse}")

            tmp_df = df_rvas[df_rvas['pdb_filename'] == item]

            control_mask = (tmp_df['ac_control'] >= 1) & (tmp_df['ac_case'] == 0)
            case_mask = (tmp_df['ac_case'] >= 1) & (tmp_df['ac_control'] == 0)
            both_mask = (tmp_df['ac_case'] >= 1) & (tmp_df['ac_control'] >= 1)

            tmp_df_control = tmp_df[control_mask].copy()
            tmp_df_case = tmp_df[case_mask].copy()
            tmp_df_both = tmp_df[both_mask].copy()

            control_pos = tmp_df_control['aa_pos'].tolist()
            case_pos = tmp_df_case['aa_pos'].tolist()
            both_pos = tmp_df_both['aa_pos'].tolist()
            control_case_pos = set(control_pos).intersection(set(case_pos))
            control_both_pos = set(control_pos).intersection(set(both_pos))
            case_both_pos = set(case_pos).intersection(set(both_pos))
            new_control_pos = set(control_pos) - control_case_pos - control_both_pos
            new_case_pos = set(case_pos) - control_case_pos - case_both_pos
            new_both_pos = set(both_pos).union(control_case_pos)
            tmp_df_control = tmp_df[tmp_df['aa_pos'].isin(new_control_pos)].copy()
            tmp_df_case = tmp_df[tmp_df['aa_pos'].isin(new_case_pos)].copy()
            tmp_df_both = tmp_df[tmp_df['aa_pos'].isin(new_both_pos)].copy()
            
            print(f"  Found {len(new_control_pos)} control-only positions, {len(new_case_pos)} case-only positions, {len(new_both_pos)} positions with both")
            
            # Debug: Show total positions with actual mutations
            positions_with_mutations = len(df_rvas[(df_rvas['ac_case'] > 0) | (df_rvas['ac_control'] > 0)])
            total_mutations_case = df_rvas['ac_case'].sum()
            total_mutations_control = df_rvas['ac_control'].sum()
            print(f"  Total positions with mutations: {positions_with_mutations} (case mutations: {total_mutations_case}, control mutations: {total_mutations_control})")
            
            tmp_df_both['ac_case_real'] = tmp_df_both.groupby('aa_pos')['ac_case'].transform('sum')
            tmp_df_both['ac_control_real'] = tmp_df_both.groupby('aa_pos')['ac_control'].transform('sum')

            tmp_df_control.to_csv(f'{reference_directory}/tmp_df_control.csv', sep='\t', index=False)
            tmp_df_case.to_csv(f'{reference_directory}/tmp_df_case.csv', sep='\t', index=False)
            tmp_df_both.to_csv(f'{reference_directory}/tmp_df_both.csv', sep='\t', index=False)    

            for _, row in tmp_df_control.iterrows():
                aa_pos_file = row['aa_pos_file']
                cmd.select(f"residue_{aa_pos_file}", f"resi {aa_pos_file} and name CA")
                cmd.show("spheres", f"residue_{aa_pos_file}")
                cmd.color("blue", f"residue_{aa_pos_file}")

            for _, row in tmp_df_case.iterrows():
                aa_pos_file = row['aa_pos_file']
                cmd.select(f"residue_{aa_pos_file}", f"resi {aa_pos_file} and name CA")
                cmd.show("spheres", f"residue_{aa_pos_file}")
                cmd.color("red", f"residue_{aa_pos_file}")
        
            for _, row in tmp_df_both.iterrows():
                aa_pos_file = row['aa_pos_file']
                cmd.select(f"residue_{aa_pos_file}", f"resi {aa_pos_file} and name CA")
                cmd.show("spheres", f"residue_{aa_pos_file}")
                cmd.color("purple", f"residue_{aa_pos_file}")
            
            cmd.ray(2400, 1800)
            cmd.set("ray_opaque_background", 1)
            cmd.png(f"{pymol_dir}/{pdb_filename}_mut.png")

            pse_pdb_p = os.path.join(pymol_dir, f"{pdb_filename}_mut.pse")
            cmd.save(pse_pdb_p)
            print(f"  Saved mutations PSE file: {pse_pdb_p}")
            print(f"  Generated mutation visualization PNG: {pymol_dir}/{pdb_filename}_mut.png")

    except Exception as e:
        print(f"[ERROR] in pymol_rvas(): {e}")

def get_pdb_filename(annot_df, info_df):
    '''
    Get the pdb filename for the annotation
    '''

    pdb_filenames = []
    for _, r1 in annot_df.iterrows():
        aa_pos = r1['aa_pos']
        for _, r2 in info_df.iterrows():
            pos_covered = r2['pos_covered']
            pos_covered = ast.literal_eval(pos_covered)
            if int(aa_pos) >= int(pos_covered[0]) and int(aa_pos) <= int(pos_covered[1]):
                pdb_filename = r2['pdb_filename']
                break
        pdb_filenames.append(pdb_filename)
    annot_df['pdb_filename'] = pdb_filenames
    return annot_df

def pymol_annotation(annot_file, reference_directory, results_directory, info_tsv=None, uniprot_id=None):
    # visualize the annotation
    '''
    Create PyMOL visualizations for annotation file. For each PDB file:
    - Label the residues that are annotated in the annotation file.
    '''
    try:
        # Create pymol_visualizations subdirectory
        pymol_dir = os.path.join(results_directory, 'pymol_visualizations')
        os.makedirs(pymol_dir, exist_ok=True)
        
        annot_df_p = os.path.join(reference_directory, annot_file)
        if not os.path.isfile(annot_df_p):
            print(f"[WARNING] Annotation file not found: {annot_df_p}")
            return
        
        annot_df = pd.read_csv(annot_df_p, sep='\t')
        
        if info_tsv is not None:
            info = os.path.join(reference_directory, info_tsv)
            info_df = pd.read_csv(info, sep='\t')
        else:
            print(f"[WARNING] Info TSV not found: {info_tsv}")
            return
        
        if uniprot_id is not None:
            print(uniprot_id)
            tmp_info = info_df[info_df['uniprot_id'] == uniprot_id]
            tmp_annot = annot_df[annot_df['uniprot_id'] == uniprot_id]
            tmp_annot = get_pdb_filename(tmp_annot, tmp_info)
            print(tmp_annot)
            tmp_annot['visual_filename'] = tmp_annot['pdb_filename'].apply(lambda x: os.path.join(pymol_dir, x.split('.')[0]+ '.pse'))
            tmp_visuals = set(tmp_annot['visual_filename'].tolist())
            for v in tmp_visuals:
                print(v)
                tmp_annot_visuals = tmp_annot[tmp_annot['visual_filename'] == v]
                print(tmp_annot_visuals)
                tmp_annot_pos = tmp_annot_visuals['aa_pos'].tolist()
                for item in tmp_annot_pos:
                    if not os.path.exists(v):
                        print(f"[WARNING] PSE file from pymol_rvas() not found: {v}")
                        continue
                    cmd.load(f"{v.split('.')[0]}_mut.pse")
                    item = str(item)
                    cmd.select(f"annotation_residue_{item}", f"resi {item}")
                    cmd.label(f"annotation_residue_{item} and name CA", f'"annotation"')
                    cmd.save(f"{v.split('.')[0]}_mut.pse")
        else:
            print('No uniprot id provided')

    except Exception as e:
        print(f"[ERROR] in pymol_annotation(): {e}")

    
def pymol_scan_test(info_tsv, uniprot_id, reference_directory, results_directory):
    # color by case/control ratio of the neighborhood

    '''
    Create PyMOL visualizations for scan test results. For each PDB file:
    - Color residues based on their case/control ratio (yellow_orange_red).
    - Outputs a third PSE file with ratio colored.
    '''
    try:
        # Create pymol_visualizations subdirectory
        pymol_dir = os.path.join(results_directory, 'pymol_visualizations')
        os.makedirs(pymol_dir, exist_ok=True)

        df_results_p = os.path.join(results_directory, 'p_values.h5')
        with h5py.File(df_results_p, 'r') as fid:
            df_results = read_p_values(fid, uniprot_id)
        
        if info_tsv is not None:
            info = os.path.join(reference_directory, info_tsv)
            info_df = pd.read_csv(info, sep='\t')
        else:
            print(f"[WARNING] Info TSV not found: {info_tsv}")
            return

        tmp_info = info_df[info_df['uniprot_id'] == uniprot_id]
        tmp_df = df_results[df_results['uniprot_id'] == uniprot_id]
        tmp_df = get_pdb_filename(tmp_df, tmp_info)
        print(f"Processing scan test results for {uniprot_id}: {len(tmp_df)} amino acid positions")
        tmp_df['visual_filename'] = tmp_df['pdb_filename'].apply(lambda x: os.path.join(pymol_dir, x.split('.')[0]+ '.pse'))
        tmp_df['ratio_normalized'] = tmp_df['ratio'] / tmp_df['ratio'].max()
        # tmp_df['ratio_normalized'].to_csv('test.csv', sep='\t', index=False)
        tmp_visuals = set(tmp_df['visual_filename'].tolist())
        for v in tmp_visuals:
            cmd.load(f"{v.split('.')[0]}_gray.pse")
            objects = cmd.get_names('objects')[-1]

            tmp_df_visuals = tmp_df[tmp_df['visual_filename'] == v]

            for _, row in tmp_df_visuals.iterrows():
                resi = int(row['aa_pos'])
                ratio = float(row['ratio_normalized'])
                selection = f"{objects} and resi {resi}"
                cmd.alter(selection, f"b={ratio}")
                cmd.rebuild()
            
            cmd.spectrum("b", "yellow_orange_red", objects, byres=1)
            cmd.show("cartoon", objects)
            cmd.hide("lines", objects)

            cmd.save(f"{v.split('.')[0]}_ratio.pse")

    except Exception as e:
        print(f"[ERROR] in pymol_scan_test(): {e}")

def pymol_rvas_quantitative(uniprot_id, reference_directory, results_directory):
    """
    Create PyMOL visualizations for quantitative trait RVAS results. For each PDB file:
    - Produces a basic gray structure PSE file for subsequent coloring.
    - This replaces pymol_rvas() for quantitative traits where we don't have case/control data.
    """
    try:
        print(f"Creating PyMOL RVAS visualizations for quantitative traits: {uniprot_id}")

        # Create pymol_visualizations subdirectory
        pymol_dir = os.path.join(results_directory, 'pymol_visualizations')
        os.makedirs(pymol_dir, exist_ok=True)

        # For quantitative traits, we don't have case/control variant data in the h5 file
        # We'll just create the basic gray structure files that other functions expect

        # Load reference info to find PDB files for this protein
        reference_dir = os.path.join(reference_directory, 'pdb_pae_file_pos_guide.tsv')
        if not os.path.exists(reference_dir):
            print(f"[WARNING] Reference file not found: {reference_dir}")
            return

        info_df = pd.read_csv(reference_dir, sep='\t')
        tmp_info = info_df[info_df['uniprot_id'] == uniprot_id]

        if tmp_info.empty:
            print(f"[WARNING] No PDB info found for UniProt ID: {uniprot_id}")
            return

        tmp_pdbs = set(tmp_info['pdb_filename'].tolist())
        print(f"  Found {len(tmp_pdbs)} PDB files for {uniprot_id}")

        for pdb_file in tmp_pdbs:
            try:
                cmd.reinitialize()

                # Load PDB structure - handle both .pdb and .pdb.gz files
                pdb_path = os.path.join(reference_directory, 'pdb_files', pdb_file)

                # Check if file exists as is, or with .gz extension
                if os.path.exists(pdb_path):
                    full_path = pdb_path
                elif os.path.exists(pdb_path + '.gz'):
                    full_path = pdb_path + '.gz'
                elif pdb_file.endswith('.pdb') and os.path.exists(pdb_path + '.gz'):
                    full_path = pdb_path + '.gz'
                else:
                    print(f"[WARNING] PDB file not found: {pdb_path} or {pdb_path}.gz")
                    continue

                cmd.load(full_path)
                cmd.show("cartoon")
                cmd.color("gray")
                cmd.hide("lines")

                # Save basic gray structure
                base_name = pdb_file.split('.')[0]  # Remove file extension
                gray_pse = os.path.join(pymol_dir, f"{base_name}_gray.pse")
                cmd.save(gray_pse)
                print(f"    Saved gray PSE file: {base_name}_gray.pse")

            except Exception as e:
                print(f"[ERROR] Processing PDB file {pdb_file}: {e}")

    except Exception as e:
        print(f"[ERROR] in pymol_rvas_quantitative(): {e}")


def pymol_scan_test_quantitative(info_tsv, uniprot_id, reference_directory, results_directory):
    # Color by regularized mean beta in the neighborhood

    '''
    Create PyMOL visualizations for quantitative trait scan test results. For each PDB file:
    - Color residues based on their regularized mean beta (blue to red gradient).
    - Outputs a PSE file with beta-based coloring.
    '''
    try:
        # Create pymol_visualizations subdirectory
        pymol_dir = os.path.join(results_directory, 'pymol_visualizations')
        os.makedirs(pymol_dir, exist_ok=True)

        df_results_p = os.path.join(results_directory, 'p_values_quantitative.h5')

        if not os.path.exists(df_results_p):
            print(f"[WARNING] Quantitative results file not found: {df_results_p}")
            return

        with h5py.File(df_results_p, 'r') as fid:
            # Check if the UniProt ID exists in the file
            if uniprot_id not in fid:
                available_proteins = [k for k in fid.keys() if '_' not in k]
                print(f"[WARNING] UniProt ID {uniprot_id} not found in quantitative results file")
                print(f"[INFO] Available proteins: {', '.join(available_proteins[:10])}{'...' if len(available_proteins) > 10 else ''}")
                return

            df_results = read_p_values_quantitative(fid, uniprot_id)

        if info_tsv is not None:
            info = os.path.join(reference_directory, info_tsv)
            info_df = pd.read_csv(info, sep='\t')
        else:
            print(f"[WARNING] Info TSV not found: {info_tsv}")
            return

        tmp_info = info_df[info_df['uniprot_id'] == uniprot_id]
        tmp_df = df_results[df_results['uniprot_id'] == uniprot_id]
        tmp_df = get_pdb_filename(tmp_df, tmp_info)
        print(f"Processing quantitative scan test results for {uniprot_id}: {len(tmp_df)} amino acid positions")

        # Calculate regularized mean beta for coloring
        # Regularization: add 2 synthetic variants with protein-wide mean beta to each neighborhood
        protein_wide_mean_beta = tmp_df['mean_beta_in'].mean()  # Use mean of neighborhood means as proxy

        # For each position, calculate regularized mean beta
        regularized_betas = []
        for _, row in tmp_df.iterrows():
            mean_beta_in = row['mean_beta_in']
            n_variants_in = row['n_variants_in']

            # Regularized mean = (sum of actual betas + 2 * protein_wide_mean) / (n_actual + 2)
            # Since we have mean_beta_in, we can recover the sum: sum = mean_beta_in * n_variants_in
            if n_variants_in > 0:
                actual_sum = mean_beta_in * n_variants_in
                regularized_mean = (actual_sum + 2 * protein_wide_mean_beta) / (n_variants_in + 2)
            else:
                # If no variants in neighborhood, use protein-wide mean
                regularized_mean = protein_wide_mean_beta

            regularized_betas.append(regularized_mean)

        tmp_df['regularized_mean_beta'] = regularized_betas

        # Normalize for coloring - center around 0 and scale
        beta_mean = tmp_df['regularized_mean_beta'].mean()
        beta_std = tmp_df['regularized_mean_beta'].std()
        if beta_std > 0:
            tmp_df['beta_normalized'] = (tmp_df['regularized_mean_beta'] - beta_mean) / beta_std
            # Clip extreme values and scale to 0-1 for PyMOL coloring
            beta_min = tmp_df['beta_normalized'].quantile(0.05)
            beta_max = tmp_df['beta_normalized'].quantile(0.95)
            tmp_df['beta_normalized'] = tmp_df['beta_normalized'].clip(beta_min, beta_max)
            tmp_df['beta_color'] = (tmp_df['beta_normalized'] - beta_min) / (beta_max - beta_min)
        else:
            tmp_df['beta_color'] = 0.5  # Neutral color if no variation

        print(f"  Regularized mean beta range: {tmp_df['regularized_mean_beta'].min():.3f} to {tmp_df['regularized_mean_beta'].max():.3f}")
        print(f"  Using protein-wide mean beta for regularization: {protein_wide_mean_beta:.3f}")

        tmp_df['visual_filename'] = tmp_df['pdb_filename'].apply(lambda x: os.path.join(pymol_dir, x.split('.')[0] + '.pse'))
        tmp_visuals = set(tmp_df['visual_filename'].tolist())

        for v in tmp_visuals:
            gray_pse_path = f"{v.split('.')[0]}_gray.pse"
            if not os.path.exists(gray_pse_path):
                print(f"[WARNING] Gray PSE file not found: {gray_pse_path}")
                continue
            cmd.load(gray_pse_path)
            objects = cmd.get_names('objects')[-1]

            tmp_df_visuals = tmp_df[tmp_df['visual_filename'] == v]

            for _, row in tmp_df_visuals.iterrows():
                resi = int(row['aa_pos'])
                beta_color = float(row['beta_color'])
                selection = f"{objects} and resi {resi}"
                cmd.alter(selection, f"b={beta_color}")
                cmd.rebuild()

            # Use blue-white-red spectrum for beta values (negative to positive effect sizes)
            cmd.spectrum("b", "blue_white_red", objects, byres=1)
            cmd.show("cartoon", objects)
            cmd.hide("lines", objects)

            cmd.save(f"{v.split('.')[0]}_beta.pse")
            print(f"  Saved quantitative beta PSE file: {v.split('.')[0]}_beta.pse")

    except Exception as e:
        print(f"[ERROR] in pymol_scan_test_quantitative(): {e}")

def pymol_neighborhood(uniprot_id, results_directory, info_tsv, reference_directory):
    # for each significant neighborhood, zoom in and show the case and control mutations
    # just in that neighborhood.
    '''
    Create PyMOL visualizations for significant neighborhood in scan test results. For each PDB file:
    - Load the PSE file with ratio colored and show spheres for residues with p-value < 0.05.
    - Outputs a image with spheres for significant residues.
    '''
    try:
        # Create pymol_visualizations subdirectory
        pymol_dir = os.path.join(results_directory, 'pymol_visualizations')
        os.makedirs(pymol_dir, exist_ok=True)
        
        df_results_p = os.path.join(results_directory, 'p_values.h5')

        with h5py.File(df_results_p, 'r') as fid:
            df_results = read_p_values(fid, uniprot_id)

        if info_tsv is not None:
            info = os.path.join(reference_directory, info_tsv)
            info_df = pd.read_csv(info, sep='\t')
        else:
            print(f"[WARNING] Info TSV not found: {info_tsv}")
            return
        
        tmp_info = info_df[info_df['uniprot_id'] == uniprot_id]
        tmp_df = df_results[df_results['uniprot_id'] == uniprot_id]
        tmp_df = get_pdb_filename(tmp_df, tmp_info)

        tmp_df['visual_filename'] = tmp_df['pdb_filename'].apply(lambda x: os.path.join(pymol_dir, x.split('.')[0]+ '.pse'))
        tmp_visuals = set(tmp_df['visual_filename'].tolist())
        for v in tmp_visuals:
            cmd.load(v.split('.')[0] + '_ratio.pse')
            # Removed sphere display for significant residues to keep clean cartoon view
            
            cmd.save(v.split('.')[0] + '_ratio.pse')

            cmd.ray(2400, 1800)
            cmd.set("ray_opaque_background", 1)
            cmd.png(f"{v.split('.')[0]}_ratio.png")

    except Exception as e:  
        print(f"[ERROR] in pymol_neighborhood(): {e}")

def pymol_neighborhood_quantitative(uniprot_id, results_directory, info_tsv, reference_directory):
    '''
    Create neighborhood visualization for quantitative traits colored by regularized mean beta.
    '''
    info = os.path.join(reference_directory, info_tsv)
    info_df = pd.read_csv(info, sep='\t')

    # Filter for the specified UniProt ID
    tmp_info = info_df[info_df['uniprot_id'] == uniprot_id]
    if tmp_info.empty:
        print(f"[WARNING] No PDB info found for UniProt ID: {uniprot_id}")
        return

    tmp_pdbs = set(tmp_info['pdb_filename'].tolist())

    # Create pymol_visualizations subdirectory
    pymol_dir = os.path.join(results_directory, 'pymol_visualizations')
    os.makedirs(pymol_dir, exist_ok=True)

    for v in tmp_pdbs:
        try:
            cmd.reinitialize()

            # Load the FDR results to get the most significant position
            fdr_file = os.path.join(results_directory, 'all_proteins.fdr.tsv')
            if not os.path.exists(fdr_file):
                print(f"[WARNING] FDR results file not found: {fdr_file}")
                continue

            df_fdr = pd.read_csv(fdr_file, sep='\t')
            df_protein = df_fdr[df_fdr.uniprot_id == uniprot_id]
            if df_protein.empty:
                print(f"[WARNING] No FDR results found for UniProt ID: {uniprot_id}")
                continue

            # Find the most significant position (lowest p-value)
            min_p_idx = df_protein['p_value'].idxmin()
            center_pos = df_protein.loc[min_p_idx, 'aa_pos']

            print(f"[INFO] Using center position {center_pos} for neighborhood visualization")

            # Load PDB structure - handle both .pdb and .pdb.gz files
            pdb_path = os.path.join(reference_directory, 'pdb_files', v)

            # Check if file exists as is, or with .gz extension
            if os.path.exists(pdb_path):
                full_path = pdb_path
            elif os.path.exists(pdb_path + '.gz'):
                full_path = pdb_path + '.gz'
            elif v.endswith('.pdb') and os.path.exists(pdb_path + '.gz'):
                full_path = pdb_path + '.gz'
            else:
                print(f"[WARNING] PDB file not found: {pdb_path} or {pdb_path}.gz")
                continue

            cmd.load(full_path)
            cmd.show("cartoon")
            cmd.set("cartoon_transparency", 0.7)
            cmd.color("gray")

            # Get PDB mapping info for this structure
            pdb_info = tmp_info[tmp_info['pdb_filename'] == v].iloc[0]

            # For AlphaFold structures, typically use chain A
            # Extract PDB ID from filename if available
            pdb_filename = pdb_info['pdb_filename']
            if 'AF-' in pdb_filename:
                # AlphaFold structure - use chain A
                chain_id = 'A'
                pdb_id = pdb_filename.split('.')[0]  # Use filename as PDB ID
            else:
                # Try to get from columns if they exist, otherwise default to chain A
                chain_id = pdb_info.get('chain_id', 'A')
                pdb_id = pdb_info.get('pdb_id', pdb_filename.split('.')[0])

            # Read PAE data if available
            pae_file = pdb_info['pae_filename']
            if pd.notna(pae_file):
                pae_path = os.path.join(reference_directory, pae_file)
                try:
                    # Handle both compressed and uncompressed PAE files
                    if pae_path.endswith('.gz'):
                        import gzip
                        with gzip.open(pae_path, 'rt') as f:
                            pae_data = json.load(f)
                    else:
                        with open(pae_path, 'r') as f:
                            pae_data = json.load(f)
                    pae_matrix = np.array(pae_data[0]['predicted_aligned_error'])
                except Exception as e:
                    print(f"[WARNING] Could not load PAE data from {pae_path}: {e}")
                    pae_matrix = None
            else:
                pae_matrix = None

            # Define neighborhood around center position (15A radius)
            neighborhood_radius = 15.0
            pae_cutoff = 15.0

            # Get residues in neighborhood
            neighborhood_residues = []

            # If we have PAE data, use it for filtering
            if pae_matrix is not None and center_pos <= len(pae_matrix):
                for i in range(len(pae_matrix)):
                    pae_val = pae_matrix[center_pos-1, i]  # Convert to 0-based indexing
                    if pae_val <= pae_cutoff:
                        # Check distance
                        try:
                            distance = cmd.get_distance(f"chain {chain_id} and resi {center_pos} and name CA",
                                                      f"chain {chain_id} and resi {i+1} and name CA")
                            if distance <= neighborhood_radius:
                                neighborhood_residues.append(i+1)
                        except:
                            pass  # Skip if distance calculation fails
            else:
                # Fall back to distance-only filtering
                try:
                    cmd.select("center_residue", f"chain {chain_id} and resi {center_pos}")
                    cmd.select("neighborhood", f"chain {chain_id} and name CA within {neighborhood_radius} of center_residue")
                    neighborhood_residues = cmd.get_model("neighborhood").get_residues()
                    neighborhood_residues = [int(r.resi) for r in neighborhood_residues]
                except:
                    print(f"[WARNING] Could not determine neighborhood for position {center_pos}")
                    continue

            # Color neighborhood residues by their regularized mean beta values
            protein_data = df_protein.copy()

            # Calculate protein-wide mean beta for regularization
            protein_wide_mean_beta = protein_data['mean_beta_in'].mean()

            for pos in neighborhood_residues:
                pos_data = protein_data[protein_data.aa_pos == pos]
                if not pos_data.empty:
                    mean_beta_in = pos_data.iloc[0]['mean_beta_in']
                    n_variants_in = pos_data.iloc[0]['n_variants_in']

                    # Calculate regularized mean beta
                    if n_variants_in > 0:
                        actual_sum = mean_beta_in * n_variants_in
                        regularized_mean = (actual_sum + 2 * protein_wide_mean_beta) / (n_variants_in + 2)
                    else:
                        regularized_mean = protein_wide_mean_beta

                    # Determine color based on regularized mean beta
                    if regularized_mean > 0.1:  # Positive effect
                        color = "red"
                    elif regularized_mean < -0.1:  # Negative effect
                        color = "blue"
                    else:  # Near zero effect
                        color = "white"

                    cmd.color(color, f"chain {chain_id} and resi {pos}")
                else:
                    # No data for this position, color gray
                    cmd.color("gray", f"chain {chain_id} and resi {pos}")

            # Highlight the center residue
            cmd.color("yellow", f"chain {chain_id} and resi {center_pos}")
            cmd.show("spheres", f"chain {chain_id} and resi {center_pos}")

            # Set view and save
            cmd.orient()
            cmd.zoom("all", 5)  # Zoom out a bit

            # Save PyMOL session
            pse_path = os.path.join(pymol_dir, f"{v.split('.')[0]}_beta_neighborhood.pse")
            cmd.save(pse_path)

            # Render and save image
            cmd.ray(2400, 1800)
            cmd.set("ray_opaque_background", 1)
            cmd.png(f"{v.split('.')[0]}_beta_neighborhood.png")

        except Exception as e:
            print(f"[ERROR] in pymol_neighborhood_quantitative(): {e}")


def make_movie_from_pse(result_directory, pse_name):
    '''
    Create a movie from a PyMOL session file (.pse).
    '''
    # Create pymol_visualizations subdirectory
    pymol_dir = os.path.join(result_directory, 'pymol_visualizations')
    os.makedirs(pymol_dir, exist_ok=True)

    pse = os.path.join(pymol_dir, f"{pse_name}.pse")
    try:
        cmd.load(pse)
        cmd.ray(2400, 1800)
        cmd.set("ray_opaque_background", 1)
        cmd.png(f"{pymol_dir}/{pse_name}.png")
        cmd.movie.add_roll(10, axis='y', start=1)
        mv = os.path.join(pymol_dir, f"{pse_name}.mov")
        cmd.movie.produce(mv)

    except Exception as e:
        print(f"[ERROR] Failed to create movie from PSE: {e}")

# def make_movie(results_directory, uniprot_id, info_tsv=None, reference_directory=None):

#     if info_tsv is not None:
#         info = os.path.join(reference_directory, info_tsv)
#         info_df = pd.read_csv(info, sep='\t')
#     else:
#         print(f"[WARNING] Info TSV not found: {info_tsv}")
#         return

#     tmp_info = info_df[info_df['uniprot_id'] == uniprot_id]
#     tmp_pdbs = set(tmp_info['pdb_filename'].tolist())
#     tmp_pses = [item.split('.')[0] for item in tmp_pdbs]

#     for item in tmp_pses:
#         gray_mv_p = os.path.join(results_directory, f'{item}_gray.mov')
#         rib_mut_mv_p = os.path.join(results_directory, f'{item}_rib_mut.mov')
#         ratio_mv_p = os.path.join(results_directory, f'{item}_ratio.mov')

#         if not os.path.exists(gray_mv_p):
#             print(f"[WARNING] pymol_rvas() base movie file not found: {gray_mv_p}")
#             return
#         if not os.path.exists(rib_mut_mv_p): 
#             print(f"[WARNING] pymol_rvas() movie file not found: {rib_mut_mv_p}")
#             return
        
#         if not os.path.exists(ratio_mv_p): 
#             print(f"[WARNING] pymol_neighborhood() movie file not found: {ratio_mv_p}")
#             return
        
#         clip1 = VideoFileClip(gray_mv_p)
#         clip2 = VideoFileClip(rib_mut_mv_p)
#         clip3 = VideoFileClip(ratio_mv_p)

#         min_duration = min(clip1.duration, clip2.duration, clip3.duration)
#         clip1 = clip1.subclipped(0, min_duration)
#         clip2 = clip2.subclipped(0, min_duration)
#         clip3 = clip3.subclipped(0, min_duration)

#         target_height = min(clip1.h, clip2.h, clip3.h)
#         clip1 = clip1.resized(height=target_height)
#         clip2 = clip2.resized(height=target_height)
#         clip3 = clip3.resized(height=target_height)

#         final_clip = clips_array([[clip1, clip2, clip3]])

#         output_file = os.path.join(results_directory, f"{uniprot_id}.mov")
#         final_clip.write_videofile(output_file, codec="libx264", fps=24)


def run_all(uniprot_id, results_directory, reference_directory):
    '''
    Run all PyMOL visualizations for a given UniProt ID.
    '''
    pymol_rvas(uniprot_id, reference_directory, results_directory)
    # pymol_annotation('ClinVar_PLP_uniprot_canonical.tsv', reference_directory , results_directory, 'pdb_pae_file_pos_guide.tsv', uniprot_id)
    pymol_scan_test('pdb_pae_file_pos_guide.tsv', uniprot_id, reference_directory, results_directory)
    pymol_neighborhood(uniprot_id, results_directory, 'pdb_pae_file_pos_guide.tsv', reference_directory)


def run_all_quantitative(uniprot_id, results_directory, reference_directory):
    '''
    Run all PyMOL visualizations for quantitative traits for a given UniProt ID.
    '''
    pymol_rvas_quantitative(uniprot_id, reference_directory, results_directory)  # Creates gray structure files
    pymol_scan_test_quantitative('pdb_pae_file_pos_guide.tsv', uniprot_id, reference_directory, results_directory)
    pymol_neighborhood_quantitative(uniprot_id, results_directory, 'pdb_pae_file_pos_guide.tsv', reference_directory)


