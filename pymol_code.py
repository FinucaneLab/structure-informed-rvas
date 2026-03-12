from pymol import cmd
import pandas as pd
import os
import ast
import re
from utils import read_p_values, read_original_mutation_data
import h5py


def _load_gene_name(uniprot_id, reference_directory):
    """Look up gene name for a UniProt ID from gene_to_uniprot_id.tsv."""
    path = os.path.join(reference_directory, 'gene_to_uniprot_id.tsv')
    try:
        df = pd.read_csv(path, sep='\t')
        match = df[df['uniprot_id'] == uniprot_id]
        if not match.empty:
            return match.iloc[0]['gene_name']
    except Exception:
        pass
    return uniprot_id  # fall back to UniProt ID if not found


def _pse_base(pdb_filename, gene_name, uniprot_id):
    """Return the base name for PSE files: {gene_name}_{uniprot_id}[_FN]."""
    match = re.search(r'-F(\d+)-', pdb_filename)
    frag = f'_F{match.group(1)}' if match else ''
    return f'{gene_name}_{uniprot_id}{frag}'


def pymol_rvas(uniprot_id, reference_directory, results_directory, gene_name=None):
    # make a pymol session with case and control mutations
    # output a gif and a .pse file
    '''
    Create PyMOL visualizations for RVAS results. For each PDB file:
    - Produces a basic image and PSE file.
    - Highlights mutations for control-only (blue), case-only (red), both (purple).
    - Outputs a second image and PSE file with mutations shown and colored.
    '''

    try:
        if gene_name is None:
            gene_name = _load_gene_name(uniprot_id, reference_directory)
        print(f"Creating PyMOL RVAS visualizations for {gene_name} ({uniprot_id})")

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
        if len(pdbs) > 1:
            print(f"  Skipping {gene_name} ({uniprot_id}): protein spans multiple PDB chunks ({len(pdbs)} files)")
            return

        # cmd.set('ribbon_as_cylinders')
        # cmd.set("ribbon_radius", 0.5)

        for item in pdbs:
            p = os.path.join(reference_directory, f'pdb_files/{item}')
            if not os.path.isfile(p):
                print(f"[ERROR] PDB file not found: {p}")
                continue
            print(f'Processing PDB file: {p}')

            cmd.reinitialize()
            cmd.load(p, "structure")
            pse_base = _pse_base(item, gene_name, uniprot_id)

            cmd.bg_color("white")
            cmd.reset()
            cmd.color("grey70")
            gray_pse = os.path.join(pymol_dir, f"{pse_base}_gray.pse")
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

            control_pos_strs = [str(row['aa_pos_file']) for _, row in tmp_df_control.iterrows()]
            case_pos_strs = [str(row['aa_pos_file']) for _, row in tmp_df_case.iterrows()]
            both_pos_strs = [str(row['aa_pos_file']) for _, row in tmp_df_both.iterrows()]

            if control_pos_strs:
                cmd.select("control_only", f"resi {'+'.join(control_pos_strs)} and name CA")
                cmd.show("spheres", "control_only")
                cmd.color("blue", "control_only")

            if case_pos_strs:
                cmd.select("case_only", f"resi {'+'.join(case_pos_strs)} and name CA")
                cmd.show("spheres", "case_only")
                cmd.color("red", "case_only")

            if both_pos_strs:
                cmd.select("case_and_control", f"resi {'+'.join(both_pos_strs)} and name CA")
                cmd.show("spheres", "case_and_control")
                cmd.color("purple", "case_and_control")
            
            pse_pdb_p = os.path.join(pymol_dir, f"{pse_base}_mut.pse")
            cmd.save(pse_pdb_p)
            print(f"  Saved mutations PSE file: {pse_pdb_p}")

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

def pymol_scan_test(info_tsv, uniprot_id, reference_directory, results_directory, gene_name=None):
    # color by case/control ratio of the neighborhood

    '''
    Create PyMOL visualizations for scan test results. For each PDB file:
    - Color residues based on their case/control ratio (green_red).
    - Outputs a third PSE file with ratio colored.
    '''
    try:
        if gene_name is None:
            gene_name = _load_gene_name(uniprot_id, reference_directory)

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
        tmp_df['visual_filename'] = tmp_df['pdb_filename'].apply(lambda x: os.path.join(pymol_dir, _pse_base(x, gene_name, uniprot_id) + '.pse'))
        tmp_df['ratio_normalized'] = tmp_df['ratio'] / tmp_df['ratio'].max()
        # tmp_df['ratio_normalized'].to_csv('test.csv', sep='\t', index=False)
        tmp_visuals = set(tmp_df['visual_filename'].tolist())
        for v in tmp_visuals:
            cmd.reinitialize()
            cmd.load(f"{v.split('.')[0]}_gray.pse")
            objects = cmd.get_names('objects')[-1]

            tmp_df_visuals = tmp_df[tmp_df['visual_filename'] == v]

            for _, row in tmp_df_visuals.iterrows():
                resi = int(row['aa_pos'])
                ratio = float(row['ratio_normalized'])
                selection = f"{objects} and resi {resi}"
                cmd.alter(selection, f"b={ratio}")
                cmd.rebuild()
            
            cmd.spectrum("b", "green_red", objects, byres=1)
            cmd.show("cartoon", objects)
            cmd.hide("lines", objects)

            cmd.save(f"{v.split('.')[0]}_ratio.pse")

    except Exception as e:
        print(f"[ERROR] in pymol_scan_test(): {e}")

def pymol_neighborhood(uniprot_id, results_directory, info_tsv, reference_directory, gene_name=None):
    # for each significant neighborhood, zoom in and show the case and control mutations
    # just in that neighborhood.
    '''
    Create PyMOL visualizations for significant neighborhood in scan test results. For each PDB file:
    - Load the PSE file with ratio colored and show spheres for residues with p-value < 0.05.
    - Outputs a image with spheres for significant residues.
    '''
    try:
        # Create pymol_visualizations subdirectory
        if gene_name is None:
            gene_name = _load_gene_name(uniprot_id, reference_directory)

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

        tmp_df['visual_filename'] = tmp_df['pdb_filename'].apply(lambda x: os.path.join(pymol_dir, _pse_base(x, gene_name, uniprot_id) + '.pse'))
        tmp_visuals = set(tmp_df['visual_filename'].tolist())
        for v in tmp_visuals:
            cmd.reinitialize()
            cmd.load(v.split('.')[0] + '_ratio.pse')
            # Removed sphere display for significant residues to keep clean cartoon view
            
            cmd.save(v.split('.')[0] + '_ratio.pse')

    except Exception as e:  
        print(f"[ERROR] in pymol_neighborhood(): {e}")

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


def run_all(uniprot_id, results_directory, reference_directory):
    '''
    Run all PyMOL visualizations for a given UniProt ID.
    '''
    gene_name = _load_gene_name(uniprot_id, reference_directory)
    pymol_rvas(uniprot_id, reference_directory, results_directory, gene_name=gene_name)
    # pymol_annotation('ClinVar_PLP_uniprot_canonical.tsv', reference_directory , results_directory, 'pdb_pae_file_pos_guide.tsv', uniprot_id)
    pymol_scan_test('pdb_pae_file_pos_guide.tsv', uniprot_id, reference_directory, results_directory, gene_name=gene_name)
    pymol_neighborhood(uniprot_id, results_directory, 'pdb_pae_file_pos_guide.tsv', reference_directory, gene_name=gene_name)


