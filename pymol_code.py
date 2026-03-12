from pymol import cmd
import pandas as pd
import os
import ast
import re
from utils import read_p_values, read_original_mutation_data
import h5py

INFO_TSV = 'pdb_pae_file_pos_guide.tsv'


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


def get_pdb_filename(annot_df, info_df):
    """Map each row in annot_df to its PDB filename based on amino acid position."""
    pdb_filenames = []
    for _, r1 in annot_df.iterrows():
        aa_pos = r1['aa_pos']
        for _, r2 in info_df.iterrows():
            pos_covered = ast.literal_eval(r2['pos_covered'])
            if int(pos_covered[0]) <= int(aa_pos) <= int(pos_covered[1]):
                pdb_filename = r2['pdb_filename']
                break
        pdb_filenames.append(pdb_filename)
    annot_df['pdb_filename'] = pdb_filenames
    return annot_df


def pymol_rvas(uniprot_id, reference_directory, results_directory, gene_name=None):
    """Create PyMOL visualizations for RVAS results.

    Produces two PSE files per protein:
    - {gene}_{uniprot}_gray.pse: plain grey cartoon structure
    - {gene}_{uniprot}_mut.pse: structure with mutations colored
      (blue = control-only, red = case-only, purple = both)
    """
    try:
        if gene_name is None:
            gene_name = _load_gene_name(uniprot_id, reference_directory)
        print(f"Creating PyMOL RVAS visualizations for {gene_name} ({uniprot_id})")

        pymol_dir = os.path.join(results_directory, 'pymol_visualizations')
        os.makedirs(pymol_dir, exist_ok=True)

        df_results_p = os.path.join(results_directory, 'p_values.h5')
        if not os.path.isfile(df_results_p):
            print(f"[WARNING] Results file not found: {df_results_p}")
            return

        with h5py.File(df_results_p, 'r') as fid:
            df_rvas = read_original_mutation_data(fid, uniprot_id)

        print(f"  Loaded scan test data: {len(df_rvas)} amino acid positions")

        info = os.path.join(reference_directory, INFO_TSV)
        if not os.path.isfile(info):
            print(f"[WARNING] Info TSV not found: {info}")
            return
        info_df = pd.read_csv(info, sep='\t')

        tmp_info = info_df[info_df['uniprot_id'] == uniprot_id]
        df_rvas = get_pdb_filename(df_rvas, tmp_info)

        if 'aa_pos_file' not in df_rvas:
            df_rvas['aa_pos_file'] = df_rvas['aa_pos']
        pdbs = set(df_rvas['pdb_filename'].tolist())
        if len(pdbs) > 1:
            print(f"  Skipping {gene_name} ({uniprot_id}): protein spans multiple PDB chunks ({len(pdbs)} files)")
            return

        for item in pdbs:
            p = os.path.join(reference_directory, 'pdb_files', item)
            if not os.path.isfile(p):
                print(f"[ERROR] PDB file not found: {p}")
                continue
            print(f'  Processing PDB file: {p}')

            cmd.reinitialize()
            cmd.load(p, "structure")
            pse_base = _pse_base(item, gene_name, uniprot_id)

            cmd.bg_color("white")
            cmd.reset()
            cmd.color("grey70")
            gray_pse = os.path.join(pymol_dir, f"{pse_base}_gray.pse")
            cmd.save(gray_pse)
            print(f"  Saved gray PSE: {gray_pse}")

            tmp_df = df_rvas[df_rvas['pdb_filename'] == item]

            control_mask = (tmp_df['ac_control'] >= 1) & (tmp_df['ac_case'] == 0)
            case_mask    = (tmp_df['ac_case'] >= 1)    & (tmp_df['ac_control'] == 0)
            both_mask    = (tmp_df['ac_case'] >= 1)    & (tmp_df['ac_control'] >= 1)

            control_pos = set(tmp_df[control_mask]['aa_pos'])
            case_pos    = set(tmp_df[case_mask]['aa_pos'])
            both_pos    = set(tmp_df[both_mask]['aa_pos'])

            # Positions in both sets move to the "both" category
            final_control = control_pos - case_pos - both_pos
            final_case    = case_pos    - control_pos - both_pos
            final_both    = both_pos    | (control_pos & case_pos)

            print(f"  {len(final_control)} control-only, {len(final_case)} case-only, {len(final_both)} both")

            def _pos_strs(pos_set):
                return [str(r['aa_pos_file']) for _, r in tmp_df[tmp_df['aa_pos'].isin(pos_set)].iterrows()]

            control_pos_strs = _pos_strs(final_control)
            case_pos_strs    = _pos_strs(final_case)
            both_pos_strs    = _pos_strs(final_both)

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
            print(f"  Saved mutations PSE: {pse_pdb_p}")

    except Exception as e:
        print(f"[ERROR] in pymol_rvas(): {e}")


def pymol_scan_test(uniprot_id, reference_directory, results_directory, gene_name=None):
    """Color protein structure by per-residue case/control ratio.

    Loads the gray PSE produced by pymol_rvas and colors it on a green-red
    spectrum by normalized neighborhood ratio, saving a {gene}_{uniprot}_ratio.pse.
    """
    try:
        if gene_name is None:
            gene_name = _load_gene_name(uniprot_id, reference_directory)

        pymol_dir = os.path.join(results_directory, 'pymol_visualizations')
        os.makedirs(pymol_dir, exist_ok=True)

        df_results_p = os.path.join(results_directory, 'p_values.h5')
        with h5py.File(df_results_p, 'r') as fid:
            df_results = read_p_values(fid, uniprot_id)

        info = os.path.join(reference_directory, INFO_TSV)
        if not os.path.isfile(info):
            print(f"[WARNING] Info TSV not found: {info}")
            return
        info_df = pd.read_csv(info, sep='\t')

        tmp_info = info_df[info_df['uniprot_id'] == uniprot_id]
        tmp_df = df_results[df_results['uniprot_id'] == uniprot_id]
        tmp_df = get_pdb_filename(tmp_df, tmp_info)
        print(f"  Processing scan test results for {gene_name} ({uniprot_id}): {len(tmp_df)} positions")

        tmp_df['visual_filename'] = tmp_df['pdb_filename'].apply(
            lambda x: os.path.join(pymol_dir, _pse_base(x, gene_name, uniprot_id) + '.pse')
        )
        tmp_df['ratio_normalized'] = tmp_df['ratio'] / tmp_df['ratio'].max()

        for v in set(tmp_df['visual_filename']):
            cmd.reinitialize()
            cmd.load(f"{v.split('.')[0]}_gray.pse")
            objects = cmd.get_names('objects')[-1]

            for _, row in tmp_df[tmp_df['visual_filename'] == v].iterrows():
                cmd.alter(f"{objects} and resi {int(row['aa_pos'])}", f"b={float(row['ratio_normalized'])}")
            cmd.rebuild()

            cmd.spectrum("b", "green_red", objects, byres=1)
            cmd.show("cartoon", objects)
            cmd.hide("lines", objects)

            cmd.save(f"{v.split('.')[0]}_ratio.pse")

    except Exception as e:
        print(f"[ERROR] in pymol_scan_test(): {e}")


def pymol_neighborhood(uniprot_id, results_directory, reference_directory, gene_name=None):
    """Re-save the ratio PSE (hook for future neighborhood-level annotations)."""
    try:
        if gene_name is None:
            gene_name = _load_gene_name(uniprot_id, reference_directory)

        pymol_dir = os.path.join(results_directory, 'pymol_visualizations')
        os.makedirs(pymol_dir, exist_ok=True)

        df_results_p = os.path.join(results_directory, 'p_values.h5')
        with h5py.File(df_results_p, 'r') as fid:
            df_results = read_p_values(fid, uniprot_id)

        info = os.path.join(reference_directory, INFO_TSV)
        if not os.path.isfile(info):
            print(f"[WARNING] Info TSV not found: {info}")
            return
        info_df = pd.read_csv(info, sep='\t')

        tmp_info = info_df[info_df['uniprot_id'] == uniprot_id]
        tmp_df = df_results[df_results['uniprot_id'] == uniprot_id]
        tmp_df = get_pdb_filename(tmp_df, tmp_info)

        tmp_df['visual_filename'] = tmp_df['pdb_filename'].apply(
            lambda x: os.path.join(pymol_dir, _pse_base(x, gene_name, uniprot_id) + '.pse')
        )

        for v in set(tmp_df['visual_filename']):
            cmd.reinitialize()
            cmd.load(v.split('.')[0] + '_ratio.pse')
            cmd.save(v.split('.')[0] + '_ratio.pse')

    except Exception as e:
        print(f"[ERROR] in pymol_neighborhood(): {e}")


def make_movie_from_pse(result_directory, pse_name):
    """Render a PNG and rotating movie from a saved PSE file.

    Looks for {result_directory}/pymol_visualizations/{pse_name}.pse and
    produces a PNG snapshot and a .mov rotating movie in the same directory.
    Requires ffmpeg on PATH.
    """
    pymol_dir = os.path.join(result_directory, 'pymol_visualizations')
    os.makedirs(pymol_dir, exist_ok=True)
    pse = os.path.join(pymol_dir, f"{pse_name}.pse")
    try:
        cmd.reinitialize()
        cmd.load(pse)
        cmd.ray(2400, 1800)
        cmd.set("ray_opaque_background", 1)
        cmd.png(os.path.join(pymol_dir, f"{pse_name}.png"))
        cmd.movie.add_roll(10, axis='y', start=1)
        cmd.movie.produce(os.path.join(pymol_dir, f"{pse_name}.mov"))
    except Exception as e:
        print(f"[ERROR] Failed to create movie from PSE: {e}")


def run_all(uniprot_id, results_directory, reference_directory):
    """Run the full PyMOL visualization pipeline for a single protein."""
    gene_name = _load_gene_name(uniprot_id, reference_directory)
    pymol_rvas(uniprot_id, reference_directory, results_directory, gene_name=gene_name)
    pymol_scan_test(uniprot_id, reference_directory, results_directory, gene_name=gene_name)
    pymol_neighborhood(uniprot_id, results_directory, reference_directory, gene_name=gene_name)
