from pymol import cmd
import pandas as pd
import os
import ast
import re
from utils import read_p_values, read_original_mutation_data, resolve_pval_node
import h5py
import numpy as np

INFO_TSV = 'pdb_pae_file_pos_guide.tsv'


def _load_gene_name(uniprot_id, reference_directory):
    """Look up gene name for a UniProt ID from protein_sequence_guide.tsv."""
    path = os.path.join(reference_directory, 'protein_sequence_guide.tsv')
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


def pymol_rvas(uniprot_id, reference_directory, results_directory, pval_file, gene_name=None,
               test='binomial'):
    """Create PyMOL visualizations for RVAS results.

    Produces two PSE files per protein:
    - {gene}_{uniprot}_gray.pse: plain grey cartoon structure
    - {gene}_{uniprot}_mut.pse: structure with mutations colored

    The standard 3DNT colors by case/control class (blue = control-only, red = case-only,
    purple = both). A mutation-rate run has no control cohort at all -- the comparison is
    against the modelled rate -- so it instead shows de novo positions in one colour,
    darkened where a residue carries more than one.
    """
    try:
        if gene_name is None:
            gene_name = _load_gene_name(uniprot_id, reference_directory)
        print(f"Creating PyMOL RVAS visualizations for {gene_name} ({uniprot_id})")

        pymol_dir = os.path.join(results_directory, 'pymol_visualizations')
        os.makedirs(pymol_dir, exist_ok=True)

        df_results_p = pval_file
        if not os.path.isfile(df_results_p):
            print(f"[WARNING] Results file not found: {df_results_p}")
            return

        with h5py.File(df_results_p, 'r') as fid:
            node, mu_mode = resolve_pval_node(fid, test)
            df_rvas = read_original_mutation_data(node, uniprot_id, mu_mode=mu_mode)

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

            def _pos_strs(pos_set):
                return [str(r['aa_pos_file']) for _, r in tmp_df[tmp_df['aa_pos'].isin(pos_set)].iterrows()]

            def _select_show(name, pos_set, color):
                pos_strs = _pos_strs(pos_set)
                if not pos_strs:
                    return
                cmd.select(name, f"resi {'+'.join(pos_strs)} and name CA")
                cmd.show("spheres", name)
                cmd.color(color, name)

            if mu_mode:
                single = set(tmp_df[tmp_df['ac_case'] == 1]['aa_pos'])
                multi  = set(tmp_df[tmp_df['ac_case'] >= 2]['aa_pos'])
                print(f"  {len(single)} residues with one de novo, {len(multi)} with more than one")
                _select_show("denovo", single, "salmon")
                _select_show("denovo_recurrent", multi, "firebrick")
            else:
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

                _select_show("control_only",     final_control, "blue")
                _select_show("case_only",        final_case,    "red")
                _select_show("case_and_control", final_both,    "purple")

            pse_pdb_p = os.path.join(pymol_dir, f"{pse_base}_mut.pse")
            cmd.save(pse_pdb_p)
            print(f"  Saved mutations PSE: {pse_pdb_p}")

    except Exception as e:
        print(f"[ERROR] in pymol_rvas(): {e}")


def _obs_exp_to_b(obs_exp):
    """
    Map observed/expected onto [0, 1] with the null value (1.0) at exactly 0.5.

    Symmetric in log space, so a two-fold depletion sits as far from the midpoint as a
    two-fold enrichment, and clipped to a fold-change of 4 so one extreme neighborhood
    cannot flatten the rest of the protein. The case/control `ratio` path normalizes by
    its own maximum instead, which has no meaningful anchor -- obs/exp does, and throwing
    that away would make two proteins' colours incomparable.
    """
    fold_cap = 4.0
    lg = np.log2(np.clip(np.asarray(obs_exp, dtype=float), 1e-6, None))
    return 0.5 + 0.5 * np.clip(lg / np.log2(fold_cap), -1.0, 1.0)


def pymol_scan_test(uniprot_id, reference_directory, results_directory, pval_file, gene_name=None,
                    test='binomial'):
    """Color protein structure by per-residue neighborhood effect size.

    Loads the gray PSE produced by pymol_rvas and colors it on a blue-red spectrum,
    saving a {gene}_{uniprot}_ratio.pse. The standard 3DNT colors by the case/control
    ratio normalized to its maximum; a mutation-rate run colors by observed/expected
    against the selected test's null, on a diverging scale centred at obs/exp = 1.
    """
    try:
        if gene_name is None:
            gene_name = _load_gene_name(uniprot_id, reference_directory)

        pymol_dir = os.path.join(results_directory, 'pymol_visualizations')
        os.makedirs(pymol_dir, exist_ok=True)

        df_results_p = pval_file
        with h5py.File(df_results_p, 'r') as fid:
            node, mu_mode = resolve_pval_node(fid, test)
            df_results = read_p_values(node, uniprot_id, mu_mode=mu_mode)

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
        if mu_mode:
            tmp_df['b_value'] = _obs_exp_to_b(tmp_df['obs_exp'])
            palette = "ratio_blue white ratio_red"
            # Fixed endpoints rather than the data range, so the white midpoint really is
            # obs/exp = 1 and the same colour means the same thing in every protein.
            spectrum_range = {'minimum': 0.0, 'maximum': 1.0}
        else:
            tmp_df['b_value'] = (tmp_df['ratio'] / tmp_df['ratio'].max()) ** 2
            palette = "ratio_blue ratio_red"
            spectrum_range = {}

        for v in set(tmp_df['visual_filename']):
            cmd.reinitialize()
            cmd.load(f"{v.split('.')[0]}_gray.pse")
            objects = cmd.get_names('objects')[-1]

            for _, row in tmp_df[tmp_df['visual_filename'] == v].iterrows():
                cmd.alter(f"{objects} and resi {int(row['aa_pos'])}", f"b={float(row['b_value'])}")
            cmd.rebuild()

            cmd.set_color("ratio_blue", [0.45, 0.78, 1.0])
            cmd.set_color("ratio_red",  [1.0, 0.55, 0.55])
            cmd.spectrum("b", palette, objects, byres=1, **spectrum_range)
            cmd.show("cartoon", objects)
            cmd.hide("lines", objects)
            cmd.set("specular", 0.0)
            cmd.set("ambient", 0.4)
            cmd.set("ray_trace_mode", 1)
            cmd.set("ray_trace_gain", 0.15)
            cmd.set("cartoon_loop_radius", 0.4)

            cmd.save(f"{v.split('.')[0]}_ratio.pse")

    except Exception as e:
        print(f"[ERROR] in pymol_scan_test(): {e}")


def pymol_neighborhood(uniprot_id, results_directory, reference_directory, pval_file, gene_name=None,
                       test='binomial'):
    """Re-save the ratio PSE (hook for future neighborhood-level annotations)."""
    try:
        if gene_name is None:
            gene_name = _load_gene_name(uniprot_id, reference_directory)

        pymol_dir = os.path.join(results_directory, 'pymol_visualizations')
        os.makedirs(pymol_dir, exist_ok=True)

        df_results_p = pval_file
        with h5py.File(df_results_p, 'r') as fid:
            node, mu_mode = resolve_pval_node(fid, test)
            df_results = read_p_values(node, uniprot_id, mu_mode=mu_mode)

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


def pymol_component(uniprot_id, reference_directory, results_directory, pval_file,
                    residues, label, gene_name=None, test='binomial'):
    """
    Render one component: its residues highlighted on the grey chain, its mutations
    shown as spheres.

    Loads the structure directly rather than the _gray.pse, so this works with
    --skip-visualization and does not depend on the order the other PSEs were made.

    `residues` is the component's 1-based residue set and `label` goes into the filename
    (e.g. 'c1_842'), giving {gene}_{uniprot}_{label}_component.pse.
    """
    try:
        if gene_name is None:
            gene_name = _load_gene_name(uniprot_id, reference_directory)
        residues = sorted(int(r) for r in residues)
        if not residues:
            print(f"  [WARNING] component for {uniprot_id} has no residues; skipping PSE")
            return

        pymol_dir = os.path.join(results_directory, 'pymol_visualizations')
        os.makedirs(pymol_dir, exist_ok=True)

        with h5py.File(pval_file, 'r') as fid:
            node, mu_mode = resolve_pval_node(fid, test)
            df_mut = read_original_mutation_data(node, uniprot_id, mu_mode=mu_mode)

        info = os.path.join(reference_directory, INFO_TSV)
        if not os.path.isfile(info):
            print(f"  [WARNING] Info TSV not found: {info}")
            return
        info_df = pd.read_csv(info, sep='\t')
        tmp_info = info_df[info_df['uniprot_id'] == uniprot_id]
        pdbs = set(tmp_info['pdb_filename'].tolist())
        if len(pdbs) != 1:
            print(f"  Skipping component PSE for {gene_name} ({uniprot_id}): "
                  f"structure spans {len(pdbs)} PDB files")
            return
        pdb_file = os.path.join(reference_directory, 'pdb_files', pdbs.pop())
        if not os.path.isfile(pdb_file):
            print(f"  [ERROR] PDB file not found: {pdb_file}")
            return

        cmd.reinitialize()
        cmd.load(pdb_file, "structure")
        cmd.bg_color("white")
        cmd.hide("everything", "all")
        cmd.show("cartoon", "all")
        cmd.color("grey80", "all")

        resi = '+'.join(str(r) for r in residues)
        cmd.select("component", f"resi {resi}")
        cmd.color("palecyan", "component")
        # The component is the region; the mutations are why it is one, so they stay on
        # top of it rather than replacing it.
        cmd.set("cartoon_transparency", 0.55, "not component")

        in_comp = df_mut[df_mut['aa_pos'].isin(residues)]

        def _spheres(name, positions, color):
            positions = [int(p) for p in positions]
            if not positions:
                return
            cmd.select(name, f"resi {'+'.join(str(p) for p in positions)} and name CA")
            cmd.show("spheres", name)
            cmd.color(color, name)

        if mu_mode:
            _spheres("denovo", in_comp.loc[in_comp.ac_case == 1, 'aa_pos'], "salmon")
            _spheres("denovo_recurrent", in_comp.loc[in_comp.ac_case >= 2, 'aa_pos'], "firebrick")
        else:
            case_only = in_comp[(in_comp.ac_case >= 1) & (in_comp.ac_control == 0)]
            both = in_comp[(in_comp.ac_case >= 1) & (in_comp.ac_control >= 1)]
            ctrl_only = in_comp[(in_comp.ac_control >= 1) & (in_comp.ac_case == 0)]
            _spheres("control_only", ctrl_only['aa_pos'], "blue")
            _spheres("case_only", case_only['aa_pos'], "red")
            _spheres("case_and_control", both['aa_pos'], "purple")

        cmd.set("specular", 0.0)
        cmd.set("ambient", 0.4)
        cmd.set("ray_trace_mode", 1)
        cmd.set("ray_trace_gain", 0.15)
        cmd.set("cartoon_loop_radius", 0.4)
        cmd.deselect()

        out = os.path.join(pymol_dir, f"{gene_name}_{uniprot_id}_{label}_component.pse")
        cmd.save(out)
        print(f"  Saved component PSE: {out}")

    except Exception as e:
        print(f"[ERROR] in pymol_component(): {e}")


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


def run_all(uniprot_id, results_directory, reference_directory, pval_file, test='binomial'):
    """
    Run the full PyMOL visualization pipeline for a single protein.

    `test` selects which mutation-rate null the pictures are drawn against, and is
    ignored for a standard 3DNT p-value file.
    """
    gene_name = _load_gene_name(uniprot_id, reference_directory)
    pymol_rvas(uniprot_id, reference_directory, results_directory, pval_file,
               gene_name=gene_name, test=test)
    pymol_scan_test(uniprot_id, reference_directory, results_directory, pval_file,
                    gene_name=gene_name, test=test)
    pymol_neighborhood(uniprot_id, results_directory, reference_directory, pval_file,
                       gene_name=gene_name, test=test)
