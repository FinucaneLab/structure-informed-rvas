from pymol import cmd
from pymol.cgo import COLOR, ALPHA, SPHERE
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
        # Every name, slash-joined, matching _load_gene_map in visualize_and_interpret
        # and the gene_name column compute_fdr writes, so no two of the three can
        # disagree for a duplicated accession. A row whose names are all blank joins to
        # '', which has to fall through to the accession rather than be returned: _safe
        # turns it into 'unnamed', and _load_gene_map drops the row entirely, so the
        # two subcommands would stem the same protein's PSEs differently and stop
        # sharing its grey chain.
        if not match.empty:
            name = '/'.join(sorted(n for n in match['gene_name'].dropna().unique()
                                   if str(n).strip()))
            if name:
                return name
    except Exception:
        pass
    return uniprot_id  # fall back to UniProt ID if not found


def _safe(name):
    """Filename-safe gene name; see safe_name in visualize_and_interpret."""
    cleaned = re.sub(r'[^A-Za-z0-9._-]+', '_', str(name)).strip('_')
    return cleaned or 'unnamed'


def _gray_pse_path(visual_pse):
    """
    The grey-chain PSE sitting beside a fragment's visual PSE.

    splitext rather than split('.'): this is a full path, so a results directory with
    a dot in it (./results, /data/v1.2/out) would otherwise be truncated at the first
    dot and the load would miss. _safe() permits dots in the stem as well.
    """
    return f'{os.path.splitext(visual_pse)[0]}_gray.pse'


def _pse_base(pdb_filename, gene_name, uniprot_id):
    """Return the base name for PSE files: {gene_name}_{uniprot_id}[_FN]."""
    match = re.search(r'-F(\d+)-', pdb_filename)
    frag = f'_F{match.group(1)}' if match else ''
    return f'{_safe(gene_name)}_{uniprot_id}{frag}'


def get_pdb_filename(annot_df, info_df):
    """
    Label each row of annot_df with the PDB file covering its residue.

    Rows whose position falls in no covered range get None rather than inheriting the
    previous row's filename, which is what the earlier version did: `pdb_filename` was
    assigned inside the inner loop and read outside it, so a miss silently reused
    whatever the last hit had been (and raised NameError if the very first row missed).

    The ranges are parsed once up front; they used to be re-parsed with literal_eval for
    every residue, which is O(residues x fragments) calls on a long protein.
    """
    ranges = [(int(ast.literal_eval(r['pos_covered'])[0]),
               int(ast.literal_eval(r['pos_covered'])[1]),
               r['pdb_filename'])
              for _, r in info_df.iterrows()]

    def _lookup(aa_pos):
        for start, end, filename in ranges:
            if start <= int(aa_pos) <= end:
                return filename
        return None

    annot_df = annot_df.copy()
    annot_df['pdb_filename'] = [_lookup(p) for p in annot_df['aa_pos']]
    n_missing = int(annot_df['pdb_filename'].isna().sum())
    if n_missing:
        print(f"  [WARNING] {n_missing} residues fall outside every PDB fragment range")
    return annot_df


# Transparent orange, matching the ball people draw by hand around a neighborhood.
COMPONENT_SPHERE_COLOR = (1.0, 0.7, 0.0)
COMPONENT_SPHERE_ALPHA = 0.20
# Thin the drawn centres to this fraction of a radius apart. Measured on the ASD
# components, 0.5 keeps 13-27 spheres out of 45-104 and covers 91-100% of the component;
# 1.0 drops to 4-8 spheres but only 74-97% coverage, and 0.25 is exact but puts every
# centre back and renders opaque.
COMPONENT_SPHERE_SEPARATION = 0.5


def _thin_centers(placed, separation_factor=COMPONENT_SPHERE_SEPARATION,
                  min_separation=None):
    """
    Greedily drop centres that sit close to one already kept.

    `placed` is [(xyz, radius), ...] in input order, and order decides who survives --
    the caller passes centres in position order, so the result is reproducible. Split out
    from _component_cgo so the geometry can be tested without PyMOL or a structure.
    """
    kept = []
    for xyz, radius in placed:
        sep = (radius * separation_factor if min_separation is None
               else float(min_separation))
        if all(np.linalg.norm(np.asarray(xyz) - np.asarray(k_xyz)) >= sep
               for k_xyz, _ in kept):
            kept.append((xyz, radius))
    return kept


def _component_cgo(centers, color=COMPONENT_SPHERE_COLOR, alpha=COMPONENT_SPHERE_ALPHA,
                   separation_factor=COMPONENT_SPHERE_SEPARATION, min_separation=None):
    """
    A CGO object drawing a component as the union of its neighborhood spheres.

    `centers` is [(aa_pos, radius), ...]. One sphere per member centre is what the
    component literally is, but transparency accumulates where spheres overlap, and a
    component with 100 members renders as a solid blob hiding the structure inside it.
    So the centres are thinned greedily in 3D: a centre is drawn only if it is at least
    `separation_factor` of its radius from every centre already drawn.

    The result is an APPROXIMATE envelope, not an exact one. A dropped centre lies within
    that separation of a kept centre, so its own sphere is not fully inside the kept one,
    and the drawn union falls a little short of the component at the edges -- 91-100% of
    residues on the ASD components at the default. The authoritative residue set is
    residues_pymol in the components file, and the component PSE colours exactly those
    residues; this object is for seeing roughly where the signal sits.

    Returns (cgo, n_drawn, n_centers).
    """
    placed = []
    for aa_pos, radius in centers:
        try:
            xyz = cmd.get_atom_coords(f'name CA and resi {int(aa_pos)}')
        except Exception:
            # A centre with no CA in the loaded structure -- skip it rather than abort
            # the whole object.
            continue
        placed.append((np.asarray(xyz, dtype=float), float(radius)))

    kept = _thin_centers(placed, separation_factor, min_separation)

    cgo = [COLOR, float(color[0]), float(color[1]), float(color[2]), ALPHA, float(alpha)]
    for xyz, radius in kept:
        cgo += [SPHERE, float(xyz[0]), float(xyz[1]), float(xyz[2]), radius]
    return cgo, len(kept), len(placed)


def _add_component_spheres(components, prefix='component'):
    """
    Load one toggleable CGO object per component into the current session.

    Each becomes its own object so it can be shown or hidden independently of the
    mutations and of the other components.

    The camera is restored afterwards: a CGO enlarges the scene's bounding box, which
    otherwise re-frames the session on the sphere and crops the rest of the protein out
    of the saved view.
    """
    if not components:
        return
    view = cmd.get_view()
    for comp in components or []:
        centers = comp.get('centers') or []
        if not centers:
            continue
        cgo, n_drawn, n_centers = _component_cgo(centers)
        if n_drawn == 0:
            continue
        name = f"{prefix}_{comp.get('label', 'c')}"
        cmd.load_cgo(cgo, name)
        note = ('' if n_drawn == n_centers
                else f' (thinned from {n_centers}; approximate envelope)')
        print(f"  Added component object '{name}': {n_drawn} spheres{note}")
    cmd.set_view(view)


def show_mutation_spheres(df_mut, mu_mode, scope='', pos_col='aa_pos'):
    """
    Show each mutation class as spheres, one named selection per class.

    The selections are what make a session useful afterwards: `denovo`,
    `case_only` and so on each hold every residue of that class, so they can be
    recoloured, hidden or zoomed as a unit rather than residue by residue.

    `scope` narrows the selection to one object (e.g. 'target_chain and'), which the
    complex-structure path needs and the monomer path does not. `pos_col` is the column
    holding the residue number to select on -- `aa_pos_file` where the PDB is numbered
    per fragment rather than per protein.

    Returns the names created, in draw order.
    """
    created = []

    def _select(name, rows, color):
        positions = [str(p) for p in rows[pos_col]]
        if not positions:
            return
        prefix = f'{scope} ' if scope else ''
        cmd.select(name, f"{prefix}resi {'+'.join(positions)} and name CA")
        cmd.show("spheres", name)
        cmd.color(color, name)
        created.append(name)

    if mu_mode:
        # No control cohort exists, so there is nothing to contrast against; recurrence
        # is the only extra dimension the data carries.
        _select("denovo", df_mut[df_mut.ac_case == 1], "salmon")
        _select("denovo_recurrent", df_mut[df_mut.ac_case >= 2], "firebrick")
    else:
        case_only = df_mut[(df_mut.ac_case >= 1) & (df_mut.ac_control == 0)]
        both = df_mut[(df_mut.ac_case >= 1) & (df_mut.ac_control >= 1)]
        ctrl_only = df_mut[(df_mut.ac_control >= 1) & (df_mut.ac_case == 0)]
        _select("control_only", ctrl_only, "blue")
        _select("case_only", case_only, "red")
        _select("case_and_control", both, "purple")
    return created


def apply_effect_coloring(positions, effect, mu_mode, selection, vmax=None):
    """
    Colour `selection` by per-residue effect size, via the B-factor column.

    Mutation-rate mode uses observed/expected on a diverging scale pinned at 1 (see
    _obs_exp_to_b); case/control uses the regularized ratio normalized to `vmax`, which
    has no meaningful anchor of its own.

    `vmax` must be the maximum over the WHOLE protein. The caller colours one PDB
    fragment at a time, and taking the maximum within each fragment would normalize
    every fragment to its own brightest residue -- so two fragments of one protein would
    use different scales, and a fragment whose strongest signal is weak would be
    coloured as though it were strong. Defaults to the maximum of what it is given,
    which is correct only when that is the whole protein.

    Residues not in `positions` are reset to the scale's neutral value first. They keep
    the structure's pLDDT B-factor otherwise (0-100), and cmd.spectrum colours the whole
    selection -- so against the mu-mode scale pinned to [0, 1] every unmeasured residue
    would clamp to maximum enrichment. pymol_scan_test drops residues falling outside
    the position guide's pos_covered, so this is reachable whenever the guide is shorter
    than the structure.
    """
    cmd.alter(selection, f"b={0.5 if mu_mode else 0.0}")

    if mu_mode:
        b_values = _obs_exp_to_b(np.asarray(effect, dtype=float))
        palette, spectrum_range = "ratio_blue white ratio_red", {'minimum': 0.0, 'maximum': 1.0}
    else:
        effect = np.asarray(effect, dtype=float)
        scale = float(vmax) if vmax is not None else float(effect.max() or 1.0)
        b_values = (effect / (scale or 1.0)) ** 2
        palette, spectrum_range = "ratio_blue ratio_red", {}

    for pos, b in zip(positions, b_values):
        cmd.alter(f"{selection} and resi {int(pos)}", f"b={float(b)}")
    cmd.rebuild()
    cmd.set_color("ratio_blue", [0.45, 0.78, 1.0])
    cmd.set_color("ratio_red", [1.0, 0.55, 0.55])
    cmd.spectrum("b", palette, selection, byres=1, **spectrum_range)


def pymol_rvas(uniprot_id, reference_directory, results_directory, pval_file, gene_name=None,
               test='binomial', components=None):
    """Create PyMOL visualizations for RVAS results.

    Produces two PSE files per protein:
    - {gene}_{uniprot}_gray.pse: plain grey cartoon structure
    - {gene}_{uniprot}_mut.pse: structure with mutations colored, plus one toggleable
      CGO object per component when `components` is given

    The standard 3DNT colors by case/control class (blue = control-only, red = case-only,
    purple = both). A mutation-rate run has no control cohort at all -- the comparison is
    against the modelled rate -- so it instead shows de novo positions in one colour,
    darkened where a residue carries more than one.

    `components` is [{'label': str, 'centers': [(aa_pos, radius), ...]}, ...]. Each is
    drawn as a transparent sphere union in its own object, so it can be shown or hidden
    without disturbing the mutations.
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
        # Residues covered by no fragment get None from get_pdb_filename. Drop them
        # rather than letting None reach a filename.
        n_unmapped = int(df_rvas['pdb_filename'].isna().sum())
        if n_unmapped:
            print(f"  {n_unmapped} residues map to no PDB fragment and are not shown")
            df_rvas = df_rvas[df_rvas['pdb_filename'].notna()]
        if df_rvas.empty:
            print(f"  Skipping {gene_name} ({uniprot_id}): no residue maps to a fragment")
            return
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

            if 'aa_pos_file' not in tmp_df:
                tmp_df = tmp_df.assign(aa_pos_file=tmp_df['aa_pos'])
            names = show_mutation_spheres(tmp_df, mu_mode, pos_col='aa_pos_file')
            print(f"  Mutation selections: {', '.join(names) if names else 'none'}")

            _add_component_spheres(components)

            cmd.deselect()
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
        n_unmapped = int(tmp_df['pdb_filename'].isna().sum())
        if n_unmapped:
            print(f"  {n_unmapped} residues map to no PDB fragment and are not coloured")
            tmp_df = tmp_df[tmp_df['pdb_filename'].notna()]
        if tmp_df.empty:
            print(f"  Skipping {gene_name} ({uniprot_id}): no residue maps to a fragment")
            return
        print(f"  Processing scan test results for {gene_name} ({uniprot_id}): {len(tmp_df)} positions")

        tmp_df['visual_filename'] = tmp_df['pdb_filename'].apply(
            lambda x: os.path.join(pymol_dir, _pse_base(x, gene_name, uniprot_id) + '.pse')
        )
        effect_col = 'obs_exp' if mu_mode else 'ratio'
        # Protein-wide, before the per-fragment loop below.
        vmax = float(tmp_df[effect_col].max())

        for v in set(tmp_df['visual_filename']):
            cmd.reinitialize()
            cmd.load(_gray_pse_path(v))
            objects = cmd.get_names('objects')[-1]
            sub = tmp_df[tmp_df['visual_filename'] == v]
            apply_effect_coloring(sub['aa_pos'], sub[effect_col], mu_mode, objects,
                                  vmax=vmax)
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


def pymol_component(uniprot_id, reference_directory, results_directory, pval_file,
                    residues, label, gene_name=None, test='binomial', centers=None):
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

        show_mutation_spheres(in_comp, mu_mode)

        if centers:
            _add_component_spheres([{'label': label, 'centers': centers}])

        cmd.set("specular", 0.0)
        cmd.set("ambient", 0.4)
        cmd.set("ray_trace_mode", 1)
        cmd.set("ray_trace_gain", 0.15)
        cmd.set("cartoon_loop_radius", 0.4)
        cmd.deselect()

        out = os.path.join(pymol_dir, f"{_safe(gene_name)}_{uniprot_id}_{label}_component.pse")
        cmd.save(out)
        print(f"  Saved component PSE: {out}")

    except Exception as e:
        print(f"[ERROR] in pymol_component(): {e}")


def run_all(uniprot_id, results_directory, reference_directory, pval_file, test='binomial',
            components=None, gene_name=None):
    """
    Run the full PyMOL visualization pipeline for a single protein.

    `test` selects which mutation-rate null the pictures are drawn against, and is
    ignored for a standard 3DNT p-value file.

    `gene_name` overrides the guide lookup. Pass the caller's name: 67 accessions map to
    several gene names (HBA1/HBA2, the histone clusters), and re-deriving one here would
    put the whole-chain PSE under a different stem from the component PSE and TSVs that
    the same run writes for the same protein.
    """
    if gene_name is None:
        gene_name = _load_gene_name(uniprot_id, reference_directory)
    pymol_rvas(uniprot_id, reference_directory, results_directory, pval_file,
               gene_name=gene_name, test=test, components=components)
    pymol_scan_test(uniprot_id, reference_directory, results_directory, pval_file,
                    gene_name=gene_name, test=test)
