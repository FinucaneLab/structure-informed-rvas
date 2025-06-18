import glob
from pymol import cmd
import pandas as pd
import os
import ast
import gzip
from Bio.PDB import PDBParser
from Bio.PDB import StructureBuilder, PDBIO, Model, Chain

import re
from utils import read_p_values
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

def pymol_rvas(df_rvas, reference_directory, results_directory):
    # make a pymol session with case and control mutations
    # output a gif and a .pse file
    '''
    Create PyMOL visualizations for RVAS results. For each PDB file:
    - Produces a basic image and PSE file.
    - Highlights mutations for control-only (blue), case-only (red), both (purple).
    - Outputs a second image and PSE file with mutations shown and colored.
    '''

    try:
        df_rvas_p =  os.path.join(reference_directory, df_rvas)
        if not os.path.isfile(df_rvas_p):
            print(f"[WARNING] RVAS file not found: {df_rvas_p}")
            return

        df_rvas = pd.read_csv(df_rvas_p, sep='\t')
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
            print('Reading pdb:', p)

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
            cmd.png(f"{results_directory}/{pdb_filename}.png")

            cmd.reset()
            cmd.color("grey")
            gray_pse = os.path.join(results_directory, f"{pdb_filename}_gray.pse")
            print(gray_pse)
            cmd.save(gray_pse)
            print("Saved gray PSE file.")

            tmp_df = df_rvas[df_rvas['pdb_filename'] == item]

            control_mask = (tmp_df['ac_control'] >= 1) & (tmp_df['ac_case'] == 0)
            case_mask = (tmp_df['ac_case'] >= 1) & (tmp_df['ac_control'] == 0)
            both_mask = (tmp_df['ac_case'] >= 1) & (tmp_df['ac_control'] >= 1)

            tmp_df_control = tmp_df[control_mask]
            tmp_df_case = tmp_df[case_mask]
            tmp_df_both = tmp_df[both_mask]

            control_pos = tmp_df_control['aa_pos'].tolist()
            case_pos = tmp_df_case['aa_pos'].tolist()
            both_pos = tmp_df_both['aa_pos'].tolist()
            control_case_pos = set(control_pos).intersection(set(case_pos))
            control_both_pos = set(control_pos).intersection(set(both_pos))
            case_both_pos = set(case_pos).intersection(set(both_pos))
            new_control_pos = set(control_pos) - control_case_pos - control_both_pos
            new_case_pos = set(case_pos) - control_case_pos - case_both_pos
            new_both_pos = set(both_pos).union(control_case_pos)
            tmp_df_control = tmp_df[tmp_df['aa_pos'].isin(new_control_pos)]
            tmp_df_case = tmp_df[tmp_df['aa_pos'].isin(new_case_pos)]
            tmp_df_both = tmp_df[tmp_df['aa_pos'].isin(new_both_pos)]
            
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
            cmd.png(f"{results_directory}/{pdb_filename}_mut.png")

            pse_pdb_p = os.path.join(results_directory, f"{pdb_filename}_mut.pse")
            cmd.save(pse_pdb_p)
            print('save pse pdb')

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
            tmp_annot['visual_filename'] = tmp_annot['pdb_filename'].apply(lambda x: os.path.join(results_directory, x.split('.')[0]+ '.pse'))
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
        print(tmp_df)
        tmp_df['visual_filename'] = tmp_df['pdb_filename'].apply(lambda x: os.path.join(results_directory, x.split('.')[0]+ '.pse'))
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

            cmd.save(f"{v.split('.')[0]}_ratio.pse")

    except Exception as e:
        print(f"[ERROR] in pymol_scan_test(): {e}")

def pymol_neighborhood(uniprot_id, results_directory, info_tsv, reference_directory):
    # for each significant neighborhood, zoom in and show the case and control mutations
    # just in that neighborhood.
    '''
    Create PyMOL visualizations for significant neighborhood in scan test results. For each PDB file:
    - Load the PSE file with ratio colored and show spheres for residues with p-value < 0.05.
    - Outputs a image with spheres for significant residues.
    '''
    try:
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

        tmp_df['visual_filename'] = tmp_df['pdb_filename'].apply(lambda x: os.path.join(results_directory, x.split('.')[0]+ '.pse'))
        tmp_visuals = set(tmp_df['visual_filename'].tolist())
        for v in tmp_visuals:
            cmd.load(v.split('.')[0] + '_ratio.pse')
            for _, row in tmp_df.iterrows():
                resi = int(row['aa_pos'])
                p_value = float(row['p_value'])
                if p_value < 0.05:
                    selection = f"resi {resi} and name CA"
                    cmd.select(f"residue_{resi}", selection)
                    cmd.show("spheres", f"residue_{resi}")

            cmd.save(v.split('.')[0] + '_ratio.pse')

            cmd.ray(2400, 1800)
            cmd.set("ray_opaque_background", 1)
            cmd.png(f"{v.split('.')[0]}_ratio.png")

    except Exception as e:  
        print(f"[ERROR] in pymol_neighborhood(): {e}")

def make_movie_from_pse(result_directory, pse_name):
    '''
    Create a movie from a PyMOL session file (.pse).
    '''

    pse = os.path.join(result_directory, f"{pse_name}.pse")
    try:
        cmd.load(pse)
        cmd.ray(2400, 1800)
        cmd.set("ray_opaque_background", 1)
        cmd.png(f"{result_directory}/{pse_name}.png")
        cmd.movie.add_roll(10, axis='y', start=1)
        mv = os.path.join(result_directory, f"{pse_name}.mov")
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
    df_rvas = f'{uniprot_id}.df_rvas.tsv'
    pymol_rvas(df_rvas, reference_directory, results_directory)
    # pymol_annotation('ClinVar_PLP_uniprot_canonical.tsv', reference_directory , results_directory, 'pdb_pae_file_pos_guide.tsv', uniprot_id)
    pymol_scan_test('pdb_pae_file_pos_guide.tsv', uniprot_id, reference_directory, results_directory)
    pymol_neighborhood(uniprot_id, results_directory, 'pdb_pae_file_pos_guide.tsv', reference_directory)


