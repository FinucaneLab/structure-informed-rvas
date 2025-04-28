import glob
from pymol import cmd
import pandas as pd
import os
import ast
import gzip
from Bio.PDB import PDBParser
from Bio.PDB import StructureBuilder, PDBIO, Model, Chain
from moviepy import VideoFileClip, clips_array
import re

# from utils import read_p_values

def write_full_pdb(full_pdb, output_path, pdb_overlap = 1200):
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

def pymol_rvas(info_tsv, df_rvas, reference_directory, results_directory):
    # make a pymol session with case and control mutations
    # output a gif and a .pse file
    try:
        df_rvas_p =  os.path.join(results_directory, df_rvas)
        if not os.path.isfile(df_rvas_p):
            print(f"[WARNING] RVAS file not found: {df_rvas_p}")
            return

        df_rvas = pd.read_csv(df_rvas_p, sep='\t')
        if not 'aa_pos_file' in df_rvas:
            df_rvas['aa_pos_file'] = df_rvas['aa_pos']
        pdbs = set(df_rvas['pdb_filename'].tolist())
        uniprot_ids = set(df_rvas['uniprot_id'].tolist())
        cmd.set('ribbon_as_cylinders')
        cmd.set("ribbon_radius", 0.5) 

        for item in pdbs:
            p = os.path.join(reference_directory, f'pdb_files/{item}')
            if not os.path.isfile(p):
                print(f"[ERROR] PDB file not found: {p}")
                continue
            print('Reading pdb:', p)

            cmd.load(p, "structure")
            cmd.color("grey")
            uniprot_id = item.split('-')[1]
            gray_pse = os.path.join(results_directory, f"{uniprot_id}_gray.pse")
            print(gray_pse)
            cmd.save(gray_pse)
            print("Saved gray PSE file.")

            # cmd.mset("1 x 60")
            # cmd.scene("001","store")
            # cmd.mview("store", 1, "001")
            # # cmd.movie.produce("1-30.png")
            cmd.movie.add_nutate(8,60,start=1)
            gray_mv_p = os.path.join(results_directory, f"{uniprot_id}_gray.mov")
            cmd.movie.produce(gray_mv_p)
            # print('scene 001 stored')

            tmp_df = df_rvas[df_rvas['pdb_filename'] == item]
            uniprot_id = tmp_df['uniprot_id'].values[0]

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

            tmp_df_control.to_csv('tmp_df_control.csv', sep='\t', index=False)
            tmp_df_case.to_csv('tmp_df_case.csv', sep='\t', index=False)
            tmp_df_both.to_csv('tmp_df_both.csv', sep='\t', index=False)    

            for _, row in tmp_df_control.iterrows():
                # aa_ref = row['aa_ref']
                # aa_alt = row['aa_alt']
                aa_pos_file = row['aa_pos_file']
                cmd.select(f"residue_{aa_pos_file}", f"resi {aa_pos_file}")
                cmd.color("blue", f"residue_{aa_pos_file}")
                # cmd.label(f"residue_{aa_pos_file} and name CA", f'"{aa_ref}->{aa_alt}"')
                # cmd.label(f"residue_{aa_pos_file} and name CA", f'{aa_pos_file}')

            for _, row in tmp_df_case.iterrows():
                # aa_ref = row['aa_ref']
                # aa_alt = row['aa_alt']
                aa_pos_file = row['aa_pos_file']
                cmd.select(f"residue_{aa_pos_file}", f"resi {aa_pos_file}")
                cmd.color("red", f"residue_{aa_pos_file}")
                # cmd.label(f"residue_{aa_pos_file} and name CA", f'"{aa_ref}->{aa_alt}"')
                # cmd.label(f"residue_{aa_pos_file} and name CA", f'{aa_pos_file}')
        
            for _, row in tmp_df_both.iterrows():
                # aa_ref = row['aa_ref']
                # aa_alt = row['aa_alt']
                aa_pos_file = row['aa_pos_file']
                cmd.select(f"residue_{aa_pos_file}", f"resi {aa_pos_file}")
                cmd.color("purple", f"residue_{aa_pos_file}")
                # cmd.label(f"residue_{aa_pos_file} and name CA", f'"{aa_ref}->{aa_alt}"')
                # cmd.label(f"residue_{aa_pos_file} and name CA", f'{aa_pos_file}')
            
            # cmd.mset("61 x 120")
            # cmd.scene("002","store")
            # cmd.mview("store", 61, "002")
            cmd.movie.add_nutate(12,60,start=1)
            # cmd.movie.produce("31-60.png")
            rib_mut_mv_p = os.path.join(results_directory, f"{uniprot_id}_rib_mut.mov")
            cmd.movie.produce(rib_mut_mv_p)

            objects = cmd.get_names("objects")
            for obj in objects:
                cmd.hide('cartoon', obj)
                cmd.show('ribbon', obj)

            pse_pdb_p = os.path.join(results_directory, f"{uniprot_id}.pse")
            cmd.save(pse_pdb_p)
            print('save pse pdb')
        
        if info_tsv is not None:
            for uniprot_id in uniprot_ids:
                get_one_pdb(info_tsv, uniprot_id, reference_directory)
                pdb_p = os.path.join(reference_directory, f'pdb_files/{uniprot_id}.pdb')
                cmd.load(pdb_p, "structure")
                tmp_df = df_rvas[df_rvas['uniprot_id'] == uniprot_id]
                uniprot_id = tmp_df['uniprot_id'].values[0]

                control_mask = (tmp_df['ac_control'] > 1) & (tmp_df['ac_case'] == 0)
                case_mask = (tmp_df['ac_case'] > 1) & (tmp_df['ac_control'] == 0)
                both_mask = (tmp_df['ac_case'] > 1) & (tmp_df['ac_control'] > 1)
                tmp_df_control = tmp_df[control_mask]
                tmp_df_case = tmp_df[case_mask]
                tmp_df_both = tmp_df[both_mask]
                
                for _, row in tmp_df_control.iterrows():
                    # aa_ref = row['aa_ref']
                    # aa_alt = row['aa_alt']
                    aa_pos = row['aa_pos']
                    cmd.select(f"residue_{aa_pos}", f"resi {aa_pos}")
                    cmd.color("blue", f"residue_{aa_pos}")
                    # cmd.label(f"residue_{aa_pos} and name CA", f'"{aa_ref}->{aa_alt}"')
                    # cmd.label(f"residue_{aa_pos} and name CA", f'{aa_pos}')

                
                for _, row in tmp_df_case.iterrows():
                    # aa_ref = row['aa_ref']
                    # aa_alt = row['aa_alt']
                    aa_pos = row['aa_pos']
                    cmd.select(f"residue_{aa_pos}", f"resi {aa_pos}")
                    cmd.color("red", f"residue_{aa_pos}")
                    # cmd.label(f"residue_{aa_pos} and name CA", f'"{aa_ref}->{aa_alt}"')
                    # cmd.label(f"residue_{aa_pos} and name CA", f'{aa_pos}')

            
                for _, row in tmp_df_both.iterrows():
                    # aa_ref = row['aa_ref']
                    # aa_alt = row['aa_alt']
                    aa_pos = row['aa_pos']
                    cmd.select(f"residue_{aa_pos}", f"resi {aa_pos}")
                    cmd.color("purple", f"residue_{aa_pos}")
                    # cmd.label(f"residue_{aa_pos} and name CA", f'"{aa_ref}->{aa_alt}"')
                    # cmd.label(f"residue_{aa_pos} and name CA", f'{aa_pos}')
                
                # cmd.mset("61 x 120")
                # cmd.scene("002", "store")
                # cmd.mview("store", 61, "002")
                # cmd.movie.produce("31-60.png")
                # cmd.movie.produce("31-60.mov")
                cmd.movie.add_nutate(12,60,start=1)
                rib_mut_mv_p = os.path.join(results_directory, f"{uniprot_id}_rib_mut.mov")
                cmd.movie.produce(rib_mut_mv_p)

                objects = cmd.get_names("objects")  
                for obj in objects:
                    cmd.hide('cartoon', obj)
                    cmd.show('ribbon', obj)

                pse_p = os.path.join(results_directory, f'{uniprot_id}.pse')
                cmd.save(pse_p)

    except Exception as e:
        print(f"[ERROR] in pymol_rvas(): {e}")

def pymol_annotation(annot_file, reference_directory, results_directory):
    # visualize the annotation
    try:
        annot_df_p = os.path.join(reference_directory, annot_file)
        if not os.path.isfile(annot_df_p):
            print(f"[WARNING] Annotation file not found: {annot_df_p}")
            return
        
        annot_df = pd.read_csv(annot_df_p, sep='\t')
        uniprot_ids = set(annot_df['uniprot_id'].tolist())
        for uniprot_id in uniprot_ids:
            p = os.path.join(results_directory,  f'{uniprot_id}.pse')
            if not os.path.exists(p):
                print(f"[WARNING] PSE file from pymol_rvas() not found: {p}")
                continue
            cmd.load(p)
            tmp_annot = annot_df[annot_df['uniprot_id'] == uniprot_id]
            tmp_annot_pos = tmp_annot['aa_pos'].tolist()
            for item in tmp_annot_pos:
                cmd.select(f"annotation_residue_{item}", f"resi {item}")
                cmd.label(f"annotation_residue_{item} and name CA", f'"annotation"')
            cmd.save(p)

    except Exception as e:
        print(f"[ERROR] in pymol_annotation(): {e}")

    
def pymol_scan_test(info_tsv, df_rvas, uniprot_id, reference_directory, results_directory):
    # color by case/control ratio of the neighborhood
    try:
        df_results_p = os.path.join(results_directory, 'p_values.h5')
        if not os.path.isfile(df_results_p):
            print(f"[WARNING] Scan test result file not found: {df_results_p}")
            return
        if info_tsv is not None:
            pse_p = os.path.join(results_directory, uniprot_id + '.pse')
        else:
            df_rvas_p = os.path.join(results_directory, df_rvas)
            df_rvas = pd.read_csv(df_rvas_p, sep='\t')
            pdb_filename = df_rvas['pdb_filename'].values[0]
            pse_p = os.path.join(results_directory, f"{uniprot_id}_{pdb_filename.split('.')[0]}.pse")
        if not os.path.isfile(pse_p):
            print(f"[WARNING] PSE file from pymol_rvas() not found: {pse_p}")
            return
        
        with h5py.File(df_results_p, 'r') as fid:
            df_results = read_p_values(fid, uniprot_id)

        df_results['ratio_normalized'] = df_results['ratio'] / df_results['ratio'].max()
        df_results['ratio_normalized'].to_csv('test.csv', sep='\t', index=False)
        
        cmd.load(pse_p)
        objects = cmd.get_names('objects')[-1]

        for sel in cmd.get_names("selections"):
            cmd.delete(sel)
        for obj in cmd.get_names("objects"):
            cmd.color("gray", obj)  
        cmd.label("all", "")

        for _, row in df_results.iterrows():
            resi = int(row['aa_pos'])
            ratio = float(row['ratio_normalized'])
            selection = f"{objects} and resi {resi}"
            cmd.alter(selection, f"b={ratio}")
            cmd.rebuild()
        
        cmd.spectrum("b", "yellow_orange_red", objects, byres=1)

        # cmd.mset("1 x 60")
        # cmd.scene("003", "store")
        # cmd.movie.produce("61-90.png")

        objects = cmd.get_names("objects")  
        for obj in objects:
            cmd.hide('ribbon', obj)
            cmd.show('cartoon', obj)

        cmd.movie.add_nutate(12,60,start=1)
        ratio_mv_p = os.path.join(results_directory, f"{uniprot_id}_ratio.mov")
        cmd.movie.produce(ratio_mv_p)


        objects = cmd.get_names("objects")  
        for obj in objects:
            cmd.hide('cartoon', obj)
            cmd.show('ribbon', obj)
        
        result_pse_p = os.path.join(results_directory, f'{uniprot_id}_result.pse')
        cmd.save(result_pse_p)
    except Exception as e:
        print(f"[ERROR] in pymol_scan_test(): {e}")

def pymol_neighborhood(df_results, results_directory):
    # for each significant neighborhood, zoom in and show the case and control mutations
    # just in that neighborhood.
    try:
        uniprot_id = df_results.split('_')[0]
        df_results_p = os.path.join(reference_directory, df_results)
        if not os.path.isfile(df_results_p):
            print(f"[WARNING] Scan test result file not found: {df_results_p}")
            return
        df_results = pd.read_csv(df_results_p, sep='\t')
        pse_p = os.path.join(reference_directory, uniprot_id + '_result.pse')
        if not os.path.isfile(pse_p):
            print(f"[WARNING] PSE file from pymol_scan_test() not found: {pse_p}")
            return
        
        cmd.load(pse_p)
        for _, row in df_results.iterrows():
            resi = int(row['aa_pos'])
            p_value = float(row['p_value'])
            if p_value < 0.05:
                nbhd_case = row['nbhd_case']
                nbhd_ctrl = row['nbhd_control']
                selection = f"resi {resi}"
                cmd.select(f"residue_{resi}", selection)
                cmd.label(f"residue_{resi} and name CA", f'"case: {nbhd_case}; control: {nbhd_ctrl}"')
                # cmd.zoom(selection)
        
        cmd.save(pse_p)
    except Exception as e:  
        print(f"[ERROR] in pymol_neighborhood(): {e}")

def make_movie(results_directory, uniprot_id):

    gray_mv_p = os.path.join(results_directory, f'{uniprot_id}_gray.mov')
    rib_mut_mv_p = os.path.join(results_directory, f'{uniprot_id}_rib_mut.mov')
    ratio_mv_p = os.path.join(results_directory, f'{uniprot_id}_ratio.mov')

    if not os.path.exists(gray_mv_p):
        print(f"[WARNING] pymol_rvas() base movie file not found: {gray_mv_p}")
        return
    if not os.path.exists(rib_mut_mv_p): 
        print(f"[WARNING] pymol_rvas() movie file not found: {rib_mut_mv_p}")
        return
    
    if not os.path.exists(ratio_mv_p): 
        print(f"[WARNING] pymol_neighborhood() movie file not found: {rib_mut_mv_p}")
        return
    
    clip1 = VideoFileClip(gray_mv_p)
    clip2 = VideoFileClip(rib_mut_mv_p)
    clip3 = VideoFileClip(ratio_mv_p)

    min_duration = min(clip1.duration, clip2.duration, clip3.duration)
    clip1 = clip1.subclipped(0, min_duration)
    clip2 = clip2.subclipped(0, min_duration)
    clip3 = clip3.subclipped(0, min_duration)

    target_height = min(clip1.h, clip2.h, clip3.h)
    clip1 = clip1.resized(height=target_height)
    clip2 = clip2.resized(height=target_height)
    clip3 = clip3.resized(height=target_height)

    final_clip = clips_array([[clip1, clip2, clip3]])

    output_file = os.path.join(results_directory, f"{uniprot_id}.mov")
    final_clip.write_videofile(output_file, codec="libx264", fps=24)


def run_all(results_directory, reference_directory, info_tsv=None):
    filelist = glob.glob(f'{results_directory}/*.df_rvas.tsv')
    uniprot_list = [x.split('/')[-1].split('.')[0] for x in filelist]
    for uniprot_id in uniprot_list:
        df_rvas = f'{uniprot_id}.df_rvas.tsv'
        pymol_rvas(info_tsv, df_rvas, reference_directory, results_directory)
        pymol_scan_test(info_tsv, df_rvas, df_results, reference_directory, results_directory)
        make_movie(results_directory, uniprot_id)
        cmd.reinitialize()
# reference_directory = '../sir-reference-data/'
reference_directory = './'
results_directory = 'results/'
run_all(results_directory, reference_directory)


