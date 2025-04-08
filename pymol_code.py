from pymol import cmd
import pandas as pd
import os
import ast
import gzip
from Bio.PDB import PDBParser
from Bio.PDB import StructureBuilder, PDBIO, Model, Chain


def write_full_pdb(full_pdb, output_path):
    try:
        builder = StructureBuilder.StructureBuilder()
        builder.init_structure("new_structure")
        builder.init_model(0)
        builder.init_chain('A')  

        model = builder.get_structure()[0]
        chain = model['A']

        for residue in full_pdb:
            chain.add(residue.copy())  

        io = PDBIO()
        io.set_structure(builder.get_structure())
        io.save(output_path)
    except Exception as e:
        print(f"[ERROR] Failed to write PDB file: {e}")


def get_one_pdb(info_tsv, uniprot_id, reference_directory):
    info_tsv_p = os.path.join(reference_directory, info_tsv)
    try:
        info_df = pd.read_csv(info_tsv_p, sep='\t')
        if not os.path.exists(info_tsv_p):
            print(f"[WARNING] Info TSV not found: {info_tsv_p}")
            return
    
        info_df['pos_covered'] = info_df['pos_covered'].apply(ast.literal_eval)
        pdbs = [item for item in os.listdir(reference_directory) if item.endswith('.gz') and uniprot_id in item]
        full_pdb = []
        pdbs.sort(key=lambda item: item.split('-')[2][-1])
        for item in pdbs:
            p = os.path.join(reference_directory, item)
            print('Reading pdb:', p)

            try:
                with gzip.open(p, "rt") as handle:
                    structure = PDBParser(QUIET=True).get_structure("protein", handle)
                residues = [res for model in structure for chain in model for res in chain]
                
                if 'F1' in item:
                    full_pdb.extend(residues)
                else:
                    new_residue = residues[1200:]
                    for i, residue in enumerate(residues):
                        residue.id = (' ', 1 + i + full_pdb[-1].id[1], ' ')
                    full_pdb.extend(new_residue)

            except Exception as e:
                print(f"[ERROR] Failed to parse {p}: {e}")

        output_path = os.path.join(reference_directory, uniprot_id + '.pdb')
        write_full_pdb(full_pdb, output_path)

    except Exception as e:
        print(f"[ERROR] in get_one_pdb(): {e}")

def pymol_rvas(info_tsv, df_rvas, reference_directory):
    # make a pymol session with case and control mutations
    # output a gif and a .pse file
    try:
        df_rvas_p =  os.path.join(reference_directory, df_rvas)
        if not os.path.exists(df_rvas_p):
            print(f"[WARNING] RVAS file not found: {df_rvas_p}")
            return

        df_rvas = pd.read_csv(df_rvas_p, sep='\t')
        pdbs = set(df_rvas['pdb_filename'].tolist())
        uniprot_ids = set(df_rvas['uniprot_id'].tolist())
        cmd.set('ribbon_as_cylinders')
        cmd.set("ribbon_radius", 0.5) 

        for item in pdbs:
            p = os.path.join(reference_directory, item)
            if not os.path.exists(p):
                print(f"[ERROR] PDB file not found: {p}")
                continue
            print('Reading pdb:', p)

            cmd.load(p, "structure")
            tmp_df = df_rvas[df_rvas['pdb_filename'] == item]
            uniprot_id = tmp_df['uniprot_id'].values[0]

            control_mask = (tmp_df['ac_control'] > 1) & (tmp_df['ac_case'] == 0)
            case_mask = (tmp_df['ac_case'] > 1) & (tmp_df['ac_control'] == 0)
            both_mask = (tmp_df['ac_case'] > 1) & (tmp_df['ac_control'] > 1)
            tmp_df_control = tmp_df[control_mask]
            tmp_df_case = tmp_df[case_mask]
            tmp_df_both = tmp_df[both_mask]
        
            for _, row in tmp_df_control.iterrows():
                aa_ref = row['aa_ref']
                aa_alt = row['aa_alt']
                aa_pos_file = row['aa_pos_file']
                cmd.select(f"residue_{aa_pos_file}", f"resi {aa_pos_file}")
                cmd.color("blue", f"residue_{aa_pos_file}")
                cmd.label(f"residue_{aa_pos_file} and name CA", f'"{aa_ref}->{aa_alt}"')

            for _, row in tmp_df_case.iterrows():
                aa_ref = row['aa_ref']
                aa_alt = row['aa_alt']
                aa_pos_file = row['aa_pos_file']
                cmd.select(f"residue_{aa_pos_file}", f"resi {aa_pos_file}")
                cmd.color("red", f"residue_{aa_pos_file}")
                cmd.label(f"residue_{aa_pos_file} and name CA", f'"{aa_ref}->{aa_alt}"')
        
            for _, row in tmp_df_both.iterrows():
                aa_ref = row['aa_ref']
                aa_alt = row['aa_alt']
                aa_pos_file = row['aa_pos_file']
                cmd.select(f"residue_{aa_pos_file}", f"resi {aa_pos_file}")
                cmd.color("purple", f"residue_{aa_pos_file}")
                cmd.label(f"residue_{aa_pos_file} and name CA", f'"{aa_ref}->{aa_alt}"')
            
            objects = cmd.get_names("objects")
            for obj in objects:
                cmd.hide('cartoon', obj)
                cmd.show('ribbon', obj)

            pse_pdb_p = os.path.join(reference_directory, f"{uniprot_id}_{item.split('.')[0]}.pse")
            cmd.save(pse_pdb_p)

        for uniprot_id in uniprot_ids:
            print(uniprot_id)
            get_one_pdb(info_tsv, uniprot_id, reference_directory)
            pdb_p = os.path.join(reference_directory, uniprot_id + '.pdb')
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
                aa_ref = row['aa_ref']
                aa_alt = row['aa_alt']
                aa_pos = row['aa_pos']
                cmd.select(f"residue_{aa_pos}", f"resi {aa_pos}")
                cmd.color("blue", f"residue_{aa_pos}")
                cmd.label(f"residue_{aa_pos} and name CA", f'"{aa_ref}->{aa_alt}"')

            
            for _, row in tmp_df_case.iterrows():
                aa_ref = row['aa_ref']
                aa_alt = row['aa_alt']
                aa_pos = row['aa_pos']
                cmd.select(f"residue_{aa_pos}", f"resi {aa_pos}")
                cmd.color("red", f"residue_{aa_pos}")
                cmd.label(f"residue_{aa_pos} and name CA", f'"{aa_ref}->{aa_alt}"')

        
            for _, row in tmp_df_both.iterrows():
                aa_ref = row['aa_ref']
                aa_alt = row['aa_alt']
                aa_pos = row['aa_pos']
                cmd.select(f"residue_{aa_pos}", f"resi {aa_pos}")
                cmd.color("purple", f"residue_{aa_pos}")
                cmd.label(f"residue_{aa_pos} and name CA", f'"{aa_ref}->{aa_alt}"')

            objects = cmd.get_names("objects")  
            for obj in objects:
                cmd.hide('cartoon', obj)
                cmd.show('ribbon', obj)

            pse_p = os.path.join(reference_directory, f'{uniprot_id}.pse')
            cmd.save(pse_p)
    except Exception as e:
        print(f"[ERROR] in pymol_rvas(): {e}")

def pymol_annotation(annot_file, reference_directory):
    # visualize the annotation
    try:
        annot_df_p = os.path.join(reference_directory, annot_file)
        if not os.path.exists(annot_df_p):
            print(f"[WARNING] Annotation file not found: {annot_df_p}")
            return
        
        annot_df = pd.read_csv(annot_df_p, sep='\t')
        uniprot_ids = set(annot_df['uniprot_id'].tolist())
        for uniprot_id in uniprot_ids:
            p = os.path.join(reference_directory,  f'{uniprot_id}.pse')
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

    
def pymol_scan_test(df_results, reference_directory):
    # color by case/control ratio of the neighborhood
    try:
        uniprot_id = df_results.split('_')[0]
        df_results_p = os.path.join(reference_directory, df_results)
        if not os.path.exists(df_results_p):
            print(f"[WARNING] Scan test result file not found: {df_results_p}")
            return
        pse_p = os.path.join(reference_directory, uniprot_id + '.pse')
        if not os.path.exists(pse_p):
            print(f"[WARNING] PSE file from pymol_rvas() not found: {pse_p}")
            return
        
        df_results = pd.read_csv(df_results_p, sep='\t')
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
        
        cmd.spectrum("b", "blue_white_red", objects, byres=1)

        result_pse_p = os.path.join(reference_directory, f'{uniprot_id}_result.pse')
        cmd.save(result_pse_p)
    except Exception as e:
        print(f"[ERROR] in pymol_scan_test(): {e}")

def pymol_neighborhood(df_results, reference_directory):
    # for each significant neighborhood, zoom in and show the case and control mutations
    # just in that neighborhood.
    try:
        uniprot_id = df_results.split('_')[0]
        df_results_p = os.path.join(reference_directory, df_results)
        if not os.path.exists(df_results_p):
            print(f"[WARNING] Scan test result file not found: {df_results_p}")
            return
        df_results = pd.read_csv(df_results_p, sep='\t')
        pse_p = os.path.join(reference_directory, uniprot_id + '_result.pse')
        if not os.path.exists(pse_p):
            print(f"[WARNING] PSE file from pymol_scan_test() not found: {pse_p}")
            return
        
        cmd.load(pse_p)
        for _, row in df_results.iterrows():
            resi = int(row['aa_pos'])
            p_value = float(row['p_value'])
            if p_value < 0.05:
                nbhd_case = row['nbhd_case']
                nbhd_ctrl = row['nbhd_ctrl']
                selection = f"resi {resi}"
                cmd.select(f"residue_{resi}", selection)
                cmd.label(f"residue_{resi} and name CA", f'"case: {nbhd_case}; control: {nbhd_ctrl}"')
                cmd.zoom(selection)
        result_pse_p = os.path.join(reference_directory, f'{uniprot_id}_result.pse')
        cmd.save(result_pse_p)
    except Exception as e:  
        print(f"[ERROR] in pymol_neighborhood(): {e}")
# pymol_rvas('info.tsv','sample_df_rvas.tsv', '/Users/liaoruqi/Desktop/structure-informed-rvas/examples')
# pymol_annotation('ClinVar_PLP_uniprot_canonical.tsv', '/Users/liaoruqi/Desktop/structure-informed-rvas/examples')
# pymol_scan_test('O15047_results.tsv', '/Users/liaoruqi/Desktop/structure-informed-rvas/examples')
# pymol_neighborhood('O15047_results.tsv', '/Users/liaoruqi/Desktop/structure-informed-rvas/examples')