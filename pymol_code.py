from pymol import cmd
import pandas as pd
import os
import ast
import gzip
import shutil
from Bio.PDB import PDBParser
from Bio.PDB import StructureBuilder, PDBIO, Model, Chain


def write_full_pdb(full_pdb, output_path):
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


def get_one_pdb(info_tsv, uniprot_id, reference_directory):
    info_df = pd.read_csv(info_tsv, sep='\t')
    info_df['pos_covered'] = info_df['pos_covered'].apply(ast.literal_eval)
    pdbs = [item for item in os.listdir(reference_directory) if item.endswith('.gz') and uniprot_id in item]
    full_pdb = []
    pdbs.sort(key=lambda item: item.split('-')[2][-1])
    # print(pdbs)
    for item in pdbs:
        # print(item[:-3])
        p = os.path.join(reference_directory, item)
        # print(p_pos_cover)
        if 'F1' in item:
            with gzip.open(p, "rt") as handle:
                structure = PDBParser(QUIET=True).get_structure("protein", handle)
            for model in structure:
                for chain in model:
                    for residue in chain:
                        full_pdb.append(residue)
        else:
            with gzip.open(p, "rt") as handle:
                structure = PDBParser(QUIET=True).get_structure("protein", handle)
            new_residue = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        new_residue.append(residue)
            # print(full_pdb[-1].id)
            new_residue = new_residue[1200:]
            # print(len(new_residue))
            for i in range(len(new_residue)):
                new_residue[i].id = (' ', 1 + i + full_pdb[-1].id[1], ' ')
            full_pdb.extend(new_residue)

    write_full_pdb(full_pdb, os.path.join(reference_directory, uniprot_id + '.pdb'))

def pymol_rvas(info_tsv, df_rvas, reference_directory):
    # make a pymol session with case and control mutations
    # output a gif and a .pse file
    df_rvas = pd.read_csv(df_rvas, sep='\t')
    pdbs = set(df_rvas['pdb_filename'].tolist())
    uniprot_ids = set(df_rvas['uniprot_id'].tolist())

    for item in pdbs:
        p = os.path.join(reference_directory, item)
        print('path:', p)
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
        
        # cmd.bg_color("black")
        # cmd.mset("1 x 20")  
        # cmd.turn("y", 5)  
        # cmd.mpng(f"{uniprot_id}_{item}_frame")
        cmd.save(f"{uniprot_id}_{item.split('.')[0]}.pse")

        uniprot_ids = set(df_rvas['uniprot_id'].tolist())
        for uniprot_id in uniprot_ids:
            print(uniprot_id)
            get_one_pdb(info_tsv, uniprot_id, reference_directory)
            cmd.load(f"{uniprot_id}.pdb", "structure")
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

                cmd.save(f"{uniprot_id}.pse")
            

def pymol_annotation(annot_file, reference_directory):
    # visualize the annotation

    # info_df = pd.read_csv(info_tsv, sep='\t')
    # info_df['uniprot_id'] = info_df['filename'].apply(lambda x: x.split('-')[1])
    # info_df['pos_covered'] = info_df['pos_covered'].apply(ast.literal_eval)
    # info_df['start_pos'] = info_df['pos_covered'].apply(lambda x: x[0])
    # info_df['end_pos'] = info_df['pos_covered'].apply(lambda x: x[1])
    # annot_df = pd.read_csv(annot_file, sep='\t')
    # annot_uniprot = set(annot_df['uniprot_id'].tolist())
    # for item in annot_uniprot:
    #     tmp_annot = annot_df[annot_df['uniprot_id'] == item]
    #     tmp_info = info_df[info_df['uniprot_id'] == item]
    #     for _, row in tmp_annot.iterrows(): 
    #         mask = (tmp_info['start_pos'].astype(int) < int(row['aa_pos'])) & (tmp_info['end_pos'].astype(int) > int(row['aa_pos']))
    #         annot_info = tmp_info[mask]
    #         uniprot_id = row['uniprot_id']
    #         pse_filenames = [f'{uniprot_id}_{item.split('.')[0]}.pse' for item in annot_info['filename'].tolist()]
    #         for filename in pse_filenames:
    #             p = os.path.join(reference_directory, filename)
    #             cmd.load(p)
    #             cmd.select(f"annotation_residue_{row['aa_pos']}", f"resi {row['aa_pos']}")
    #             cmd.label(f"annotation_residue_{row['aa_pos']} and name CA", f'"annotation"')
    #             cmd.save(p)

    annot_df = pd.read_csv(annot_file, sep='\t')
    uniprot_ids = set(annot_df['uniprot_id'].tolist())
    for uniprot_id in uniprot_ids:
        p = os.path.join(reference_directory,  f'{uniprot_id}.pse')
        if os.path.exists(p):
            cmd.load(f'{uniprot_id}.pse')
            tmp_annot = annot_df[annot_df['uniprot_id'] == uniprot_id]
            tmp_annot_pos = tmp_annot['aa_pos'].tolist()
            for item in tmp_annot_pos:
                cmd.select(f"annotation_residue_{item}", f"resi {item}")
                cmd.label(f"annotation_residue_{item} and name CA", f'"annotation"')
            cmd.save(p)

    
def pymol_scan_test(df_results, reference_directory):
    # color by case/control ratio of the neighborhood
    uniprot_id = df_results.split('_')[0]
    df_results_p = os.path.join(reference_directory, df_results)
    pse_p = os.path.join(reference_directory, uniprot_id + '.pse')
    df_results = pd.read_csv(df_results_p, sep='\t')
    df_results['ratio_normalized'] = df_results['ratio'] / df_results['ratio'].max()
    df_results['ratio_normalized'].to_csv('test.csv', sep='\t', index=False)
    # color_dict = dict(zip(df_results['aa_pos'], df_results['ratio_normalized']))
    
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
    # model = cmd.get_model(objects)
    # for atom in model.atom:
    #     print(f"Residue {atom.resi}, Atom {atom.name}, B-factor: {atom.b:.2f}")


    # for pos, value in color_dict.items():
    #     cmd.select(f'residue_{pos}', f"resi {pos} and chain A")
    #     cmd.color('orange', f'residue_{pos}')
    #     cmd.set("transparency", 1.0 - value, f'residue_{pos}')

    result_pse_p = os.path.join(reference_directory, f'{uniprot_id}_result.pse')
    cmd.save(result_pse_p)


def pymol_neighborhood(df_results, reference_directory):
    # for each significant neighborhood, zoom in and show the case and control mutations
    # just in that neighborhood.
    uniprot_id = df_results.split('_')[0]
    df_results_p = os.path.join(reference_directory, df_results)
    df_results = pd.read_csv(df_results_p, sep='\t')
    pse_p = os.path.join(reference_directory, uniprot_id + '_result.pse')
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
    cmd.save(f"{uniprot_id}_result.pse")

    # df_rvas = pd.read_csv(df_rvas, sep='\t')
    # tmp_df_rvas = df_rvas[df_rvas['uniprot_id'] == uniprot_id]
    # df_merged = pd.merge(df_results, tmp_df_rvas, on='aa_pos', how='inner')
    # print(df_merged)


pymol_rvas('info.tsv','sample_df_rvas.tsv', '/Users/liaoruqi/Desktop/structure-informed-rvas/')
pymol_annotation('ClinVar_PLP_uniprot_canonical.tsv', '/Users/liaoruqi/Desktop/structure-informed-rvas/')
pymol_scan_test('O15047_results.tsv', '/Users/liaoruqi/Desktop/structure-informed-rvas/')
pymol_neighborhood('O15047_results.tsv', '/Users/liaoruqi/Desktop/structure-informed-rvas/')