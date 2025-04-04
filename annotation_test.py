def annotation_test(
        df_rvas,
        annotation_file,
        reference_directory,
        neighborhood_radius,
        filter_file,
    ):

    from Bio.PDB import PDBParser
    import os
    import gzip
    from utils import valid_for_fisher, get_pairwise_distances
    
    def loop_protein(uniprotID):

        ## Expand residues
        annot_residues_inprotein = annotation_file.loc[annotation_file.uniprot_id == uniprotID, 'aa_pos'] #residues with reference to protein
        annot_residues_file = df_rvas.loc[(df_rvas.uniprot_id == uniprotID) && df_rvas.aa_pos.isin(annot_residues_inprotein), 
                                        ['aa_pos','pdb_file','file_index'].set_index(file_index) # df mapping between protein-file residue pos 
        annot_residues_infile = annot_residues_file.file_index #residues with reference to pdb file 
                             
        expanded_annot_residues = list()
        for ipdb in annot_residues_file.pdb_file.unique():
            pdb_file = f'{reference_directory}/{ipdb}'
            if not os.path.exists(pdb_file):
                return None
    
            parser = PDBParser(QUIET=True)
            with gzip.open(pdb_file, 'rt') as pdb_file:
                structure = parser.get_structure("protein", pdb_file)
        
            ca_atoms = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue:
                            ca_atoms.append(residue['CA'].get_coord())
            ca_atoms = np.array(ca_atoms)
            residue_cas = ca_atoms[annot_residues_infile-1,:]
            center = np.mean(residue_cas, axis=0)
            distance_from_center = np.sqrt(np.sum((ca_atoms - center) ** 2, axis=-1))
            expanded_residues_file = np.where(distance_from_center<=neighborhood_radius)[0]+1
            expanded_residues_protein = annot_residues_file.loc(expanded_residues_file,'aa_pos')
            expanded_annot_residues.append(expanded_residues_protein)

        ## Filter rvas data frame
        df_rvas_curr = df_rvas[df_rvas.uniprot_id == uniprotID].copy()
        df_rvas_curr['hasAnnot'] = 0
        df_rvas_curr.loc[df_rvas_curr.aa_pos.isin(expanded_annot_residues), 'hasAnnot'] = 1
        df_rvas_curr = df_rvas_curr.merge(filter_file, on=['uniprot_id', 'aa_pos', 'aa_ref', 'aa_alt'], how='inner')

        ### Perform Fischer's exact test
        inAnnotCas = df_rvas_curr.loc[df_rvas_curr.hasAnnot, 'ac_cas'].sum()
        inAnnotCon = df_rvas_curr.loc[df_rvas_curr.hasAnnot, 'ac_con'].sum()
        outAnnotCas = df_rvas_curr.loc[~df_rvas_curr.hasAnnot, 'ac_cas'].sum()
        outAnnotCon = df_rvas_curr.loc[~df_rvas_curr.hasAnnot, 'ac_con'].sum()

        contigency_table = [ [inAnnotCas, outAnnotCas], [inAnnotCon, outAnnotCon] ]
        if ~valid_for_fisher(contigency_table):
            o = np.nan
            p = np.nan
        else:
            o, p = stats.fisher_exact(contigency_table)
            
        return (uniprotID, contigency_table, o, p)
    
        res = list(map(loop_protein, df_rvas.uniprot_id.unique()))
        # this list will contain an entry per protein, which will be a tuple constisting of:
        # - the uniprot_id
        # - the contingency table
        # - the odds ratio
        # - the pvalue of the Fischer's exact test

    
    '''
    perform annotation test. annotation file and filter file have columns uniprot_id,
    aa_pos, aa_ref, aa_alt, which specify the members of the annotation/filter. 
    reference_directory has pdb_files. 

    this function loops over proteins. for each protein, it takes the annotation, uses the 
    pdb files to extend by the neighborhood radius, then filters using the filter file. then 
    performs fisher's exact to compare the resulting set of variants to the background of the 
    whole protein.

    df_rvas: pandas dataframe with columns uniprot_id, aa_pos, aa_ref, aa_alt, pdb_file, file_index, ac_case, and ac_control
    '''