from Bio.PDB import PDBParser
import MDAnalysis as mda
from MDAnalysis.analysis import align
import numpy as np

def parse_pdb(pdb_filename):
    parser = PDBParser()
    structure = parser.get_structure('PDB_structure', pdb_filename)
    for model in structure:
        for chain in model:
            print(f"Chain {chain.id}")
            for residue in chain:
                print(f"Residue {residue.get_resname()} at position {residue.get_id()[1]}")



u = mda.Universe('17gs.pdb')

# Basic operations : RMSD calculation
rmsd = align.alignto(u, reference, select="backbone")

def calculate_rmsd(structure_1, structure_2):
    R = align.rotation_matrix(structure_1.positions, structure_2.positions)
    rmsd_value = np.sqrt(np.mean(np.square(structure_1.positions - structure_2.positions.dot(R[0]))))
    return rmsd_value


