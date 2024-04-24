from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import matplotlib.pyplot as plt

# Read PDB file
p = PDBParser()
structure = p.get_structure("protein_name", "~/17gs.pdb")

# Analyzing secondary structures using DSSP
model = structure[0]
dssp = DSSP(model, "~/17gs.pdb")
# Extracting fragments of alpha helices and beta folds
helices = [res for res in dssp if res[2] == 'H']  # Helix
sheets = [res for res in dssp if res[2] == 'E']  # Beta

# Print segment information
print("Alpha helices:")
for res in helices:
    print(f"Residue {res[0]}: {res[1]}")

print("Beta sheets:")
for res in sheets:
    print(f"Residue {res[0]}: {res[1]}")

# visilization
plt.figure(figsize=(10, 6))
plt.plot([res[1] for res in dssp], label='DSSP index')
plt.scatter([res[0] for res in helices], [res[1] for res in helices], color='r', label='Alpha helices')
plt.scatter([res[0] for res in sheets], [res[1] for res in sheets], color='b', label='Beta sheets')
plt.xlabel('Residue number')
plt.ylabel('DSSP index')
plt.legend()
plt.savefig("01.png")
plt.show()