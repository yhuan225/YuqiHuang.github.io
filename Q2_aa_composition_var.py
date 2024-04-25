import os
from Bio.PDB import PDBList, PDBParser
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, kruskal

classifications = {
    '17gs': 'Glutathione S-Transferase P1-1',
    '4HHB': 'Globular protein',
    '1HSG': 'Enzyme'}


# Function to fetch PDB files and extract amino acid compositions
def fetch_and_analyze_structures(pdb_ids):
    pdbl = PDBList()
    parser = PDBParser(QUIET=True)
    aa_compositions = []

    for pdb_id in pdb_ids:
        file_path = pdbl.retrieve_pdb_file(pdb_id, pdir='.', file_format='pdb', overwrite=False)
        structure = parser.get_structure(pdb_id, file_path)

        # Calculate amino acid composition
        aa_counts = {}
        for model in structure:
            for chain in model:
                for residue in chain.get_residues():
                    if residue.id[0] != ' ':
                        continue
                    res_name = residue.get_resname()
                    if res_name in aa_counts:
                        aa_counts[res_name] += 1
                    else:
                        aa_counts[res_name] = 1

        # Normalize counts to composition
        total = sum(aa_counts.values())
        composition = {aa: count / total for aa, count in aa_counts.items()}
        composition['PDB_ID'] = pdb_id
        composition['Class'] = classifications[pdb_id]  # Assign class
        aa_compositions.append(composition)

    return aa_compositions


# List of PDB IDs to analyze
pdb_ids = ['17gs', '4HHB', '1HSG']

# Fetch data and calculate compositions
compositions = fetch_and_analyze_structures(pdb_ids)

# Convert to DataFrame for easier manipulation and visualization
df = pd.DataFrame(compositions).fillna(0)

# Melt the DataFrame for easier plotting
df_melted = df.melt(id_vars=['PDB_ID', 'Class'], var_name='Amino Acid', value_name='Frequency')

# Plotting with seaborn
plt.figure(figsize=(14, 7))
sns.barplot(data=df_melted, x='Amino Acid', y='Frequency', hue='Class')
plt.title('Comparative Amino Acid Composition by Protein Fold Class')
plt.ylabel('Normalized Frequency')
plt.xlabel('Amino Acid')
plt.legend(title='Protein Class')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Prepare data for statistical testing
df_melted = df.melt(id_vars=['PDB_ID', 'Class'], var_name='Amino Acid', value_name='Frequency')

# Chi-Square Test - aggregate data for chi-square test
contingency_table = pd.pivot_table(df_melted, values='Frequency', index='Amino Acid', columns='Class', aggfunc=sum).fillna(0)
chi2, p, dof, expected = chi2_contingency(contingency_table.values)
print("Chi-Square Test:")
print("Chi-squared Statistic:", chi2)
print("P-value:", p)

# Kruskal-Wallis Test - apply test for each amino acid
print("\nKruskal-Wallis Test Results:")
for amino_acid in contingency_table.index:
    samples = [group.dropna().values for name, group in df_melted[df_melted['Amino Acid'] == amino_acid].groupby('Class')['Frequency']]
    stat, p_value = kruskal(*samples)
    print(f"{amino_acid}: H-statistic={stat}, P-value={p_value}")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Preparing the data for machine learning
X = df.drop(['PDB_ID', 'Class'], axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))