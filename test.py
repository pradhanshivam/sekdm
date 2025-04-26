from Bio.PDB import PDBParser, PPBuilder
import esm
import torch
import numpy as np
import os
import torch
import numpy as np
import esm
from Bio.PDB import PDBParser, PPBuilder
from tqdm import tqdm  # Add progress bar

def extract_sequence_from_pdb(pdb_path, chain_id=None):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    ppb = PPBuilder()
    for model in structure:
        for chain in model:
            if chain_id is None or chain.id == chain_id:
                for pp in ppb.build_peptides(chain):
                    return str(pp.get_sequence()), chain.id
    return None, None

def generate_esm_from_pdb(pdb_path, model, batch_converter, save_dir="data/ESM"):
    os.makedirs(save_dir, exist_ok=True)
    sequence, chain_id = extract_sequence_from_pdb(pdb_path)
    if sequence is None:
        print(f"❌ Could not extract sequence from {pdb_path}")
        return

    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
    output_path = os.path.join(save_dir, f"{pdb_name}.npy")
    if os.path.exists(output_path):
        print(f"[✓] Already exists: {output_path}")
        return

    data = [(pdb_name, sequence)]
    _, _, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)

    token_representations = results["representations"][33]
    embedding = token_representations[0, 1:len(sequence)+1].cpu().numpy()

    np.save(output_path, embedding)
    print(f"[+] Saved ESM embedding: {output_path}")

# ==========================
# Batch Process All .pdb Files with Progress Bar
# ==========================
if __name__ == "__main__":
    pdb_folder = "data_/PDB"  # Your folder with .pdb files
    save_dir = "data/ESM"

    # Load ESM model once
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    pdb_files = [f for f in os.listdir(pdb_folder) if f.endswith(".pdb")]

    for file in tqdm(pdb_files, desc="Generating ESM features"):
        full_path = os.path.join(pdb_folder, file)
        generate_esm_from_pdb(full_path, model, batch_converter, save_dir)


# import pickle

# # Load the file
# with open("data_/N_Train_335.pkl", "rb") as f:
#     data = pickle.load(f)

# # Print the type of object
# print("Type:", type(data))

# # Show first few entries if it's a dict, list, or DataFrame
# if isinstance(data, dict):
#     print("Keys:", data.keys())
#     print("Sample:", next(iter(data.values())))
# elif isinstance(data, list):
#     print("Length:", len(data))
#     print("First item:", data[0])
# elif hasattr(data, 'head'):
#     print(data.head())  # Likely a pandas DataFrame
# else:
#     print("Content:", data)

# import numpy as np

# # Load the .npy file
# data = np.load('data/ESM/1a0b_A.npy')

# # Print the contents
# print(data)

# # Check the shape and data type of the array
# print("Shape:", data.shape)
# print("Data type:", data.dtype)

