import os
import torch
import numpy as np
import esm
from PDB_Parser import StructureDataParser

def generate_esm_from_pdb(pdb_path, save_dir="data/ESM"):
    protein_id = os.path.basename(pdb_path).split('.')[0]
    output_path = os.path.join(save_dir, f"{protein_id}.npy")
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(output_path):
        print(f"[✓] Skipping {protein_id} — already exists.")
        return

    try:
        # Step 1: Parse PDB and get sequence
        parser = StructureDataParser(pdb_path, protein_id, file_type='pdb')
        seq_list = parser.get_sequence()
        sequence = ''.join(seq_list)
        if len(sequence) == 0:
            print(f"[✗] Empty sequence in {protein_id}. Skipping.")
            return

        # Step 2: Load ESM model
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model.eval()
        batch_converter = alphabet.get_batch_converter()

        # Step 3: Prepare sequence input
        data = [(protein_id, sequence)]
        _, _, batch_tokens = batch_converter(data)

        # Step 4: Get representation from ESM
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)

        embedding = results["representations"][33][0, 1:len(sequence)+1].cpu().numpy()

        # Step 5: Save as .npy
        np.save(output_path, embedding)
        print(f"[+] Saved {output_path}")
    except Exception as e:
        print(f"[✗] Error processing {protein_id}: {e}")


if __name__ == "__main__":
    pdb_folder = "data_/PDB"  # Adjust if your folder is different
    for filename in os.listdir(pdb_folder):
        if filename.endswith(".pdb"):
            pdb_path = os.path.join(pdb_folder, filename)
            generate_esm_from_pdb(pdb_path)
