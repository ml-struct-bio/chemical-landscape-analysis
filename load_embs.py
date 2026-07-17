# import torch
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem import rdFingerprintGenerator
# from tqdm import tqdm

# names = ['spectranp', 'nmrexp', 'uspto']
# paths = {n: f'/scratch/gpfs/ZHONGE/jc4587/nmr_embs_ALL_DATA/{n}_embeddings_multilayer.pt' for n in names}
# raw = {n: torch.load(p) for n, p in paths.items()}

# x = torch.cat([raw[n]['layer_timestep_data'][(3, 0.001)]['global_cond'] for n in names], dim=0)
# smiles = sum([raw[n]['smiles'] for n in names], [])
# dataset_names = sum([[n] * len(raw[n]['smiles']) for n in names], [])

# print("shape:", x.shape)
# print("mean:", x.mean().item(), "std:", x.std().item())
# print("min:", x.min().item(), "max:", x.max().item())
# print("n smiles:", len(smiles), "| n dataset_names:", len(dataset_names))

# # --- regenerate ECFPs: radius=3, size=1024 ---
# RADIUS, NBITS = 3, 1024
# mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=NBITS)

# def smiles_to_ecfp(smi):
#     mol = Chem.MolFromSmiles(smi)
#     if mol is None:
#         return np.zeros(NBITS, dtype=np.uint8)
#     return mfgen.GetFingerprintAsNumPy(mol).astype(np.uint8)

# ecfp = np.stack([smiles_to_ecfp(s) for s in tqdm(smiles)]).astype(np.uint8)
# print("ecfp shape:", ecfp.shape)

# np.save('/scratch/gpfs/ZHONGE/jc4587/nmr_embs_ALL_DATA/all_ecfp_r3_1024.npy', ecfp)
# print("saved: all_ecfp_r3_1024.npy")

import torch
import numpy as np

names = ['spectranp', 'nmrexp', 'uspto']
paths = {n: f'/scratch/gpfs/ZHONGE/jc4587/nmr_embs_ALL_DATA/{n}_embeddings_multilayer.pt' for n in names}
raw = {n: torch.load(p) for n, p in paths.items()}

global_cond = torch.cat([raw[n]['layer_timestep_data'][(11, 0.001)]['global_cond'] for n in names], dim=0)
smiles = sum([raw[n]['smiles'] for n in names], [])
dataset_names = sum([[n] * len(raw[n]['smiles']) for n in names], [])

ecfp = np.load('/scratch/gpfs/ZHONGE/jc4587/nmr_embs_ALL_DATA/all_ecfp_r3_1024.npy')

print(global_cond.shape)
print(smiles[:10])
print(ecfp.shape)
print(dataset_names[:10])

# print("shape:", global_cond.shape)
# print("n smiles:", len(smiles), "| n dataset_names:", len(dataset_names), "| ecfp shape:", ecfp.shape)