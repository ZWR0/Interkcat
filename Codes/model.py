from transformers import AutoTokenizer, AutoModel
from rdkit import Chem

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, MessagePassing

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

import pandas as pd
import numpy as np
from typing import List, NamedTuple, Optional

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset class for kcat prediction
class KcatDataset(Dataset):
    def __init__(self, sequences, smiles_list, kcats):
        assert len(sequences) == len(smiles_list) == len(kcats)
        self.sequences = sequences
        self.smiles_list = smiles_list
        self.kcats = kcats

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.smiles_list[idx], self.kcats[idx]
    
def collate_fn(batch):
    sequences, smiles_list, kcats = zip(*batch)
    kcats = torch.tensor(kcats, dtype=torch.float32)
    return list(sequences), list(smiles_list), kcats

# Protein Embedding Layer using pre-trained transformer model
class ProteinEmbeddingLayer(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/esm2_t30_150M_UR50D",
        freeze: bool = True,
        unfreeze_last_n: int = 0,
        device: str = None
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        for param in self.model.parameters():
            param.requires_grad = False

        if not freeze:
            if unfreeze_last_n == 0:
                # Fully unfreeze the model
                for param in self.model.parameters():
                    param.requires_grad = True
                self.model.train()
            else:
                # unfreeze_last_n > 0
                num_layers = len(self.model.encoder.layer)
                unfreeze_start_idx = num_layers - unfreeze_last_n
                
                # unfreeze transformer layers
                for i in range(unfreeze_start_idx, num_layers):
                    for param in self.model.encoder.layer[i].parameters():
                        param.requires_grad = True
                
                if hasattr(self.model, 'contact_head'):
                    for param in self.model.contact_head.parameters():
                        param.requires_grad = True
                
                self.model.train()
        else:
            self.model.eval()

    def forward(self, sequences, return_symbols: bool = False):
        if isinstance(sequences, str):
            sequences = [sequences]

        encoded = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
            add_special_tokens=True
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        is_train = any(p.requires_grad for p in self.model.parameters())
        with torch.set_grad_enabled(is_train):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            last_hidden = outputs.last_hidden_state  # (B, L_with_special, D)

        embeddings_list = []
        mask_list = []
        symbols_list = [] if return_symbols else None

        for i in range(last_hidden.size(0)):
            seq_len_with_special = attention_mask[i].sum().item()
            real_residues = last_hidden[i, 1:seq_len_with_special - 1, :]  # (L_real, D)
            embeddings_list.append(real_residues)
            real_mask = torch.ones(real_residues.size(0), dtype=torch.bool, device=self.device)
            mask_list.append(real_mask)

            if return_symbols:
                input_ids_i = input_ids[i]  # (L_with_special,)
                real_input_ids = input_ids_i[1:seq_len_with_special - 1].cpu().tolist()
                real_symbols = self.tokenizer.convert_ids_to_tokens(real_input_ids)
                symbols_list.append(real_symbols)

        # Padding embeddings
        embeddings_padded = torch.nn.utils.rnn.pad_sequence(
            embeddings_list, batch_first=True, padding_value=0.0
        )  # (B, L_max, D)

        # Padding masks
        mask_padded = torch.nn.utils.rnn.pad_sequence(
            mask_list, batch_first=True, padding_value=False
        )  # (B, L_max)

        if return_symbols:
            return embeddings_padded, mask_padded, symbols_list
        else:
            return embeddings_padded, mask_padded
        
# Molecular Embedding Layer
# Functions to convert SMILES to graph representation
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # Suppress RDKit warnings

# Functions to convert SMILES to graph representation
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER', 'misc'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'misc'],
    'possible_bond_stereo_list': [
        'STEREONONE', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS', 'STEREOANY'
    ],
    'possible_is_conjugated_list': [False, True],
}

def safe_index(l, e):
    try:
        return l.index(e)
    except ValueError:
        return len(l) - 1

def atom_to_feature_vector(atom):
    return [
        safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
        safe_index(allowable_features['possible_chirality_list'], str(atom.GetChiralTag())),
        safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
        safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
        safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
        safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
        safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
        allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
        allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
    ]

def bond_to_feature_vector(bond):
    return [
        safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
        allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
        allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
    ]

def smiles2graph(smiles_string, removeHs=True, max_nodes=None):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None
    mol = mol if removeHs else Chem.AddHs(mol)
    if max_nodes is not None and mol.GetNumAtoms() > max_nodes:
        return None

    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    node_feat = np.array(atom_features_list, dtype=np.int64)

    num_bond_features = 3
    if len(mol.GetBonds()) > 0:
        edges_list, edge_features_list = [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            edges_list.extend([(i, j), (j, i)])
            edge_features_list.extend([edge_feature, edge_feature])
        edge_index = np.array(edges_list, dtype=np.int64).T
        edge_feat = np.array(edge_features_list, dtype=np.int64)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_feat = np.empty((0, num_bond_features), dtype=np.int64)

    return {
        'node_feat': node_feat,
        'edge_index': edge_index,
        'edge_feat': edge_feat,
        'num_nodes': len(node_feat)
    }

# model components
class AtomEncoder(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        keys = [
            'possible_atomic_num_list', 'possible_chirality_list', 'possible_degree_list',
            'possible_formal_charge_list', 'possible_numH_list', 'possible_number_radical_e_list',
            'possible_hybridization_list', 'possible_is_aromatic_list', 'possible_is_in_ring_list'
        ]
        self.embeddings = nn.ModuleList([
            nn.Embedding(len(allowable_features[k]), emb_dim) for k in keys
        ])
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, x):
        return sum(emb(x[:, i]) for i, emb in enumerate(self.embeddings))


class BondEncoder(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        keys = ['possible_bond_type_list', 'possible_bond_stereo_list', 'possible_is_conjugated_list']
        self.embeddings = nn.ModuleList([
            nn.Embedding(len(allowable_features[k]), emb_dim) for k in keys
        ])
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, edge_attr):
        return sum(emb(edge_attr[:, i]) for i, emb in enumerate(self.embeddings))


class GNNCov(MessagePassing):
    def __init__(self, emb_dim: int, use_gat: bool = False, heads: int = 4):
        super(GNNCov, self).__init__(aggr="add")
        self.use_gat = use_gat
        
        if use_gat:
            self.conv = GATConv(emb_dim, emb_dim // heads, heads=heads, edge_dim=emb_dim)
        else:
            self.conv = GCNConv(emb_dim, emb_dim)
            
        self.batch_norm = nn.BatchNorm1d(emb_dim)
        self.res_fc = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        if self.use_gat:
            h = self.conv(x, edge_index, edge_attr=edge_attr)
        else:
            h = self.conv(x, edge_index) + self.message_fusion(x, edge_index, edge_attr)
            
        # residual connection
        x = F.relu(self.batch_norm(x + h))
        return x

    def message_fusion(self, x, edge_index, edge_attr):
        row, col = edge_index
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, edge_attr):
        return F.relu(edge_attr)


class MoleculeEmbeddingOutput(NamedTuple):
    atom_embeddings: torch.Tensor      # (B, L, D)
    atom_mask: torch.Tensor           # (B, L)
    atom_symbols: Optional[List[List[str]]]

class MolecularEmbeddingLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 300,
        num_layers: int = 3,
        gnn_type: str = "gat", # "gcn" or "gat"
        max_nodes: int = 100
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)
        
        # GNN multiple layers
        self.layers = nn.ModuleList([
            GNNCov(hidden_dim, use_gat=(gnn_type == "gat")) 
            for _ in range(num_layers)
        ])
        
        self.post_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.1)
        )
        
    def _smiles_to_graph(self, smiles: str):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() == 0 or mol.GetNumAtoms() > self.max_nodes:
                raise ValueError
            mol = Chem.RemoveHs(mol)
            symbols = [a.GetSymbol() for a in mol.GetAtoms()]
            graph = smiles2graph(smiles, removeHs=True, max_nodes=self.max_nodes)
            if graph is None:
                raise ValueError
            x = torch.from_numpy(graph['node_feat']).long()
            edge_index = torch.from_numpy(graph['edge_index']).long()
            edge_attr = torch.from_numpy(graph['edge_feat']).long()
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), symbols
        except Exception:
            x = torch.zeros((1, 9), dtype=torch.long)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.long)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), ['X']

    def forward(
        self,
        smiles_list: List[str],
        return_symbols: bool = True
    ) -> MoleculeEmbeddingOutput:
        # 1. SMILES to Graph
        datas, symbols_list = [], []
        for smi in smiles_list:
            data, symbols = self._smiles_to_graph(smi)
            datas.append(data)
            symbols_list.append(symbols)

        device = next(self.parameters()).device
        batch = Batch.from_data_list(datas).to(device)
        x, edge_index, edge_attr, batch_vec = batch.x, batch.edge_index, batch.edge_attr, batch.batch

        # 2. initial feature embedding
        node_emb = self.atom_encoder(x)
        edge_emb = self.bond_encoder(edge_attr)

        # 3. GNN extract features
        for layer in self.layers:
            node_emb = layer(node_emb, edge_index, edge_emb)
        
        node_emb = self.post_mlp(node_emb)

        # 4. reshape to (B, L, D)
        B = len(smiles_list)
        L = self.max_nodes
        atom_embeddings = torch.zeros(B, L, self.hidden_dim, device=device)
        atom_mask = torch.zeros(B, L, dtype=torch.bool, device=device)

        for i in range(B):
            mask_i = (batch_vec == i)
            atoms_i = node_emb[mask_i]
            n_real = min(atoms_i.size(0), L)
            atom_embeddings[i, :n_real] = atoms_i[:n_real]
            atom_mask[i, :n_real] = True

        return MoleculeEmbeddingOutput(
            atom_embeddings=atom_embeddings,
            atom_mask=atom_mask,
            atom_symbols=symbols_list if return_symbols else None, 
        )
    
# InterKcat Model
class InterKcat(nn.Module):
    def __init__(self, prot_dim: int, mol_dim: int, d: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.prot_dim = prot_dim
        self.mol_dim = mol_dim
        self.dropout = dropout

        self.prot_emb = ProteinEmbeddingLayer(freeze=False, unfreeze_last_n=2)
        self.mol_emb = MolecularEmbeddingLayer(hidden_dim=mol_dim, max_nodes=100)

        # Cross-attention projections
        self.prot_proj_q = nn.Linear(self.prot_dim, self.d)
        self.mol_proj_k = nn.Linear(self.mol_dim, self.d)
        self.mol_proj_v = nn.Linear(self.mol_dim, self.d)

        self.mol_proj_q = nn.Linear(self.mol_dim, self.d)
        self.prot_proj_k = nn.Linear(self.prot_dim, self.d)
        self.prot_proj_v = nn.Linear(self.prot_dim, self.d)

        # Prediction head
        self.proj_h = nn.Linear(2 * d, 2 * d)
        self.norm1 = nn.LayerNorm(2 * d)
        self.dropout1 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(2 * d, d)
        self.norm2 = nn.LayerNorm(d)
        self.dropout2 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(d, 1)

    def forward(
        self,
        protein_sequences,    # List[str]
        smiles_list,          # List[str]
        return_attn=False,
        inter_model="both"    # "p2m", "m2p", or "both"
    ):

        # Get embeddings and masks
        prot_emb, prot_mask = self.prot_emb(protein_sequences)  # (B, N, D_p), (B, N)
        mol_out = self.mol_emb(smiles_list)
        mol_emb = mol_out.atom_embeddings  # (B, M, D_m)
        mol_mask = mol_out.atom_mask       # (B, M)

        B = prot_emb.size(0)

        # Initialize enhanced representations as original if not computed
        prot_enhanced = prot_emb  # fallback: no enhancement
        mol_enhanced = mol_emb    # fallback: no enhancement

        attn_p2m = None
        attn_m2p = None

        # ===== Option 1: Protein ← Molecule Attention (p2m) =====
        if inter_model in {"p2m", "both"}:
            Q_p = self.prot_proj_q(prot_emb)      # (B, N, d)
            K_m = self.mol_proj_k(mol_emb)        # (B, M, d)
            V_m = self.mol_proj_v(mol_emb)        # (B, M, d)

            att_logits_p2m = torch.matmul(Q_p, K_m.transpose(-2, -1)) / math.sqrt(self.d)
            if mol_mask is not None:
                att_logits_p2m = att_logits_p2m.masked_fill(~mol_mask.unsqueeze(1), float('-inf'))
            attn_p2m = torch.softmax(att_logits_p2m, dim=-1)  # (B, N, M)
            prot_enhanced = torch.matmul(attn_p2m, V_m)       # (B, N, d)

        # ===== Option 2: Molecule ← Protein Attention (m2p) =====
        if inter_model in {"m2p", "both"}:
            Q_m = self.mol_proj_q(mol_emb)        # (B, M, d)
            K_p = self.prot_proj_k(prot_emb)      # (B, N, d)
            V_p = self.prot_proj_v(prot_emb)      # (B, N, d)

            att_logits_m2p = torch.matmul(Q_m, K_p.transpose(-2, -1)) / math.sqrt(self.d)
            if prot_mask is not None:
                att_logits_m2p = att_logits_m2p.masked_fill(~prot_mask.unsqueeze(1), float('-inf'))
            attn_m2p = torch.softmax(att_logits_m2p, dim=-1)  # (B, M, N)
            mol_enhanced = torch.matmul(attn_m2p, V_p)        # (B, M, d)

        # ===== Mask-aware global pooling =====
        if prot_mask is not None:
            h_p = (prot_enhanced * prot_mask.unsqueeze(-1)).sum(1) / prot_mask.sum(1, keepdim=True).clamp(min=1)
        else:
            h_p = prot_enhanced.mean(1)

        if mol_mask is not None:
            h_m = (mol_enhanced * mol_mask.unsqueeze(-1)).sum(1) / mol_mask.sum(1, keepdim=True).clamp(min=1)
        else:
            h_m = mol_enhanced.mean(1)

        # ===== Prediction head =====
        h = torch.cat([h_p, h_m], dim=-1)      # (B, 2d)
        residual = h
        x = self.proj_h(h)                     # (B, 2d)
        x = self.norm1(x + residual)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc1(x)                        # (B, d)
        x = self.norm2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        pred = self.fc2(x).squeeze(-1)

        if return_attn:
            return pred, attn_p2m, attn_m2p
        return pred
    
if __name__ == "__main__":
    # Example usage
    model = InterKcat(prot_dim=640, mol_dim=256).to(DEVICE)
    sequences = ["MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPDWQNYTPGPGIRYPLKF"]
    smiles_list = ["CCO"]
    pred = model(sequences, smiles_list)
    print(pred)