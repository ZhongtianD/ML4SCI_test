import torch
import numpy as np
import energyflow
from scipy.sparse import coo_matrix
from torch.utils.data import TensorDataset, DataLoader

def get_adj_matrix(n_nodes, batch_size, edge_mask):
    rows, cols = [], []
    for batch_idx in range(batch_size):
        nn = batch_idx*n_nodes
        x = coo_matrix(edge_mask[batch_idx])
        rows.append(nn + x.row)
        cols.append(nn + x.col)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)

    edges = [torch.LongTensor(rows), torch.LongTensor(cols)]
    return edges

def collate_fn(data):
    data = list(zip(*data)) # label p4s nodes atom_mask
    data = [torch.stack(item) for item in data]
    batch_size, n_nodes, _ = data[1].size()
    atom_mask = data[-1]
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edges = get_adj_matrix(n_nodes, batch_size, edge_mask)
    return data + [edge_mask, edges]

def retrieve_dataloaders(batch_size, num_data = -1, cache_dir = './data', num_workers=4):
    raw = energyflow.qg_jets.load(num_data=num_data, pad=True, ncol=4, generator='pythia',
                            with_bc=False, cache_dir=cache_dir)
    splits = ['train', 'val', 'test']
    data = {type:{'raw':None,'label':None} for type in splits}
    (data['train']['raw'],  data['val']['raw'],   data['test']['raw'],
    data['train']['label'], data['val']['label'], data['test']['label']) = \
        energyflow.utils.data_split(*raw, train=0.8, val=0.1, test=0.1, shuffle = False)
    
    for split, value in data.items():
        pid = torch.from_numpy(np.abs(np.asarray(value['raw'][...,3], dtype=int))).unsqueeze(-1)
        p4s = torch.from_numpy(energyflow.p4s_from_ptyphipids(value['raw'],error_on_unknown=True))
        mass = torch.from_numpy(energyflow.ms_from_p4s(p4s)).unsqueeze(-1)
        charge = torch.from_numpy(energyflow.pids2chrgs(pid))
        nodes = torch.cat((mass,charge),dim=-1)
        nodes = torch.sign(nodes) * torch.log(torch.abs(nodes) + 1)
        atom_mask = (pid[...,0] != 0)
        value['p4s'] = p4s
        value['nodes'] = nodes
        value['label'] = torch.from_numpy(value['label'])
        value['atom_mask'] = atom_mask.to(torch.bool)

    datasets = {split: TensorDataset(value['label'], value['p4s'],
                                     value['nodes'], value['atom_mask'])
                for split, value in data.items()}

    # Construct PyTorch dataloaders from datasets
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=batch_size if (split == 'train') else batch_size,
                                     pin_memory=True,
                                     persistent_workers=True,
                                     drop_last=True if (split == 'train') else False,
                                     num_workers=num_workers,
                                     collate_fn=collate_fn)
                        for split, dataset in datasets.items()}

    return dataloaders