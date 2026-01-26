import os,time
def _printdw(str0, str1="WARNING"):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]{str1} {str0}", flush=True)
def printd(str0, str1="INFO", types=0):
    str1_default = ["INFO","WARNING","ERROR"]
    if str1 == "INFO" and int(types)>0 and int(types)<len(str1_default):
        str1=str1_default[int(types)] 
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}]{str1} {str0}", flush=True)
def calculate_metrics(y_true, y_proba, y_type='NA', y_thres=0.5):
    import numpy as np
    from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,recall_score,precision_score,f1_score,matthews_corrcoef,roc_curve
    y_pred = (y_proba > y_thres).astype(int)
    unique_labels = np.unique(y_true)
    n_classes = len(unique_labels)
    if len(np.unique(y_pred))  == 1 or n_classes == 1: mcc = 0.0
    else: mcc = matthews_corrcoef(y_true, y_pred)
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        youden = max(tpr - fpr) if len(fpr) > 1 else 0.0
    except ValueError: youden = 0.0
    return {'Type': y_type,
        'AUC': roc_auc_score(y_true, y_proba) if n_classes > 1 else float('nan'),
        'AP': average_precision_score(y_true, y_proba),
        'Acc': accuracy_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'Prec': precision_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'MCC': mcc,
        'Youden': youden}
def find_file_groups(path):
    import glob,re
    bact_files = sorted(glob.glob(os.path.join(path, 'bact*.csv')))
    matched_groups = []
    for bact_path in bact_files:
        match = re.search(r'bact(.*)\.csv',  os.path.basename(bact_path)) 
        if not match: continue 
        wildcard_part = match.group(1) 
        phage_file = f"phage{wildcard_part}.csv"
        edge_file = f"edge{wildcard_part}.csv"
        phage_path = os.path.join(path, phage_file)
        edge_path = os.path.join(path, edge_file)
        if os.path.isfile(phage_path) and os.path.isfile(edge_path): 
            matched_groups.append({'bact': bact_path, 'phage': phage_path, 'edge': edge_path, 'wildcard': wildcard_part}) 
    return matched_groups
def load_csv_data(d1, d2, e):
    import pandas as pd
    import torch
    from torch_geometric.data import Data
    x1 = torch.from_numpy(pd.read_csv(d1,index_col=0,header=None).to_numpy()).float()
    x2 = torch.from_numpy(pd.read_csv(d2,index_col=0,header=None).to_numpy()).float()
    edge = pd.read_csv(e,header=None)
    label_half = torch.from_numpy(edge.iloc[:,2].to_numpy()).float()
    index_half = torch.from_numpy(edge.iloc[:,[4,5]].to_numpy()).long()
    return Data(x1=x1, x2=x2, index=index_half, label=label_half)
def split_sample(bacteria, phage, edge_index, edge_labels, train_ratio=0.7):
    import sys,torch
    if edge_index.shape[0] == 2 and edge_index.shape[1] != 2: edge_index = edge_index.t().contiguous()
    def split_nodes(n, ratio):
        n_train = max(1, int(n * ratio))
        n_remain = n - n_train
        n_valid = n_remain // 2
        perm = torch.randperm(n)
        return perm[:n_train], perm[n_train:n_train+n_valid], perm[n_train+n_valid:]
    bac_train, bac_valid, bac_test = split_nodes(bacteria.size(0),  train_ratio)
    phage_train, phage_valid, phage_test = split_nodes(phage.size(0),  train_ratio)
    if len(bac_valid) <=0 and len(phage_valid) <=0:
        _printdw("None Node Exists", "ERROR")
        sys.exit(1)
    if len(bac_test) <=0 and len(phage_test) <=0:
        _printdw("None Node Exists", "ERROR")
        sys.exit(1)
    def get_edge_mask(node_set, node_type=0):
        if node_type == 0: 
            return torch.isin(edge_index[:,  0], node_set)
        return torch.isin(edge_index[:,  1], node_set)
    train_mask = get_edge_mask(bac_train) & get_edge_mask(phage_train, 1)
    test_mask = (get_edge_mask(bac_test) if len(bac_test) > 0 else torch.tensor(False)) | (get_edge_mask(phage_test, 1) if len(phage_test) > 0 else torch.tensor(False))
    valid_mask = ~(test_mask | train_mask)
    def build_train_graph():
        edge_mask = train_mask
        sub_edges = edge_index[edge_mask]
        sub_labels = edge_labels[edge_mask]
        bac_map = {gid.item():  i for i, gid in enumerate(bac_train)}
        phage_map = {gid.item():  i + len(bac_train) for i, gid in enumerate(phage_train)}
        local_edges = torch.zeros_like(sub_edges)
        for i, (bac_g, phage_g) in enumerate(sub_edges):
            local_edges[i, 0] = bac_map[bac_g.item()] 
            local_edges[i, 1] = phage_map[phage_g.item()] 
        return {"node_features": torch.cat([bacteria[bac_train], phage[phage_train]], dim=0),
            "edge_index": torch.cat([local_edges, local_edges.flip(1)],dim=0),
            "all_labels": torch.cat([sub_labels, sub_labels]),
            "bact_global_ids": bac_train,
            "phage_global_ids": phage_train}
    def build_global_graph():
        bac_map = {**{gid.item():  i for i, gid in enumerate(bac_train)},
                **{gid.item():  i + len(bac_train) + len(phage_train) for i, gid in enumerate(bac_valid)},
                **{gid.item():  i + len(bac_train) + len(phage_train) + len(bac_valid) + len(phage_valid) for i, gid in enumerate(bac_test)}}
        phage_map = {**{gid.item():  i + len(bac_train) for i, gid in enumerate(phage_train)},
                    **{gid.item():  i + len(bac_train) + len(phage_train) + len(bac_valid) for i, gid in enumerate(phage_valid)},
                    **{gid.item():  i + len(bac_train) + len(phage_train) + len(bac_valid) + len(phage_valid) + len(bac_test) for i, gid in enumerate(phage_test)}}
        def convert_edges(edges, bac_map, phage_map):
            return torch.tensor([[bac_map[bac.item()], phage_map[phage.item()]] for bac, phage in edges]).contiguous()
        train_edges = convert_edges(edge_index[train_mask], bac_map, phage_map)
        valid_edges = convert_edges(edge_index[valid_mask], bac_map, phage_map)
        test_edges = convert_edges(edge_index[test_mask], bac_map, phage_map)
        all_edges = torch.cat([train_edges,  valid_edges, test_edges], dim=0)
        if all_edges.size(0) == 2 and all_edges.size(1) != 2: all_edges = all_edges.t().contiguous()
        all_labels = torch.cat([edge_labels[train_mask], edge_labels[valid_mask], edge_labels[test_mask]])
        return {"node_features": torch.cat([bacteria[bac_train], phage[phage_train], bacteria[bac_valid], phage[phage_valid], bacteria[bac_test], phage[phage_test]], dim=0), 
            "all_edges": torch.cat([all_edges, all_edges.flip(1)],dim=0),
            "all_labels": torch.cat([all_labels, all_labels]),
            "valid_edges": torch.cat([valid_edges, valid_edges.flip(1)],dim=0),
            "test_edges": torch.cat([test_edges, test_edges.flip(1)],dim=0),
            "valid_labels": torch.cat([edge_labels[valid_mask], edge_labels[valid_mask]]),
            "test_labels": torch.cat([edge_labels[test_mask], edge_labels[test_mask]]),
            "valid_global_ids": {"bact": bac_valid, "phage": phage_valid},
            "test_global_ids": {"bact": bac_test, "phage": phage_test}}
    return build_train_graph(), build_global_graph()
def prepare_task_data(group, args, mode="train", device='cpu'):
    import torch
    from torch_geometric.data import Data
    def pad_dim(small_tensor, total_dim=220, fill_value=-999):
        pad_size = total_dim - small_tensor.size(1) 
        padding = torch.full(size=(small_tensor.size(0), pad_size), fill_value=fill_value, dtype=small_tensor.dtype)
        return torch.cat([small_tensor, padding], dim=1)
    def generate_mask(x, small_dim=40, pad_value=-999):
        all_mask = (x[:, small_dim:] != pad_value).any(dim=1)
        return all_mask.long().unsqueeze(-1)
    data = load_csv_data(group['bact'], group['phage'], group['edge'])
    data.x_equal_dim = (data.x1.shape[1] == data.x2.shape[1]) 
    if data.x1.shape[1] > data.x2.shape[1]: 
        small_idx = 2
        small_dim = data.x2.shape[1]
        data.x2 = pad_dim(data.x2, data.x1.shape[1]) 
    else:
        small_idx = 1
        small_dim = data.x1.shape[1]
        data.x1 = pad_dim(data.x1, data.x2.shape[1])
    prop = args.prop if mode == 'train' else 0.5
    n_bacteria = max(int(data.x1.size(0) * prop), 1)
    n_phage = max(int(data.x2.size(0) * prop), 1)
    _printdw(f"{mode.capitalize()},  Genus: {group['wildcard']}, Sampled Bacteria: {n_bacteria}, Sampled Phage: {n_phage}")
    train_graph, other_graph = split_sample(data.x1, data.x2, data.index, data.label, train_ratio=prop)
    pos_edges = train_graph['edge_index'][train_graph['all_labels'] == 1,:]
    x = train_graph['node_features']
    if data.x_equal_dim: x_type = torch.cat([torch.zeros(len(train_graph['bact_global_ids'])), torch.ones(len(train_graph['phage_global_ids']))])
    else: x_type = generate_mask(x, small_dim)
    train_data = Data(x=x, edge_index=pos_edges, small_idx=small_idx, types=x_type, edge_label=train_graph['edge_index'], true_label=train_graph['all_labels'])
    x = other_graph['node_features']
    if data.x_equal_dim:
        x_type = torch.cat([torch.zeros(len(train_graph['bact_global_ids'])), torch.ones(len(train_graph['phage_global_ids'])), 
                            torch.zeros(len(other_graph['valid_global_ids']['bact'])), torch.ones(len(other_graph['valid_global_ids']['phage'])), 
                            torch.zeros(len(other_graph['test_global_ids']['bact'])), torch.ones(len(other_graph['test_global_ids']['phage']))])
    else: x_type = generate_mask(x, small_dim)
    valid_data = Data(x=x, edge_index=pos_edges, small_idx=small_idx, types=x_type, edge_label=other_graph['valid_edges'], true_label=other_graph['valid_labels'])
    test_data = Data(x=x, edge_index=pos_edges, small_idx=small_idx, types=x_type, edge_label=other_graph['test_edges'], true_label=other_graph['test_labels'])
    return train_data.to(device), valid_data.to(device), test_data.to(device)
def save_graph_data(data_obj, genus, mode, output_dir):
    import pandas as pd
    feature_file = os.path.join(output_dir, f"{genus}_{mode}_features.csv") 
    pd.DataFrame(data_obj.x.cpu().numpy()).to_csv(feature_file,  index=False, header=False)
    edge_file = os.path.join(output_dir, f"{genus}_{mode}_edges.csv") 
    edges = data_obj.edge_label.cpu().numpy()
    labels = data_obj.true_label.cpu().numpy()
    edge_df = pd.DataFrame({'node_index_1': edges[:, 0], 'node_index_2': edges[:, 1], 'label': labels})
    edge_df.to_csv(edge_file,  index=False, header=False)