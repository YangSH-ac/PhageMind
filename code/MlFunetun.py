import argparse,sys
def parse_args():
    parser = argparse.ArgumentParser(description='MAML fine-tuning for link prediction')
    parser.add_argument('-ip', required=True, type=str, metavar='PATH', help='Input directory')
    parser.add_argument('-o', required=True, type=str, metavar='PATH', help='Output directory')
    parser.add_argument('-i1', type=str, metavar='STR', help='Input file prefix')
    parser.add_argument('-si', default=2, choices=[1,2], type=int, metavar='INT', help='Small index (default: 2)')
    parser.add_argument('-sd', default=756, type=int, metavar='INT', help='Small dimension (default: 756)')
    parser.add_argument('-s2', default=86, type=int, metavar='INT', help='Seed for torch (default: 86)')
    parser.add_argument('-ss', default='64,32,16,8', type=str, metavar='INT,INT,INT,INT', help='Number of nodes (default: 64,32,16,8)')
    parser.add_argument('-a', default=16, type=int, metavar='INT', help='Adaptor dim (default: No adaptor)')
    parser.add_argument('-ep', default=1000, type=int, metavar='INT', help='Number of step for training (default: 1000)')
    parser.add_argument('-prop', default=0.75, type=float, metavar='FLOAT', help='Proportion of nodes to sample (default: 0.75)')
    parser.add_argument('-sr', default=1, type=float, metavar='FLOAT', help='Support set ratio (default: 1)')
    parser.add_argument('-gr', default=1, type=float, metavar='FLOAT', help='Training graph set ratio (default: 1=use all)')
    parser.add_argument('-er', default=1, type=float, metavar='FLOAT', help='Training edge set ratio (default: 1=use all)')
    parser.add_argument('-m', type=str, metavar='FILE', help='Pretrained model file')
    parser.add_argument('-mt', default=1, type=int, choices=[1,2], metavar='INT', help='Model type (1(default): with GCN, 2: without GCN)')
    parser.add_argument('-iep', default=5, type=int, metavar='INT', help='Number of inner update steps (default: 5)')
    parser.add_argument('-lr', default=0.0001, type=float, metavar='FLOAT', help='Learning rate (default: 0.0001)')
    parser.add_argument('-g', default=4, type=int, metavar='INT', help='Gamma for focal loss (default: 4)')
    parser.add_argument('-p', default=50, type=int, metavar='INT', help='Number of patience (default: 50)')
    args = parser.parse_args()
    args.ss = list(map(int, args.ss.split(',')))
    if len(args.ss) != 4:
        parser.print_help(sys.stderr)
        print(f"\nERROR: length of nodes is not enough") 
        sys.exit(1)
    return args
if __name__ == "__main__": args = parse_args()
import os,torch,glob,re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.my_train import calculate_metrics,printd
from src.my_loss import focal_loss as loss_fn
from torch_geometric.data import Data
import pandas as pd
from types import SimpleNamespace
def split_train(data0, split_ratio=1.0, graph_ratio=1.0, edge_ratio=1.0):
    data = data0.cpu()
    if graph_ratio < 1.0:
        def get_edge_mask(node_set, node_type=0):
            if node_type == 0:
                return torch.isin(part_edge[:, 0], node_set)
            return torch.isin(part_edge[:, 1], node_set)
        tmp_num = int(data.edge_label.shape[0]/2)
        bac_num = max(data.edge_label[:tmp_num,0]) + 1
        phage_num = data.x.size(0) - bac_num
        n_bac = max(int(bac_num * graph_ratio), 1)
        n_phage = max(int(phage_num * graph_ratio), 1)
        selected_bac = torch.randperm(bac_num)[:n_bac]
        selected_phage = torch.randperm(phage_num)[:n_phage] + bac_num
        n_edges = int(data.edge_label.size(0)/2)
        part_edge = data.edge_label[:n_edges]
        part_true = data.true_label[:n_edges]
        train_mask = get_edge_mask(selected_bac, 0) & get_edge_mask(selected_phage, 1)
        edge_label = torch.cat([part_edge[train_mask], part_edge[train_mask].flip(1)], dim=0)
        true_label = torch.cat([part_true[train_mask], part_true[train_mask]], dim=0)
        data = Data(x=data.x, edge_index=edge_label[true_label == 1,:].contiguous(), small_idx=data.small_idx, types=data.types, edge_label=edge_label, true_label=true_label)
    n_edges = int(data.edge_label.shape[0]/2)
    support_size = int(n_edges * split_ratio)
    perm = torch.randperm(n_edges)
    part_edge = data.edge_label[:n_edges]
    part_true = data.true_label[:n_edges]
    if edge_ratio < 1.0:
        idx_num = int(data.edge_index.shape[0]/2)
        n_idx = max(int(idx_num * edge_ratio), 1)
        selected_bac = torch.randperm(idx_num)[:n_idx]
        part_idx = data.edge_index[:idx_num]
        edge_index = torch.cat([part_idx[selected_bac], part_idx[selected_bac].flip(1)], dim=0)
    else:
        edge_index = data.edge_index
    support_data = Data(x=data.x, edge_index=edge_index, small_idx=data.small_idx, types=data.types, 
                        edge_label=torch.cat([part_edge[perm[:support_size]], part_edge[perm[:support_size]].flip(1)], dim=0),
                        true_label=torch.cat([part_true[perm[:support_size]], part_true[perm[:support_size]]]))
    return support_data.to(device)
def generate_mask(x, small_dim=40, pad_value=-999):
    all_mask = (x[:, small_dim:] != pad_value).any(dim=1)
    return all_mask.long().unsqueeze(-1)
def load_single_dataset(path, prefix1, small_idx=2, small_dim=728):
    def load_csv_data(d, e):
        x = torch.from_numpy(pd.read_csv(d, header=None).to_numpy()).float()
        edge = pd.read_csv(e, header=None)
        index = torch.from_numpy(edge.iloc[:, [0,1]].to_numpy()).long()
        label = torch.from_numpy(edge.iloc[:, 2].to_numpy()).float()
        return Data(x=x, index=index, label=label)
    tmp_path = os.path.join(path, prefix1)
    data_all = {}
    for types in ['train', 'valid', 'test']:
        fea_path = tmp_path + f'_{types}_features.csv'
        edge_path = tmp_path + f'_{types}_edges.csv'
        if not os.path.exists(fea_path) or not os.path.exists(edge_path):
            printd(f"File not exist - {fea_path} or {edge_path}", types=1)
            return None
        data_all[types] = load_csv_data(fea_path, edge_path)
    x_all = torch.cat([data_all['train'].x, data_all['valid'].x, data_all['test'].x], dim=0)
    pos_edges = data_all['train'].index[data_all['train'].label == 1, :]
    x_type_train = generate_mask(data_all['train'].x, small_dim)
    x_type_all = generate_mask(x_all, small_dim)
    data_train = Data(x=data_all['train'].x, edge_index=pos_edges, small_idx=small_idx, types=x_type_train,
                      edge_label=data_all['train'].index, true_label=data_all['train'].label, dataset_id=0, dataset_name=f"{prefix1}_{prefix2}")
    data_valid = Data(x=x_all, edge_index=pos_edges, small_idx=small_idx, types=x_type_all, 
                      edge_label=data_all['valid'].index, true_label=data_all['valid'].label, dataset_id=0, dataset_name=f"{prefix1}_{prefix2}")
    data_test = Data(x=x_all, edge_index=pos_edges, small_idx=small_idx, types=x_type_all, 
                     edge_label=data_all['test'].index, true_label=data_all['test'].label, dataset_id=0, dataset_name=f"{prefix1}_{prefix2}")
    return {'train': data_train,'valid': data_valid,'test': data_test,'num_nodes_train': data_all['train'].x.shape[0],'num_nodes_all': data_all['test'].x.shape[0], 'dataset_name': f"{prefix1}"}
def find_all_datasets(path):
    feature_files = glob.glob(os.path.join(path, "*features.csv"))
    dataset_patterns = {}
    for f in feature_files:
        basename = os.path.basename(f)
        match = re.match(r'(.+)_(train|valid|test)_features\.csv', basename)
        if match:
            prefix = match.group(1)
            suffix = match.group(3)
            type_key = match.group(2)
            dataset_key = f"{prefix}_{suffix}"
            if dataset_key not in dataset_patterns:
                dataset_patterns[dataset_key] = {'prefix': prefix, 'suffix': suffix, 'files': {'train': None, 'valid': None, 'test': None}}
            edge_file = f.replace('_features.csv', '_edges.csv')
            if os.path.exists(edge_file):
                dataset_patterns[dataset_key]['files'][type_key] = {'features': f, 'edges': edge_file}
    complete_datasets = []
    for dataset_key, info in dataset_patterns.items():
        if all(info['files'][t] is not None for t in ['train', 'valid', 'test']):
            complete_datasets.append((info['prefix'], info['suffix']))
    printd(f"Find {len(complete_datasets)} complete datasets:")
    for i, (prefix, suffix) in enumerate(complete_datasets):
        printd(f"  Dataset {i+1}: prefix='{prefix}', suffix='{suffix}'")
    return complete_datasets
def merge_datasets(datasets_list, small_idx=2, small_dim=728):
    train_data_list = []
    valid_data_list = []
    test_data_list = []
    individual_datasets = []
    node_offset_train = 0
    node_offset_all = 0
    for i, (prefix, suffix) in enumerate(datasets_list):
        dataset = load_single_dataset(args.ip, prefix, suffix, small_idx, small_dim)
        if dataset is None:
            continue
        individual_datasets.append({'train': dataset['train'].clone(),'valid': dataset['valid'].clone(),
                                    'test': dataset['test'].clone(),'name': dataset['dataset_name'], 'id': i})
        train_data = dataset['train']
        if train_data.edge_index.numel() > 0:
            train_data.edge_index = train_data.edge_index + node_offset_train
        train_data.edge_label = train_data.edge_label + node_offset_train
        valid_data = dataset['valid']
        test_data = dataset['test']
        if valid_data.edge_index.numel() > 0:
            valid_data.edge_index = valid_data.edge_index + node_offset_all
        valid_data.edge_label = valid_data.edge_label + node_offset_all
        if test_data.edge_index.numel() > 0:
            test_data.edge_index = test_data.edge_index + node_offset_all
        test_data.edge_label = test_data.edge_label + node_offset_all
        train_data.dataset_id = i
        valid_data.dataset_id = i
        test_data.dataset_id = i
        train_data_list.append(train_data)
        valid_data_list.append(valid_data)
        test_data_list.append(test_data)
        node_offset_train += dataset['num_nodes_train']
        node_offset_all += dataset['num_nodes_all']
    if not train_data_list:
        raise ValueError("Not found any valid complete datasets.")
    def merge_data_list(data_list):
        x_list = [d.x for d in data_list]
        x_merged = torch.cat(x_list, dim=0)
        edge_index_list = [d.edge_index for d in data_list if d.edge_index.numel() > 0]
        edge_index_merged = torch.cat(edge_index_list, dim=0) if edge_index_list else torch.empty((0, 2), dtype=torch.long)
        edge_label_list = [d.edge_label for d in data_list if d.edge_label.numel() > 0]
        edge_label_merged = torch.cat(edge_label_list, dim=0) if edge_label_list else torch.empty((0, 2), dtype=torch.long)
        true_label_list = [d.true_label for d in data_list if d.true_label.numel() > 0]
        true_label_merged = torch.cat(true_label_list, dim=0) if true_label_list else torch.empty((0,), dtype=torch.float)
        types_list = [d.types for d in data_list]
        types_merged = torch.cat(types_list, dim=0)
        dataset_ids_list = []
        for idx, d in enumerate(data_list):
            dataset_ids_list.append(torch.full((d.x.shape[0],), idx, dtype=torch.long))
        dataset_ids_merged = torch.cat(dataset_ids_list, dim=0)
        return Data(x=x_merged, edge_index=edge_index_merged, small_idx=small_idx, types=types_merged,
                    edge_label=edge_label_merged, true_label=true_label_merged, dataset_ids=dataset_ids_merged)
    data_all_train = merge_data_list(train_data_list)
    data_all_valid = merge_data_list(valid_data_list)
    data_all_test = merge_data_list(test_data_list)
    printd(f" {len(individual_datasets)} datasets merged:")
    for ds in individual_datasets:
        printd(f"  Dataset {ds['id']}: {ds['name']}")
    return data_all_train, data_all_valid, data_all_test, individual_datasets
def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model.link_prediction(data).squeeze().detach()
        loss = loss_fn(out, data.true_label, gamma=args.g).cpu().item()
    metrics = calculate_metrics(data.true_label.cpu().numpy(), out.cpu().numpy())
    return loss, metrics
def generate_result(params, crt_dir, num_nodes = 2, Name=''): 
    import numpy as np
    import pandas as pd
    np.set_printoptions(threshold=np.inf)
    def n2s(arr):
        return np.array2string(arr, max_line_width=np.inf, threshold=np.inf, separator=',')
    prediction = []
    if int(num_nodes) < 1 or int(num_nodes) > 2:
        printd(f"generate_result: Unsupported num_nodes={num_nodes}", "ERROR")
        return
    if int(num_nodes) == 2:
        prediction.append({'Type':'Train','Node1':n2s(params.train_index[:,0].numpy()),'Node2':n2s(params.train_index[:,1].numpy()),'PredProba':n2s(params.out_train),'Label':n2s(params.train_label.numpy())})
        prediction.append({'Type':'Val','Node1':n2s(params.val_index[:,0].numpy()),'Node2':n2s(params.val_index[:,1].numpy()),'PredProba':n2s(params.out_val),'Label':n2s(params.val_label.numpy())})
        prediction.append({'Type':'Test','Node1':n2s(params.test_index[:,0].numpy()),'Node2':n2s(params.test_index[:,1].numpy()),'PredProba':n2s(params.out_test),'Label':n2s(params.test_label.numpy())})
    pd.DataFrame(prediction).to_csv(os.path.join(crt_dir, 'prediction'+Name+'.csv'))
    metrics = []
    metrics.append(calculate_metrics(params.train_label, params.out_train, 'Train'))
    metrics.append(calculate_metrics(params.val_label, params.out_val, 'Val'))
    metrics.append(calculate_metrics(params.test_label, params.out_test, 'Test'))
    pd.DataFrame(metrics).to_csv(os.path.join(crt_dir, 'metrics'+Name+'.csv'))
if __name__ == "__main__":
    os.makedirs(args.o, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    printd(f"Using device: {device}")
    if args.i1:
        printd(f"Using single dataset: prefix='{args.i1}'")
        data_train, data_valid, data_test = load_single_dataset(args.ip, args.i1, small_dim=args.sd, small_idx=args.si)
        individual_datasets = [{'train': data_train.clone(),'valid': data_valid.clone(),'test': data_test.clone(),
                                'name': f"{args.i1}",'id': 0}]
    else: 
        printd(f"Auto-detecting all datasets in {args.ip}")
        datasets_list = find_all_datasets(args.ip)
        if not datasets_list:
            raise ValueError(f"Not found any complete datasets in {args.ip}.")
        data_train, data_valid, data_test, individual_datasets = merge_datasets(datasets_list, small_idx=args.si, small_dim=args.sd)
    torch.manual_seed(args.s2)
    torch.cuda.manual_seed_all(args.s2)
    torch.autograd.set_detect_anomaly(True)
    if args.mt == 1: 
        printd("Using GCNLinkPredictShare model.")
        from my_model import GCNLinkPredictShare as ModelClass
    if args.mt == 2: 
        printd("Using MLPLinkPredictShare model.")
        from my_model import MLPLinkPredictShare as ModelClass
    model = ModelClass(args.sd, data_train.x.shape[1], args.ss, args.a).to(device)
    if args.m:
        printd(f"Loading pretrained model from {args.m}...")
        model.load_state_dict(torch.load(args.m, weights_only=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(int(args.p/2),1), T_mult=1, eta_min=args.lr*0.001)
    best_val_losses = [float("inf")] * len(individual_datasets)
    patience_counters = [0] * len(individual_datasets)
    best_epochs = [0] * len(individual_datasets)
    for step in range(args.ep):
        model.train()
        optimizer.zero_grad()
        total_loss = 0
        for i, ds in enumerate(individual_datasets):
            data_train_task = split_train(ds['train'].clone(), split_ratio=args.sr,
                                          graph_ratio=args.gr, edge_ratio=args.er)
            out = model.link_prediction(data_train_task)
            loss = loss_fn(out, data_train_task.true_label, gamma=args.g)
            total_loss += loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        model.eval()
        all_datasets_stopped = True
        for i, ds in enumerate(individual_datasets):
            ds_valid = ds['valid'].clone().to(device)
            val_loss, _ = evaluate(model, ds_valid)
            printd(f"Dataset {i} ({ds['name']}) - Step {step+1}: Validation Loss={val_loss:.4f}")
            if val_loss < best_val_losses[i]:
                best_val_losses[i] = val_loss
                patience_counters[i] = 0
                best_epochs[i] = step + 1
                model_path = os.path.join(args.o, f'{ds["name"]}_best_model.pth')
                torch.save(model.state_dict(), model_path)
            else:
                patience_counters[i] += 1
                if patience_counters[i] < args.p:
                    all_datasets_stopped = False
        if all_datasets_stopped and step > 10 * args.p:
            printd("All datasets triggered early stopping. Training stopped.")
            break
    printd("Using the best model of each dataset for testing")
    all_results = []
    for i, ds in enumerate(individual_datasets):
        model_path = os.path.join(args.o, f'{ds["name"]}_best_model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.eval()
            ds_train = ds['train'].clone().to(device)
            ds_valid = ds['valid'].clone().to(device)
            ds_test = ds['test'].clone().to(device)
            train_loss, train_metrics = evaluate(model, ds_train)
            val_loss, val_metrics = evaluate(model, ds_valid)
            test_loss, test_metrics = evaluate(model, ds_test)
            printd(f"Final Results of Dataset {i} ({ds['name']}):")
            printd(f"  Training: Loss={train_loss:.4f}")
            printd(f"  Validation: Loss={val_loss:.4f}")
            printd(f"  Testing: Loss={test_loss:.4f}")
            all_results.append({'dataset_id': i,'dataset_name': ds['name'],'train_auc': train_metrics['AUC'],'val_auc': val_metrics['AUC'],
                'test_auc': test_metrics['AUC'],'train_loss': train_loss,'val_loss': val_loss,'test_loss': test_loss,'best_epoch': best_epochs[i]})
            with torch.no_grad():
                out_train = model.link_prediction(ds_train).squeeze().detach().cpu().numpy()
                out_val = model.link_prediction(ds_valid).squeeze().detach().cpu().numpy()
                out_test = model.link_prediction(ds_test).squeeze().detach().cpu().numpy()
            params = SimpleNamespace(out_train = out_train, out_val = out_val, out_test = out_test, 
                                     train_index = ds_train.edge_label.cpu(), train_label = ds_train.true_label.cpu(), 
                                     val_index = ds_valid.edge_label.cpu(), val_label = ds_valid.true_label.cpu(),
                                     test_index = ds_test.edge_label.cpu(), test_label = ds_test.true_label.cpu())
            dataset_output_dir = os.path.join(args.o, f"{ds['name']}")
            os.makedirs(dataset_output_dir, exist_ok=True)
            generate_result(params, dataset_output_dir, num_nodes=2, Name='_maml')
        else:
            printd(f"Best model file does not exist for dataset {i} ({ds['name']})", types=1)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(args.o, 'all_datasets_summary.csv'), index=False)
    printd(f"All results saved to {args.o} and {os.path.join(args.o, 'all_datasets_summary.csv')}")

