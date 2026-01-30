import argparse,sys
def parse_args():
    parser = argparse.ArgumentParser(description='MAML Meta Learning for Link Prediction')
    parser.add_argument('-i', required=True, type=str, metavar='PATH', help='Input directory')
    parser.add_argument('-o', required=True, type=str, metavar='PATH', help='Output directory')
    parser.add_argument('-s1', default=86, type=int, metavar='INT', help='Seed for numpy (default: 86)')
    parser.add_argument('-t', default=2, type=int, metavar='INT', help='Number of task per step (default: 2)')
    parser.add_argument('-sn', type=str, metavar='INT,...', help='Dataset index for test, seperated by ",", will override -s1 and -t')
    parser.add_argument('-s2', default=86, type=int, metavar='INT', help='Seed for torch (default: 86)')
    parser.add_argument('-ss', default='64,32,16,8', type=str, metavar='INT,INT,INT,INT', help='Number of nodes (default: 64,32,16,8)')
    parser.add_argument('-a', default=16, type=int, metavar='INT', help='Adaptor dim (default: 16)')
    parser.add_argument('-ep', default=1000, type=int, metavar='INT', help='Number of step for training (default: 1000)')
    parser.add_argument('-prop', default=0.75, type=float, metavar='FLOAT', help='Proportion of nodes to sample (default: 0.75)')
    parser.add_argument('-sr', default=0.3, type=float, metavar='FLOAT', help='Support set ratio (default: 0.3)')
    parser.add_argument('-gr', default=1, type=float, metavar='FLOAT', help='Training graph set ratio (default: 1=use all)')
    parser.add_argument('-er', default=1, type=float, metavar='FLOAT', help='Training edge set ratio (default: 1=use all)')
    parser.add_argument('-m', default=0, type=int, choices=[0,1,2,3], help='Mark for inner update params (default: 0)')
    parser.add_argument('-ilr', default=0.0001, type=float, metavar='FLOAT', help='Inner loop learning rate (default: 0.0001)')
    parser.add_argument('-iep', default=5, type=int, metavar='INT', help='Number of inner update steps (default: 5)')
    parser.add_argument('-lr', default=0.0001, type=float, metavar='FLOAT', help='Learning rate (default: 0.0001)')
    parser.add_argument('-g', default=4, type=int, metavar='INT', help='Gamma for focal loss (default: 4)')
    parser.add_argument('-p', default=50, type=int, metavar='INT', help='Number of patience (default: 50)')
    args = parser.parse_args()
    args.ss = list(map(int, args.ss.split(',')))
    if len(args.ss) != 4:
        parser.print_help(sys.stderr)   # 输出到标准错误流 
        printd(f"\nERROR: length of nodes is not enough") 
        sys.exit(1)
    return args
if __name__ == "__main__": args = parse_args()
import os,torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.my_train import calculate_metrics,find_file_groups,load_csv_data,prepare_task_data,printd
from src.my_loss import focal_loss as loss_fn
from src.my_model import GCNLinkPredictShare as ModelClass
from torch_geometric.data import Data
import numpy as np
def split_train(data0, split_ratio=0.3, graph_ratio=1.0, edge_ratio=1.0):
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
    query_data = Data(x=data.x, edge_index=edge_index, small_idx=data.small_idx, types=data.types, 
                      edge_label=torch.cat([part_edge[perm[support_size:]], part_edge[perm[support_size:]].flip(1)], dim=0),
                      true_label=torch.cat([part_true[perm[support_size:]], part_true[perm[support_size:]]]))
    return support_data.to(device), query_data.to(device)
def get_inner_update_params(model, mark=0):
    mark_param_map = {0: ["hete0", "hete1", "conv1", "conv2", "lin0", "lin1"],
        1: ["conv1", "conv2"], 
        2: ["hete0", "hete1"], 
        3: ["lin0", "lin1"]} 
    target_prefixes = mark_param_map.get(mark,  mark_param_map[0]) 
    update_params = [] 
    for name, _ in model.named_parameters(): 
        if "adapt" in name: 
            update_params.append(name)  
            continue 
        for prefix in target_prefixes: 
            if name.startswith(prefix):  
                update_params.append(name)  
                break
    return list(set(update_params))
def adapt_task(model, support_data, mark=0, inner_lr=0.01, inner_steps=5, training=True):
    device = next(model.parameters()).device
    if support_data.x.device != device:
        support_data = support_data.to(device) 
    param_names = get_inner_update_params(model, mark=mark)
    fast_weights = {name: model.state_dict()[name].clone().to(device).requires_grad_(True) for name in param_names}
    prev_mode = model.training
    model.train(training)
    original_state = model.state_dict().copy()
    for step in range(inner_steps):
        with torch.no_grad():
            current_state = model.state_dict() 
            for name in param_names:
                current_state[name] = fast_weights[name]
            model.load_state_dict(current_state,  strict=False)
        pred = model.link_prediction(support_data) 
        labels = support_data.true_label.to(pred.device) 
        loss = loss_fn(pred, labels, gamma=args.g)
        for param in fast_weights.values(): 
            if param.grad is not None:
                param.grad.zero_() 
        grads = torch.autograd.grad(loss, list(fast_weights.values()), create_graph=True, retain_graph=True, allow_unused=True)
        total_norm = 0.0 
        for grad in grads:
            if grad is not None:
                total_norm += grad.norm(2).item()  ** 2 
        total_norm = total_norm ** 0.5 
        clip_coef = max(1.0, total_norm / 5.0)
        updated_params = {}
        for (name, param), grad in zip(fast_weights.items(),  grads):
            if grad is not None:
                clipped_grad = grad / clip_coef
                updated_param = param - inner_lr * clipped_grad
                updated_params[name] = updated_param.detach().requires_grad_(True) 
            else:
                updated_params[name] = param
        fast_weights.clear() 
        fast_weights.update(updated_params) 
    with torch.no_grad():
        model.load_state_dict(original_state,  strict=False)
    model.train(prev_mode) 
    return {k: v.to(device) for k, v in fast_weights.items()}
def evaluate_task(model, query_data, args, training=True): 
    original_state = {name: param.clone() for name, param in model.named_parameters()} 
    if training: 
        fast_weights = adapt_task0(model, query_data, mark=args.m, inner_lr=args.ilr,  inner_steps=args.iep,  training=training)
        for name, param in fast_weights.items(): 
            setattr(model, name, param)
    with torch.no_grad() if not training else torch.enable_grad(): 
        query_logits = model.link_prediction(query_data) 
        query_loss = loss_fn(query_logits, query_data.true_label, gamma=args.g)
    query_pred = torch.sigmoid(query_logits).detach().cpu()  
    query_true = query_data.true_label.detach().cpu() 
    return query_loss, query_pred, query_true, original_state
def run_meta_batch(model, data, args, mode="train"):
    return evaluate_task(model, data, args, training=(mode=="train"))
def adapt_task0(model, support_data, mark=0, inner_lr=0.01, inner_steps=5, training=True):
    device = next(model.parameters()).device 
    if support_data.x.device != device:
        support_data = support_data.to(device)  
    param_names = get_inner_update_params(model, mark=mark)
    fast_params = [param.clone() for name, param in model.named_parameters()  if name in param_names]
    original_params = [param for name, param in model.named_parameters()  if name in param_names]
    prev_mode = model.training  
    model.train(training) 
    for step in range(inner_steps):
        for i, name in enumerate(param_names):
            original_params[i].data.copy_(fast_params[i]) 
        pred = model.link_prediction(support_data)
        labels = support_data.true_label.to(pred.device)  
        loss = loss_fn(pred, labels, gamma=args.g)
        grads = torch.autograd.grad(loss, fast_params, create_graph=True, retain_graph=True, allow_unused=True)
        total_norm = sum(g.norm(2).item()**2  for g in grads if g is not None)**0.5 
        clip_coef = max(1.0, total_norm / 5.0)
        with torch.no_grad(): 
            for i, grad in enumerate(grads):
                if grad is not None:
                    clipped_grad = grad / clip_coef 
                    fast_params[i] = fast_params[i] - inner_lr * clipped_grad 
                    fast_params[i].requires_grad_(True)
    model.train(prev_mode) 
    return {name: param for name, param in zip(param_names, fast_params)}
def evaluate_task0(model, support_data, query_data, args, training=True):
    original_state = {name: param.clone()  for name, param in model.named_parameters()} 
    if training: 
        fast_weights = adapt_task0(model, support_data, mark=args.m, inner_lr=args.ilr,  inner_steps=args.iep,  training=training)
        for name, param in fast_weights.items(): 
            setattr(model, name, param)
    with torch.no_grad() if not training else torch.enable_grad(): 
        query_logits = model.link_prediction(query_data) 
        query_loss = loss_fn(query_logits, query_data.true_label, gamma=args.g)
    for name, param in original_state.items(): 
        setattr(model, name, param)
    query_pred = torch.sigmoid(query_logits).detach().cpu()  
    query_true = query_data.true_label.detach().cpu()
    return query_loss, query_pred, query_true 
def run_meta_batch0(model, data, args, mode="train"):
    if mode == "train":
        support_data, query_data = split_train(data, args.sr, args.gr, args.er)
    else:
        support_data = None; query_data = data
    return evaluate_task0(model, support_data, query_data, args, training=(mode=="train"))
if __name__ == "__main__":
    matched_groups = find_file_groups(args.i)
    if not matched_groups:
        printd(f"No matched data files found in directory {args.d}") 
        sys.exit(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    printd(f"Using device: {device}")
    np.random.seed(args.s1)
    torch.manual_seed(args.s2)
    torch.cuda.manual_seed_all(args.s2)
    torch.autograd.set_detect_anomaly(True) 
    group = matched_groups[0]
    data = load_csv_data(group['bact'], group['phage'], group['edge'])
    model = ModelClass(min(data.x1.shape[1], data.x2.shape[1]), max(data.x1.shape[1], data.x2.shape[1]), args.ss, args.a).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(int(args.p/2),1), T_mult=1, eta_min=args.lr*0.001)
    if args.sn:
        valid_dirs = np.array(list(map(int,args.sn.strip().split(','))))
        sampled_dirs = np.array([i for i in range(len(matched_groups)) if i not in valid_dirs])
    else:
        all_dirs = np.random.choice(len(matched_groups), size=len(matched_groups), replace=False)
        sampled_dirs = all_dirs[:args.t]
        valid_dirs = all_dirs[args.t:]
    all_train_data = []; all_val_data = []
    for g in sampled_dirs:
        train_d, valid_d, test_d = prepare_task_data(matched_groups[g], args, device=device)
        all_train_data.append({'train_data': train_d, 'valid_data': valid_d, 'test_data': test_d, 'genus': matched_groups[g]['wildcard']})
    for g in valid_dirs:
        train_d, valid_d, test_d = prepare_task_data(matched_groups[g], args, device=device)
        all_val_data.append({'train_data': train_d, 'valid_data': valid_d, 'test_data': test_d, 'genus': matched_groups[g]['wildcard']})
    best_valid_loss = float('inf')
    patience_counter = 0
    for meta_step in range(args.ep):
        model.train()
        total_query_loss = 0.0
        total_query_pred = []; total_query_true = []
        for g in all_train_data:
            query_loss, query_pred, query_true = run_meta_batch0(model, g['train_data'], args, mode="train")
            total_query_loss += query_loss
            total_query_pred.append(query_pred)
            total_query_true.append(query_true)
        optimizer.zero_grad()
        total_query_loss.backward() 
        optimizer.step()
        scheduler.step()
        printd(f"Meta Step {meta_step+1}/{args.ep}, Query Loss: {total_query_loss.item():.4f}")
        if True:
            model.eval()
            total_valid_loss = 0.0
            total_valid_pred = []; total_valid_true = []
            for g in all_train_data:
                valid_loss, valid_pred, valid_true = run_meta_batch0(model, g['valid_data'], args, mode="valid")
                total_valid_loss += valid_loss
                total_valid_pred.append(valid_pred)
                total_valid_true.append(valid_true)
            printd(f"Validation Loss: {total_valid_loss.item():.4f}")
            if total_valid_loss.item() < best_valid_loss:
                best_valid_loss = total_valid_loss.item()
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(args.o, 'best_maml_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= args.p and meta_step > 10 * args.p:
                    printd("Early stopping triggered.")
                    break
    printd("Loading best model for testing...")
    model.load_state_dict(torch.load(os.path.join(args.o, 'MAML_best_model.pth'), weights_only=True))
    model.eval()
    for g in [*all_train_data, *all_val_data]:
        printd(f"Testing on dataset: {g['genus']}")
        for h in ['train','valid','test']:
            if h == 'train':
                tloss, tpred, ttrue, tstate = run_meta_batch(model, g[h+'_data'], args, mode=h)
            else:
                tloss, tpred, ttrue, _ = run_meta_batch(model, g[h+'_data'], args, mode=h)
            qrtmat = calculate_metrics(ttrue.numpy(), tpred.numpy(), h)
            printd(f"{h.capitalize()} Loss: {tloss.item():.4f}, Metrics: {qrtmat}")
            if h == 'test':
                for name, param in tstate.items(): 
                    setattr(model, name, param)