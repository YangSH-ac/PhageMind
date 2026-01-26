import argparse,sys
def parse_args():
    parser = argparse.ArgumentParser(description='Save data for data spliting')
    parser.add_argument('-i', required=True, type=str, metavar='PATH', help='Input directory')
    parser.add_argument('-o', required=True, type=str, metavar='PATH', help='Output directory')
    parser.add_argument('-s1', default=86, type=int, metavar='INT', help='Seed for numpy (default: 86)')
    parser.add_argument('-s2', default=86, type=int, metavar='INT', help='Seed for torch (default: 86)')
    parser.add_argument('-t', default=2, type=int, metavar='INT', help='Number of task per step (default: 2)')
    parser.add_argument('-sn', type=str, metavar='INT,...', help='Dataset index for test, seperated by ",", will override -s1 and -t')
    parser.add_argument('-prop', default=0.75, type=float, metavar='FLOAT', help='Proportion of nodes to sample (default: 0.75)')
    args = parser.parse_args()
    return args
if __name__ == "__main__": 
    args = parse_args()
import os,torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.my_train_related import find_file_groups,prepare_task_data,save_graph_data,printd
import numpy as np
if __name__ == "__main__":
    matched_groups = find_file_groups(args.i)
    if not matched_groups:
        printd(f"No matched data files found in directory {args.i}") 
        sys.exit(1)
    device = 'cpu'
    np.random.seed(args.s1)
    torch.manual_seed(args.s2)
    if args.sn:
        valid_dirs = np.array(list(map(int,args.sn.strip().split(','))))
        sampled_dirs = np.array([i for i in range(len(matched_groups)) if i not in valid_dirs])
    else:
        all_dirs = np.random.choice(len(matched_groups), size=len(matched_groups), replace=False)
        sampled_dirs = all_dirs[:args.t]
        valid_dirs = all_dirs[args.t:]
    all_train_data = []; all_val_data = []
    for g in sampled_dirs:
        train_d, valid_d, test_d = prepare_task_data(matched_groups[g], args, device=device, mode="train")
        all_train_data.append({'train_data': train_d, 'valid_data': valid_d, 'test_data': test_d, 'genus': matched_groups[g]['wildcard']})
    for g in valid_dirs:
        train_d, valid_d, test_d = prepare_task_data(matched_groups[g], args, device=device, mode="valid")
        all_val_data.append({'train_data': train_d, 'valid_data': valid_d, 'test_data': test_d, 'genus': matched_groups[g]['wildcard']})
    output_dir = args.o
    os.makedirs(output_dir,  exist_ok=True)
    for task in all_train_data:
        genus = task['genus']
        for mode in ['train', 'valid', 'test']:
            data_key = f"{mode}_data"
            data_obj = task[data_key]
            save_graph_data(data_obj, genus, mode+'_t', output_dir)
    for task in all_val_data:
        genus = task['genus']
        for mode in ['train', 'valid', 'test']:
            data_key = f"{mode}_data"
            data_obj = task[data_key]
            save_graph_data(data_obj, genus, mode+'_v', output_dir)
    printd(f"Finished, output directory: {output_dir}")