import argparse
from scipy.sparse.csgraph import shortest_path
import numpy as np
import pandas as pd
import torch
import dgl
#from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from sklearn import metrics
import os
from ogb.linkproppred import Evaluator
from hmdd_dataset_dgl import DglLinkPropPredDataset
from torch.utils.data import Dataset
import pdb

import sys
sys.path.append('../')
from data.preparation.GIP import *
from data.preparation.miRNA_sim import *
def parse_arguments():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description='SEAL')
    parser.add_argument('--dataset', type=str, default='HMDD3.2')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--hop', type=int, default=1)
    parser.add_argument('--model', type=str, default='dgcnn')
    parser.add_argument('--gcn_type', type=str, default='gcn')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_units', type=int, default=32)
    parser.add_argument('--sort_k', type=int, default=30)
    parser.add_argument('--pooling', type=str, default='sum')
    parser.add_argument('--dropout', type=str, default=0.5)
    parser.add_argument('--hits_k', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--neg_samples', type=int, default=1)
    parser.add_argument('--subsample_ratio', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument('--save_dir', type=str, default='./processed')
    args = parser.parse_args()

    return args

class DMDataset(Dataset):
    """
    GCN Model
    Attributes:
        matrix(tensor): similarity of mirna, or disease
    """
    def __init__(self, matrix):
        self.data = matrix
        self.len  = self.data.size(0)
        
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len
    

def load_ogb_dataset(dataset):
    """
    Load OGB dataset
    Args:
        dataset(str): name of dataset (ogbl-collab, ogbl-ddi, ogbl-citation)

    Returns:
        graph(DGLGraph): graph
        split_edge(dict): split edge

    """
    dataset = DglLinkPropPredDataset(name=dataset)
    split_edge = dataset.get_edge_split()
    graph = dataset[0]

    return graph, split_edge

def load_hmdd_dataset(dataset, prefix, sim_type = 'gip', neg_rate=0.):
    """
    Load HMDD dataset
    Args:
        dataset(str): name of dataset (ogbl-collab, ogbl-ddi, ogbl-citation)
        prefix: directory
        sim_type: similarity of mirna and disease used of the k-fold
        neg_rate: negative edges used in testing, default is 0, stands for all negative samples.
    Returns:
        graph(DGLGraph): graph
        split_edge(dict): split edge

    """
    dataset = DglLinkPropPredDataset(name=dataset)
    split_edge = dataset.get_edge_split(prefix,neg_rate)
    graph = dataset[0]

    disease_sim_path = None
    miRNA_sim_path   = None
    if sim_type == 'gip':
        miRNA_sim_path = prefix + 'mirna_gip_sim.csv'
        disease_sim_path = prefix + 'disease_gip_sim.csv'
    elif sim_type == 'seq':
        miRNA_sim_path = os.path.dirname(prefix) +  '/mirna_seq_sim.csv'
        disease_sim_path = prefix + 'disease_sem2_sim.csv'
    elif sim_type == 'functional2':
        miRNA_sim_path = prefix + 'mirna_fun2_sim.csv'
        disease_sim_path = prefix + 'disease_sem2_sim.csv'
    elif sim_type == 'functional1':
        miRNA_sim_path = prefix + 'mirna_fun_sim.csv'
        disease_sim_path = prefix + 'disease_sem1_sim.csv'

    #print(miRNA_sim_path)   
    #pdb.set_trace()
    if sim_type == 'none':
        disease_sim = None
        miRNA_sim   = None
    elif sim_type == 'all':
        miRNA_sim_path = prefix + 'mirna_gip_sim.csv'
        miRNA_sim1 = pd.read_csv(miRNA_sim_path, header=None).values
        miRNA_sim_path = prefix + 'mirna_fun2_sim.csv'
        miRNA_sim2 = pd.read_csv(miRNA_sim_path, header=None).values
        miRNA_sim_path = os.path.dirname(prefix) +  '/mirna_seq_sim.csv'
        miRNA_sim3 = pd.read_csv(miRNA_sim_path, header=None).values
        
        disease_sim_path = prefix + 'disease_gip_sim.csv'
        disease_sim1 = pd.read_csv(disease_sim_path, header=None).values
        disease_sim_path = prefix + 'disease_sem2_sim.csv'
        disease_sim2 = pd.read_csv(disease_sim_path, header=None).values

        disease_sim = (disease_sim1 + disease_sim2)/2
        miRNA_sim = (miRNA_sim1 + miRNA_sim2 + miRNA_sim3)/3
    else:
        disease_sim = pd.read_csv(disease_sim_path, header=None).values
        miRNA_sim = pd.read_csv(miRNA_sim_path, header=None).values
    
    return graph, split_edge, disease_sim, miRNA_sim

def load_hmdd_dataset_case_study(dataset, prefix, neg_rate=0.):
    """
    Load HMDD dataset
    Args:
        dataset(str): name of dataset (ogbl-collab, ogbl-ddi, ogbl-citation)
        prefix: directory
        neg_rate: negative edges used in testing, default is 0, stands for all negative samples.
    Returns:
        graph(DGLGraph): graph
        split_edge(dict): split edge

    """
    dataset = DglLinkPropPredDataset(name=dataset)
    split_edge = dataset.get_edge_split_case_study(prefix,neg_rate)
    graph = dataset[0]
 
    return graph, split_edge

def assoc_list_to_adj(n_disease, n_miRNA, assoc_list, assoc_lbl):
    adj = np.zeros((n_miRNA, n_disease))
    zero_index = list()
    one_index = list()

    for pair, lbl in zip(assoc_list, assoc_lbl):
        if type(lbl) is list:
            lbl = lbl[0]
        adj[pair[0], pair[1]] = lbl
        if lbl == 0:
            zero_index.append(pair)
        else:
            one_index.append(pair)
    return adj, zero_index, one_index

def get_score(y_test, preds):
    auc_score = metrics.roc_auc_score(y_test, preds, average='micro')

    aupr_score = metrics.average_precision_score(y_test, preds, average='micro')
    return auc_score, aupr_score

def get_score_v2(y_true,y_pred):
    fpr,tpr,thresholds = metrics.roc_curve(y_true,y_pred)#sample_weight
    roc_auc=metrics.auc(fpr,tpr)

    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)#, sample_weight=weights)#sample_weight=
    roc_pr=metrics.auc(recall, precision)
    ap = metrics.average_precision_score(y_true,y_pred)#AP
    
    #acc_score = metrics.accuracy_score(y_true,y_pred)
    #reca_score = metrics.recall_score(y_true,y_pred)#y_pred.gt(0.5).int()
    #f_score = metrics.f1_score(y_true,y_pred)
    #metric = (roc_auc,roc_pr,acc_score,ap,reca_score,f_score,(fpr,tpr,precision, recall))

    metric = (roc_auc,roc_pr,ap,(fpr,tpr,precision, recall))
    return metric

def save_scores(y_preds, y_true, path):
    join_list = [[pred,target] for pred, target in zip(y_preds, y_true)]
    df = pd.DataFrame(np.array(join_list))
    df.to_csv(path, header=False, index=False)

def standardize_dir(dir):
    res_dir = dir
    if not res_dir.endswith('/') and not res_dir.endswith('\\'):
        res_dir += '/'
    
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    return res_dir

def cal_faulty_sim(args):
    data_dir = standardize_dir(args.data_dir)
    adj = pd.read_csv(data_dir + 'm-d.csv', header=None).values
    onto_disease_sim_path = data_dir + 'disease_sem_sim.csv'
    onto_disease_sim_path2 = data_dir + 'disease_sem2_sim.csv'
    disease_not_found_path = data_dir + 'disease_not_found_list.txt'
    miRNA_seq_sim_path = data_dir + 'mirna_seq_sim.csv'

    if args.sim_type == 'gip':
        miRNA_gip, disease_gip = calculate_gip(adj)
        return miRNA_gip, disease_gip

    if args.sim_type == 'seq':
        disease_sim, _ = cal_miRNA_func_sim(adj, onto_disease_sim_path2, disease_not_found_path, onto_disease_sim_path)
        miRNA_sim = pd.read_csv(miRNA_seq_sim_path, header=None).values
        return miRNA_sim, disease_sim

    if args.sim_type == 'functional1':
        disease_sim, miRNA_sim = cal_miRNA_func_sim(adj, onto_disease_sim_path, disease_not_found_path, onto_disease_sim_path)
        return miRNA_sim, disease_sim

    if args.sim_type == 'functional2':
        disease_sim, miRNA_sim = cal_miRNA_func_sim(adj, onto_disease_sim_path2, disease_not_found_path, onto_disease_sim_path)
        return miRNA_sim, disease_sim
    
    if args.sim_type == 'all':
        miRNA_gip, disease_gip = calculate_gip(adj)
        miRNA_sim_seq = pd.read_csv(miRNA_seq_sim_path, header=None).values
        disease_sim_fun, miRNA_sim_fun = cal_miRNA_func_sim(adj, onto_disease_sim_path2, disease_not_found_path, onto_disease_sim_path)
        miRNA_sim, disease_sim = (miRNA_gip+miRNA_sim_seq+miRNA_sim_fun)/3, (disease_gip+disease_sim_fun)/2
        return miRNA_sim, disease_sim

def coalesce_graph(graph, aggr_type='sum', copy_data=False):
    """
    Coalesce multi-edge graph
    Args:
        graph(DGLGraph): graph
        aggr_type(str): type of aggregator for multi edge weights
        copy_data(bool): if copy ndata and edata in new graph

    Returns:
        graph(DGLGraph): graph


    """
    src, dst = graph.edges()
    graph_df = pd.DataFrame({'src': src, 'dst': dst})
    graph_df['edge_weight'] = graph.edata['edge_weight'].numpy()

    if aggr_type == 'sum':
        tmp = graph_df.groupby(['src', 'dst'])['edge_weight'].sum().reset_index()
    elif aggr_type == 'mean':
        tmp = graph_df.groupby(['src', 'dst'])['edge_weight'].mean().reset_index()
    else:
        raise ValueError("aggr type error")

    if copy_data:
        graph = dgl.to_simple(graph, copy_ndata=True, copy_edata=True)
    else:
        graph = dgl.to_simple(graph)

    src, dst = graph.edges()
    graph_df = pd.DataFrame({'src': src, 'dst': dst})
    graph_df = pd.merge(graph_df, tmp, how='left', on=['src', 'dst'])
    graph.edata['edge_weight'] = torch.from_numpy(graph_df['edge_weight'].values).unsqueeze(1)

    graph.edata.pop('count')
    return graph


def drnl_node_labeling(subgraph, src, dst):
    """
    Double Radius Node labeling
    d = r(i,u)+r(i,v)
    label = 1+ min(r(i,u),r(i,v))+ (d//2)*(d//2+d%2-1)
    Isolated nodes in subgraph will be set as zero.
    Extreme large graph may cause memory error.

    Args:
        subgraph(DGLGraph): The graph
        src(int): node id of one of src node in new subgraph
        dst(int): node id of one of dst node in new subgraph
    Returns:
        z(Tensor): node labeling tensor
    """
    adj = subgraph.adj().to_dense().numpy()
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx] #adj except idx row and col

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx] #adj except idx row and col

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)


def evaluate_hits(name, pos_pred, neg_pred, K):
    """
    Compute hits
    Args:
        name(str): name of dataset
        pos_pred(Tensor): predict value of positive edges
        neg_pred(Tensor): predict value of negative edges
        K(int): num of hits

    Returns:
        hits(float): score of hits


    """
    evaluator = Evaluator(name)
    evaluator.K = K
    hits = evaluator.eval({
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    })[f'hits@{K}']

    return hits
