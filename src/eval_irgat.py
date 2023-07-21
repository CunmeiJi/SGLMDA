import torch as t
from torch import nn, optim
import numpy as np
from model import *
from multiprocess import Process, Queue
import argparse
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from dgl import NID, EID
from dgl.dataloading import GraphDataLoader
from torch.utils.data import Dataset, DataLoader
from utils import parse_arguments
from utils import *#load_ogb_dataset,load_hmdd_dataset, evaluate_hits,standardize_dir
from sampler import SEALData
from model import AE, GCN, DGCNN
from logger import LightLogging
import pdb
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


np.random.seed(1337)
device = t.device('cuda:0')#('cuda' if t.cuda.is_available() else 'cpu')

def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    edge_tensor = t.LongTensor(edge_index)
    edge_tensor = edge_tensor.to(device)
    return edge_tensor


class Myloss(nn.Module):
    def __init__(self, alpha=0.4):
        super(Myloss, self).__init__()
        self.alpha = alpha

    def forward(self, one_index, zero_index, target, input):
        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(input, target)
        return (1-self.alpha)*loss_sum[one_index].sum()+self.alpha*loss_sum[zero_index].sum()

def train_help(model, optimizer, train_loader):
    for x in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        x_prd = model(x)
        loss = nn.MSELoss(reduction='sum')(x_prd,x)#.cpu()
        loss.backward()
        optimizer.step()

def train_features(features,args):
        disease_sim, miRNA_sim = features['d'], features['m']
        model_d = AE(disease_sim.size(0), args.hidden_units).to(device)
        model_m  = AE(miRNA_sim.size(0), args.hidden_units).to(device)
        optimizer_d = t.optim.Adam(model_d.parameters(), lr=args.lr, weight_decay=args.decay)
        optimizer_m = t.optim.Adam(model_m.parameters(), lr=args.lr, weight_decay=args.decay)
        train_loader_d = DataLoader(dataset=DMDataset(disease_sim),batch_size= args.batch_size, shuffle=True)
        train_loader_m = DataLoader(dataset=DMDataset(miRNA_sim),batch_size=args.batch_size, shuffle=True)

        model_d.train()
        model_m.train()
        for epoch in range(0, args.epochs):    
            train_help(model_d, optimizer_d, train_loader_d)

        for epoch in range(0, args.epochs):    
            train_help(model_m, optimizer_m, train_loader_m)

        model_d.eval()
        model_m.eval()
        feature_d = model_d.encode(disease_sim)
        feature_m = model_m.encode(miRNA_sim)
        return t.cat((feature_d, feature_m))


def evaluate(model, dataloader, device):
    model.eval()

    y_pred, y_true = [], []
    with torch.no_grad():
        for g, labels in tqdm(dataloader, ncols=100):
            g = g.to(device)
            try:
                logits = model(g, g.ndata['z'], g.ndata[NID], g.edata[EID])
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    else:
                        raise e

            y_pred.append(logits.view(-1).cpu())
            y_true.append(labels.view(-1).cpu().to(t.float))

    y_pred, y_true = t.cat(y_pred), t.cat(y_true)
    print('evaluate: ','shape of y_pred:',y_pred.shape)
    return y_pred, y_true

def eval_fold(foldIdx,auc_queue, aupr_queue, metric_queue, args):

    prefix = args.fold_dir + str(args.randseed) + '_' + str(foldIdx) + '_'
    graph, split_edge, disease_sim, miRNA_sim = load_hmdd_dataset(args.data_dir,prefix,args.sim_type,0.)#jicm, using all negative samples.

    if args.sim_type == 'none':
        features = None
        m_d_path = args.data_dir + 'm-d.csv'
        m_d = pd.read_csv(m_d_path, header=None).values
        n_miRNA,n_disease = m_d.shape[0],m_d.shape[1]
    else:
        if args.faulty:
            miRNA_sim, disease_sim = cal_faulty_sim(args)
    
        features = dict()
        disease_sim_tensor = t.FloatTensor(disease_sim)
        disease_sim_tensor = disease_sim_tensor.to(device)        
        features['d'] = disease_sim_tensor#, 'edge_index': dd_edge_index}

        miRNA_sim_tensor = t.FloatTensor(miRNA_sim)
        miRNA_sim_tensor = miRNA_sim_tensor.to(device)
        features['m'] = miRNA_sim_tensor#, 'edge_index': mm_edge_index}

        features = train_features(features,args)
        n_miRNA,n_disease = miRNA_sim.shape[0],disease_sim.shape[0]


    seal_data = SEALData(g=graph, split_edge=split_edge, n_miRNA=n_miRNA, n_disease=n_disease, hop=args.hop, neg_samples=args.neg_samples, subsample_ratio=args.subsample_ratio, use_coalesce=False, random_seed=args.randseed, prefix=args.dataset+str(args.randseed) + '_' + str(foldIdx),save_dir=args.save_dir, num_workers=args.num_workers, print_fn=print)
    num_nodes = graph.num_nodes()
    node_attribute = None #jicm
    edge_weight = None #jicm

    train_data = seal_data('train')
    test_data = seal_data('test')
    print("len of train_data: ", len(train_data), ",len of test data: ", len(test_data))
    #pdb.set_trace() 

    train_loader = GraphDataLoader(train_data, batch_size=args.batch_size)#, num_workers=args.num_workers)
    test_loader = GraphDataLoader(test_data, batch_size=args.batch_size)#, num_workers=args.num_workers)
    if  'irgat-gcn' in args.method:
        model = GCN(num_layers=args.num_layers,
                    hidden_units=args.hidden_units,
                    gcn_type=args.gcn_type,
                    pooling_type=args.pooling,
                    node_attributes=features,#node_attribute,
                    edge_weights=edge_weight,
                    node_embedding=None,
                    use_embedding=False,
                    num_nodes=num_nodes,
                    dropout=args.dropout)
    elif  'irgat-dgcnn' in args.method:
        model = DGCNN(num_layers=args.num_layers, 
                    hidden_units=args.hidden_units,
                    k=args.sort_k,
                    gcn_type=args.gcn_type,
                    node_attributes=features,
                    edge_weights=edge_weight,
                    node_embedding=None,
                    use_embedding=False,
                    num_nodes=num_nodes,
                    dropout=args.dropout)
    else:
        print('Invalid method name, please input the right name!')
        return
    
    model = model.to(device)
    parameters = model.parameters()
    optimizer = t.optim.Adam(parameters, lr=args.lr, weight_decay=args.decay)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.1, patience=3, verbose=True) 
    
    regression_crit = Myloss()
    model = model.to(device)
    regression_crit = regression_crit.to(device)
    
    model.train()
    loss_fn = BCEWithLogitsLoss()
    for epoch in range(0, args.epochs):    
        total_loss = 0
        pbar = tqdm(train_loader,ncols=100)
        for i, (g, labels) in enumerate(pbar):
            g = g.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            logits = model(g, g.ndata['z'], g.ndata[NID], g.edata[EID])
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * args.batch_size#
            pbar.set_postfix({'loss:':loss.item()})
        
        scheduler.step(total_loss)

        if epoch > 250 and epoch % 40 == 0:
            test_pred_lbl, test_pair_lbl = evaluate(model=model,dataloader=test_loader,device=device)#jicm
            auc_score, auprc_score = get_score(test_pair_lbl.detach().numpy(), test_pred_lbl.detach().numpy())
            print('foldIdx: ', foldIdx, 'auc_score: ', auc_score, 'auprc_score: ', auprc_score, 'test.shape:',test_pred_lbl.shape)
    
    test_pred_lbl, test_pair_lbl = evaluate(model=model,dataloader=test_loader,device=device)#jicm
    #auc_score, auprc_score = get_score(test_pair_lbl.detach().numpy(), test_pred_lbl.detach().numpy())
    #print('foldIdx: ', foldIdx, 'auc_score: ', auc_score, 'auprc_score: ', auprc_score, 'test.shape:',test_pred_lbl.shape)
    #auc_queue.put(auc_score)
    #aupr_queue.put(auprc_score)
    
    roc_auc,roc_pr,ap,(fpr,tpr,precision, recall) = get_score_v2(test_pair_lbl.detach().numpy(), test_pred_lbl.detach().numpy())
    metric=roc_auc,roc_pr,ap,(fpr,tpr,precision, recall)
    metric_queue.put(metric)
    
    auc_score,auprc_score =  roc_auc,  ap 
    print('foldIdx: ', foldIdx, 'auc_score: ', roc_auc, 'auprc_score: ', roc_pr,  ',ap:', ap, 'test.shape:',test_pred_lbl.shape)
    auc_queue.put(auc_score)
    aupr_queue.put(auprc_score)
    
    #save scores
    if args.save_score:
        score_save_dir = args.result_dir + str(args.randseed)
        score_save_dir = standardize_dir(score_save_dir)
        score_path = score_save_dir + str(foldIdx) + '_' + args.method + '_' + args.sim_type + '.csv'
        if args.faulty:
            score_path = score_path.replace( args.sim_type, args.sim_type + 'Faulty')
        if args.imbalanced:
            score_path = score_path.replace('.csv', '_imbalanced.csv')
        save_scores(test_pred_lbl.tolist(), test_pair_lbl.tolist(), score_path)

def eval(args):
    data_dir = standardize_dir(args.data_dir)
    fold_dir = standardize_dir(args.fold_dir)
    result_dir = standardize_dir(args.result_dir)

    numFold = args.numFold # default is 5 for 5FoldCV
    auc_queue = Queue(numFold)# for multiprocessing
    aupr_queue = Queue(numFold)
    metric_queue = Queue(numFold)
    processList = list()

    for foldIdx in range(numFold):
        eval_fold(foldIdx,auc_queue, aupr_queue, metric_queue, args)

    auc_list = [auc_queue.get() for i in range(numFold)]
    auprc_list = [aupr_queue.get() for i in range(numFold)]
    rankHG = [metric_queue.get()[-1] for i in range(numFold)]
    avg_auc = sum(auc_list)/len(auc_list)
    avg_auprc = sum(auprc_list)/len(auprc_list)

    print(args.method, ' average performance: auc:', avg_auc, 'auprc: ', avg_auprc)
    save_path = result_dir + str(args.randseed) + '_' + args.method
    
    #if args.faulty:
    #    save_path += 'Faulty'+ '.csv'
    #else:
    #    save_path +=  '_' + args.sim_type + '.csv'
    
    save_path +=  '_' + args.sim_type + '.csv'
    if args.faulty:
        save_path = save_path.replace( args.sim_type, args.sim_type + 'Faulty')

    with open(save_path, 'w') as f:
        f.write('Fold,auc,auprc\n')
        for i in range(5):
            f.write(str(i) + ',' + str(auc_list[i]) + ',' + str(auprc_list[i]) + '\n')
        f.write('Average,' + str(avg_auc) + ',' + str(avg_auprc) + '\n')

    rankHG = np.array(rankHG)
    save_path_rank = save_path.replace('csv','npy')
    np.save(save_path_rank,rankHG)

hmdd = 'HMDD2/'
def main():
    parser = argparse.ArgumentParser(description='Subgraph neural networks learning for miRNA-disease association prediction')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')#130
    parser.add_argument('--data_dir', default='../data/'+hmdd, help='dataset directory')
    parser.add_argument('--dataset', default=hmdd, help='dataset name')
    parser.add_argument('--fold_dir', default='../data/'+hmdd+'folds/', help='dataset directory')
    parser.add_argument('--result_dir', default='../data/'+hmdd+'results/', help='saved result directory')
    parser.add_argument('--method', default='irgat-dgcnn-concat', help='method')
    parser.add_argument('--save_score', default=True, help='whether to save the predicted score or not')
    parser.add_argument('--sim_type', default='functional1', help='the miRNA and disease sim, pass in "functional2" for miRNA functional + disease semantic(with phenotype info added),'
                                                                  '"none" for none miRNA and disease additional info used,'
                                                                  '"functional1" for miRNA functional and disease semantic only,'
                                                                  '"gip" for miRNA and disease GIP kernel similarity,'
                                                                  '"seq" for miRNA sequence and disease semantic,'
                                                                  '"all" for all miRNA and disease features')
    parser.add_argument('--randseed', default=112, help='the random seed')
    parser.add_argument('--numFold', default=5, help='value of K for K-foldCV, default is 5')
    parser.add_argument('--faulty', default=False, help='Faulty calculation or not')
    parser.add_argument('--imbalanced', default=False, help='Faulty calculation or not')
    
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_units', type=int, default=32)
    parser.add_argument('--gcn_type', type=str, default='gcn')
    parser.add_argument('--pooling', type=str, default='sum')
    parser.add_argument('--dropout', type=str, default=0.0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--decay', type=float, default=0.0001)
    parser.add_argument('--hop', type=int, default=1)
    parser.add_argument('--sort_k', type=int, default=20)
    parser.add_argument('--neg_samples', type=int, default=1,help='negative samples per positive sample')
    parser.add_argument('--subsample_ratio', type=float, default=0.2)#1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument('--save_dir', type=str, default='./processed')
    args = parser.parse_args()
    args.save_score = True if str(args.save_score) == 'True' else False
    args.faulty = True if str(args.faulty) == 'True' else False
    args.imbalanced = True if str(args.imbalanced) == 'True' else False

    randseeds = [123]#[123, 456,789, 101, 112] #jicm
    data_dirs = ['../data/'+hmdd]
    fold_dirs = ['../data/'+hmdd+'folds/']
    result_dirs = ['../data/'+hmdd+'results']
    methods = ['irgat-dgcnn']#['irgat-dgcnn']#,'irgat-gcn']
    sim_types = ['functional2']#'none','functional1','gip','seq','all']#'functional2'
    faulties = [ False]
    for randseed in randseeds:
        args.randseed = randseed
        for method in methods:
            args.method = method
            for idata, data_dir in enumerate(data_dirs):
                args.data_dir = data_dir
                args.fold_dir = fold_dirs[idata]
                print('In ', data_dir, fold_dirs[idata])
                args.result_dir = result_dirs[idata]+'_subsample_'+str(args.subsample_ratio)+'/'
                args.save_score = True

                #args.faulty = True
                #eval(args)
                #args.faulty = False
                
                for simtype in sim_types:
                    args.sim_type = simtype
                    if simtype == 'none':
                        args.faulty = False # False and True has the same results.
                        eval(args)
                    else:
                        for faulty in faulties:
                            args.faulty = faulty
                            eval(args)

if __name__ == "__main__":
    main()


