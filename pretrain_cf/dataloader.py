from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
import random
from torch.nn.utils.rnn import pad_sequence

class POI_Dataset(Dataset):
    def __init__(self, train_df, config, device) -> None:
        super(POI_Dataset, self).__init__()
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.device = device
        self.path = config['matrix_path']
        self.neg_num = 10

        self.get_info(train_df)
    
    def get_info(self, df:pd.DataFrame):
        
        self.n_users, self.m_items = pd.unique(df['uid']).shape[0], pd.unique(df['loc']).shape[0]

        self.user_pos = [[] for _ in range(self.n_users)]
        create_sparse_row = []
        create_sparse_col = []
        for u_id, item in df.groupby('uid'):
            pos_item = pd.unique(item['loc']).tolist()
            self.user_pos[u_id] = torch.tensor(pos_item)
            create_sparse_row.extend([u_id]*len(pos_item))
            create_sparse_col.extend(pos_item)
        self.user_pos = pad_sequence(self.user_pos, batch_first=True, padding_value=-1)
        
        create_sparse_row = np.array(create_sparse_row)
        create_sparse_col = np.array(create_sparse_col)
        self.ui_list = np.stack([create_sparse_row, create_sparse_col], axis=1)
        random.shuffle(self.ui_list)
        self.length = len(self.ui_list)

        self.getSparseGraph(create_sparse_row, create_sparse_col)
        # (users,items), bipartite graph
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        uid, positem = self.ui_list[index]
        
        posForUser = self.user_pos[uid]
        neglist = []
        while len(neglist) < self.neg_num:
            negitem = np.random.randint(0, self.m_items, 2*self.neg_num)
            neglist.extend([i for i in negitem if i not in posForUser])
        neglist = neglist[:self.neg_num]

        return uid, positem, negitem


    def getSparseGraph(self, row, col):
        print("loading adjacency matrix")
        try:
            pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
            print("successfully loaded...")
            norm_adj = pre_adj_mat
        except :
            print("generating adjacency matrix")
            adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = sp.csr_matrix((np.ones(len(row)), (row, col)),
                                      shape=(self.n_users, self.m_items)).tolil()
            adj_mat[:self.n_users, self.n_users:] = R
            adj_mat[self.n_users:, :self.n_users] = R.T
            adj_mat = adj_mat.todok()
            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
            
            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            
            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

        if self.split == True:
            self.Graph = self._split_A_hat(norm_adj)
            print("done split matrix")
        else:
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(self.device)
            print("don't split the matrix")

        return self.Graph
    
    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self.device))
        return A_fold
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

class Test_Dataset(Dataset):
    def __init__(self, df, train_user_pos) -> None:
        super(Test_Dataset, self).__init__()
        self.train_user_pos = train_user_pos
        
        self.user_list = df['uid'].tolist()
        random.shuffle(self.user_list)
        self.length = len(self.user_list)
        self.user_pos = [[] for _ in range(max(df['uid'])+1)]
        for u_id, item in df.groupby('uid'):
            pos_item = pd.unique(item['loc'])
            self.user_pos[u_id] = torch.tensor(pos_item.tolist())
        self.user_pos = pad_sequence(self.user_pos, batch_first=True, padding_value=-1)

    def __getitem__(self, index):
        uid = self.user_list[index]
        
        return uid, self.train_user_pos[uid], self.user_pos[uid]
        
    def __len__(self):
        return self.length