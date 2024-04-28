import torch
import numpy as np
import random
from torch import nn, optim
from lightgcn import LightGCN
from dataloader import POI_Dataset, Test_Dataset
import pandas as pd
import os
import multiprocessing
import utils
from torch.utils.data import DataLoader

CORES = multiprocessing.cpu_count() // 2

def BPR_train_original(device, dataloader, Recmodel, config):
    Recmodel.train()
    weight_decay = config['decay']
    lr = config['lr']
    opt = optim.Adam(Recmodel.parameters(), lr=lr)

    aver_loss = 0.

    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(dataloader):
        loss, reg_loss = Recmodel.bpr_loss(batch_users.to(device), batch_pos.to(device), batch_neg.to(device))
        reg_loss = reg_loss*weight_decay
        loss = loss + reg_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        aver_loss += loss.cpu().item()

    aver_loss = aver_loss / len(dataloader)
    
    return f"loss{aver_loss:.3f}"

def Test(dataloader, Recmodel, device, config, multicore=0):
    u_batch_size = config['test_u_batch_size']

    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(config['topks'])
    
    results = {'precision': np.zeros(len(config['topks'])),
               'recall': np.zeros(len(config['topks'])),
               'ndcg': np.zeros(len(config['topks']))}
    pre, recall, ndcg = [[0]]*len(config['topks']), [[0]]*len(config['topks']), [[0]]*len(config['topks'])
    with torch.no_grad():

        for batch_users, Train_Known, groundTrue in dataloader:
            batch_users_gpu = batch_users.to(device)  # bs, pad_len
            True_gt = []

            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(Train_Known):
                items = items[items>-1]
                True_gt.append(items)
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            
            r = np.array([
                list(map(lambda x: x in gt, predictTopK)) for gt, predictTopK in zip(groundTrue, rating_K.cpu())
            ]).astype('float')
            for i, k in enumerate(config['topks']):
                ret = utils.RecallPrecision_ATk(groundTrue, r, k)
                pre[i].append(ret['precision'])
                recall[i].append(ret['recall'])
                ndcg[i].append(utils.NDCGatK_r(groundTrue,r,k))
            
        all_num = len(dataloader) * u_batch_size

        for i in range(len(config['topks'])):
            results['recall'][i] = sum(recall[i])/float(all_num)
            results['precision'][i] = sum(pre[i])/float(all_num)
            results['ndcg'][i] = sum(ndcg[i])/float(all_num)
        
        print(results)
        return results


if __name__ == "__main__":
    config = utils.parse_args()
    config['A_split'] = False
    config['bigdata'] = False
    config['matrix_path'] = os.path.join(config['matrix_path'], config['dataset'])
    config['topks'] = eval(config['topks'])
    epoch_num = config['epochs']
    device = torch.device(0)

    train_df = pd.read_csv(os.path.join(config['matrix_path'], 'train.csv'), header=0)
    test_df = pd.read_csv(os.path.join(config['matrix_path'], 'test.csv'), header=0)
    dataset = POI_Dataset(train_df, config, device)
    train_loader = DataLoader(dataset, config['bpr_batch_size'], shuffle=True)
    test_loader = DataLoader(Test_Dataset(test_df, dataset.user_pos), config['test_u_batch_size'])

    Recmodel = LightGCN(config, dataset).to(device)

    file = f"lgn-{config['lightGCN_n_layers']}-{config['latent_dim_rec']}.pth.tar"
    weight_file = os.path.join(config['matrix_path'], f'cf_emb/{file}')
    if config['load']:
        try:
            Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
            print(f"loaded model weights from {weight_file}")
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")

    best_rlt = 0
    last_rlt = 0
    patience_num = 0
    for epoch in range(1, 1+epoch_num):
        if epoch % 10 == 0:
            print("[TEST]")
            test_rlt = Test(test_loader, Recmodel, device, config, config['multicore'])
            if best_rlt < test_rlt['ndcg'][-1]:
                best_rlt = test_rlt['ndcg'][-1]
                torch.save(Recmodel.state_dict(), weight_file)
                patience_num = 0
            elif test_rlt['ndcg'][-1] > last_rlt:
                patience_num += 1
            else:
                patience_num = 0
            if patience_num >= 10:
                print('early stoped.')
                break
        output_information = BPR_train_original(device, train_loader, Recmodel, config)
        print(f'EPOCH[{epoch}/{epoch_num}] {output_information}')
    print('Best result', best_rlt) 