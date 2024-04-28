import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def split_it(df:pd.DataFrame, test_prop, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    user_num, item_num = pd.unique(df['uid']).shape[0], pd.unique(df['loc']).shape[0]
    user_map = dict(zip(pd.unique(df['uid']), range(user_num)))
    item_map = dict(zip(pd.unique(df['loc']), range(item_num)))
    np.save(os.path.join(save_path, 'user_map.npy'), user_map, allow_pickle=True)
    np.save(os.path.join(save_path, 'item_map.npy'), item_map, allow_pickle=True)
    df['uid'] = df['uid'].map(user_map)
    df['loc'] = df['loc'].map(item_map)

    assert max(df['loc']) < item_num

    train_df = pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(columns=df.columns)
    
    for uid, item in tqdm(df.groupby('uid')):
        length = len(item['loc'].tolist())
        train_num = length - int(length * test_prop)

        if not len(train_df):
            train_df = item.iloc[: train_num].reset_index()
        else:
            train_df = pd.concat([train_df, item.iloc[: train_num]], ignore_index=True)
        
        if not len(test_df):
            test_df = item.iloc[train_num: ].reset_index()
        else:
            test_df = pd.concat([test_df, item.iloc[train_num: ]], ignore_index=True)

    test_df.reset_index()
    train_df.reset_index()

    train_loc = train_df['loc'].tolist()
    for uid, item in tqdm(test_df.groupby('uid')):
        test_loc = item['loc'].tolist()
        exclude = [i for i,loc in enumerate(test_loc) if loc not in train_loc]
        if len(exclude):
            train_df = pd.concat([train_df, item.iloc[exclude]], ignore_index=True)
            test_df.drop(item.iloc[exclude].index, inplace=True)
            train_loc = train_df['loc'].tolist()

    assert max(train_loc) <= pd.unique(train_df['loc']).shape[0]

    train_len, test_len = len(train_df), len(test_df)
    print(train_len, test_len)
    print(f'10:{round(test_len*10/train_len, 3)}')

    train_df.to_csv(os.path.join(save_path, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(save_path, 'test.csv'), index=False)

if __name__ == "__main__":
    data_path = '/data/wangshirui_data/original_poi/'
    save_path = './df_data/'

    # NYC
    # 171327 36496 (10:1.522)
    NYC_df = pd.read_csv(data_path+'dataset_TSMC2014_NYC_1210.csv', header=0)
    split_it(NYC_df, 0.15, save_path+'NYC')
    # TKY
    # 467266 106263 (10:1.611)
    TKY_df = pd.read_csv(data_path+'dataset_TSMC2014_TKY_1210.csv', header=0)
    split_it(TKY_df, 0.15, save_path+'TKY')