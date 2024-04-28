from dataclasses import replace
from geopy.geocoders import Nominatim
import numpy as np
from tqdm import tqdm
import random
import pandas as pd
import json
import gc

def get_name_TKY(name_dict:dict):
    if 'city' in name_dict:
        main_info = name_dict['city']
    elif 'City' in name_dict:
        main_info = name_dict['City']
    elif 'city:' in name_dict:
        main_info = name_dict['city:']
    elif 'neighbourhood' in name_dict:
        main_info = name_dict['neighbourhood']
    elif 'Neighbourhood' in name_dict:
        main_info = name_dict['Neighbourhood']
    elif 'Quarter' in name_dict:
        main_info = name_dict['Quarter']
    else:
        main_info = name_dict['quarter']

    if 'road' in name_dict:
        main_info += ' '+name_dict['road']
        if 'house_number' in name_dict:
            main_info += f' No.{name_dict["house_number"]}'
    
    return main_info

def get_name_NYC(name_dict:dict):
    if 'suburb' in name_dict:
        main_info = name_dict['suburb']
    else:
        main_info = name_dict['county']

    if 'road' in name_dict:
        main_info += ' '+name_dict['road']
        if 'house_number' in name_dict:
            main_info += f' No.{name_dict["house_number"]}'

    return main_info



def save_instruction(df:pd.DataFrame, get_name, poi2name, save_path,
                     user_map, type="train", neg_num = 5,
                     seq_len = 10, overlap=1):
    cat_name = df[['cat_name','loc']].groupby('loc')['cat_name'].first()
    generate_name= lambda loc_id: f'{get_name(poi2name[loc_id])} ({cat_name[loc_id]})'
    ori_poi = pd.unique(df['loc'])

    json_list = []
    # TODO: df是修改后的新id，记得修改正负样本的对应！！！
    for uid, item in tqdm(df.groupby('uid')):
        loc = item['loc'].array
        assert loc != 47010
        user = user_map[uid]
        
        for i in range(seq_len, len(item['loc']), overlap):
            history = list(map(generate_name, loc[i-seq_len: i]))
            history_str = ", ".join(history)
            neg_poi = np.random.choice(ori_poi, 2*neg_num, replace=False)
            neg_poi = neg_poi[~np.isin(neg_poi, loc)]
            while len(neg_poi) < neg_num:
                choices = np.random.choice(ori_poi, 2*neg_num, replace=False)
                choices = choices[~np.isin(choices, loc)]
                choices = choices[~np.isin(choices, neg_poi)]
                neg_poi.append(choices)
            neg_poi = neg_poi[:neg_num].tolist()

            data_point = {
                "instruction": "Given the user's visiting POI history, suggest him 5 next locations in this city to visit. Each POI is formatted as location description (type). Response Format:[No. a recommending POI].",
                "input": f"User Visiting History: {history_str}",
                "uid": user,
                "labels": loc[i].item(),
            }
            if type == "test":
                data_point.update({
                    "known_labels": loc[i-seq_len: i].astype(np.int32).tolist()   # 可能将用于去掉已知
                })
            else:
                data_point.update({
                    "neg_labels": neg_poi
                })
            json_list.append(data_point)
            
    random.shuffle(json_list)
    print(len(json_list))
    with open(f'{save_path}/{type}.json', 'w') as f:
        json.dump(json_list, f, indent=4)

    description_dict = {}
    for poi_id in pd.unique(df['loc']):
        descrption = generate_name(poi_id)
        description_dict[poi_id] = descrption
    np.save(f'{save_path}/descrption_{type}.npy', description_dict)


if __name__ == "__main__":
    data_dir = '/data/wangshirui_data/original_poi'
    processed_dir = '/home/wangshirui/Secor'

    # 85056, 
    for dataset, func in zip(['NYC', 'TKY'],[get_name_NYC, get_name_TKY]):
        this_dir = f'{processed_dir}/df_data/{dataset}'
        poi_file = f'{data_dir}/{dataset}_id2name.npy'
        poi2name = np.load(poi_file,allow_pickle=True).item()
        if dataset == 'TKY':
            poi_file = f'{data_dir}/{dataset}_id2name_1210.npy'
            poi2name_ori = poi2name
            loc_map = np.load(f'{this_dir}/item_map.npy',allow_pickle=True).item()
            new2ori = dict(zip(loc_map.values(), loc_map.keys()))
            poi2name = dict(zip(new2ori.keys(), map(lambda x:poi2name_ori[new2ori[x]], new2ori.keys())))
            del loc_map, new2ori, poi2name_ori
            gc.collect()

        for type in ['train', 'test']:
            df = pd.read_csv(f'{this_dir}/{type}.csv', header=0)
            save_instruction(
                            df, func, poi2name, this_dir+'_constrast',
                            np.load(f'{this_dir}/user_map.npy',allow_pickle=True).item(),
                            type=type, neg_num = 5, overlap=2
                        )
        
