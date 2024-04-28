import sys

import fire
import gradio as gr
import torch
torch.set_num_threads(1)
import numpy as np
import json
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from data_collator import POI_dataset, DataLoader
from model.model import Secor
from pretrain_base import utils
from sklearn.metrics import roc_auc_score
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

from tqdm import tqdm

def check_rlt(lora_weights, test_data_path, result_json_data, to_write_data=None):
    
    model_type = lora_weights.split('/')[-2]    # ends with /
    model_name = '_'.join(model_type.split('_')[:2])

    if model_type.find('NYC') > -1:
        train_sce = 'NYC'
    else:
        train_sce = 'TKY'
    
    if test_data_path.find('NYC') > -1:
        test_sce = 'NYC'
    else:
        test_sce = 'TKY'
    
    temp_list = model_type.split('_')
    seed = temp_list[-2]
    sample = temp_list[-1]
    
    if os.path.exists(result_json_data):
        f = open(result_json_data, 'r')
        data = json.load(f)
        f.close()
    else:
        data = dict()

    if not data.__contains__(train_sce):
        data[train_sce] = {}
    if not data[train_sce].__contains__(test_sce):
        data[train_sce][test_sce] = {}
    if not data[train_sce][test_sce].__contains__(model_name):
        data[train_sce][test_sce][model_name] = {}
    if not data[train_sce][test_sce][model_name].__contains__(seed):
        data[train_sce][test_sce][model_name][seed] = {}
    if data[train_sce][test_sce][model_name][seed].__contains__(sample):
        return False
        # data[train_sce][test_sce][model_name][seed][sample] = 
    elif to_write_data:
        data[train_sce][test_sce][model_name][seed][sample] = to_write_data
        with open(result_json_data, 'w') as f:
            json.dump(data, f, indent=4)
    return True


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    embedding_weights: str = "",
    test_data_path: str = "data/test.json",
    result_json_data: str = "temp.json",
    batch_size: int = 8,
    top_k: list = [10],
    share_gradio: bool = False,
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    # if not check_rlt(lora_weights, test_data_path, result_json_data):
    #     exit(0)

    max_K = max(top_k)
    pre, recall, ndcg = [[]]*len(top_k), [[]]*len(top_k), [[]]*len(top_k)

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    save_embedding = torch.load(embedding_weights)
    if device == "cuda":
        model = Secor.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.init_setting(save_embedding['embedding_user.weight'], save_embedding['embedding_item.weight'], pretrained_path=lora_weights)
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={'': 0}
        )
    elif device == "mps":
        model = Secor.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model.init_setting(save_embedding['embedding_user.weight'], save_embedding['embedding_item.weight'], pretrained_path=lora_weights)
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = Secor.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model.init_setting(save_embedding['embedding_user.weight'], save_embedding['embedding_item.weight'], pretrained_path=lora_weights)
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    tokenizer.padding_side = "left"
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def collator(data):
        to_padding = [{'input_ids':f['input_ids'], 'attention_mask': f['attention_mask']} for f in data]
        batch = tokenizer.pad(to_padding, return_tensors="pt", padding=True, pad_to_multiple_of=8).to(device)
        return batch["input_ids"], batch["attention_mask"], \
             torch.tensor([f["uid"] for f in data], device=device), \
             torch.tensor([f["labels"] for f in data], device=device) #, torch.tensor([f["known_labels"] for f in data]).to(device)

    dataset = POI_dataset(test_data_path, tokenizer)
    loader = DataLoader(dataset, batch_size, collate_fn=collator)

    for input_ids, attention_mask, uids, labels in tqdm(loader):
        ratings = model.test_forward(input_ids, attention_mask, uids)
        _, rating_K = torch.topk(ratings, k=max_K)
        ratings = ratings.cpu().detach().numpy()
        del ratings
        
        r = np.array([
            list(map(lambda x: x in gt, predictTopK)) for gt, predictTopK in zip(labels, rating_K.cpu())
        ]).astype('float')
        labels = labels.cpu().detach().numpy()
        for i, k in enumerate(top_k):
            ret = utils.RecallPrecision_ATk(labels, r, k)
            pre[i].append(ret['precision'])
            recall[i].append(ret['recall'])
            ndcg[i].append(utils.NDCGatK_r(labels,r,k))

    all_num = len(loader) * batch_size

    results = {'precision': [0]*len(top_k),
               'recall': [0]*len(top_k),
               'ndcg': [0]*len(top_k)}
    for i in range(len(top_k)):
        results['recall'][i] = sum(recall[i])/float(all_num)
        results['precision'][i] = sum(pre[i])/float(all_num)
        results['ndcg'][i] = sum(ndcg[i])/float(all_num)
    
    print(results)
    check_rlt(lora_weights, test_data_path, result_json_data, results)
    


if __name__ == "__main__":
    # fire.Fire(main)
    main(
        base_model='/data/llmweights/llama-7b-hf',
        lora_weights='/data/wangshirui_data/poi_llm/test0423',
        embedding_weights='/home/wangshirui/llm/lora-poi/df_data/TKY/cf_emb/lgn-3-128.pth.tar',
        test_data_path='/home/wangshirui/llm/lora-poi/df_data/TKY_constrast/test.json',
        result_json_data='/data/wangshirui_data/poi_llm/test0428.log',
        batch_size=6
    )