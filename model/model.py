import torch
from typing import List, Optional, Tuple, Union
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F
from torch import nn
import os

class FLModel(LlamaForCausalLM):
    '''This model is without CF module to compare with Secor.
    '''
    def __init__(self, config):
        super(FLModel, self).__init__(config)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, CausalLMOutputWithPast]:
            # logits: 16, 304, 32000
            # labels: 16, 304
            # input_ids: 16, 304

            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.lm_head(hidden_states)
            logits = logits.float()

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                # Focal loss for output
                labels_index = torch.argwhere(torch.bitwise_or(labels == 8241, labels == 3782))
                gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == 3782, 0, 1)
                gold = gold[2::3]
                labels_index[: , 1] = labels_index[: , 1] - 1
                this_logits = logits.softmax(dim=-1).to(labels_index.device)
                this_logits = torch.softmax(this_logits[labels_index[:, 0], labels_index[:, 1]][:,[3782, 8241]], dim = -1)
                this_logits = this_logits[:,1][2::3]
                focal_loss = -(1-this_logits)**2 *torch.log(this_logits) * gold - this_logits**2 *torch.log(1-this_logits) * (1-gold)
                loss += torch.mean(focal_loss.to(loss.device))

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


class PrefixModel(LlamaForCausalLM):
    '''This model utilizes random prefix (without cf module) to compare with Secor.
    The results are shown in section 4.3
    '''
    def __init__(self, config):
        super(PrefixModel, self).__init__(config)
        self.prefix = nn.Parameter(torch.FloatTensor((1, config.hidden_size)), requires_grad=True)
        nn.init.uniform_(self.prefix.data)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            uids: torch.LongTensor = None,
        ) -> Union[Tuple, CausalLMOutputWithPast]:
            # labels: 16, 2
            # input_ids: 16, 304

            token_embeds = self.model.embed_tokens(input_ids)   # 16, 304, 4096
            batch_size = token_embeds.shape[0]
            # add random-initialization trainable prefix
            input_embeds = torch.cat((self.prefix, token_embeds), dim=1)
            new_attention_mask = torch.cat((torch.ones((batch_size, 1)).to(attention_mask.device), attention_mask), dim=-1)

            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=None,
                attention_mask=new_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=input_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            #outputs.last_hidden_state

            hidden_states = outputs[0]
            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.lm_head(hidden_states)
            logits = logits.float()

            user_embeddings = hidden_states[:, 0].to(self.prefix.device)
            pos_embeddings = self.cf_emb_p[labels[:, 0]]
            neg_embeddings = self.cf_emb_p[labels[:, 1]]
            pos_scores = torch.mul(user_embeddings, pos_embeddings).sum(dim=1)
            neg_scores = torch.mul(user_embeddings, neg_embeddings).sum(dim=1)

            loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
            #cross-entropy?

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    def test_forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            uids: torch.LongTensor = None
        ):
            token_embeds = self.model.embed_tokens(input_ids)   # 16, 304, 4096
            batch_size = token_embeds.shape[0]
            
            user_embeds = self.cf_emb_u[uids].unsqueeze(1)
            input_embeds = torch.cat((user_embeds, token_embeds), dim=1).to(torch.float16)
            new_attention_mask = torch.cat((torch.ones((batch_size, 1)).to(attention_mask.device), attention_mask), dim=-1)

            outputs = self.model(
                attention_mask=new_attention_mask,
                inputs_embeds=input_embeds,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
                return_dict=self.config.return_dict,
            )

            user_embeddings = outputs[0][:, 0].to(self.cf_emb_p.device)
            ratings = torch.matmul(user_embeddings, self.cf_emb_p.to(torch.float16).t()).sigmoid()

            return ratings


class Secor(LlamaForCausalLM):
    def __init__(self, config):
        super(Secor, self).__init__(config)

    def init_setting(self, embedding_user, embedding_item, pretrained_path=None, cf_dim = 128, tau = 0.3, lambda1 = 0.1, lambda2 = 1e-4):
        device = self.model.embed_tokens.weight.device
        self.cf_emb_u = embedding_user.to(device)
        self.cf_emb_p = embedding_item.to(device)
        self.tau = tau
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mapping = nn.Linear(cf_dim, self.config.hidden_size).to(device)
        # save the final encoding for ranking on testset
        self.hybrid_item = torch.zeros((self.cf_emb_p.shape[0], self.config.hidden_size), device=device)
        if pretrained_path:
            self.mapping.load_state_dict(torch.load(os.path.join(pretrained_path, "mapping.pt")))
            self.hybrid_item = torch.load(os.path.join(pretrained_path, "hybrid_item.pt")).to(device)

    def update_item_emb(self, item_embeds, description_tokens, attention_mask):
        # tokens process
        token_embeds = self.model.embed_tokens(description_tokens)   # 96, 304, 4096
        batch_size = token_embeds.shape[0]

        input_embeds = torch.cat((item_embeds, token_embeds), dim=1)
        new_attention_mask = torch.cat((torch.ones((batch_size, 1)).to(attention_mask.device), attention_mask), dim=-1)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=None,
            attention_mask=new_attention_mask,
            inputs_embeds=input_embeds,
            output_attentions=self.config.output_attentions,
            output_hidden_states=self.config.output_hidden_states,
            return_dict=self.config.return_dict,
        )
        new_item_emb = outputs[0][:, 0].to(self.cf_emb_p.device)

        # return the whole poi embeddings for all-rank
        return new_item_emb    

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            uids: torch.LongTensor = None,
            neg_labels: torch.LongTensor = None,
            poi_input_ids: torch.LongTensor = None,
            poi_attention_mask: Optional[torch.Tensor] = None
        ) -> Union[Tuple, CausalLMOutputWithPast]:
            # labels: 16
            # neg_labels: 16, 5
            # input_ids: 16, 304
            # description: 16, 6, 304

            token_embeds = self.model.embed_tokens(input_ids)   # 16, 304, 4096
            batch_size = token_embeds.shape[0]

            all_labels = torch.concat([labels, neg_labels], dim=1).view((-1, 1))
            # CF Module
            map_user = self.mapping(self.cf_emb_u[uids]).unsqueeze(1)
            map_item = self.mapping(self.cf_emb_p[all_labels])

            # User process
            ## LLM
            input_embeds = torch.cat((map_user, token_embeds), dim=1)
            new_attention_mask = torch.cat((torch.ones((batch_size, 1)).to(attention_mask.device), attention_mask), dim=-1)
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            ### decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=None,
                attention_mask=new_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=input_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=False,
            )
            #outputs.last_hidden_state

            # poi process
            ## See details in 'update_item_emb'
            item_embeddings = self.update_item_emb(map_item, poi_input_ids, poi_attention_mask)
            ## Save final hybrid embeddings for test
            self.hybrid_item[all_labels.view(-1)] = item_embeddings

            # Get Score
            user_embeddings = outputs[0][:, 0].to(item_embeddings.device)
            pos_embeddings = item_embeddings[:batch_size]
            neg_embeddings = item_embeddings[batch_size:].unflatten(0, (batch_size, -1))
            pos_scores = torch.mul(user_embeddings, pos_embeddings).sum(dim=1)
            neg_scores = torch.mul(user_embeddings.unsqueeze(1), neg_embeddings).sum(dim=-1).mean(dim=-1)

            # bpr + infonce
            loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores)) + self.hybrid_info_nce(map_user, map_item, user_embeddings, item_embeddings)

            return CausalLMOutputWithPast(loss=loss)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        if z1.size()[0] == z2.size()[0]:
            return F.cosine_similarity(z1, z2)
        else:
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            return torch.mm(z1, z2.t())

    def hybrid_info_nce(self, map_user, map_item, hybrid_user, hybrid_item):
        ## 全体的视为负的，不涉及ids
        # 维度问题，原本cf特征维度不同
        def f(x): return torch.exp(x / self.tau)

        neg_d = f(self.sim(hybrid_user, self.mapping(self.cf_emb_u))).sum(1)
        pos_d = f(self.sim(hybrid_user, map_user.squeeze(1)))

        neg_d_i = f(self.sim(hybrid_item, self.mapping(self.cf_emb_p))).sum(1)
        pos_d_i = f(self.sim(hybrid_item, map_item.squeeze(1)))

        return self.lambda1 * (torch.mean(-torch.log(pos_d/neg_d)) + torch.mean(-torch.log(pos_d_i/neg_d_i)))

    def test_forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            uids: torch.LongTensor = None
        ):
            self.eval()
            token_embeds = self.model.embed_tokens(input_ids)   # 16, 304, 4096
            batch_size = token_embeds.shape[0]
            
            user_embeds = self.mapping(self.cf_emb_u[uids]).unsqueeze(1)
            input_embeds = torch.cat((user_embeds, token_embeds), dim=1).to(torch.float16)
            new_attention_mask = torch.cat((torch.ones((batch_size, 1)).to(attention_mask.device), attention_mask), dim=-1)

            outputs = self.model(
                attention_mask=new_attention_mask,
                inputs_embeds=input_embeds,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
                return_dict=self.config.return_dict,
            )

            user_embeddings = outputs[0][:, 0].to(self.hybrid_item.device)

            ratings = torch.matmul(user_embeddings, self.hybrid_item.to(torch.float16).t()).sigmoid()

            return ratings