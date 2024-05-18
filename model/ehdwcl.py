import numpy as np
from utils.utils import get_model
import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.layers import TransformerEncoder
from recbole.model.abstract_recommender import SequentialRecommender
from torch.nn.init import xavier_uniform_, xavier_normal_
from functools import reduce

def swish(x):
    return x.mul(torch.sigmoid(x))

class EHDWCL(SequentialRecommender):

    def __init__(self, config, dataset):
        super(EHDWCL, self).__init__(config, dataset)

        # load parameters info
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.embedding_size = config['embedding_size']
        self.our_ae_drop_out = config['our_ae_drop_out']
        self.hard_noise_num = config['hard_noise_num']
        self.user_interest_num = config['user_interest_num']
        self.gaussian_kernel_num = config['gaussian_kernel_num']

        # 下面是我添加的参数
        self.reweight_loss_gamma = config['reweight_loss_gamma']
        self.reweight_loss_zeta = config['reweight_loss_zeta']
        self.reweight_loss_eta = config['reweight_loss_eta']
        self.train_batch_size = config['train_batch_size']
        self.transformer_encoder_heads = config['transformer_encoder_heads']
        self.transformer_encoder_layers = config['transformer_encoder_layers']
        self.transformer_encoder_dim_feedforward = config['transformer_encoder_dim_feedforward']
        self.transformer_encoder_layer_norm_eps = config['transformer_encoder_layer_norm_eps']

        self.gumbel_tau = 0.5  # 初始化
        self.spu_cl_tau = 0.5  # 初始化
        self.train_set_ratio = config['train_set_ratio']

        self.n_users = dataset.user_num

        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.embedding_size)

        self.LayerNorm = nn.LayerNorm(self.hidden_size)
        self.emb_dropout = nn.Dropout(self.our_ae_drop_out)
        self.relu = nn.ReLU()

        # 下面是两个MLP来将hidden转化为对应的噪音信号和兴趣信号
        self.multiattention_2_noise_scores_fc1 = nn.Linear(self.embedding_size, self.embedding_size, True)
        self.multiattention_2_noise_scores_fc2 = nn.Linear(self.embedding_size, 2, False)

        self.multiattention_2_interest_scores_fc1 = nn.Linear(self.embedding_size, self.embedding_size, True)
        self.user_interests_emb = torch.nn.Parameter(torch.randn(self.user_interest_num, self.embedding_size))

        # 下面是将用户兴趣与原来序列做一个attention的初始化
        self.attention_read_out = AttnReadout(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=2,
            layer_norm=True,
            feat_drop=self.our_ae_drop_out,
        )

        # 下面是添加的transformerencoder代码来读取长期信息，初始化模型
        self.raw_gaussian_kernel_transformer_encoder = TransformerEncoder(
            n_layers=self.transformer_encoder_layers,
            n_heads=self.transformer_encoder_heads,
            hidden_size=self.hidden_size,
            inner_size=self.transformer_encoder_dim_feedforward,
            hidden_dropout_prob=self.our_ae_drop_out,
            attn_dropout_prob=self.our_ae_drop_out,
            hidden_act='relu',
            layer_norm_eps=self.transformer_encoder_layer_norm_eps
        )

        # 下面是添加的transformerencoder代码来读取长期信息，初始化模型
        self.item_wise_transformer_encoder = TransformerEncoder(
            n_layers=self.transformer_encoder_layers,
            n_heads=self.transformer_encoder_heads,
            hidden_size=self.hidden_size,
            inner_size=self.transformer_encoder_dim_feedforward,
            hidden_dropout_prob=self.our_ae_drop_out,
            attn_dropout_prob=self.our_ae_drop_out,
            hidden_act='relu',
            layer_norm_eps=self.transformer_encoder_layer_norm_eps
        )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        # self.binary_softmax = nn.Softmax(dim=-1)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.recommender_ce_loss = nn.CrossEntropyLoss(reduction='none')

        self.apply(self._init_weights)

        # 初始化sub_model
        self.sub_model = get_model(config['sub_model'])(config, dataset).to(config['device'])
        self.sub_model_name = config['sub_model']
        self.item_embedding = self.sub_model.item_embedding

        if config['load_pre_train_emb'] is not None and config['load_pre_train_emb']:
            checkpoint_file = config['pre_train_model_dict'][config['dataset']][config['sub_model']]
            checkpoint = torch.load(checkpoint_file)
            if config['sub_model'] == 'DSAN':
                embedding_weight = checkpoint['state_dict']['embedding.weight']
                self.item_embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False)
            else:
                item_embedding_weight = checkpoint['state_dict']['item_embedding.weight']
                self.item_embedding = nn.Embedding.from_pretrained(item_embedding_weight, freeze=False)

    # 在apply里面使用的初始化函数
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    # 原版（添加）硬噪声并重构序列，1不是噪音，0是噪音
    def hard_noise_adding_and_reconstraction(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        total_num_item_set = self.item_embedding.num_embeddings - 1
        hard_noise_item_id = torch.randint(1, total_num_item_set, (item_seq.shape[0], self.hard_noise_num), device=self.device)
        hard_noise_item_index, _ = torch.stack([
            torch.randperm(i+self.hard_noise_num, dtype=torch.long, device=self.device)[:self.hard_noise_num]
            for i in item_seq_len], dim=0).sort(descending=False)
        temp_item_seq = item_seq.clone()
        # temp_item_seq = torch.nn.ZeroPad2d(padding=(0, self.hard_noise_num, 0, 0))(item_seq.clone())

        for i in range(self.hard_noise_num):
            temp_mask = torch.stack([
                                    torch.cat((torch.ones(p, device=self.device), torch.zeros(temp_item_seq.shape[1]-p, device=self.device)), dim=-1)
                                    for p in hard_noise_item_index[:, i]], dim=0).to(torch.long)
            
            temp_head_mask = torch.nn.ZeroPad2d(padding=(0, 1, 0, 0))(temp_mask)
            temp_item_seq_head = torch.nn.ZeroPad2d(padding=(0, 1, 0, 0))(temp_item_seq)
            temp_item_seq_head = temp_item_seq_head * temp_head_mask
            temp_item_seq_head[torch.arange(temp_item_seq.shape[0], dtype=torch.long, device=self.device),
                               hard_noise_item_index[:, i]] = hard_noise_item_id[:, i]

            temp_tail_mask = torch.nn.ZeroPad2d(padding=(1, 0, 0, 0))(1 - temp_mask)
            temp_item_seq_tail = torch.nn.ZeroPad2d(padding=(1, 0, 0, 0))(temp_item_seq)
            temp_item_seq_tail = temp_item_seq_tail * temp_tail_mask
            
            temp_item_seq = temp_item_seq_head + temp_item_seq_tail
            
        hard_noise_item_mask = torch.ones(temp_item_seq.shape, dtype=torch.long, device=temp_item_seq.device) * temp_item_seq.gt(0)
        hard_noise_item_len = hard_noise_item_mask.sum(-1)

        # 下面的代码是获取噪音位置的mask矩阵，1不是噪音，0是噪音
        row_index_mask = torch.arange(temp_item_seq.shape[0], dtype=torch.long, device=self.device).unsqueeze(-1).expand(temp_item_seq.shape[0], self.hard_noise_num).reshape(-1)
        column_index_mask = hard_noise_item_index.reshape(-1)
        hard_noise_item_mask[row_index_mask, column_index_mask] = 0

        # 下面的代码是针对seq和mask获取长度为max length的序列
        final_item_seq = torch.zeros_like(item_seq, dtype=torch.long, device=item_seq.device)
        final_item_mask = torch.zeros_like(item_seq, dtype=torch.long, device=item_seq.device)
        # 序列长度小于等于max length
        seq_len_i = (hard_noise_item_len <= self.max_seq_length)
        final_item_seq[seq_len_i] = temp_item_seq[seq_len_i][:, 0:self.max_seq_length]
        final_item_mask[seq_len_i] = hard_noise_item_mask[seq_len_i][:, 0:self.max_seq_length]
         # 序列长度大于max length
        for i in range(self.hard_noise_num):
            seq_len_i = (hard_noise_item_len == self.max_seq_length + i + 1)
            final_item_seq[seq_len_i] = temp_item_seq[seq_len_i][:, i + 1:self.max_seq_length + i + 1]
            final_item_mask[seq_len_i] = hard_noise_item_mask[seq_len_i][:, i + 1:self.max_seq_length + i + 1]


        # hard_noise_item_mask = hard_noise_item_mask
        # hard_noise_item_mask = torch.stack([torch.multinomial(torch.tensor([self.hard_noise_ratio, 1-self.hard_noise_ratio], device=self.device),
        #                                                       num_samples=self.max_seq_length, replacement=True).to(self.device)
        #                                                       for _ in range(item_seq.shape[0])], dim=0) * origin_mask
        
        # hard_noise_item_set_id = torch.randint(1, total_num_item_set, (item_seq.shape), device=self.device) * (1 - hard_noise_item_mask) * origin_mask
        # item_seq_maksed_retain = item_seq * hard_noise_item_mask

        # final_item_seq = item_seq_maksed_retain + hard_noise_item_set_id
        processed_item_seq = final_item_seq
        processed_item_seq_len = (final_item_seq > 0).sum(-1)
        processed_item_seq_mask = final_item_mask

        return processed_item_seq, processed_item_seq_len, processed_item_seq_mask

    # 使用高斯核的硬降噪过程
    def sequence_encoder_with_gaussian_kernel(self, item_seq, item_seq_emb, target_item):
        item_seq_emb = self.emb_dropout(item_seq_emb)
        target_item_emb = self.item_embedding(target_item)
        if item_seq_emb.isnan().any() or target_item_emb.isnan().any():
            print(44)

        # init mu and sigma
        gaussian_mu = [1]
        gaussian_sigma = [0.001]  # for exact match. small variance -> exact match
        if self.gaussian_kernel_num > 1:
            bin_size = 2.0 / (self.gaussian_kernel_num - 1) # score range from [-1, 1]
            gaussian_mu.append(1 - bin_size / 2)  # mu: middle of the bin
            for i in range(1, self.gaussian_kernel_num - 1):
                gaussian_mu.append(gaussian_mu[i] - bin_size)
        gaussian_sigma += [0.1] * (self.gaussian_kernel_num - 1)

        gaussian_mu = torch.tensor(gaussian_mu)
        gaussian_sigma = torch.tensor(gaussian_sigma)       # 这里还有其它文章使用了 0.01，但李潇可使用了0.1

        norm_item_seq_emb = item_seq_emb.div(torch.norm(item_seq_emb, p=2, dim=-1, keepdim=True))
        norm_target_item_emb = target_item_emb.div(torch.norm(target_item_emb, p=2, dim=-1, keepdim=True))

        if norm_item_seq_emb.isnan().any() or norm_target_item_emb.isnan().any():
            print(88)
            
        mask = torch.ones(item_seq.shape, dtype=torch.float, device=item_seq.device) * item_seq.gt(0)
        norm_item_seq_emb = norm_item_seq_emb * mask.unsqueeze(-1)
        norm_target_item_emb = norm_target_item_emb.expand_as(
            norm_item_seq_emb
        ).to(torch.float)
        # 这步得到它的相似度矩阵
        similarity_matrix = F.cosine_similarity(norm_item_seq_emb, norm_target_item_emb, dim=2)

        gaussian_feat = []
        for mu, sigma in zip(gaussian_mu, gaussian_sigma):
            tmp = torch.exp(- torch.square(similarity_matrix-mu) / (2*torch.square(sigma)))  # exp(- (x-mu)^2 /(2*sigmma^2) )
            gaussian_feat.append(tmp)
        
        temp_final_gaussian_feat = torch.stack(gaussian_feat, dim=-1)
        masked_item_seq_emb = item_seq_emb * mask.unsqueeze(-1)
        temp_raw_item_seq_emb = masked_item_seq_emb.unsqueeze(-2) * temp_final_gaussian_feat.unsqueeze(-1)

        if temp_raw_item_seq_emb.isnan().any():
            print(77)
        temp_raw_item_seq_emb_sumed = temp_raw_item_seq_emb.sum(dim=2)

        # 下面语句可检查guassian kernel的有效性
        # temp_raw_item_seq_emb_sumed = item_seq_emb

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        temp_whole_item_seq_emb = temp_raw_item_seq_emb_sumed + position_embedding
        temp_whole_item_seq_emb_masked = self.emb_dropout(temp_whole_item_seq_emb) * mask.unsqueeze(-1)
        extended_attention_mask = self.get_attention_mask(item_seq)

        if temp_whole_item_seq_emb_masked.isnan().any():
            print(55)
        outputs = self.raw_gaussian_kernel_transformer_encoder(
            temp_whole_item_seq_emb_masked, extended_attention_mask, output_all_encoded_layers=True)
        final_item_seq_emb = outputs[-1] * mask.unsqueeze(-1)

        if final_item_seq_emb.isnan().any():
            print(66)
        return final_item_seq_emb

    # 下面是使用任意的下游模型来做recommender以得到序列表征，forward部分
    def sub_model_forward(self, item_seq, item_seq_emb, item_seq_len, user):
        if self.sub_model_name == 'BERT4Rec':
            sub_model_items_output = self.sub_model.forward(item_seq, item_seq_emb)
        elif self.sub_model_name == 'GRU4Rec':
            sub_model_items_output = self.sub_model.forward(item_seq_emb, item_seq_len)
        elif self.sub_model_name == 'SASRec':
            sub_model_items_output = self.sub_model.forward(item_seq, item_seq_emb, item_seq_len)
        elif self.sub_model_name == 'Caser':
            sub_model_items_output = self.sub_model.forward(user, item_seq_emb)
        elif self.sub_model_name == 'NARM':
            sub_model_items_output = self.sub_model.forward(item_seq, item_seq_emb, item_seq_len)
        elif self.sub_model_name == 'STAMP':
            sub_model_items_output = self.sub_model.forward(item_seq_emb, item_seq_len)
        else:
            raise ValueError(f'Sub_model [{self.sub_model_name}] not support.')

        # 下面是取出序列捏最后一个item的embedding作为序列表征
        seq_emb = self.seq_last_one_emb_extract(
            sub_model_items_output=sub_model_items_output,
            item_seq_len=item_seq_len)
        
        return seq_emb

    r"""该函数是为了将降噪的item序列里面的最后一个拿出来
    generated_seq：经降噪的模型
    seq_output：子模型（如bertrec等）的输出
    """

    def seq_last_one_emb_extract(self, sub_model_items_output, item_seq_len):
        if self.sub_model_name in ['Caser', 'GRU4Rec', 'NARM', 'DSAN', 'STAMP']:
            sub_model_seq_output = sub_model_items_output
        else:
            sub_model_seq_output = self.gather_indexes(sub_model_items_output, item_seq_len - 1)
        return sub_model_seq_output  # [B H]

    # 下面是两个MLP来将multiattention_feat转化为对应的噪音信号和兴趣信号
    def multiattention_feat_transformation_2_noise_signal(self, multiattention_feat, mask):
        if multiattention_feat.isnan().any() or mask.isnan().any():
            print(22)
        multiattention_feat = self.emb_dropout(multiattention_feat)
        noise_scores = self.relu(self.multiattention_2_noise_scores_fc1(multiattention_feat))
        normaled_noise_scores = self.softmax(self.multiattention_2_noise_scores_fc2(noise_scores))
        final_noise_scores = normaled_noise_scores * mask
        seq_len = mask.squeeze().sum(-1).to(torch.long)

        score_gumbel_softmax = F.gumbel_softmax(final_noise_scores, tau=self.gumbel_tau, hard=True, dim=-1)
        final_hard_noise_scores = score_gumbel_softmax[:, :, 0] * mask.squeeze()
 
        r"""
        以防止某个序列内没有正样本，而导致的supervised contrastive learning的分母为0
        """
        pos_seq_len = torch.sum(final_hard_noise_scores, dim=-1)
        final_hard_noise_scores[pos_seq_len.eq(0), seq_len[pos_seq_len.eq(0)] - 1] = 1

        if final_noise_scores.isnan().any() or final_hard_noise_scores.isnan().any():
            print(11)
        return final_noise_scores, final_hard_noise_scores

    def multiattention_feat_transformation_2_interest_signal(self, multiattention_feat, target_item, mask):
        multiattention_feat = self.emb_dropout(multiattention_feat)
        target_item_emb = self.item_embedding(target_item.squeeze())
        target_item_emb = self.emb_dropout(target_item_emb)
        interest_scores = self.relu(self.multiattention_2_interest_scores_fc1(multiattention_feat))
        normaled_interest_scores = self.softmax(torch.matmul(interest_scores, self.user_interests_emb.transpose(-1, -2)))
        final_interest_scores = normaled_interest_scores * mask
        final_interest_weight = final_interest_scores.sum(-2) / mask.sum(-2)

        full_interests_emb = self.user_interests_emb.unsqueeze(0).expand(final_interest_weight.shape + (self.embedding_size,))
        attn_interest_scores = self.attention_read_out(full_interests_emb, target_item_emb)

        score_gumbel_softmax = F.gumbel_softmax(attn_interest_scores, tau=self.gumbel_tau, hard=True, dim=-1)
        final_hard_interest_flag = score_gumbel_softmax[:, :, 0]

        r"""
        以防止某个序列内没有正样本，而导致的supervised contrastive learning的分母为0
        """
        pos_seq_len = torch.sum(final_hard_interest_flag, dim=-1)
        final_hard_interest_flag[pos_seq_len.eq(0), 0] = 1

        return final_interest_weight, final_hard_interest_flag
    
    # 下面是使用target的信息来监督hard的信息的生成
    def cross_entropy_loss(self, interaction, seq_emb, islist=True):
        all_items_emb = self.item_embedding.weight[:self.n_items]  # delete masked token

        # 下面开始将e与每个item进行交互（点乘），并计算cross entropy loss
        all_items_scores = torch.matmul(seq_emb, all_items_emb.transpose(0, 1))  # [B, item_num]
        target_item = interaction[self.ITEM_ID].unsqueeze(1)
        target_item = target_item.squeeze()
        recommender_ce_loss_list = self.recommender_ce_loss(all_items_scores, target_item)

        return recommender_ce_loss_list if islist else recommender_ce_loss_list.mean()
    
    # 下面是使用了supervised contrastive learning来对分类好的item计算loss，forward部分（batch mean）
    def supervised_contrastive_learning(
            self,
            seq_emb,
            item_seq_emb,
            item_pos_flag,
            item_weight,
            mask,
            tau,
            mode='item'):
        item_seq_emb = item_seq_emb * mask
                
        # 下面是为了检查加权对比学习的能力
        if mode == 'item':
            # item_weight = torch.ones_like(item_weight, device=self.device) * 0.5
            sim_weights = item_weight[:, :, 0]
            no_sim_weight = 1 - sim_weights
        else:
            # item_weight = torch.ones_like(item_weight, device=self.device) / self.user_interest_num
            sim_weights = item_weight
            no_sim_weight = item_weight

        target_seq_emb = seq_emb.unsqueeze(1).expand_as(
            item_seq_emb
        ).to(torch.float)
        # 这步得到它的相似度矩阵
        similarity_matrix = F.cosine_similarity(item_seq_emb, target_seq_emb, dim=2)

        # 这步给相似度矩阵求exp, 并且除以温度参数T, 注意要乘mask
        similarity_matrix_after_exp = torch.exp(similarity_matrix / tau) * mask.squeeze()

        # 这步产生了正样本（五噪音item）的相似度矩阵，其他位置都是0
        # sim = item_pos_flag * similarity_matrix_after_exp
        sim = item_pos_flag * similarity_matrix_after_exp * sim_weights

        # 用原先的相似度矩阵减去正样本矩阵得到负样本（噪音item）的相似度矩阵
        # no_sim = (1 - item_pos_flag) * mask.squeeze() * similarity_matrix_after_exp
        no_sim = (1 - item_pos_flag) * mask.squeeze() * similarity_matrix_after_exp * no_sim_weight
        # no_sim = similarity_matrix_after_exp - sim

        # 把负样本矩阵按行求和，得到的是对比损失的分母(还差一个与分子相同的那个相似度，后面会加上)
        no_sim_sum = torch.sum(no_sim, dim=1, keepdim=True)

        '''
        将上面的矩阵扩展一下，再转置，加到sim（也就是正样本矩阵上），然后再把sim矩阵与sim_num矩阵做除法。
        至于为什么这么做，就是因为对比损失的分母存在一个正样本的相似度，就是分子的数据。
        '''
        no_sim_sum_expend = no_sim_sum.expand_as(item_pos_flag)
        sim_sum = sim + no_sim_sum_expend

        # 为了防止自监督对比学习的分母为0
        zero_anomaly_process = sim_sum.le(0.).float() * 1e-10
        anomaly_process_sim_sum = sim_sum + zero_anomaly_process

        sim_div = torch.div(sim, anomaly_process_sim_sum)

        '''
        由于loss矩阵中，存在0数值，那么在求-log的时候会出错。这时候，我们就将loss矩阵里面为0的地方
        全部加上1，然后再去求loss矩阵的值，那么-log1 = 0 ，就是我们想要的。
        '''
        sim_div_sum = sim_div + sim_div.eq(0)

        # 接下来就是算一个批次中的sup_con_loss了，batch里面per item的loss
        sim_div_sum_log = -torch.log(sim_div_sum)  # 求-log
        sup_con_losses = torch.sum(sim_div_sum_log, dim=1) / (torch.sum(item_pos_flag, dim=-1))

        return sup_con_losses

    # 下面是使用transformer来对item denoised序列进行encoder
    def item_wise_denoisd_seq_encoder(self, item_seq, item_emb, soft_signal, hard_signal):
        if soft_signal.isnan().any() or hard_signal.isnan().any() or item_seq.isnan().any():
            print(11)
        pos_seq = item_seq * hard_signal
        pos_seq_len = torch.sum(hard_signal, dim=-1)
        pos_seq_len = pos_seq_len.type(torch.long).to(self.device)
        row_indexes, col_id = torch.where(pos_seq.gt(0))
        hard_denoising_seq = torch.zeros_like(pos_seq, device=self.device)
        soft_signal_first_dim = soft_signal[:, :, 0]
        hard_denoising_seq_weight = torch.zeros_like(soft_signal_first_dim, device=self.device)
        if pos_seq_len.isnan().any():
            print(33)
        pos_id_list = [torch.arange(i).tolist() for i in pos_seq_len]
        pos_id_list_concat = reduce((lambda x, y: x + y), pos_id_list)
        pos_id_concat = torch.tensor(pos_id_list_concat, device=self.device)
        
        hard_denoising_item_emb = torch.zeros_like(item_emb, device=self.device)
        hard_denoising_item_emb[row_indexes, pos_id_concat] = item_emb[row_indexes, col_id]
        hard_denoising_seq[row_indexes, pos_id_concat] = pos_seq[row_indexes, col_id]
        hard_denoising_seq_weight[row_indexes, pos_id_concat] = soft_signal_first_dim[row_indexes, col_id]

        hard_denoising_seq = hard_denoising_seq.long()
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        hard_denoising_seq_emb_weighted = hard_denoising_item_emb * hard_denoising_seq_weight.unsqueeze(-1)
        denoised_items_emb = hard_denoising_seq_emb_weighted + position_embedding
        hard_denoising_seq_mask = torch.ones(item_seq.shape, dtype=torch.float, device=item_seq.device) * hard_denoising_seq.gt(0)
        denoised_items_emb = self.emb_dropout(denoised_items_emb) * hard_denoising_seq_mask.unsqueeze(-1)
        extended_attention_mask = self.get_attention_mask(hard_denoising_seq)

        outputs = self.item_wise_transformer_encoder(
            denoised_items_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = outputs[-1]
        output = self.gather_indexes(output, pos_seq_len - 1)
        return output

    # 下面是sum来对interest denoised序列进行encoder
    def interest_wise_denoisd_seq_encoder(self, interest_emb, soft_signal, hard_signal):
        weighted_interest_emb = interest_emb.unsqueeze(0) * soft_signal.unsqueeze(-1)
        hard_denoised_weighted_interest_emb = weighted_interest_emb * hard_signal.unsqueeze(-1)
        final_interest_emb = hard_denoised_weighted_interest_emb.sum(1)

        return final_interest_emb

    def forward(self, interaction, train_flag=True):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user = interaction[self.USER_ID]
        item_seq_emb = self.item_embedding(item_seq)

        target_item = None
        if train_flag:
            target_item = interaction[self.ITEM_ID].unsqueeze(1)
        else:
            target_item = (item_seq[torch.arange(item_seq.shape[0], dtype=torch.long, device=self.device), item_seq_len - 1]).unsqueeze(-1)

        # calculate the mask
        mask = torch.ones(item_seq.shape, dtype=torch.float, device=item_seq.device) * item_seq.gt(0)
        # size = [batchSize, user_num_per_batch, 1]
        mask = mask.unsqueeze(2)
        
        gaussian_processed_item_seq_emb = self.sequence_encoder_with_gaussian_kernel(item_seq, item_seq_emb, target_item)

        # 下面是两个MLP来将multiattention_feat转化为对应的噪音信号和兴趣信号
        final_noise_weight, final_hard_noise_flag = self.multiattention_feat_transformation_2_noise_signal(gaussian_processed_item_seq_emb, mask)
        final_interest_weight, final_hard_interest_flag = self.multiattention_feat_transformation_2_interest_signal(gaussian_processed_item_seq_emb, 
                                                                                                                    target_item, mask)
        
        # 下面是使用任意的下游模型来做recommender以得到序列表征，forward部分
        seq_emb_output = self.sub_model_forward(
            item_seq=item_seq,
            item_seq_emb=gaussian_processed_item_seq_emb,
            item_seq_len=item_seq_len,
            user=user
        )
        
        # 下面是使用transformer来对item denoised序列进行encoder
        item_denoised_user_emb = self.item_wise_denoisd_seq_encoder(item_seq, gaussian_processed_item_seq_emb, final_noise_weight, final_hard_noise_flag)

        # 下面是sum来对interest denoised序列进行encoder
        interest_denoised_user_emb = self.interest_wise_denoisd_seq_encoder(self.user_interests_emb, final_interest_weight, final_hard_interest_flag)

        # 下面是为了检查interest或item来检查模型
        terminate_user_emb = seq_emb_output + item_denoised_user_emb + interest_denoised_user_emb
        # terminate_user_emb = seq_emb_output + interest_denoised_user_emb
        # terminate_user_emb = seq_emb_output + item_denoised_user_emb
        
        return seq_emb_output, terminate_user_emb, final_noise_weight, final_hard_noise_flag, final_interest_weight, final_hard_interest_flag

    def calculate_reweight_loss(self, interaction, gumbel_tau, spu_cl_tau):
        self.gumbel_tau = gumbel_tau
        self.spu_cl_tau = spu_cl_tau
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_emb = self.item_embedding(item_seq)
        target_item = interaction[self.ITEM_ID].unsqueeze(1)
        # target_item_emb = self.item_embedding(target_item)

        # calculate the mask
        mask = torch.ones(item_seq.shape, dtype=torch.float, device=item_seq.device) * item_seq.gt(0)
        mask = mask.unsqueeze(2)

        seq_emb_output, terminate_user_emb, final_noise_weight, final_hard_noise_flag, final_interest_weight, final_hard_interest_flag = self.forward(interaction, train_flag=True)

        # 下面是使用了supervised contrastive learning（batch mean）对分类好的item计算loss，forward部分
        item_sup_con_losses_list = self.supervised_contrastive_learning(
            seq_emb=seq_emb_output,
            item_seq_emb=item_seq_emb,
            item_pos_flag=final_hard_noise_flag,
            item_weight=final_noise_weight,
            mask=mask,
            tau=self.spu_cl_tau,
            mode='item')

        # 下面是使用了supervised contrastive learning（batch mean）对分类好的interest计算loss，forward部分
        interest_full_mask = torch.ones((item_seq.shape[0], self.user_interest_num), dtype=torch.long, device=self.device).unsqueeze(2)
        full_interests_emb = self.user_interests_emb.unsqueeze(0).expand(final_interest_weight.shape + (self.embedding_size,))
        interest_sup_con_losses_list = self.supervised_contrastive_learning(
            seq_emb=seq_emb_output,
            item_seq_emb=full_interests_emb,
            item_pos_flag=final_hard_interest_flag,
            item_weight=final_interest_weight,
            mask=interest_full_mask,
            tau=self.spu_cl_tau,
            mode='interest')
        
        # 添加硬噪声并重构序列并降噪，1不是噪音，0是噪音
        if self.hard_noise_num != 0:
            processed_item_seq, processed_item_seq_len, processed_item_seq_mask = self.hard_noise_adding_and_reconstraction(interaction)
            after_add_noise_mask = torch.ones(processed_item_seq.shape, dtype=torch.float, device=processed_item_seq.device) * processed_item_seq.gt(0)
            processed_item_seq_emb = self.item_embedding(processed_item_seq)

            gaussian_processed_item_seq_emb = self.sequence_encoder_with_gaussian_kernel(processed_item_seq, processed_item_seq_emb, target_item)
            after_add_noise_final_noise_weight, _ = self.multiattention_feat_transformation_2_noise_signal(gaussian_processed_item_seq_emb, after_add_noise_mask.unsqueeze(2))
            after_add_noise_final_noise_weight_positive = after_add_noise_final_noise_weight[:, :, 0]
            denoise_mse_losses_list = self.mse_loss(after_add_noise_final_noise_weight_positive, processed_item_seq_mask.to(torch.float)).sum(dim=-1) / processed_item_seq_len

        recommender_ce_losses_list = self.cross_entropy_loss(interaction, terminate_user_emb, True)
        


        # # 下面是做不同训练集大小的实验结果的代码
        mask_weights = torch.ones_like(recommender_ce_losses_list) # create a Tensor of weights
        mask_weights = torch.multinomial(mask_weights, int(recommender_ce_losses_list.shape[0] * self.train_set_ratio))
        final_ratio_mask = torch.zeros_like(recommender_ce_losses_list) # create a Tensor of weights
        final_ratio_mask[mask_weights] = 1

        item_sup_con_losses_list = item_sup_con_losses_list * final_ratio_mask
        interest_sup_con_losses_list = interest_sup_con_losses_list * final_ratio_mask
        denoise_mse_losses_list = denoise_mse_losses_list * final_ratio_mask
        recommender_ce_losses_list = recommender_ce_losses_list * final_ratio_mask

        # 求的batch里面的instance层次的loss
        item_sup_con_loss = torch.sum(item_sup_con_losses_list) / final_ratio_mask.count_nonzero()
        interest_sup_con_loss = torch.sum(interest_sup_con_losses_list) / final_ratio_mask.count_nonzero()
        denoise_mse_loss = torch.sum(denoise_mse_losses_list) / final_ratio_mask.count_nonzero()
        recommender_ce_loss = torch.sum(recommender_ce_losses_list) / final_ratio_mask.count_nonzero()


        # item_sup_con_loss = item_sup_con_losses_list.mean()
        # interest_sup_con_loss = interest_sup_con_losses_list.mean()
        # denoise_mse_loss = denoise_mse_losses_list.mean()
        # recommender_ce_loss = recommender_ce_losses_list.mean()

        # 下面是为了检查interest，辅助任务或item来检查模型
        total_loss = recommender_ce_loss + \
            denoise_mse_loss * self.reweight_loss_gamma + \
            item_sup_con_loss * self.reweight_loss_zeta + \
            interest_sup_con_loss * self.reweight_loss_eta
        
        # total_loss = recommender_ce_loss + \
        #     denoise_mse_loss * self.reweight_loss_gamma
                
        # total_loss = recommender_ce_loss + \
        #     denoise_mse_loss * self.reweight_loss_gamma + \
        #     item_sup_con_loss * self.reweight_loss_zeta
        
        # total_loss = recommender_ce_loss + \
        #     denoise_mse_loss * self.reweight_loss_gamma + \
        #     interest_sup_con_loss * self.reweight_loss_eta

        # total_loss = recommender_ce_loss + \
        #     item_sup_con_loss * self.reweight_loss_zeta + \
        #     interest_sup_con_loss * self.reweight_loss_eta
        
        return total_loss

    def predict(self, interaction):
        _, terminate_user_emb, _, _, _ = self.forward(interaction, train_flag=False)
        test_item = interaction[self.ITEM_ID]
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(terminate_user_emb, test_item_emb).sum(dim=1)  # [B]

        return scores

    # 下面函数与recbole里面的代码很不一样
    def full_sort_predict(self, interaction):
        _, terminate_user_emb, _, _, _, _ = self.forward(interaction, train_flag=False)

        test_items_emb = self.item_embedding.weight[:self.n_items]  # delete masked token
        scores = torch.matmul(terminate_user_emb, test_items_emb.transpose(0, 1))  # [B, item_num]

        return scores

class AttnReadout(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            layer_norm=True,
            feat_drop=0.0,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim) if layer_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_key_map = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_attention = nn.Linear(hidden_dim * 4,
                                      output_dim, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, interests_emb, user_emb):
        if self.layer_norm is not None:
            interests_emb = self.layer_norm(interests_emb)
        interests_emb = self.feat_drop(interests_emb)
        interests_emb_key = self.fc_key_map(interests_emb)
        user_emb_expand = user_emb.unsqueeze(1).expand_as(interests_emb)

        emb_subtract = interests_emb_key - user_emb_expand
        emb_multiply = interests_emb_key * user_emb_expand
        interests_two_dim_score = self.fc_attention(
            torch.cat((interests_emb_key, user_emb_expand,
                       emb_subtract, emb_multiply), dim=-1)
        )

        score = self.softmax(interests_two_dim_score)

        return score
