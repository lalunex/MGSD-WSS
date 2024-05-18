import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='EHDWCL', type=str, help="model selection"
                                                                      "you can choose one of ['EHDWCL', 'GRU4Rec', 'NARM', 'Caser', 'BERT4Rec', 'SASRec', 'STAMP']")
    parser.add_argument('--dataset', default='ml-100k', type=str, help="Dataset to use"
                                                                       "you can choose one of ml-100k, ml-1m, amazon-beauty, amazon-sports-outdoors, yelp")
    parser.add_argument('--sub_model', default='BERT4Rec', type=str, help='sub-model selection'
                                                                               "you can choose one of ['GRU4Rec', 'NARM', 'Caser', 'BERT4Rec', 'SASRec', 'STAMP']")
    # parser.add_argument('--local_rank', default=0, type=int,
    #                     help='node rank for distributed training')
    parser.add_argument('--seed', default=2024, type=int, help='seed for experiment')
    parser.add_argument('--verbose', default=False, type=bool, help='whether save the log file')
    parser.add_argument('--embedding_size', default=100, type=int, help='embedding size for all layer')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help="weight decay for adam optimizer")
    parser.add_argument('--train_batch_size', default=256, type=int, help='train batch size')
    parser.add_argument('--hard_noise_num', default=2, type=int, help='add hard noise ratio')
    parser.add_argument('--user_interest_num', default=5, type=int, help='user interest num')
    parser.add_argument('--gaussian_kernel_num', default=10, type=int, help='gaussian kernel num')
    parser.add_argument('--is_gumbel_tau_anneal', default=True, type=bool, help='whether use anneal for gumbel')
    parser.add_argument('--gumbel_temperature', default=0.5, type=float, help='gumbel temperature')
    parser.add_argument('--is_spu_cl_tau_anneal', default=True, type=bool, help='whether use anneal for gumbel')
    parser.add_argument('--supervised_contrastive_temperature', default=0.5, type=float, help='gumbel temperature')
    parser.add_argument('--our_ae_drop_out', default=0.1, type=float, help='our average drop-out rate')
    parser.add_argument('--load_pre_train_emb', default=False, type=bool, help='whether load pre-train model embedding')
    parser.add_argument('--reweight_loss_gamma', default=5, type=float, help='reweight loss alpha for ave') # 1-5
    parser.add_argument('--reweight_loss_zeta', default=10, type=float, help='reweight loss lambda for gaussian') # 1-10
    parser.add_argument('--reweight_loss_eta', default=1, type=float, help='reweight loss lambda for gaussian') # 1-20
    parser.add_argument('--gpu_id', default=0, type=int, help='gpu id')
    parser.add_argument('--warm_up_ratio', default=0, type=int, help='warm up ratio for lr')

    parser.add_argument('--train_set_ratio', default=1, type=float, help='train set ratio for model')

    args = parser.parse_args()
    return args
