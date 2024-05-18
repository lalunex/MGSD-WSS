#！/bin/bash

# for dataset in 'ml-100k' 'ml-1m' 'amazon-beauty' 'amazon-sports-outdoors' 'yelp'; do
#  CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --dataset ${dataset}
# done

# for seed in 2022; do
#     for hard_noise_num in 1 2 3 4 5; do
#         CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --hard_noise_num ${hard_noise_num} --seed ${seed}
#     done
# done

for seed in 2024 ; do
    for user_interest_num in 25; do
        CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --user_interest_num ${user_interest_num} --seed ${seed} --gpu_id 0 --hard_noise_num 1
    done
done

# for seed in 2020 2021; do
#     for user_interest_num in 5 10 15 20 25; do
#         CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --user_interest_num ${user_interest_num} --seed ${seed}
#     done
# done


# for user_interest_num in 1 2 3 4 5 6 7 8 9 10; do
#  CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --user_interest_num ${user_interest_num}
# done
# for seed in 2020 2021 2022 2023 2024; do
#  CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --seed ${seed} --gpu_id 0 --train_set_ratio .4
# done
# for seed in 2020 2021 2022 2023; do
#  CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --seed ${seed} --gpu_id 0 --train_set_ratio .6
# done
# for seed in 2020 2021 2022 2023; do
#  CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --seed ${seed} --gpu_id 0 --train_set_ratio .8
# done
# for seed in 2020 2021 2022 2023 2024; do
#  CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --seed ${seed} --gpu_id 0 --train_set_ratio 1
# done
#  for seed in 2021 2022 2023; do
#   CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --seed ${seed}
# done

# for train_set_ratio in .2 .4 .6 .8 1; do
#  for seed in 2020; do
#   CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --train_set_ratio ${train_set_ratio} --seed ${seed}
#   done
# done
# for embedding_size in 256 32 64 100 128; do
#  CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --embedding_size ${embedding_size}
# done
#for sub_model in 'GRU4Rec' 'NARM' 'STAMP' 'Caser' 'SASRec'; do
#  CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --sub_model ${sub_model}
#done
#for learning_rate in 1e-1 1e-2 1e-3 1e-4; do
#  CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --learning_rate ${learning_rate}
#done
#for warm_up_ratio in 0 .1 .2 .3 .4 .5 .6 .7 .8 .9; do
#  CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --warm_up_ratio ${warm_up_ratio}
#done
# for latent_size in 10 20 30 40 50; do
#  CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --latent_size ${latent_size}
# done
#for gumbel_temperature in .1 .3 .5 .7 .9; do
#  CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --gumbel_temperature ${gumbel_temperature}
#done
#for supervised_contrastive_temperature in .1 .3 .5 .7 .9; do
#  CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --supervised_contrastive_temperature ${supervised_contrastive_temperature}
#done
# for sequence_last_m in 1 2 3 4 5 6; do
#   CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --sequence_last_m ${sequence_last_m}
# done
#for sigmoid_extent in 1 2 3 4 5 6; do
#  CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --sigmoid_extent ${sigmoid_extent}
#done
#for reweight_loss_alpha in .1 .3 .5 .7 .9; do
#  CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --reweight_loss_alpha ${reweight_loss_alpha}
#done
# for reweight_loss_lambda in 0 0.2 0.5 0.8 1 5 10; do
#  CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --reweight_loss_lambda ${reweight_loss_lambda}
# done
# for our_att_drop_out in 0 .1 .2 .3 .4 .5 .6 .7 .8 .9; do
#  CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --our_att_drop_out ${our_att_drop_out}
# done
# for our_ae_drop_out in 0 .1 .2 .3 .4 .5 .6 .7 .8 .9; do
#  CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --our_ae_drop_out ${our_ae_drop_out}
# done
# for weight_decay in 0 1e-3 1e-4 1e-5; do
#  CUDA_VISIBLE_DEVICES=0 python run_ehdwcl.py --weight_decay ${weight_decay}
# done

#a = ['amazon-books', 'amazon-toys-games', 'amazon-clothing-shoes-jewelry'
#     'amazon-video-games', 'avazu', 'criteo', 'food', 'netflix']