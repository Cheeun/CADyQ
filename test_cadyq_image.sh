CUDA_VISIBLE_DEVICES=4 python3 main.py \
--test_only \
--data_test Urban100 --dir_data /mnt/disk1/cheeun914/datasets/ --n_GPUs 1 \
--scale 4 --k_bits 8 --model CARN \
--cadyq --search_space 4+6+8 --save test_carn_cadyq_image \
--n_feats 64 --n_resblocks 9 --group 1 --multi_scale \
--student_weights experiment/carn_cadyq_x4/model/model_best.pth.tar \
#