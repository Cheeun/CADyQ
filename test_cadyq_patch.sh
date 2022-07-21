CUDA_VISIBLE_DEVICES=7 python3 main.py \
--test_only \
--data_test test2k+test4k --dir_data /mnt/disk1/cheeun914/datasets/ --n_GPUs 1 \
--scale 4 --k_bits 8 --model CARN \
--cadyq --search_space 4+6+8 --save test_carn_cadyq_patch_2 \
--n_feats 64 --n_resblocks 9 --group 1 --multi_scale \
--student_weights experiment/carn_cadyq_x4/model/model_best.pth.tar \
--test_patch --step_size 90 --patch_size 96 \
# 