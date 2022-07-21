CUDA_VISIBLE_DEVICES=3 python3 main.py \
--data_test Urban100+div2k_valid --dir_data /mnt/disk1/cheeun914/datasets/ --n_GPUs 1 \
--scale 4 --k_bits 8 --model CARN \
--cadyq --search_space 4+6+8 --save carn_cadyq_x4_2 \
--n_feats 64 --n_resblocks 9 --group 1 --multi_scale \
--patch_size 192 --batch_size 16 \
--epochs 600 --lr 1e-4 --decay 400 --bitsel_lr 1e-3 --bitsel_decay 400 \
--loss_kd --loss_kdf --w_bit 1e-5 --w_bit_decay 1e-7 \
--teacher_weights pretrained/carn_x4_pams_w8a8.pt \
--student_weights pretrained/carn_x4_pams_w8a8.pt \
#