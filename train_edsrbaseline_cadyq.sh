CUDA_VISIBLE_DEVICES=7 python3 main.py \
--data_test Urban100+div2k_valid --dir_data /mnt/disk1/cheeun914/datasets/ --n_GPUs 1 \
--scale 4 --k_bits 8 --model EDSR \
--cadyq --search_space 4+6+8 --save edsrbaseline_cadyq_x4 \
--n_feats 64 --n_resblocks 16 --res_scale 1 \
--patch_size 192 --batch_size 16 \
--epochs 300  --decay 150 --lr 1e-4 --bitsel_lr 1e-4 --bitsel_decay 150 \
--loss_kd --loss_kdf --w_bit 1e-4 --w_bit_decay 1e-6 \
--teacher_weights pretrained/edsr_x4_pams_w8a8.pt \
--student_weights pretrained/edsr_x4_pams_w8a8.pt \
# 