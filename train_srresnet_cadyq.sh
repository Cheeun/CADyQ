CUDA_VISIBLE_DEVICES=6,7 python3 main.py \
--data_test Urban100+div2k_valid --dir_data /mnt/disk1/cheeun914/datasets/ --n_GPUs 2 \
--scale 4 --k_bits 8 --model SRResNet \
--cadyq --search_space 4+6+8 --save srresnet_cadyq_x4 \
--n_feats 64 --n_resblocks 16 --res_scale 1 \
--patch_size 192 --batch_size 16 \
--epochs 300 --lr 1e-4 --decay 150 --bitsel_lr 1e-4 --bitsel_decay 150 \
--loss_kd --loss_kdf --w_bit 1e-4 --w_bit_decay 1e-6 \
--teacher_weights pretrained/srresnet_x4_pams_w8a8.pth.tar \
--student_weights pretrained/srresnet_x4_pams_w8a8.pth.tar \
#