CUDA_VISIBLE_DEVICES=4 python3 main.py \
--data_test Urban100+div2k_valid --dir_data /mnt/disk1/cheeun914/datasets/ --n_GPUs 1 \
--scale 4 --k_bits 8 --model IDN \
--cadyq --search_space 4+6+8 --save idn_cadyq_x4_2 \
--idn_d 16 --idn_s 4 --n_resblocks 4 \
--patch_size 192 --batch_size 16 \
--epochs 300 --lr 1e-4 --decay 150 --bitsel_lr 1e-4 --bitsel_decay 150 \
--loss_kd --loss_kdf --w_bit 1e-5 --w_bit_decay 1e-7 \
--teacher_weights pretrained/idn_x4_pams_w8a8.pt \
--student_weights pretrained/idn_x4_pams_w8a8.pt \
#