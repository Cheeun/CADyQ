# for bit in "${bits[@]}"
scale=4
bit=6
datasets=('Set5' 'Set14' 'B100' 'Urban100')

for dataset in ${datasets[@]}
do
    matlab -nodesktop -nosplash -r "calculate_PSNR_SSIM('$dataset',$scale,$bit);quit"
done
