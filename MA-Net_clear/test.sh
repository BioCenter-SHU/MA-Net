cuda_id=$1

python3 test.py --is_shift --dataset sthv2 --clip_len 60 --shift_div 8 \
                --cuda_id $cuda_id --batch_size 2 --test_crops 1 --scale_size 256 \
                --crop_size 256  --base_model resnet18 \
                --root_path 'frame_120/test_60_video.txt'
                --checkpoint 'checkpoint/2022-06-15-04-35-23/clip_len_60_4_Fold_checkpoint.pth.tar'