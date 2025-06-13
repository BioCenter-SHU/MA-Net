cuda_id=$1

python3 ./main_down.py --is_train --k 5\
                 --dataset sthv2 --clip_len 30\
                 --frame_path "/YOUR_PATHset/0_single_plaque/video_302_frame_120/" \
                 --save_path 'KF2/method4_1' --module 'KF2_4_1'\
                 --wd 1e-2 --dropout 0.9 \
                 --cuda_id $cuda_id --batch_size 2 \
                 --lr 1e-4 --base_model resnet50 --epochs 100 --num_workers 1