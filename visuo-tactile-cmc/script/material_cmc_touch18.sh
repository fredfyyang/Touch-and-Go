CUDA_VISIBLE_DEVICES=0 python LinearProbing_touch.py --dataset touch_and_go --data_folder dataset/ --save_path ckpt/cls --model_path ckpt/cmc/memory_nce_16384_resnet18t2_lr_0.05_decay_0.0001_bsz_128_amount100_comment_view_Touch/ckpt_epoch_240.pth --model resnet18t2 --learning_rate 1.0 --layer 5