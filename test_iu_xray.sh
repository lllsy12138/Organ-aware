export PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:31232
srun -p Model_CV -N 1 --gres=gpu:1 --async \
python main_test.py \
    --image_dir /mnt/lustre/lishiyu/datasets/report_gen/iu_xray/images \
    --ann_path /mnt/lustre/lishiyu/datasets/report_gen/iu_xray/iu_annotation.json \
    --dataset_name iu_xray \
    --max_seq_length 60 \
    --threshold 3 \
    --epochs 100 \
    --batch_size 16 \
    --lr_ve 1e-4 \
    --lr_ed 5e-4 \
    --step_size 10 \
    --gamma 0.8 \
    --num_layers 3 \
    --topk 32 \
    --cmm_size 2048 \
    --cmm_dim 512 \
    --seed 7580 \
    --beam_size 3 \
    --save_dir results/iu_xray/ \
    --log_period 50 \
    --load data/model_iu_xray.pth
