export PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:31232
srun -p Model_CV -N 1 --gres=gpu:1 \
python main.py \
--image_dir /mnt/lustre/lishiyu/datasets/report_gen/mimic_cxr/images \
--ann_path /mnt/lustre/lishiyu/datasets/report_gen/mimic_cxr/mimic_annotation.json \
--dataset_name mimic_cxr \
--max_seq_length 60 \
--threshold 10 \
--num_heads 16 \
--batch_size 64 \
--epochs 20 \
--save_dir results/mimic_cxr_multi_cls_ABL \
--step_size 1 \
--gamma 0.9 \
--seed 123456  \
--pretrained 1 \
--training_setting Multi_Cls_ABL
 