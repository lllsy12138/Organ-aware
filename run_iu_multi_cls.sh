export PYTHONPATH=$PWD
srun -p Model_CV -N 1 --gres=gpu:1 --async \
python main.py \
--image_dir /mnt/lustre/lishiyu/datasets/report_gen/iu_xray/images \
--ann_path /mnt/lustre/lishiyu/datasets/report_gen/iu_xray/iu_annotation.json \
--dataset_name iu_xray \
--max_seq_length 30 \
--threshold 3 \
--num_heads 16 \
--batch_size 32 \
--epochs 100 \
--save_dir results/iu_xray_multi_cls_ \
--step_size 5 \
--gamma 0.9 \
--seed 12345678  \
--training_setting Multi_Cls_ABL