#!/bin/sh

date=1208
model_type=codet5
dataset=datapath


cd ..
CUDA_VISIBLE_DEVICES=2,1 python run_updater_mutil_gpu.py \
--lang java \
--model_name_or_path /codet5-base-path \
--model_type $model_type \
--output_dir ${model_type}_${dataset}_${date}_models \
--data_dir ${dataset} \
--cache_path ${model_type}_${dataset}_${date}_cach \
--eval_dir ${model_type}_${dataset}_${date}_eval-results \
--summary_dir ${model_type}_${dataset}_${date}_summary \
--train_filename train.jsonl \
--eval_filename valid.jsonl \
--test_filename test.jsonl \
--max_src_test_length 256 \
--max_dst_test_length 256 \
--max_edit_seq_length 256 \
--max_src_ass_length 150 \
--max_dst_ass_length 150 \
--do_train \
--do_eval \
--write_to_pred \
--train_batch_size 8 \
--eval_batch_size 8 \
--num_train_epochs 15 \
--gradient_accumulation_steps 1 \
--weight_decay 0.001 \
--adam_epsilon 1e-4 \
--n_gpu 2 \



