#!/bin/sh

date=1208
model_type=codet5
dataset=datapath
chheckpoint=checkpoint-last

cd ..
CUDA_VISIBLE_DEVICES=0 python run_updater_single_gpu.py \
--lang java \
--model_name_or_path /codet5-base-path \
--load_model_path ${model_type}_${dataset}_${date}_models/$chheckpoint \
--model_type $model_type \
--results_dir ./${model_type}_${dataset}_${date}_${chheckpoint}_test_results \
--data_dir /${dataset} \
--cache_path ${model_type}_${dataset}_${date}_cach \
--summary_dir ${model_type}_${dataset}_${date}_summary \
--test_filename test.jsonl \
--do_test \
--max_src_test_length 256 \
--max_dst_test_length 256 \
--max_edit_seq_length 256 \
--max_src_ass_length 150 \
--max_dst_ass_length 150 \
--test_batch_size 8 \



