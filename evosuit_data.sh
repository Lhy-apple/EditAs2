#!/bin/bash

#for i in `seq 1 1`;do
#    echo ${i}
#    file1_path=/data/lhy/Editas2/codet5_json_data_old_nocon_jaccard_1126_noedit_checkpoint-last_test_results/${i}/test.output
#    file2_path=/data/lhy/Editas2/codet5_json_data_old_nocon_jaccard_1126_noedit_checkpoint-last_test_results/${i}/test.gold
#    input_path=/data/lhy/TEval-plus/data/evosuite_buggy_tests_old/${i}/assert_model_inputs.csv
#    output_path=/data/lhy/TEval-plus/data/evosuite_buggy_tests_old/${i}/ours_oracle_preds.csv
#    python evosuit_data.py ${file1_path} ${file2_path} ${input_path}  ${output_path}
#rm -rf /tmp/run_bug_detection.pl_*
#done


#for i in `seq 5 5`;do
#    echo ${i}
#    file1_path=/data/lhy/Editas2/codet5_json_data_new_nocon_jaccard_1126_checkpoint-last_test_results/nju/${i}/test.output
#    file2_path=/data/lhy/Editas2/codet5_json_data_new_nocon_jaccard_1126_checkpoint-last_test_results/nju/${i}/test.gold
#    input_path=/data/lhy/TEval-plus/data/evosuite_buggy_tests_swf/${i}/assert_model_inputs.csv
#    output_path=/data/lhy/TEval-plus/data/evosuite_buggy_tests_swf/${i}/ours_oracle_preds.csv
#    python evosuit_data.py ${file1_path} ${file2_path} ${input_path}  ${output_path}
#rm -rf /tmp/run_bug_detection.pl_*
#done

###################150edit noequal
for i in `seq 1 10`;do
    echo ${i}
    file1_path=/data/lhy/Editas2/codet5_json_data_old_nocon_jaccard_noequal_1205longernoequal_checkpoint-last_test_results/nju/${i}/test.output
    file2_path=/data/lhy/Editas2/codet5_json_data_old_nocon_jaccard_noequal_1205longernoequal_checkpoint-last_test_results/nju/${i}/test.gold
    input_path=/data/lhy/TEval-plus/data/evosuite_buggy_tests_swf_old/${i}/assert_model_inputs.csv
    output_path=/data/lhy/TEval-plus/data/evosuite_buggy_tests_swf_old/${i}/ours2_oracle_preds.csv
    python evosuit_data.py ${file1_path} ${file2_path} ${input_path}  ${output_path}
rm -rf /tmp/run_bug_detection.pl_*
done


#for i in `seq 1 10`;do
#    echo ${i}
#    file1_path=/data/lhy/Editas2/codet5_json_data_new_nocon_jaccard_noequal_1208longernoequal_checkpoint-last_test_results/nju/${i}/test.output
#    file2_path=/data/lhy/Editas2/codet5_json_data_new_nocon_jaccard_noequal_1208longernoequal_checkpoint-last_test_results/nju/${i}/test.gold
#    input_path=/data/lhy/TEval-plus/data/evosuite_buggy_tests_swf/${i}/assert_model_inputs.csv
#    output_path=/data/lhy/TEval-plus/data/evosuite_buggy_tests_swf/${i}/ours2_oracle_preds.csv
#    python evosuit_data.py ${file1_path} ${file2_path} ${input_path}  ${output_path}
#rm -rf /tmp/run_bug_detection.pl_*
#done


##############noedit
#
#for i in `seq 3 10`;do
#    echo ${i}
#    file1_path=/data/lhy/Editas2/codet5_json_data_old_nocon_jaccard_1228_noedit_256_150_checkpoint-last_test_results/${i}/test.output
#    file2_path=/data/lhy/Editas2/codet5_json_data_old_nocon_jaccard_1228_noedit_256_150_checkpoint-last_test_results/${i}/test.gold
#    input_path=/data/lhy/TEval-plus/data/evosuite_buggy_tests_old/${i}/assert_model_inputs.csv
#    output_path=/data/lhy/TEval-plus/data/evosuite_buggy_tests_old/${i}/ours_noedit_oracle_preds.csv
#    python evosuit_data.py ${file1_path} ${file2_path} ${input_path}  ${output_path}
#rm -rf /tmp/run_bug_detection.pl_*
#done

#for i in $(seq 3 10);do
##for i in `seq 3 10`;do
#    echo ${i}
##    file1_path=/data/lhy/Editas2/codet5_json_data_new_nocon_jaccard_1228_noedit_256_150_checkpoint-last_test_results/${i}/test.output
##    file2_path=/data/lhy/Editas2/codet5_json_data_new_nocon_jaccard_1228_noedit_256_150_checkpoint-last_test_results/${i}/test.gold
##    input_path=/data/lhy/TEval-plus/data/evosuite_buggy_tests/${i}/assert_model_inputs.csv
##    output_path=/data/lhy/TEval-plus/data/evosuite_buggy_tests/${i}/ours_noedit_oracle_preds.csv
##    python evosuit_data.py ${file1_path} ${file2_path} ${input_path}  ${output_path}
##rm -rf /tmp/run_bug_detection.pl_*
#done
