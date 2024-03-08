
# encoding=utf-8
import os
import re
import argparse
import pandas as pd
import tqdm
from .gen_tests_from_metadata import TogaGenerator, NaiveGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='input_dir')
    parser.add_argument(dest='result_dir')
    parser.add_argument(dest='model_name')
    args = parser.parse_args()
    gen_dir = args.input_dir
    result_dir = args.result_dir
    work_dir = os.path.join(gen_dir, "..")
    failed_result_df = pd.read_csv(os.path.join(gen_dir, result_dir, "failed_test_data.csv"))
    full_result_df = pd.read_csv(os.path.join(gen_dir, result_dir, "full_test_data.csv"))
    #生成一个找到了bug的文件
    buggyfound_test_data = failed_result_df.loc[failed_result_df['TP'] == True]
    print(buggyfound_test_data)
    print(len(buggyfound_test_data))
    buggyfound_test_data.to_csv(os.path.join(gen_dir, result_dir, "buggyfound_test_data.csv"), index=False)
    #生成一个csv文件，记bugfound,p,R，F1,FPR
    bugfound=len(buggyfound_test_data)
    TPs = full_result_df['TP'].sum()
    FPs = full_result_df['FP'].sum()
    TNs = full_result_df['TN'].sum()
    FNs = full_result_df['FN'].sum()

    test_P = TPs / (TPs + FPs)
    test_R = TPs / (TPs + FNs)
    test_F1 = 2 * test_P * test_R / (test_P + test_R)
    FP_rate = FPs / (FPs + TNs)
    print(test_P)
    print(test_P)
    print(test_F1)
    print('******************')
    result_df=pd.DataFrame(data=[[bugfound,test_P,test_R,test_F1,FP_rate]],
             columns = ['bugfound','test_P','test_R','test_F1','FP_rate'],
             index=[0])

#             
#    result_df = pd.DataFrame({
#        'bugfound': bugfound,
#        'test_P': test_P,
#        'test_R': test_R,
#        'test_F1': test_F1,
#        'FP_rate': FP_rate
#    })
    print(result_df)
    result_df.to_csv(os.path.join(gen_dir, result_dir, "buggyfound_results.csv"), index=False)
    
