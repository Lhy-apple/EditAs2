# encoding=utf-8
import os
import re
import argparse
import pandas as pd
import tqdm
import json
import numpy
from ast import literal_eval
from .gen_tests_from_metadata import TogaGenerator, NaiveGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='input_dir')
    parser.add_argument(dest='result_dir')
    parser.add_argument(dest='model_name')
    args = parser.parse_args()
    gen_dir = args.input_dir
    result_dir = args.result_dir
    model_name=args.model_name
    #把结果文件merge出来
    final_df = pd.DataFrame(columns = ['bugfound','test_P','test_R','test_F1','FP_rate'])
    print(final_df)
    ids=[]
    index=[]
    for i in range(1,11):
      print('*****************')
      temp_dir = gen_dir.replace(gen_dir.split('/')[-2],str(i))
      work_dir=os.path.join(temp_dir, result_dir, "buggyfound_results.csv")
      if os.path.exists(work_dir):
        index.append(i)
        print(work_dir)
        temp_df = pd.read_csv(work_dir)
        temp_id=literal_eval(temp_df.loc[0,'bug_ids'])
        ids.extend(temp_id)
        final_df=pd.concat([final_df,temp_df])
    print(final_df)
    print(ids)
    ids=list(set(ids))
    final_df['index']=index
    final_df.to_csv(os.path.join("/data/lhy/TEval-plus/buggy_results",model_name+"buggy_result.csv"), index=False)
    ########
    f=open(os.path.join("/data/lhy/TEval-plus/buggy_results",model_name+"buggy_result.txt"),"w")
    f.writelines(str(ids))
    f.close()
    