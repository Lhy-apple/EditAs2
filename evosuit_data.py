import re
import argparse
import pandas as pd
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)

parser = argparse.ArgumentParser()
parser.add_argument(dest='file1_path')
parser.add_argument(dest='file2_path')
parser.add_argument(dest='input_path')
parser.add_argument(dest='output_path')

args = parser.parse_args()
file1_path = args.file1_path
file2_path = args.file2_path
input_path = args.input_path
output_path = args.output_path


codet5_tokenizer = RobertaTokenizer.from_pretrained('/data/lhy/LargeModel/codet5-base')


# input_string = "This is a string with spacesthat should be 'processed separately'ith spacesthat sh."
# output_string = remove_spaces_except_quotes(input_string)
# print(output_string)

def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    return tokens

def compare_strings_ignore_whitespace(str1, str2):
    # 移除字符串中的空格
    str1_without_space = re.sub(r'\s', '', str1)
    str2_without_space = re.sub(r'\s', '', str2)

    # 使用re.match()函数进行匹配比较
    if re.match(f'^{re.escape(str1_without_space)}$', str2_without_space):
        return True
    else:
        return False


with open(file1_path, "r",encoding='utf-8') as file1, open(file2_path, "r",encoding='utf-8') as file2:
    lines1 = file1.readlines()
    lines2 = file2.readlines()
raw_predictions=[]
acc=0
tol=0
match = []

for line1, line2 in zip(lines1, lines2):
    encoded_input = codet5_tokenizer.encode(line2, padding=True, truncation=True,max_length=150, return_tensors='pt')
    deconded_input = codet5_tokenizer.decode(encoded_input[0], skip_special_tokens=False)
    tol=tol+1
    raw_predictions.append(clean_tokens(line1))
#    if clean_tokens(line1)==clean_tokens(deconded_input):
#        acc=acc+1
#        match.append(1)
#    else:
#        match.append(0)
    if compare_strings_ignore_whitespace(clean_tokens(line1),clean_tokens(deconded_input)):
        match.append(1)
        acc=acc+1
    else:
        match.append(0)

#    print(clean_tokens(line1))
#    print(clean_tokens(deconded_input))
print(acc/tol)

df = pd.read_csv(input_path)
print(len(df))
print(len(raw_predictions))
#print(df.head(5))
#print(raw_predictions[0:5])
df["assert_pred"] = raw_predictions
df["match"] = match
df.drop('source',axis=1)

#统一格式
except_preds = [0] * len(df)
df['except_pred'] = except_preds
df.to_csv(output_path)

















        