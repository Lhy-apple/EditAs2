
import re
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)


codet5_tokenizer = RobertaTokenizer.from_pretrained('/codet5-base')

def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    return tokens

with open("test.output", "r",encoding='utf-8') as file1, open("test.gold", "r",encoding='utf-8') as file2:
    lines1 = file1.readlines()
    lines2 = file2.readlines()
acc=0
tol=0
for line1, line2 in zip(lines1, lines2):
    encoded_input = codet5_tokenizer.encode(line2, padding=True, truncation=True,max_length=150, return_tensors='pt')
    deconded_input = codet5_tokenizer.decode(encoded_input[0], skip_special_tokens=False)
    tol=tol+1
    if clean_tokens(line1)==clean_tokens(deconded_input):
        acc=acc+1
    else:
        print(clean_tokens(line1))
        print(clean_tokens(deconded_input))
print(acc)
print(acc/tol)


