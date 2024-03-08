import logging
import os
import random
import numpy as np
from tqdm import tqdm
import jsonlines
logger = logging.getLogger(__name__)

import torch
from torch.utils.data import TensorDataset


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 edit_seq,
                 src_test,
                 dst_test,
                 src_ass,
                 dst_ass,
                 ):
        self.idx = idx
        self.edit_seq = edit_seq
        self.src_test = src_test
        self.dst_test = dst_test
        self.src_ass = src_ass
        self.dst_ass = dst_ass


def read_examples_jsonl(file_path, data_num=-1, task='train'):
    examples = []
    with jsonlines.open(file_path) as reader:
        for idx,line in enumerate(reader):
            examples.append(
                 Example(
                    idx = line["sample_id"],
                    edit_seq = line['code_change_seq'],
                    src_test = line["src_method"],
                    dst_test = line["dst_method"],
                    src_ass = line["src_desc"],
                    dst_ass = line["dst_desc"],
                )
            )

            if idx < 2:
                print("=="*50)
                print("idx: {}".format(examples[idx].idx))
                print("edit_seq: {}".format(examples[idx].edit_seq))
                print("src_test: {}".format(examples[idx].src_test))
                print("dst_test: {}".format(examples[idx].dst_test))
                print("src_ass: {}".format(examples[idx].src_ass))
                print("dst_ass: {}".format(examples[idx].dst_ass))
                

    if data_num != -1:
        return examples[:data_num]
    
    return examples

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,                 
                 target_ids,
                 source_mask,
                 target_mask, 

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask 

def convert_examples_to_features(item):
    example, example_index, tokenizer, args = item
    edit_seq= example.edit_seq
    src_test = example.src_test
    dst_test = example.dst_test
    src_ass = example.src_ass
    dst_ass = example.dst_ass
    edit_input_ids = []
    edit_attention_mask = []

    for item in edit_seq:
        if len(item) >= 3:
            item = item[:3]
        encoded_tmp = tokenizer(item, padding=True, truncation=True,max_length=args.max_edit_seq_length, return_tensors='pt')
        edit_input_ids.extend(np.array(encoded_tmp['input_ids']).flatten())
        edit_attention_mask.extend(np.array(encoded_tmp['attention_mask']).flatten())

    edit_input_ids = edit_input_ids[:args.max_edit_seq_length]
    edit_attention_mask = edit_attention_mask[:args.max_edit_seq_length]
    padding_length1 = args.max_edit_seq_length - len(edit_input_ids)
    padding_length2 = args.max_edit_seq_length - len(edit_attention_mask)
    edit_input_ids += [tokenizer.pad_token_id]*padding_length1
    edit_attention_mask += [0]*padding_length2

    encoding_src_test = tokenizer(src_test, max_length=args.max_src_test_length,truncation=True,return_tensors="pt")
    src_test_input_ids= encoding_src_test['input_ids'][0].tolist()
    src_test_attention_mask = encoding_src_test['attention_mask'][0].tolist()
    padding_length = args.max_src_test_length - len(src_test_input_ids)
    src_test_input_ids += [tokenizer.pad_token_id]*padding_length
    src_test_attention_mask += [0]*padding_length

    encoding_dst_test = tokenizer(dst_test, max_length=args.max_dst_test_length,truncation=True,return_tensors="pt")
    dst_test_input_ids= encoding_dst_test['input_ids'][0].tolist()
    dst_test_attention_mask = encoding_dst_test['attention_mask'][0].tolist()
    padding_length = args.max_dst_test_length - len(dst_test_input_ids)
    dst_test_input_ids += [tokenizer.pad_token_id]*padding_length
    dst_test_attention_mask += [0]*padding_length


    encoding_src_ass = tokenizer(src_ass, max_length=args.max_src_ass_length,truncation=True,return_tensors="pt")
    src_ass_input_ids= encoding_src_ass['input_ids'][0].tolist()
    src_ass_attention_mask = encoding_src_ass['attention_mask'][0].tolist()
    padding_length = args.max_src_ass_length - len(src_ass_input_ids)
    src_ass_input_ids += [tokenizer.pad_token_id]*padding_length
    src_ass_attention_mask += [0]*padding_length

 
    source_ids = edit_input_ids + src_test_input_ids + dst_test_input_ids + src_ass_input_ids
    source_mask = edit_attention_mask + src_test_attention_mask + dst_test_attention_mask + src_ass_attention_mask




    tgt_seq = dst_ass
    encoding_tgt = tokenizer(tgt_seq, max_length=args.max_dst_ass_length,truncation=True,return_tensors="pt")
    target_ids = encoding_tgt.input_ids[0].tolist()
    target_mask = encoding_tgt.attention_mask[0].tolist()
    padding_length = args.max_dst_ass_length - len(target_ids)
    target_ids += [tokenizer.pad_token_id]*padding_length
    target_mask += [0]*padding_length
    target_ids = torch.tensor(target_ids)
    target_ids[target_ids == tokenizer.pad_token_id] = -100
    target_ids = target_ids.tolist()


    return InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask
        )

def load_and_cache_data(args, filename, pool, tokenizer, split_tag, only_src=False, is_sample=False):
    if not os.path.exists(args.cache_path):
        os.makedirs(args.cache_path)
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + '_all' if args.data_num == -1 else '_%d' % args.data_num)
    file_path = os.path.join(args.data_dir,filename)

    examples = read_examples_jsonl(file_path, args.data_num, args.task)

    if is_sample:
        examples = random.sample(examples, int(len(examples) * 0.1))

    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 20 percent of data from %s", filename)
        logger.info("Create cache data into %s", cache_fn)

        tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]
        features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in features], dtype=torch.long)

        print('=--='*100)
        print(all_source_ids.shape)
        print(all_source_mask.shape)
        print(all_target_ids.shape)
        print(all_target_mask.shape)
        assert(all_source_ids.shape[0] == all_source_mask.shape[0] and all_source_ids.shape[0] != 0)
        
        data = TensorDataset(all_source_ids, all_source_mask,all_target_ids,all_target_mask)
        torch.save(data, cache_fn)

    return examples, data

