from tqdm import tqdm
from abc import ABC, abstractmethod
import os
import argparse
import numpy as np
from io import open
import time
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
import multiprocessing
import torch.nn as nn
CPU_COUNT = 4

import torch
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
from sklearn.metrics import auc  
from bleu import _bleu, compute_sentence_level_blue

from configs import add_args, set_seed
from models import EditModel
from utils import load_and_cache_data


def main():
    print("main")
    parser = argparse.ArgumentParser()
    t0 = time.time()
    args = add_args(parser)
    logger.info(args)
    set_seed(args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")


    
    # Build model
    codet5_tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    codet5_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path).to(device)
    model = EditModel(decoder_model=codet5_model,t5_tokenizer=codet5_tokenizer,args=args)
    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(os.path.join(args.load_model_path,"pytorch_model.bin")))

    #
    if torch.cuda.device_count() > 1:
        print("使用{}个GPU".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model = model.to(device)

    pool = multiprocessing.Pool(CPU_COUNT)
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)
    # fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        
        summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
        tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        train_examples, train_dataset = load_and_cache_data(args, args.train_filename, pool, codet5_tokenizer, 'train', is_sample=False)
        train_sampler = SequentialSampler(train_dataset) 
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
        len_dataset = len(train_examples)
        t_total = (len_dataset // args.train_batch_size) * args.num_train_epochs if len_dataset % args.train_batch_size == 0 else (len_dataset // args.train_batch_size + 1) * args.num_train_epochs
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                        'weight_decay': args.weight_decay},
                        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                    ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total*0.1), num_training_steps=t_total)
        dev_dataset={}
        
        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)
        global_step,best_bleu,best_loss = 0,0,1e6
        best_eval_loss = 1e6
        for epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader,total=len(train_dataloader), desc="Training" )
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(device) for t in batch)
                source_ids,source_mask,target_ids,target_mask = batch
                loss,_ = model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids,target_mask=target_mask)

                if args.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                    print("multi-gpu")
                tr_loss += loss.item()
                nb_tr_steps += 1
                train_loss=round(tr_loss/(nb_tr_steps+1),4)
                bar.set_description("epoch {} loss {}".format(epoch,train_loss))
                # print('loss: ',train_loss)
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()
                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                            #Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

            # validation at the end of an epoch
            if args.do_eval:
                eval_examples, eval_data = load_and_cache_data(args, args.eval_filename, pool, codet5_tokenizer, 'eval', is_sample=False)
                eval_sampler = SequentialSampler(eval_data) 
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4)

                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0                     
                eval_flag=False    
    
                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                #Start Evaling model
                model.eval()
                eval_loss,step_num = 0,1
                for batch in eval_dataloader:   
                    batch = tuple(t.to(device) for t in batch)
                    source_ids,source_mask,target_ids,target_mask = batch
                    with torch.no_grad():
                        loss,_ = model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids,target_mask=target_mask)
                    if args.n_gpu > 1:
                        loss = loss.mean() # mean() to average on multi-gpu.
                    eval_loss = loss.item()
                    eval_loss = eval_loss / step_num
                    step_num += 1
                    result = {'eval_ppl': round(np.exp(eval_loss),5),
                                'global_step': global_step+1,
                                'train_loss': round(train_loss,5),
                                'eval_loss': round(eval_loss,5)}
                    
                    for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                    logger.info("  "+"*"*20)

                # save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)

                # save best-ppl checkpoint
                if eval_loss<best_eval_loss:
                    logger.info("  Best ppl:%s",round(np.exp(eval_loss),5))
                    logger.info("  "+"*"*20)
                    best_eva_loss=eval_loss
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)


    if args.do_test:

        test_examples, test_data = load_and_cache_data(args, args.test_filename, pool, codet5_tokenizer, 'test', is_sample=False)
        test_sampler = SequentialSampler(test_data) 
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size,num_workers=4)

        logger.info("\n***** Running testing *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.test_batch_size)

        model.eval()
        best_bleu = -100 
        pre_text=[]
        bar = tqdm(test_dataloader,total=len(test_dataloader))
        for batch in bar:
            batch = tuple(t.to(device) for t in batch)
            source_ids,source_mask,target_ids,target_mask = batch
            with torch.no_grad():
                generate_ids, batch_text = model(source_ids=source_ids, source_mask=source_mask, target_ids=None, target_mask=None)
            for text in batch_text:
                pre_text.append(text)
        
        print('-=-='*40)
        gold_name_list = []
        pre_name_list = []
        if not os.path.exists(args.results_dir):
            os.mkdir(args.results_dir)
        with open(os.path.join(args.results_dir,"test.output"),'w',encoding='utf-8') as f, open(os.path.join(args.results_dir,"test.gold"),'w',encoding='utf-8') as f1:
            for pre_name,gold in zip(pre_text,test_examples):
                    f.write(pre_name+'\n')
                    f1.write(gold.dst_name+'\n')
       
    # fa.close()

if __name__ == "__main__":
    main()
