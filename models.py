import torch
import torch.nn as nn

class EditModel(nn.Module):
    def __init__(self, decoder_model,t5_tokenizer,args):
        super(EditModel,self).__init__()
        self.t5_model = decoder_model
        self.t5_tokenizer = t5_tokenizer
        self.args = args
    
    def forward(self, source_ids=None,source_mask=None,target_ids=None,target_mask=None):
        if target_ids is not None:
            outputs = self.t5_model(input_ids=source_ids, attention_mask=source_mask, labels=target_ids)
            loss, logits = outputs.loss, outputs.logits
            return loss, logits
        else:
            generate_ids = self.t5_model.generate(input_ids=source_ids,attention_mask=source_mask, max_length=self.args.max_dst_name_length)
            pres = self.t5_tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
            return generate_ids, pres
    