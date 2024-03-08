# EditAs2

## Environment

our operating environment is based on the Pytorch Framework.
CUDA 11.3 can be applied to most current graphics card driver versions. And you can use `pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`to install pytorch-gpu. For other versions, you can see  [PyTorch](https://pytorch.org/) .


#  Dataset
Our dataset,trained model and results can be downloaded from [here](https://workdrive.zohopublic.com.cn/folder/qwvthfb71c50db6484ac2a2f02af012240baa)


## Train
```bash
cd scripts
./fintuned.sh 
```

## Infer 
```bash
cd scripts
./test.sh 
```

## Evaluate
```bash
cd evaluator
```

### Accuracy 
```bash
compute_accuracy.py
```

### CodeBLEU
```bash
cd CodeBLEU
python calc_code_bleu.py --refs reference_files --hyp candidate_file --lang java --params 0.25,0.25,0.25,0.25
```

### BugFound
Please read the README.md file in the "TEval-plus" directory.

