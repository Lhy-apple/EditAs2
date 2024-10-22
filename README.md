# EditAs2

## Environment

our operating environment is based on the Pytorch Framework.
CUDA 11.3 can be applied to most current graphics card driver versions. And you can use `pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`to install pytorch-gpu. For other versions, you can see  [PyTorch](https://pytorch.org/) .


##  Dataset
Data_atlas and Data_integration Dataset are sourced from the paper "Automated Assertion Generation via Information Retrieval and Its Integration with Deep Learning."
Data_atlas, Data_integration, newly constructed cross-project test set, trained models and results can be downloaded from [here](https://workdrive.zohopublic.com.cn/folder/qwvthfb71c50db6484ac2a2f02af012240baa)


## Training the Model
To train the model, navigate to the scripts directory and run “fintuned.sh”:
```bash
cd scripts
./fintuned.sh 
```

## Testing the Model 
For inference, you can use the following script to run tests:
```bash
cd scripts
./test.sh 
```

## Evaluate
We provide several evaluation metrics to assess the performance of the model:
```bash
cd evaluator
```

### (1) Accuracy 
To compute the accuracy of the model, navigate to the “evaluator” directory and run the “compute_accuracy.py” script:
```bash
compute_accuracy.py
```

### (2) CodeBLEU
We also evaluate the model using the CodeBLEU metric, which is specifically designed for evaluating code generation tasks. To compute CodeBLEU, use the following command:
```bash
cd CodeBLEU
python calc_code_bleu.py --refs reference_files --hyp candidate_file --lang java --params 0.25,0.25,0.25,0.25
```

### (3) BugFound
Please refer to the README.md file in the "TEval-plus" directory for detailed instructions on how to set up and run the evaluation.


