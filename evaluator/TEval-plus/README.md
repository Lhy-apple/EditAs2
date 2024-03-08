# README

## Prepare the environment
```bash
conda create -n revisit_toga python=3.9
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip3 install -r requirements.txt
pip3 uninstall protobuf
pip3 install protobuf==3.20
```

## Install Defects4J
- See: https://github.com/rjust/defects4j
- Install v2.0.0
- Patch Defects4j to fix compilation errors

## Prepare Test Prefixes and generate assertions

```bash
# run EvoSuite 10 times on the buggy versions
# generate assertions using assertion generation method
# Our randomly generated test cases can be downloaded from [here](https://workdrive.zohopublic.com.cn/folder/qwvthfb71c50db6484ac2a2f02af012240baa)
bsah rq0.sh
```

##  run experiments  
```bash
bash run_rq1_2.sh
```
