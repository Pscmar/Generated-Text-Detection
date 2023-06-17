# 坑の说明

## 环境配置

选择python>=3.8的pytorch镜像，安装apex。

找个临时文件夹，执行以下：
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

环境就配置好了，接下来（换个不临时的文件夹）克隆这个库以及安装其他包：
```bash
git clone git@github.com:Congyuwang/academic-budget-bert.git
cd academic-budget-bert
pip install -r requirements.txt
```

DONE.

## 预训练  

预训练分为两个部分，第一阶段是用wiki的数据进行预训练，得到初始BERT；第二阶段是在第一阶段的checkpoint下，用自己领域的语料库继续预训练，在这里我们使用labor数据（招聘信息）。  

所有的执行命令在./scripts/文件夹下，我们跳转到这个目录，并在执行前记得先用`chmod +x`给执行权限。以下是一个示例：
```bash
cd scripts
chmod +x xxx.sh
```
### 第一阶段 - wiki  

- wiki数据下载  

`./download_wiki.sh`

- BERT词汇表下载

`./download_vocab.sh`

- wiki数据处理

    `./extract_wiki.sh`  

    这一步把我们下载好的wiki数据转换成一个txt文件，方便后面的数据处理。

    `./sharding.sh`  

    在这一步之前，txt文件是一个article一行，执行这一步之后，txt文件变为一个sentence一行，也就变成BERT能够接受的文本形式了。此外，在这一步中还划分了训练集和测试集，为后续的预训练做准备。

    `./generate_samples.sh`  

    直接的文本数据还不能用，我们需要把它转化成dataloader能够用的样子。

- 预训练  

万事俱备，只用`./pretraining.sh`就可以开始我们的第一阶段预训练过程了！

### 第二阶段 -labor

- labor数据准备

Need to be completed...

- labor数据处理  

执行以下：
```bash
cd /
cd academic-budget-bert/laborscripts
./sharding.sh
./generate_samples.sh
./pretraining.sh
```
注：如果执行.sh文件出现问题，记得查看是不是没有用`chmod +x`给权限。
