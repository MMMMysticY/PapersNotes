# BERT
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  

## 背景
如何对token做word embedding一直是一个NLP领域中火热的话题  
Pre-trained word embedding是现代NLP最重要的部分之一 与从头开始的学习相比有了明显提升  
一般有两种方法去做 一种是单向的 训练过程只能看到之前的信息 另一种是双向的 之前之后的信息都可以看到  
ELMO类的方法做word embedding时用的是从左向右 从右向左地提取上下文特征 token的学习是二者结合的结果  
一般的训练方法就是完形填空类似的任务 是不需要label的数据就可以做的unsupervised learning  
而之后只需要少量的有监督样本去fine-tune到特定的任务上 就可以取得很好的效果  
pre-train + fine-tune模式可以如图所示：  
![bert1](img/bert.png)  

## 模型
模型所使用的的就是Transformer Encoder  
BERT BASE的参数量： Layer-num = 12 Hidden-size = 768 Num-head = 12 Params = 110M  
BERT LARGE的参数量：Layer-num = 24 Hidden-size = 1024 Num-head = 16 Params = 340M  
## 模型细节
![bert](img/bert_model.png)  
对于一个给定的token 它在进入BERT前是由三个部分组成的: Token embedding/ Segment embedding/ Position Embedding  

## Input/Output
### Input
BERT为了尽可能适应所有的下游任务 其输入可以是单个句子也可以是一对句子  
Sentence的概念是一段连续文本的任意一段 而不需要是绝对意义上的一个句子  
Sequence的概念是BERT的input token sequence 可以是一个也可以是两个Sentence  
1. 使用的是WordPiece embedding 有30000个token词
2. 每个句子的开头都有一个[CLS] 符号 与这个标记对应的输出结果可以用作分类问题使用
3. 如果是两个句子的话 使用[SEP] 符号进行分割 之后通过一层embedding的学习 segment embedding自动地将两个句子分开
### output
output部分就是Ti CLS对应的就是C  
![output](img/bert1.png)  

## Pre-train
### MLM masked Language Model
之前不能双向训练而只能从左往右从右往左的原因在于 双向结构允许网络看到"自己" 那么模型就可以轻松地预测出"自己"应该是什么  
为了能够双向训练 那么就随机地mask掉一些token 任务就是预测那些被mask的token应该是什么 这就是MLM  
通过mask一些token之后对应位置的预测结果通过softmax进行分类确定词 进行训练  

#### MASK方法
MASK的比例使用的是15%  
如果直接进行MASK 那么[MASK] 这个token在fine-tune的地方并不存在这样就造成了明显的pre-train和fine-tune部分的不匹配  
为了解决这个问题 采用的方法是mask方式按比例变化：  
1. 80%的token用[MASK]
2. 10%随机成另一个token
3. 10%不变  

这个方法的优点在于Transformer encoder不知道它将被要求预测哪些词或哪些词已被随机词替换 因此它被迫考虑每个输入标记的所有分布上下文表示  
而因为随机替换有可能导致问题仅仅占1.5% 对模型影响不大  


### Next Sentence Prediction
许多下游任务 类似于问答Question Answering 和 自然语言推理Natural Language Inference都是要了解两个句子之间的关系  
为了尝试完成这样的工作 就提出了next sentence prediction的模型进行训练  
在训练时有两个句子A B 有50%的语料中B是A后面的句子 而另外50%的语料中B不是A后面的句子  
那么在结构中C的部分就是next sentence prediction的结果  
![nsp](img/nsp.png)  


## bert使用场景
![bert_fine-tune3](img/bert_fine-tune3.png)  

### seq -> class
输入是序列 输出是类别 举例而言就是情绪识别问题 输入一个句子 输出是积极还是消极  
![bert_case1](img/bert_case1.png)  
总体的模型还是supervised learning 还是需要该任务提供输入样本和label对  
模型结构是bert+linear 其中bert的参数是pre-train的 linear参数是随机初始化的  
训练过程就是在输入文本前加入CLS token 经过linear + softmax就进行了n分类  
在更新参数的时候是都更新的  
pre-train的bert参数比随机初始化参数要好很多  

### seq -> seq (两个seq长度相同)
输入是序列 输出也是序列 而且两个序列长度相同 举例而言就是词性识别问题 对于句子中每个词 进行词性标注  
![bert_case2](img/bert_case2.png)  
模型结构类似 还是有CLS 不过结果和CLS关系并不直接  

### 2 seq -> class
输入是两个序列 输出是一个类别  
举例而言有自然语言推论问题 即给一个句子作为前提 另一个句子作为推断结果 去判断能否由第一个前提推断出第二个结果  
举例而言还有立场分析 第一个输入是一个文章 第二个输入是一个评论 去判断该评论是支持还是反对该文章的观点  
输入是CLS + SEN1 + SEP + SE2 需要的部分和场景一一样 只需要CLS部分的输出  
![bert_case3](img/bert_case3.png)  

### 2 seq -> int值
输入是两个序列 输出是一个int值  
举例而言是一个答案在文章中的问答系统 输入是一个文章 和 一个问题  
而输出是[s,e]就是文章中[s,e]部分的内容就是这个问题的答案  
![bert_case4](img/bert_case4.png)  
输入是CLS + SEN1 + SEP + SE2 和场景3一样  
经过bert之后 需要随机初始化两个向量 一个去计算start 一个去计算end  
start和文章部分的向量去做运算经过softmax得到的就是初始位置s end向量和文章部分向量做运算经过softmax得到的是结束位置e  
![bert_case4_1](img/bert_case4_1.png)  


## how to fine-tune
微调模型的方法(仍然是以NLP任务举例)  
NLP任务分类:  
![nlp_tasks](img/nlp_tasks.png)  
input分为两类：一个seq / 多个seq output分为四类：one-class / 对于每个token一个分类 / input的赋值 / 一般的seq(seq2seq)  
### input分类
1. 一个seq 就是正常地输入到网络中
2. 在多个seq之间加入特殊token:分隔符sep

### output分类
1. one-class 需要在所有输入之前加入特殊token:CLS
2. 每个token一个class就是最正常的LSTM或者transformer encoder就可以
3. 可能的情况就像bert使用场景中的2seq->int一样
4. seq2seq问题简单地解决就是将bert部分当做encoder 输出结果作为attention放入另一个decoder网络中  
![bert_seq2seq_1](img/bert_seq2seq1.png)  
但是这样在训练的时候encoder是参数初始化的 decoder是参数未初始化的 在一般fine-tune的时候 样本数比较少 可能无法训练好decoder  
更合理的办法是将pre-train model同时当做encoder和decoder 具体做法如下:  
![bert_seq2seq_2](img/bert_seq2seq2.png)  

### fine-tune方法
![bert_fine-tune](img/bert_fine-tune.png)  
fine-tune有两种方法:  
1. 固定pre-train model 不作调整
2. 同时调整pre-train model和新加入的部分
结果而言后者往往更好一些  
### 存储方法
由于模型往往很大 所以不同的任务如果对于每个任务都存储一个模型 那存储空间过大 而且存储的大部分都是相同的内容  
所以 每次存储只要存储相对于原始Pre-train model来说 调整的部分即可  
![bert_fine-tune2](img/bert_fine-tune2.png)  


