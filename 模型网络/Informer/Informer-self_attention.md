# Informer笔记

## 背景
时序预测问题(Time-series forecasting)应用于很多领域中：
1. 传感器网络监控(sensor network monitoring)
2. 能源和智能电网管理(energy and smart grid management)
3. 经济与金融(economics and finance)
4. 病害传播分析(disease propagation analysis)

在这些问题中，可以利用**大量过去**行为的时间序列来进行**长期预测**，称为长序列时间预测(long sequence time-series forecasting (LSTF))  

LSTF问题的主要挑战是：
1. 非凡的长序列处理能力 (要能够处理过去很长的信息 也要能预测未来很长的信息)
2. 对长序列输入和输出的高校操作

Transformer对于LSTF问题的解决：  
Transformer中的自注意力机制(self-attention)可以将网络信号传播路径的最大长度减少到理论上最短的 O(1) 并避免循环结构  
但是Transformer模型违反了上述第二点 因为其对于算力和内存的需求过高  

因此 希望找到对transformer结构的优化 使得其能应用于LSTF问题  

## 问题描述
在具有固定窗口长度的时序问题中，在某一时间t输入一个输入向量X(t) = [x1, x2, ... xm] 目的为输出一个预测值Y(t) = [y1, y2, ... yn]  
Encoder-Decoder结构中 将X(t)编码为隐层状态H(t) = [h1, h2, ... hl] 之后通过解码器一步一步进行解码 解码器的输入是h(t) 和可能的输入计算预测值y(t)  


## 方法
### 对self-attention机制的优化
#### 基本的self-attention机制
对于输入[a1, a2, a3 ... an]来说 每个位置都要和其他位置进行计算attention dot-product attention  
![self-attention1](img/self-attention-1.png)  
先通过维度变化Wq和Wk将输入变换为可以点积的维度，然后进行直接点积(dot) 得到attention score  
跟所有的位置(包括自己)进行计算后，再经过softmax就得到了该位置相对于其他位置的所有attention score  
![self-attention2](img/self-attention-2.png)  
在得到了所有的attention score后要应用于数据中 将输入乘以Wv变为合适的维度 然后和之间的attention score加权求和 便得到了计算结果  
![self-attention11](img/self-attention-11.png)  
同样地 每个位置都要和别的所有位置进行计算得到[b1, b2, ... , bn]  
**self-attention的计算完全不用考虑时序性质，可以进行并行计算**  
  
从矩阵的角度来看，对于所有输入[a1, a2, ... an] 都要计算获得key query value 即[key1, key2, ... keyn] [query1, query2, ... queryn] [value1, value2, ... valuen]  
计算的矩阵为Wk, Wq, Wv (这三个矩阵为网络的参数 是需要学习的)  图示如下：  
![self-attention3](img/self-attention-3.png)  
  
之后要计算attention score 对于某个单一位置来说 attention score的计算时使用当前的query去和所有的key相乘  
![self-attention4](img/self-attention-4.png)  
那么对于所有的位置来说，就是对上述过程进行重复 数学表示就是Q乘以K的转置得到了attention_score矩阵
![self-attention5](img/self-attention-5.png)  
  
对于attention score的使用 就是和所有的V进行乘法求和  
![self-attention6](img/self-attention-6.png)  
  
整体效果如下图所示：  
![self-attention7](img/self-attention-7.png)  
  
如果从维度和代码的角度来看 输入为input [batch, seq_len, feature] 而一般的Wk, Wq, Wv就是线性层 将其转化为[batch, seq_len, dk] Q, K, V都是这样的维度  
那么Q*K的转置 得到的就是[batch, seq_len, seq_len] 经过softmax维度不变，再和V相乘之后的维度是[batch, seq_len, dk] 最后再恢复到[batch, seq_len, feature](维度恢复)  
Transformer中的self-attention 和上述唯一的差别是多了一个除法 除以根号下dk  
原因在于当dk非常大时，Q*K(T)点积的量级很大，将softmax推到极小的梯度区域去， 除以根号下dk 可以缓解这个问题  
  
self-attention的复杂度分三个部分：
1. 相似度计算时间(QK(T)) 为(seq_len, dk) * (dk, seq_len) -> (seq_len, seq_len) 复杂度为O(seq_len ^ 2 * dk)  (矩阵乘法复杂度 就是相同的那个维度乘dk次，然后有目标矩阵seq_len^2个位置)
2. softmax时间复杂度本身是O(n) 然后有seq_len^2个位置 则为O(seq_len ^ 2)
3. 加权平均的时间复杂度 (seq_len, seq_len) * (seq_len, dk) -> (seq_len, dk) 复杂度还是O(seq_len ^2 * dk)
所以时间复杂度就是O(seq_len^2 * dk)  

#### multi-head attention机制
multi-head attention的本质就是有多组Wk, Wq, Wv 直觉上考虑就是有多种的attention模式 多重对于输入不同的注意方式，多重不同的相关性  
![self-attention8](img/self-attention-8.png)  

有了多组的key query 对后 就可以得到多组attention score 和对应的多组value计算加权求和 可以得到多组的b  
![self-attention9](img/self-attention-9.png)  
在得到多组的b之后 通过一个新的矩阵Wo 就可以将维度恢复 进入下一层  
![self-attention10](img/self-attention-10.png)  
从维度和代码的角度来看 input仍然是[batch, seq_len, feature] Wk, Wq Wv的维度为[batch, seq_len, hidden] Q, K, V都是这样的维度  
之后因为有multi-head，所以要将hidden分解为num_head * dk 即 hidden = num_head(m) * dk(d), [batch, seq_len, num_head, dk] 再转置之后[batch, num_head, seq_len, dk]  
Q*K(T)得到[batch, num_head, seq_len, seq_len] 经过softmax维度不变， 再和V相乘之后得到[batch, num_head, seq_len, dk] 转置之后变为[batch, seq_len, num_head, dk]  
再拼接内部两维维度恢复为[batch, seq_len, hidden] 最后再维度恢复为[batch, seq_len, feature]  

#### Encoder输出作为attention作用于Decoder上时
query为decoder矩阵 key value都是encoder的输出  
encoder的输出为[batch, seq_len_encoder, num_head, dk(两边统一即可)]  decoder的输入为[batch, seq_len_decoder, num_head, dk(两边统一即可)]  
attention score为：Q*K(T) [batch, num_head, seq_len_decoder, seq_len_encoder]  再和 value同样还是encoder输出的[batch, num_head, seq_len_encoder, dk]点积  
得到的结果就是[batch, num_head, seq_len_decoder, dk]  这样encoder的维度就不存在了 但是其已经作用于decoder上了  


#### 优化方式
由于self-attention是每个位置的query和所有位置的key进行点积 之后再经过softmax算成概率分布  
当对于所有位置的考虑度相近的时候 概率分布就变成了均匀分布 但是事实上 并不是均匀分布 而是对有些key考虑非常大 很多key的考虑很小  
考虑度可以用 Kullback-Leibler divergence进行计算  
在得到了相关性之后，就可以对self-attention进行优化 变为问题稀疏的self-attention(ProbSpare self-attention)  
做的方法就是在每次进行self-attention时 query部分使用原来query的top-u个其他的都是空 top-u是个超参数  
这样在计算的时候 除了top-u的其他都是空的不计算 将时间复杂度从(seq_len, dk) * (dk, seq_len) -> (seq_len, seq_len) 中第一个seq_len只计算top-u个数  
虽然矩阵维度不变，仍然是(seq_len', dk) * (dk, seq_len) -> (seq_len, seq_len) 但是复杂度可以变成O(seq_len' * seq_len * dk) 作者说是ln(seq_len) * seq_len * dk  

### Encoder
![encoder](img/encoder.png)  
encoder中先是使用了上述优化过的ProbSpare self-attention机制作为self-attention部分  
然后使用了self-attention Distilling 自注意蒸馏的方法  
具体做法是在经过ProbSpare self-attention之后 加入一个Conv1d和maxPool(stride = 2)层 这样就可以经过之后在时间维度上每次都可以减少一半(L->L/2->L/4)  
同时使用一半的数据(L/2)经过一次ProbSpare self-attention和Conv1d maxPool之后 也变成(L/4) 这样拼一块就是Encoder的输出  
输入中的Scalar和Stamp中的Scalar代表数据值 如股票价格等 Stamp就是时间信息 如周几等  

### Decoder
![decoder](img/decoder.png)  
decoder的改进策略是在decoder的输入中不止有start token 同时将目标长度的token-0拼接在start token之后 这样经过masked ProbSpare self-attention 得到的维度的后半部分就是输出  
start token不同于NLP中的Start Of Sentence SOS 标识  在时序预测领域中 使用的是Input的后一部分值作为token  



