"""
 @Author: TCcjx
 @Email: tcc2025@163.com
 @FileName: main.py
 @DateTime: 2025-02-06 11:57
 @SoftWare: PyCharm
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import textwrap

# 超参数
batch_size = 16 # 同时处理多少条数据
block_size = 128 # 训练、验证的字符串长度
n_embedding = 384 # token的embedding长度
wrap_width = 50
num_heads = 8
head_size = n_embedding // num_heads

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
torch.manual_seed(1337) # 随机种子
file_name = "../hong_lou_meng.txt"
# 文本 -> 词典/字典(按字划分,按词划分) -> Token -> Embedding(词嵌入)
# -----------数据预处理--------------f
with open(file_name,'r',encoding='utf-8') as f:
    text = f.read()
# 有序、不重复的列表
chars = sorted(list(set(text)))
vocab_size = len(chars)  # 词表大小

# 做一个字符和数字之间的投影
stoi = {char:i for i,char in enumerate(chars)} # 符号到整数
itos = {i:char for i,char in enumerate(chars)} # 整数到符号

encode = lambda s:[stoi[c] for c in s] # 将字符串转换成数字列表
decode = lambda list1:"".join([itos[i] for i in list1]) # 将数字串转换成字符串

# 训练和验证分组
data = torch.tensor(encode(text),dtype=torch.long) # 用整数列表表示文本字符串
n = int(0.9*len(data)) # 前百分之九十的数据用于训练
train_data = data[:n] # 训练数据
val_data = data[n:] # 验证数据

# print('train_data  length:',len(train_data))
# print('val_data length:',len(val_data))
print(f'文件{file_name}读取完成')
# 获取批量数据


# 获取批量数据
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix],dim=0)
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    # x, y = x.to(device), y.to(device)
    return x,y

# --损失测评--
@torch.no_grad() # 不做梯度计算的decorator,作用域为整个函数
def estimate_loss(model,eval_iters=100):
    out = {}
    model.eval() # 评估模式

    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            X,Y = X.to(device),Y.to(device)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean() # 分别计算train\val的loss均值

    model.train() # 恢复成训练模式
    return out

# Head
class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embedding, head_size, bias=False)
        self.query = nn.Linear(n_embedding, head_size, bias=False)
        self.value = nn.Linear(n_embedding, head_size, bias=False) # 线性变换层
        self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size))) # 不可训练的结构(约等于常量)
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, x):
        B, T, C = x.shape # batch_size,block_size,n_embedding
        k = self.key(x)
        q = self.query(x)

        # wei = torch.ones((T,T),device=device) # 上下文三角掩码矩阵
        wei = q @ k.transpose(-2, -1) / (k.shape[0]**-0.5) # 注意力方阵 (B, T, T)
        wei = wei.masked_fill(self.tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # print(f'wei:{wei.shape},x:{x.shape}')
        v = self.value(x)
        out = wei @ v
        # print(f'out-shape:{out.shape}')
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in  range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embedding)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embedding):
        super(FeedForward,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedding,n_embedding*4),
            nn.ReLU(),
            nn.Linear(n_embedding*4,n_embedding),
            nn.Dropout(p=0.2)
        )
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embedding, num_heads):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads,head_size) # 多头注意力网络(自注意力)
        self.ffwd = FeedForward(n_embedding)
        self.ln1 = nn.LayerNorm(n_embedding)
        self.ln2 = nn.LayerNorm(n_embedding)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # 残差多头注意力网络
        x = x + self.ffwd(self.ln2(x)) # 残差线性前馈层
        return x

n_layer = 3
# --傻瓜语言模型--
class LanguageModel(nn.Module):
    '''
    输入: number_embedding
    输出：vocab_size (词表大小)
    '''
    def __init__(self):
        # super().__init__()
        super(LanguageModel, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedding)
        self.position_embedding_table = nn.Embedding(block_size, n_embedding)
        # self.head = Head(n_embedding)
        # self.multi_head = MultiHeadAttention(num_heads=num_heads,head_size=head_size)
        self.blocks = nn.Sequential(*[Block(n_embedding,num_heads) for _ in range(n_layer)]) # 残差多头注意力机制
        self.ln_f = nn.LayerNorm(n_embedding)
        self.lm_head = nn.Linear(n_embedding, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape # B->batch_size;T->block_size  数据为token(整数)形式
        token_embd = self.token_embedding_table(idx)
        position_idx = torch.arange(T,device=device)
        position_embd = self.position_embedding_table(position_idx)
        x = token_embd + position_embd # (B,T,n_embedding) 词嵌入 + 位置嵌入
        # head_out = self.head(x)
        # head_out = self.multi_head(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # logits = self.network2(F.relu(self.network1(head_out))) # (B,T,vocab_size) vocab_size: 词表大小
        # print(f'x:{x.shape},logits:{logits.shape}')
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        # random_tensor = torch.rand(B,T,vocab_size) # vocab_size：词表大小
        # logits = random_tensor / random_tensor.sum(dim=-1,keepdim=True) # 进行归一化
        # print(random_tensor.shape)
        # print(random_tensor.sum(dim=-1,keepdim=True).shape)
        return logits, loss

    def generate(self, token_sequ, max_new_tokens): # token_sequ 已知的上文,max_new_tokens 续写的长度
        for _ in range(max_new_tokens): # 循环次数：续写文本长度
            tokens_input = token_sequ[:,-block_size:]
            logits, loss = self.forward(tokens_input)
            logits = logits[:, -1 ,:] # 取出字符串最后一位
            probs = F.softmax(logits,dim=-1)
            token_next = torch.multinomial(probs,num_samples=1) # 把概率分布向量 --> 整数token
            token_next = token_next.to(device)
            token_sequ = torch.cat((token_sequ, token_next),dim=1) # 将新预测的字符，拼接到尾部，形成新的字符串
        new_tokens = token_sequ[:,-max_new_tokens:]
        return new_tokens

#--------------运行--------------

learning_rate = 1e-2
max_iter = 1000
def main():
    print(f'训练内容:{file_name}')
    model = LanguageModel() # 实例化模型
    model = model.to(device)
    print('Model Parameters',sum(p.numel() for p in model.parameters()) / 1e6) # 打印参数

    # 设定一个优化器
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

    # 训练
    for i in range(max_iter):
        xb, yb = get_batch('train')
        xb, yb= xb.to(device), yb.to(device)
        logits, loss = model(xb,yb) #前馈运算
        optimizer.zero_grad(set_to_none=True) # 把旧的梯度归零
        loss.backward() # 反向传播
        optimizer.step() # 梯度更新计算
        if i % 50 ==0 or i == max_iter-1:
            losses = estimate_loss(model)
            print(f'epoch:{i+1},train_loss:{losses["train"]:.4f},val_loss:{losses["val"]:.4f}')


    max_new_tokens = 500
    start_idx = random.randint(0,len(val_data)-block_size-max_new_tokens)

    # 上文内容
    context = torch.zeros((1,block_size), dtype=torch.long, device=device) # (B,T) B=1,T=Block_size
    context[0,:] = val_data[start_idx:start_idx+block_size]
    context_str = decode(context[0].tolist()) # 一阶张量
    wrapped_context_str = textwrap.fill(context_str,width=wrap_width)

    #真实下文
    real_next_token = torch.zeros((1,max_new_tokens),dtype=torch.long,device=device)
    real_next_token[0,:] = val_data[start_idx+block_size:start_idx+block_size+max_new_tokens]
    real_next_token_str = decode(real_next_token[0].tolist()) # 一阶张量
    wrapped_real_next_token_str = textwrap.fill(real_next_token_str,width=wrap_width)

    # 生成内容
    generated_tokens = model.generate(context,max_new_tokens)
    generated_str = decode(generated_tokens[0].tolist())
    wrapped_generated_str = textwrap.fill(generated_str,width=wrap_width)

    print('上文内容:',wrapped_context_str)
    print('\n\n')
    print('生成内容:',wrapped_generated_str)
    print('\n\n')
    print('真实内容:',wrapped_real_next_token_str)

if __name__ == '__main__':
    main()
    # head = Head(head_size=5)
    # print('下三角矩阵：')
    # print(head.tril)
    # wei = torch.ones(block_size,block_size) # 注意力矩阵
    # print('注意力矩阵：')
    # print(wei)
    # wei = wei.masked_fill(head.tril==0,float('-inf'))
    # print('掩码后的注意力矩阵：')
    # print(wei)
    # wei = F.softmax(wei,dim=-1)
    # print('softmax之后的注意力矩阵：')
    # print(wei)

# x = x.to(device)
# output,loss = model(x)
# print(output)
# print(output.shape)

# 嵌入表示 Embedding
# token_embedding_table = nn.Embedding(vocab_size,n_embedding) # (嵌入数,嵌入维度)
# token_embd = token_embedding_table(x)
# position_embedding_table = nn.Embedding(block_size,n_embedding) # 位置信息嵌入
# position_idx = torch.arange(block_size) # 位置嵌入
# position_embd = position_embedding_table(position_idx)
# embedding = token_embd + position_embd # 词嵌入 + 位置编码

# print(position_embd)

# x_list = x.tolist()
# for str_list in x_list:
#     print(str_list)
#     print(decode(str_list))
# print(token_embedding_table(x))


# print('data:',data)
# print(encode('你好'))
# print(decode([520,1314]))
# print(get_batch('train'))

