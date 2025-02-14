"""
 @Author: TCcjx
 @Email: tcc2025@163.com
 @FileName: 02数据预处理.py
 @DateTime: 2025-02-06 11:57
 @SoftWare: PyCharm
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import textwrap

# 超参数
<<<<<<< HEAD
batch_size = 3 # 同时处理多少条数据
block_size = 16 # 训练、验证的字符串长度
n_embedding = 3 # token的embedding长度
wrap_width = 50

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
=======
batch_size = 8 # 同时处理多少条数据
block_size = 5 # 训练、验证的字符串长度
n_embedding = 3 # token的embedding长度

device = 'cuda' if torch.cuda.is_available() else 'cpu'
>>>>>>> f4516b6df3c34264f1b678c2936be845395c7329
torch.manual_seed(1337) # 随机种子
file_name = "../hong_lou_meng.txt"
# 文本 -> 词典/字典(按字划分,按词划分) -> Token -> Embedding(词嵌入)
# -----------数据预处理--------------
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
<<<<<<< HEAD
# print('train_data  length:',len(train_data))
# print('val_data length:',len(val_data))
print(f'文件{file_name}读取完成')
# 获取批量数据
def get_batch(split):
    data = train_data if split == 'train' else val_data
=======
print(len(train_data))
print(len(val_data))

print(f'文件{file_name}读取完成')
# 获取批量数据
def get_batch(split):
    data = train_data if split == 'train' else 'val_data'
>>>>>>> f4516b6df3c34264f1b678c2936be845395c7329
    ix = torch.randint(len(data) - block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix],dim=0)
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    # x, y = x.to(device), y.to(device)
    return x,y

<<<<<<< HEAD
# --傻瓜模型--
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
        self.network = nn.Linear(n_embedding, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape # B->batch_size;T->block_size  数据为token(整数)形式
        token_embd = self.token_embedding_table(idx)
        position_idx = torch.arange(T,device=device)
        position_embd = self.position_embedding_table(position_idx)
        x = token_embd + position_embd # (B,T,n_embedding)
        logits = self.network(x) # (B,T,vocab_size)
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
=======
x,y = get_batch('train')
token_embedding_table = nn.Embedding(vocab_size,n_embedding)
token_embd = token_embedding_table(x)
position_embedding_table = nn.Embedding(block_size,n_embedding)
position_idx = torch.arange(block_size) # 位置嵌入
position_embd = position_embedding_table(position_idx)
print(position_embd)
# print(x)
# x_list = x.tolist()
# for str_list in x_list:
#     print(str_list)
#     print(decode(str_list))
# print(token_embedding_table(x))


# print(data)
# print(encode('你好'))
# print(decode([520,1314]))
# print(get_batch('train'))
>>>>>>> f4516b6df3c34264f1b678c2936be845395c7329


#--------------运行--------------
x,y = get_batch('val')
print('x:',x.shape)




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
        print(f'epoch:{i+1},loss:{loss}')


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

