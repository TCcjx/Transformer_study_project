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

# 超参数
batch_size = 8 # 同时处理多少条数据
block_size = 5 # 训练、验证的字符串长度
n_embedding = 3 # token的embedding长度

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337) # 随机种子
file_name = "../hong_lou_meng.txt"
# 文本 -> 词典/字典(按字划分,按词划分) -> Token -> Embedding(词嵌入)
# -----------数据预处理--------------
with open(file_name,'r',encoding='utf-8') as f:
    text = f.read()
# 有序、不重复的列表
chars = sorted(list(set(text)))
vocab_size = len(chars)

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
print(len(train_data))
print(len(val_data))

print(f'文件{file_name}读取完成')
# 获取批量数据
def get_batch(split):
    data = train_data if split == 'train' else 'val_data'
    ix = torch.randint(len(data) - block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix],dim=0)
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    # x, y = x.to(device), y.to(device)
    return x,y

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


