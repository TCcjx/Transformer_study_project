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

print(encode('你好'))
print(decode([520,1314]))


