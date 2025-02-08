"""
 @Author: TCcjx
 @Email: tcc2025@163.com
 @FileName: 05词嵌入.py
 @DateTime: 2025-02-07 17:41
 @SoftWare: PyCharm
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1314)
size = 10 # 需要做嵌入的值的数目 0~size-1
n_embedding = 3 # 嵌入后的维度
embedding_table = nn.Embedding(size,n_embedding)
idx = torch.tensor(9)
print(embedding_table(idx))
print(embedding_table)



