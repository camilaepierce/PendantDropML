import torch
from torch import nn
import re


filename = 'fjdksl/fkjdsjfk/jfkd1fjdsk/params001.txt'

filename = filename.split('/')[-1]
digits = re.findall(r'\d+', filename)

print(digits[0])