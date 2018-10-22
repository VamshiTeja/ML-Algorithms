

import os
import numpy as np

text = "abc"

li = list(text)
print li


ascci_val= []
for c in li:
	ascci_val.append(ord(c))

print ascci_val

one_hot_enc = []
for val in ascci_val:
	one_hot = np.zeros(shape=(256,))
	one_hot[val] = 1
	one_hot_enc.append(one_hot)

print one_hot_enc

