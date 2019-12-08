# -*- coding: utf-8 -*-
#%%
import numpy as np
data1 = [1, 2, 3, 4, 5]
arr1 =np.array (data1)

data2 = [range (1,5), range(5,9)]

arr2 = np.array(data2) # 2d array
arr2.tolist()

arr = np.arange(10, dtype=float).reshape((2, 5))
print(arr.shape)
print(arr.reshape(5, 2))

a = np.array([0, 1])
a_col = a[:, np.newaxis]
print(a_col)
#or
a_col = a[:, None]
print(a_col.T)

#%%

a = np.array([[ 0, 0, 0],
[10, 10, 10],
[20, 20, 20],
[30, 30, 30]])
b = np.array([0, 1, 2])
print(a + b)

#%%

from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

columns = ['name', 'age', 'gender', 'job']
user1 = pd.DataFrame([['alice', 19, "F", "student"],
['john', 26, "M", "student"]],
columns=columns)
user2 = pd.DataFrame([['eric', 22, "M", "student"],
['paul', 58, "F", "manager"]],
columns=columns)
user3 = pd.DataFrame(dict(name=['peter', 'julie'],
age=[33, 44], gender=['M', 'F'],
job=['engineer', 'scientist']))
print(user3)

#%%
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

user1.append(user2)
users = pd.concat([user1, user2, user3]).reset_index ()
print(users)

user4 = pd.DataFrame(dict(name=['alice', 'john', 'eric', 'julie'],
height=[165, 180, 175, 171]))
print(user4)

merge_inter = pd.merge(users, user4, on="name")
print(merge_inter)
#%%

url1 ="http://rcs.bu.edu/examples/python/data_analysis/Salaries.csv"
salary = pd.read_csv(url1)

#%%
# inline plot (for jupyter)
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 10, 50)
sinus = np.sin(x)
plt.plot(x, sinus)
plt.show()

plt.plot(x, sinus, "o")
plt.show()
# use plt.plot to get color / marker abbreviations

# Rapid multiplot
cosinus = np.cos(x)
plt.plot(x, sinus, "-b", x, sinus, "ob", x, cosinus, "-r", x, cosinus, "or")
plt.xlabel('this is x!')
plt.ylabel('this is y!')
plt.title('My First Plot')
plt.show()
