#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''       

from test_kfold_genrator import test_kfold


args = sys.argv

##prepare Data->Datasets/Data/ Label -> Datasets/Label/
Data=sorted(glob.glob("Datasets/Data/*"))
Label=sorted(glob.glob("Datasets/Label/*"))
Name=args[1]
##Data,Label,Name
test_kfold.kfold(Data, Label,Name)
