import numpy as np 
import torch
import pandas as pd 
import sklearn
import networkx as nx

data_list=[]
drug_list=[]

file=open('mat_drug_drug.txt')
for line in file.readlines():
    curline=line.strip().split(" ")
    intline=map(int,curline)
    intline=list(intline)
    data_list.append(intline)
file.close()
data1=np.array(data_list)

file=open('drug.txt')
for line in file.readlines():
    drug_line=line.strip()
    drug_list.append(drug_line)
file.close()


DG = nx.from_numpy_array(data1, create_using=nx.DiGraph)
nx.write_edgelist(DG,"data\yeast\G_drug.edgelist",data=False)
print(DG)