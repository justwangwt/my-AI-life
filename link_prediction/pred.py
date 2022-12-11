from src.arga.predict import Predict
from sklearn import metrics

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse



predict = Predict()
predict.load_model_adj('config.cfg')
adj_orig, adj_rec = predict.predict()
# 会返回原始的图邻接矩阵和经过模型编码后的hidden embedding经过内积解码的邻接矩阵，可以对这两个矩阵进行比对，得出link prediction.
#print('adj_orig: {}, \n adj_rec: {}'.format(adj_orig, adj_rec))
adj_orig_coo=adj_orig.todense()
orig=np.array(adj_orig_coo)
y_tur=list(orig.flatten())
y_tur=y_tur[0:5999]+y_tur[60000:65999]+y_tur[120000:125999]+y_tur[180000:185999]+y_tur[240000:245999]
rec=np.array(adj_rec)
y_score=list(rec.flatten())
y_score=y_score[0:5999]+y_score[60000:65999]+y_score[120000:125999]+y_score[180000:185999]+y_score[240000:245999]
'''
plt.figure(1)
fpr,tpr,thresholds=metrics.roc_curve(y_tur,y_score)
roc_auc=metrics.auc(fpr,tpr)
print(roc_auc)
plt.plot(fpr,tpr,'b',label='AUC=%0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlabel("FPR")
plt.ylabel('TPR')

plt.figure(2)
precision,recall,_=metrics.precision_recall_curve(y_tur,y_score)
aupr=metrics.auc(recall,precision)
print(aupr)
plt.plot(recall,precision,'r',label='AUC=%0.2f' % aupr)
plt.legend(loc='lower left')
plt.plot([1,0],'r--',color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')

plt.show()
'''

for i in range(len(adj_rec[0])):
    for j in range(len(adj_rec[0])):
        if adj_rec[i][j]>0.9:
            adj_rec[i][j]=1
        else:
            adj_rec[i][j]=0
sum=0
for i in range(len(adj_rec[0])):
    for j in range(len(adj_rec[0])):
        if adj_rec[i][j]==orig[i][j]:
            sum=sum+1

pr=sum/(572*572)
print('precision:',pr)

np.savetxt("rec.txt",adj_rec,fmt='%d',delimiter=' ')
np.savetxt("orgin.txt",orig,fmt='%d',delimiter=' ')


