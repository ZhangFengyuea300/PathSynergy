#绘制混淆矩阵
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#SVM_conf_matrix = np.array([[544, 83], [395, 230]])
#XGB_conf_matrix = np.array([[510, 117], [235, 390]])
#RF_conf_matrix = np.array([[533, 94], [243, 382]])
#GBM_conf_matrix = np.array([[513, 114], [217, 408]])
#PathSynergy_conf_matrix = np.array([[586, 41], [181, 444]])
#GAECDS_conf_matrix = np.array([[15, 612], [14, 611]])
DeepDDS_conf_matrix = np.array([[474, 153], [276, 349]])

# 使用Seaborn的heatmap函数绘制混淆矩阵
plt.figure(figsize=(8, 6))
#sns.heatmap(SVM_conf_matrix, annot=True, fmt='d', cmap='Blues')
#sns.heatmap(XGB_conf_matrix, annot=True, fmt='d', cmap='Blues')
#sns.heatmap(RF_conf_matrix, annot=True, fmt='d', cmap='Blues')
#sns.heatmap(GBM_conf_matrix, annot=True, fmt='d', cmap='Blues')
#sns.heatmap(PathSynergy_conf_matrix, annot=True, fmt='d', cmap='Blues')
#sns.heatmap(GAECDS_conf_matrix, annot=True, fmt='d', cmap='Blues')
sns.heatmap(DeepDDS_conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
#plt.title('SVM')
#plt.title('XGB')
#plt.title('RF')
#plt.title('GBM')
#plt.title('PathSynergy')
#plt.title('GAECDS')
plt.title('DeepDDS')


# 保存为PDF格式
#plt.savefig('SVM_confusion_matrix.pdf', bbox_inches='tight')
#plt.savefig('XGB_confusion_matrix.pdf', bbox_inches='tight')
#plt.savefig('RF_confusion_matrix.pdf', bbox_inches='tight')
#plt.savefig('GBM_confusion_matrix.pdf', bbox_inches='tight')
#plt.savefig('PathSynergy_confusion_matrix.pdf', bbox_inches='tight')
#plt.savefig('GAECDS_confusion_matrix.pdf', bbox_inches='tight')
plt.savefig('DeepDDS_confusion_matrix.pdf', bbox_inches='tight')

plt.show()