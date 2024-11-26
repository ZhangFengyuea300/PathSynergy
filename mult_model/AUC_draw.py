# 绘制AUC
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import io
from PIL import Image

def AUC_draw(data,label,color,save=False):

    data = np.array(data)
    model_all = []
    auc_all = []
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(5):
        idx_p = 2 * i
        idx_t = idx_p + 1
        data_p = data[idx_p]
        data_t = data[idx_t]

        fpr, tpr, thresholds = metrics.roc_curve(data_t, data_p, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        auc_all.append(roc_auc)
        model_all.append(np.interp(mean_fpr, fpr, tpr))
        model_all[-1][0] = 0.0

    plt.figure(1)

    mean_tpr = np.mean(model_all, axis=0)
    mean_tpr[-1] = 1.0
    #mean_auc = auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(auc_all)
    
    plt.plot(mean_fpr, mean_tpr, color=color, label= label+' (AUC = %0.4f)' % mean_auc, lw=2, alpha=.8)
    plt.legend(loc='lower right', prop={'size': 7.5}) 

    plt.plot([0, 1], [0, 1], '--', lw=2, color='grey')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    # Save the image in memory in PNG format
    png1 = io.BytesIO()
    plt.savefig(png1, format="jpg", dpi=600, pad_inches=.1, bbox_inches='tight')
    # Load this image into PIL
    png2 = Image.open(png1)
    # Save as TIFF
    if save:
        png2.save("./Figure 3.tiff")
    png1.close()


data_gbm = pd.read_csv('./mult_model/machine/GBM.csv')
data_rf = pd.read_csv('./mult_model/machine/RF.csv')
data_svm = pd.read_csv('./mult_model/machine/SVM.csv')
data_xgb = pd.read_csv('./mult_model/machine/XGB.csv')

data_gat_cnn = pd.read_csv('./mult_model/gatcnn/GAT+CNN.csv')
data_DeepDDS = pd.read_csv('./mult_model/deeplearning/DeepDDs/DeepDDs.csv')
data_GAECDS = pd.read_csv('./mult_model/deeplearning/GAECDS/GAECDS.csv')


AUC_draw(data_svm,'SVM','#F47D1E',save=False)
AUC_draw(data_rf,'RF','#34A046',save=False)
AUC_draw(data_gbm,'GBM','#693D98',save=False)
AUC_draw(data_xgb,'XGB','#1D78B4',save=False)
AUC_draw(data_gat_cnn,'PathSynergy','#E11D25',save=False)
AUC_draw(data_GAECDS,'GAECDS','#2D8875',save=False)
AUC_draw(data_DeepDDS,'DeepDDS','#B15928',save=True)




