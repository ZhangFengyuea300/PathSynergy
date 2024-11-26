from scipy.sparse import *
from possess_CNN import *
import numpy as np
import pandas as pd

# 读取数据
# result = read_csvs('./predict_4905/drug_pathway_predict.csv')     # 药物-通路数据
# cell_line = read_csvs('./predict_4905/cell_line.csv')    # 细胞系-基因表达量数据
# result_list_1_2 = get_result_list('orthogonality_1_2.csv')    # 第一个背景图谱
# result_list_1_3 = get_result_list('orthogonality_1_3.csv')    # 第二个背景图谱
# result_list_2_3 = get_result_list('orthogonality_2_3.csv')    # 第三个背景图普
# drug_synergy = pd.read_csv('./predict_4905/predict_top1.csv')    # 样本数据
# pathways = pd.read_csv('pathways.csv') 

result = read_csvs('./data_generation/predict/drug_pathway_predict.csv')     # 药物-通路数据
cell_line = read_csvs('./data_generation/predict/cell_line.csv')    # 细胞系-基因表达量数据
result_list_1_2 = get_result_list('./data_generation/orthogonality_1_2.csv')    # 第一个背景图谱
result_list_1_3 = get_result_list('./data_generation/orthogonality_1_3.csv')    # 第二个背景图谱
result_list_2_3 = get_result_list('./data_generation/orthogonality_2_3.csv')    # 第三个背景图普
drug_synergy = pd.read_csv('./data_generation/predict/predict_top1.csv')    # 样本数据
pathways = pd.read_csv('./data_generation/pathways.csv') 

# 取药物对
drug1 = np.array(drug_synergy['drug1'])    # 取药物1
drug2 = np.array(drug_synergy['drug2'])    # 取药物2
cell =np.array(drug_synergy['cell'])
pathway = np.array(pathways['pathways']) 

def get_drug_matrix(result,result_list_1_2,result_list_1_3,result_list_2_3):
    matrix_drug_cell = []
 
    for i in range(len(drug1)):
        epoch = i
        print(epoch)
        key1 = drug1[i]
        key2 = drug2[i]
        key3 = cell[i]
        value1 = result.get(key1)   # value是一个list
        value2 = result.get(key2) 
        value3 = cell_line.get(key3)

        empty_matrix_1_2_1 = get_matrix_drug(value1,result_list_1_2)
        empty_matrix_1_2_2 = get_matrix_drug(value2,result_list_1_2)
        empty_matrix_1_3_1 = get_matrix_drug(value1,result_list_1_3)
        empty_matrix_1_3_2 = get_matrix_drug(value2,result_list_1_3)
        empty_matrix_2_3_1 = get_matrix_drug(value1,result_list_2_3)
        empty_matrix_2_3_2 = get_matrix_drug(value2,result_list_2_3)

        X1 = empty_matrix_1_2_1 + empty_matrix_1_2_2   # drug1和drug2相加之后的矩阵
        Y1 = get_matrix_cell(value3,pathway,result_list_1_2)  # 细胞系矩阵
        X2 = empty_matrix_1_3_1 + empty_matrix_1_3_2   # drug1和drug2相加之后的矩阵
        Y2 = get_matrix_cell(value3,pathway,result_list_1_3)  # 细胞系矩阵
        X3 = empty_matrix_2_3_1 + empty_matrix_2_3_2   # drug1和drug2相加之后的矩阵
        Y3 = get_matrix_cell(value3,pathway,result_list_2_3)  # 细胞系矩阵

        # 计算每个矩阵的均值和标准差
        X1_mean = np.mean(X1)
        X1_std = np.std(X1)
        Y1_mean = np.mean(Y1)
        Y1_std = np.std(Y1)
        X2_mean = np.mean(X2)
        X2_std = np.std(X2)
        Y2_mean = np.mean(Y2)
        Y2_std = np.std(Y2)
        X3_mean = np.mean(X3)
        X3_std = np.std(X3)
        Y3_mean = np.mean(Y3)
        Y3_std = np.std(Y3)

        # 标准化矩阵X和Y
        X1_normalized = (X1 - X1_mean) / X1_std
        Y1_normalized = (Y1 - Y1_mean) / Y1_std
        #print("标准化后的矩阵X:\n", X_normalized)
        #print("标准化后的矩阵Y:\n", Y_normalized)
        X2_normalized = (X2 - X2_mean) / X2_std
        Y2_normalized = (Y2 - Y2_mean) / Y2_std
        X3_normalized = (X3 - X3_mean) / X3_std
        Y3_normalized = (Y3 - Y3_mean) / Y3_std

        # 标准化后的矩阵相加
        Z1 = X1_normalized + Y1_normalized
        #print(Z)
        Z2 = X2_normalized + Y2_normalized
        Z3 = X3_normalized + Y3_normalized

        matrix = np.stack((Z1, Z2, Z3), axis=0) 
        matrix_drug_cell.append(matrix)

    #len = 300 de list

    return matrix_drug_cell

matrix = get_drug_matrix(result,result_list_1_2,result_list_1_3,result_list_2_3)
np.save('top_1.npy', matrix) 