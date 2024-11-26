from scipy.sparse import *
from possess_CNN import *
import numpy as np
import pandas as pd


#读取数据
result = read_csvs('drug_pathway.csv')
result_list_1_2 = get_result_list('orthogonality_1_2.csv')  
result_list_1_3 = get_result_list('orthogonality_1_3.csv')  
result_list_2_3 = get_result_list('orthogonality_2_3.csv') 

drug_synergy = pd.read_csv('6271_drug_synergy_CNN.csv')  #样本数据
#取药物对
drug1 = np.array(drug_synergy['drug1'])  #取药物1
drug2 = np.array(drug_synergy['drug2'])  #取药物2

def get_drug_matrix(result,result_list_1_2,result_list_1_3,result_list_2_3):
    # 调用函数进行测试
    # print(result) 
    # 写入神经网络之后，key要循环样本数据
    # key = drug1[1]
    matrixs = []
    for i in range(len(drug1)):
        print(i)
        key1 = drug1[i]
        key2 = drug2[i]
        value1 = result.get(key1)  #value是一个list
        value2 = result.get(key2) 

        empty_matrix_1_2_1 = get_matrix_drug(value1,result_list_1_2)
        # print(empty_matrix_1_2_1.shape)  #(58, 3562)
        empty_matrix_1_3_1 = get_matrix_drug(value1,result_list_1_3)
        empty_matrix_2_3_1 = get_matrix_drug(value1,result_list_2_3)

        empty_matrix_1_2_2 = get_matrix_drug(value2,result_list_1_2)
        # print(empty_matrix_1_2_2.shape)  #(58, 3562)
        empty_matrix_1_3_2 = get_matrix_drug(value2,result_list_1_3)
        empty_matrix_2_3_2 = get_matrix_drug(value2,result_list_2_3)

        matrix1 = empty_matrix_1_2_1 + empty_matrix_1_2_2
        # print(matrix1.shape)  #(58, 3562)
        matrix2 = empty_matrix_1_3_1 + empty_matrix_1_3_2
        matrix3 = empty_matrix_2_3_1 + empty_matrix_2_3_2
        
        #matrix = cv2.merge([matrix1, matrix2, matrix3])   #(58, 3562, 3)
        # 在通道维度合并矩阵
        matrix = np.stack((matrix1, matrix2, matrix3), axis=0)   #(3, 58, 3562)

        matrixs.append(matrix)
    return matrixs

matrix = get_drug_matrix(result,result_list_1_2,result_list_1_3,result_list_2_3)
# print(len(matrix))  #300  
# print(matrix[0].shape)  
np.save('matrix_drug_nocell.npy', matrix)  #len = 300 de list