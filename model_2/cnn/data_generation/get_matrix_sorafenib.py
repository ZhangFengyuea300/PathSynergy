
from scipy.sparse import *
from possess_CNN import *
import numpy as np
import pandas as pd

#load data
result = read_csvs('drug_pathway_predict.csv')     # drug-pathway
#result = read_csvs('drug-pathway.csv')
cell_line = read_csvs('cell_line.csv')    # cell line
result_list_1_2 = get_result_list('orthogonality_1_2.csv')    # first background map
result_list_1_3 = get_result_list('orthogonality_1_3.csv')    # second background map
result_list_2_3 = get_result_list('orthogonality_2_3.csv')    # third background map
drug_synergy = pd.read_csv('CNN_drug_Sorafenib_4905.csv')    # drug synergy
#drug_synergy = pd.read_csv('drug_synergy.csv')
#pathways = pd.read_csv('./predict_1228/pathways.csv') 
pathways = pd.read_csv('pathways.csv') 

drug1 = np.array(drug_synergy['drug1'])    #drug1
drug2 = np.array(drug_synergy['drug2'])    #drug2
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
        value1 = result.get(key1)  #value is a list
        value2 = result.get(key2) 
        value3 = cell_line.get(key3)

        empty_matrix_1_2_1 = get_matrix_drug(value1,result_list_1_2)
        empty_matrix_1_2_2 = get_matrix_drug(value2,result_list_1_2)
        empty_matrix_1_3_1 = get_matrix_drug(value1,result_list_1_3)
        empty_matrix_1_3_2 = get_matrix_drug(value2,result_list_1_3)
        empty_matrix_2_3_1 = get_matrix_drug(value1,result_list_2_3)
        empty_matrix_2_3_2 = get_matrix_drug(value2,result_list_2_3)

        X1 = empty_matrix_1_2_1 + empty_matrix_1_2_2   #drug1+drug2
        Y1 = get_matrix_cell(value3,pathway,result_list_1_2)  
        X2 = empty_matrix_1_3_1 + empty_matrix_1_3_2   
        Y2 = get_matrix_cell(value3,pathway,result_list_1_3)  
        X3 = empty_matrix_2_3_1 + empty_matrix_2_3_2   
        Y3 = get_matrix_cell(value3,pathway,result_list_2_3)  

        # Calculate the mean and standard deviation of each matrix
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

        # Normalized matrices X and Y
        X1_normalized = (X1 - X1_mean) / X1_std
        Y1_normalized = (Y1 - Y1_mean) / Y1_std
        
        X2_normalized = (X2 - X2_mean) / X2_std
        Y2_normalized = (Y2 - Y2_mean) / Y2_std
        X3_normalized = (X3 - X3_mean) / X3_std
        Y3_normalized = (Y3 - Y3_mean) / Y3_std

    
        Z1 = X1_normalized + Y1_normalized
        #print(Z)
        Z2 = X2_normalized + Y2_normalized
        Z3 = X3_normalized + Y3_normalized

        matrix = np.stack((Z1, Z2, Z3), axis=0) 

        matrix_drug_cell.append(matrix)

    #len = 300 de list

    return matrix_drug_cell

matrix = get_drug_matrix(result,result_list_1_2,result_list_1_3,result_list_2_3)

np.save('matrix_drug_cell_4905.npy', matrix) 
