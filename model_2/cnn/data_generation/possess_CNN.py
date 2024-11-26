import csv
import pandas as pd
import numpy as np

# 定义函数来读取CSV文件并转换为字典形式         （ 读取药物-通路数据，细胞系数据 )
def read_csvs(filepath):
    result = {}      # 创建空字典用于保存结果
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            name = row[0]      # 获取第一个元素作为列表名称
            if name not in result:
                result[name] = []      # 如果该名称不在字典中则添加新的空列表
            elements = row[1:]
            result[name].append(elements)     # 将列表添加到相应的名称所对应的列表中
    return result

##########################3生成背景图谱(三个不同的背景图谱)##########################
def get_result_list(filepath):
    with open(filepath, 'r') as file:
        data = pd.read_csv(file)
    #读取数据
    scatter_data = data.to_dict(orient='records') #字典形式
    # 获取最大值和最小值作为矩阵边界
    max_x = max(int(d['x']) for d in scatter_data) + 1
    min_x = min(int(d['x']) for d in scatter_data) - 1
    max_y = max(int(d['y']) for d in scatter_data) + 1
    min_y = min(int(d['y']) for d in scatter_data) - 1
    # 创建空白矩阵
    matrix = [[None] * (max_y - min_y + 1) for _ in range((max_x - min_x + 1))]
    # 根据散点数据更新矩阵
    for data in scatter_data:
        x = int(round(data['x'])) - min_x
        y = int(round(data['y'])) - min_y
        if matrix[x][y]:
            # 如果已经存在其他label，则以列表形式保存多个label
            labels = matrix[x][y].split(',')
            if data['label'] not in labels:
                labels.append(data['label'])
            matrix[x][y] = ','.join(labels)
        else:
            matrix[x][y] = data['label']
    #print(matrix)
    # 打印结果
    # print("Matrix:\n")
    # for row in matrix:
    #     print(' '.join([str(cell or '') for cell in row]))
    # ################################### 遍历矩阵，生成通路-坐标表格 ########################################
    result_list = []  #创建一个空白矩阵
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            element = matrix[i][j]
            position = (i, j)
            if element is not None:
                
                if ',' in element is not None:  # 检查元素是否包含逗号
                    for sub_element in element.split(','):  # 将元素分割成列表并遍历
                        result_list.append((sub_element, position))
                else:
                    result_list.append((element, position))         
    #print(result_list)
    #print(type(result_list))
    #result_list = pd.DataFrame(result_list, columns=['value', 'coordinates'])
    #result_list.to_csv('result_list.csv', index=False)
    return result_list

def find_element(result_list, target):
    for row in result_list:
        if row[0] == target:
            return row[1]
    return None

def get_matrix_drug(value,result_list):
    # 获取原始矩阵的行数和列数
    # rows = len(matrix)
    # cols = len(matrix[0])
    rows = 58
    cols = 3562
    empty_matrix = np.zeros((rows, cols), dtype=int)
    # print("空矩阵:\n", empty_matrix)
    # print("空矩阵的形状:", empty_matrix.shape)     #(58, 1957)
    ############# 查找指定字符串在矩阵中的位置 #############
    for item in value:  
        for target in item:
            if target != '':
                #print(target)
                results = find_element(result_list, target)  #results是元组(truple)形式
                #print("结果为：", results)
                # 修改指定位置上的元素
                row_index = results[0]
                #print("行索引:", row_index)
                column_index = results[1]
                #print("列索引:", column_index)
                new_value = empty_matrix[row_index][column_index] + 1
                empty_matrix[row_index][column_index] = new_value
    return empty_matrix
#########################################################################################################################
def get_matrix_cell(value,pathway,result_list):
    # 获取原始矩阵的行数和列数
    # rows = len(matrix)
    # cols = len(matrix[0])
    rows = 58
    cols = 3562
    empty_matrix = np.zeros((rows, cols), dtype=int)
    #坐标
    results_condidate = []
    for target in pathway:  
        if target != '':
            #print(target)
            results = find_element(result_list, target)   #results是元组(truple)形式
            #print("结果为：", results)
            results_condidate.append(results)
    coordinates = results_condidate
    #对应数值
    value_data = []
    # 外层循环遍历外部列表
    for sublist in value:
        #print("sublist:", sublist)
        for item in sublist:
            item = float(item)
            value_data.append(item)
    #print(len(coordinates))
    #print(len(value_data))
    # 确保坐标和值的数量相同
    assert len(coordinates) == len(value_data)
    
    # 给矩阵对应位置赋值
    for coord, val in zip(coordinates, value_data):
        empty_matrix[coord] = val

    return empty_matrix

