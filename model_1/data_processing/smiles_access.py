import urllib
import pubchempy
import pandas as pd
import numpy as np
with open('./drug_symbol_independent.txt','r',encoding='utf-8-sig') as file1:
    file_lines = file1.readlines()
    name_list=[]
    a=[]
    cc=[]
    d=[]
    e=[]
    f=[]
    g=[]
#readlines读取的每行是字符串格式，采用以下代码将其转换成列表格式
    for i in file_lines:
       j=i.strip() #去掉每行头尾空白
       name_list.append(str(j))
    for k in  name_list:
       results = pubchempy.get_compounds(k, 'name')
       for l in results:
        try:
            print('CID: {}\tMass: {}\tName: {}\tMolfor: {}\tSmi: {}\tSyn: {}'.format(l.cid,l.exact_mass,l.iupac_name,l.molecular_formula,l.isomeric_smiles,l.synonyms))
            MFs=l.molecular_formula
            MWs=l.molecular_weight
            ISs=l.isomeric_smiles
            Sys=l.synonyms
            Cis=l.cid
            a.append(k)
            cc.append(MFs)
            d.append(ISs)
            e.append(Sys)
            f.append(Cis)
            g.append(MWs)
        except (pubchempy.BadRequestError,TimeoutError,urllib.error.URLError,ValueError):
            pass
        dataframe=pd.DataFrame({'name':a,'molecular_formula':cc,'molecular_weight':g,'smiles':d,'synonyms':e,'cid':f})
        dataframe.to_csv ("smiles_independent.csv",index=False,sep=',')