##############变量筛选:ks>0.02并上 IV>0.02的 
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
class Varlable_filter:
    def __init__(self,df,col):
        self.df = df
        self.col =  col
        tmp = list(df)
        tmp.remove(col)
        self.col_list = tmp
        
    def woe_IV(self,column_name): 

#这个计算，在变量分布稀疏的时候回存在问题，这里面把相应变量为1的全部合并为一组了，这样会造成连续变量不按顺序合并
#幸好 crosstab这个函数 可以处理连续非连续的变量的交叉表，所以这个函数的输入要求应为粗分箱或者分类变量
#分类变量的相应类为 1  如坏客户或者欺诈客户 
#编码的时候相应类在分子，统计分析的时候符号应该为负

        small = pd.crosstab(self.df[column_name],self.df[self.col],margins = True)
        dele = small[small[1]<10]
        combin = list(dele.index)
        #print(column_name,"需要合并的字段有：",combin)  
        temp = small[small[1]<10].sum() 
        small.drop(combin,axis = 0,inplace = True)
        a = small.T
        a['AAA']=temp
        new = a.T
        al = new.ix['All']
        cou = new.drop('All',axis = 0) 
        bad_p = cou[1]/al[1]
        good_p = cou[0]/al[0]
        woe = np.log(bad_p/good_p)
        #print(woe)
        p = bad_p - good_p
        IV = (p*woe).sum()
        #return(pd.Series([woe,IV],index = ['woe','VI']) )
        return(IV)
        
    def cmt_iv(self,limited_value = -np.inf):
        times = 0
        iv_list = []
        for x in self.col_list:
            iv_v = self.woe_IV(x)
            if iv_v >limited_value:
                print(x,iv_v)  
                iv_list.append(x)
                times+=1
        print('the number of bigger then 0.02 if %f'%times)
        return iv_list
        
    def col_ks(self,column_name):
        return ks_2samp(self.df[self.df[self.col]==1][column_name], self.df[self.df[self.col]==0][column_name]).statistic
    
    def cmt_ks(self,limited_value = -np.inf):
        times = 0
        ks_list = []
        for x in self.col_list:
            ks_v = self.col_ks(x)
            if ks_v > limited_value:
                print(x,ks_v)  
                ks_list.append(x)
                times+=1
        print('the number of bigger then 0.02 if %f'%times)
        return ks_list
    
        
    def filtered_df(self,ks_limited = 0.02,iv_limited = 0.02):
        iv_list = self.cmt_iv(iv_limited)
        ks_list = self.cmt_ks(ks_limited)
        return self.df[iv_list+ks_list+[self.col]]