import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
########################
#这个包含了 三个类，分组类Grouping，Test_wave，Test_too_close
#Grouping：包含了连续变量的等频分组，卡方分组。以及离散变量的卡方分组。
#Test_wave：继承了分组类，功能是检测是否wave后输出已经符合要求的df
#Test_too_close：继承了分组类，功能输入阈值，即两组之间的最大逾期率差，检测后输出符合要求的df
#################
class Grouping:

#分组类功能有 连续变量的等频分组，卡方最优分组，
#输入： 
#df_need_cut ：为需要分组的df
#remove_col： 为相应变量  即：y

    def __init__(self,df_need_cut,remove_col):# = 'cus_type'
#在初始化的时候，把数据做一个简单的处理，
#这个处理是把  输入的df中去掉相应变量的变量赋值给类的一个属性，这样在函数中调用就很方便，它是公用的，类内可以直接调用。
#初始化函数的作用凸显出来了，这个要比直接在类名字处写上输入变量要合理，这样做可以进行处理。   
        self.df_need_cut = df_need_cut
        float_df_need_cut_tmp = df_need_cut.columns.tolist()
        float_df_need_cut_tmp.remove(remove_col)
        self.float_df_need_cut_col = float_df_need_cut_tmp
        self.remove_col = remove_col
  
    def gouping_function_float(self,D_equal_freq,sub_name = '_cut'):
#这个函数用字典作为一个保存新的分组，那有没有保留原来的index呢？
#如果没有，那可否保证输出结果可以对应到原来的id呢？
#有待验证哦
#D_equal_freq 为 每个变量分箱的字典
        D_group = {}
        for name in self.float_df_need_cut_col:
            cut_name = name + sub_name
            if len(D_equal_freq[name])>3 or len(self.df_need_cut[name].unique())>3:
                list_tmp = []
                list_tmp.append(-np.inf)#-999999999
                list_tmp.extend(D_equal_freq[name])
                list_tmp.append(np.inf)#9999999999
                D_group[cut_name] = pd.cut(self.df_need_cut[name],list_tmp)
            else:
                D_group[cut_name] = self.df_need_cut[name]
        
        df_cuted = pd.DataFrame(D_group)
        return df_cuted.join(self.df_need_cut[self.remove_col])

        
    def __euquifrequency_coding(self,se,k):
#这个是内部调用的函数 输入为要分组的series
#k是要分组的数量
        w = [1.0*i/10 for i in list(range(k+1))]
        w1 = se.dropna().describe(percentiles = w)[4:4+k+1].values.tolist()
        w_new = list(set(w1))
        w_new.sort()
        if len(w_new)>2:
            w_new.remove(max(w_new))
            w_new.remove(min(w_new))
        return w_new
        
    def D_euquifrequency_making(self,set_k = False):#原来叫D_making()
#k是根据series计算而来的 
#默认set_k 是False，如果不为False 那这个函数可以按照指定的列名和分组数量进行等频分组
#输入 se_name = 分组的list
#set_k 在不为False 的情况下 变为要分箱的数量
        D_equal_freq = {}
        if set_k == False:
            for x in self.float_df_need_cut_col:
                if self.df_need_cut[x].count()<2000:
                    k = 10
                else:
                    k = int(self.df_need_cut[x].count()/2000) 
                D_equal_freq[x] = self.__euquifrequency_coding(self.df_need_cut[x],k)
        else:
            for name in self.float_df_need_cut_col:
                D_equal_freq[name] = self.__euquifrequency_coding(self.df_need_cut[name],set_k)
        return D_equal_freq
#######把卡方分箱方法添加 ,该方法适用于离散变量
    def Chi2(self,df, total_col, bad_col, overallRate):
#计算卡方值
        '''
        :param df: the dataset containing the total count and bad count
        :param total_col: total count of each value in the variable
        :param bad_col: bad count of each value in the variable
        :param overallRate: the overall bad rate of the training set
        :return: the chi-square value
        '''
        df2 = df.copy()
        df2['expected'] = df[total_col].apply(lambda x: x*overallRate)
        combined = zip(df2['expected'], df2[bad_col])
        chi = [(i[0]-i[1])**2/i[0] for i in combined]
        chi2 = sum(chi)
        return chi2
    def ChiMerge_MaxInterval(self, df ,col, max_interval):
#离散变量的卡方分箱
#这个有问题，应该按照卡方最近接的合并，而不是相邻组合并有时间改一下吧。
        '''
        :param self.df_need_cut: the dataframe containing splitted column, and self.remove_col column with 1-0
        :param col: splitted column
        :param self.remove_col: self.remove_col column with 1-0
        :param max_interval: the maximum number of intervals. If the raw column has attributes less than this parameter, the function will not work
        :return: the combined bins
        '''
        colLevels = set(df[col])
        if len(colLevels) <= max_interval:  #If the raw column has attributes less than this parameter, the function will not work
            print("The number of original levels for {} is less than or equal to max intervals".format(col))
            return []
        else:
            #Step 1: group the dataset by col and work out the total count & bad count in each level of the raw column
            total = df.groupby([col])[self.remove_col].count()
            #total = pd.DataFrame({'total':total})  
            total = pd.DataFrame(total).rename(columns = {self.remove_col:'total'})
            bad = df.groupby([col])[self.remove_col].sum()
            #bad = pd.DataFrame({'bad':bad}) 
            bad = pd.DataFrame(bad).rename(columns = {self.remove_col:'bad'})
            regroup =  total.merge(bad,left_index=True,right_index=True, how='left')
            regroup.reset_index(level=0, inplace=True)
            N = sum(regroup['total'])
            B = sum(regroup['bad'])
            #the overall bad rate will be used in calculating expected bad count
            overallRate = B*1.0/N
            # since we always combined the neighbours of intervals, we need to sort the attributes
            if df[col].dtype != 'object':
                colLevels =sorted(list(colLevels)) #colLevels = regroup.index.tolist()
            else:
                colLevels =list(colLevels)
            # initially, each single attribute forms a single interval
            groupIntervals = [[i] for i in colLevels]
            groupNum = len(groupIntervals)
            times = 0
            while(len(groupIntervals)>max_interval):   #the termination condition: the number of intervals is equal to the pre-specified threshold
                # in each step of iteration, we calcualte the chi-square value of each atttribute
                print(times)
                times  +=1
                chisqList = []
                for interval in groupIntervals:
                    df_need_cut2 = regroup.loc[regroup[col].isin(interval)]
                    chisq = self.Chi2(df_need_cut2, 'total','bad',overallRate)
                    chisqList.append(chisq)
                #find the interval corresponding to minimum chi-square, and combine with the neighbore with smaller chi-square
                min_position = chisqList.index(min(chisqList))
                if min_position == 0:
                    combinedPosition = 1
                elif min_position == groupNum - 1:
                    combinedPosition = min_position -1
                else:
                    if chisqList[min_position - 1]<=chisqList[min_position + 1]:
                        combinedPosition = min_position - 1
                    else:
                        combinedPosition = min_position + 1
                groupIntervals[min_position] = groupIntervals[min_position]+groupIntervals[combinedPosition]
                # after combining two intervals, we need to remove one of them
                groupIntervals.remove(groupIntervals[combinedPosition])
                groupNum = len(groupIntervals)
            return groupIntervals
    def D_making_chi2_object(self):
#这个是 离散变量 卡方分组的 ，输出的是一个字典 以及这些变量，为什么要这个输出，忘记了，
        D_cut_list_info = {}
        object_ori_list = []
        for x in self.float_df_need_cut_col:
            df = self.df_need_cut[[self.remove_col, x ]]
            df.dropna(inplace = True)
            cut_list = []
            basic_num = len(self.df_need_cut)/10
            max_interval = int(len(self.df_need_cut)/basic_num)
            cut_temp = self.ChiMerge_MaxInterval(df, x , max_interval) 
            if cut_temp !=[]:
                D_cut_list_info[x] = cut_temp
            else:
                object_ori_list.append(x)
        return D_cut_list_info,object_ori_list
############
    def gouping_function_object(self,D_cut_list_info):
#离散变量分组，输入的是一个分组字典，输出的是一个分好组的df
        df_copy = self.df_need_cut
        for x in D_cut_list_info.keys():
            print(x)
            list_tmp = D_cut_list_info[x]
            dic_tmp = {}
            for y in list_tmp:
                number_len = len(y)
                number_list = [y]*number_len
                zip_tmp = zip(y,number_list)
                for z in zip_tmp:
                    dic_tmp[z[0]]= z[1]
            df_copy[x+'_cut'] = df_copy[x].map(dic_tmp.get)
        df_cuted = df_copy.loc[:,df_copy.columns.str.contains('_cut')]
        return df_cuted.join(df_copy[self.remove_col])
        
    def gouping_function_object_int(self,D_cut_list_info):
#这个是另一个离散变量分组的分组编码方式，是按照1,2,3来的
#这个其实有些麻烦了，可以直接用sklearn里面的 encoding ,加上1 ，2 3,4 
        df_copy = self.df_need_cut
        for x in D_cut_list_info.keys():
            print(x)
            list_tmp = D_cut_list_info[x]
            dic_tmp = {}
            i = 0
            for y in list_tmp:
                number_len = len(y)
                number_list = [i]*number_len
                zip_tmp = zip(y,number_list)
                for z in zip_tmp:
                    dic_tmp[z[0]]= z[1]
                i = i+1
            df_copy[x+'_cut'] = df_copy[x].map(dic_tmp.get)
        df_cuted = df_copy.loc[:,df_copy.columns.str.contains('_cut')]
        return df_cuted.join(df_copy[self.remove_col])
##########开始连续变量的卡方分组 ###########################
#########3有得地方也叫做 最优分组 #################################
    def __ChiMerge_MaxInterval_float(self,regroup,col, max_interval):#df,
#为什么没有用df_needed_cut作为输入 呢？这样啰嗦哦，应该不用这个 ，看了后面的，df用在这里
        '''
        :param df: the dataframe containing splitted column, and target column with 1-0
        :param col: splitted column
        :param target: target column with 1-0
        :param max_interval: the maximum number of intervals. If the raw column has attributes less than this parameter, the function will not work
        :return: the combined bins
        '''
        #colLevels = set(regroup[col])
        df = self.df_need_cut[[self.remove_col, col ]]
        if len(regroup[col]) <= max_interval:  #If the raw column has attributes less than this parameter, the function will not work
            print("The number of original levels for {} is less than or equal to max intervals".format(col))
            return list(regroup[col])
        else:
             #Step 1: group the dataset by col and work out the total count & bad count in each level of the raw column
            total = df.groupby([col])[self.remove_col].count()
            #total = pd.DataFrame({'total':total})  
            total = pd.DataFrame(total).rename(columns = {self.remove_col:'total'})
            bad = df.groupby([col])[self.remove_col].sum()
            #bad = pd.DataFrame({'bad':bad}) 
            bad = pd.DataFrame(bad).rename(columns = {self.remove_col:'bad'})
            regroup_assist=  total.merge(bad,left_index=True,right_index=True, how='left')
            regroup_assist.reset_index(level=0, inplace=True)
            N = sum(regroup['total'])
            B = sum(regroup['bad'])
            #the overall bad rate will be used in calculating expected bad count
            overallRate = B*1.0/N
            # since we always combined the neighbours of intervals, we need to sort the attributes
            ##colLevels =sorted(list(colLevels)) #colLevels = regroup.index.tolist()
            # initially, each single attribute forms a single interval
            groupIntervals = regroup[col].values.tolist()
            groupNum = len(groupIntervals)
            times = 0
            while(len(groupIntervals)>max_interval):   #the termination condition: the number of intervals is equal to the pre-specified threshold
                # in each step of iteration, we calcualte the chi-square value of each atttribute
                print(times)
                times  +=1
                chisqList = []
                for interval in groupIntervals:
                    df2 = regroup_assist.loc[regroup_assist[col].isin(interval)]                 
                    chisq = self.Chi2(df2, 'total','bad',overallRate)
                    chisqList.append(chisq)
                #find the interval corresponding to minimum chi-square, and combine with the neighbore with smaller chi-square
                min_position = chisqList.index(min(chisqList))
                if min_position == 0:
                    combinedPosition = 1
                elif min_position == groupNum - 1:
                    combinedPosition = min_position -1
                else:
                    if chisqList[min_position - 1]<=chisqList[min_position + 1]:
                        combinedPosition = min_position - 1
                    else:
                        combinedPosition = min_position + 1
                groupIntervals[min_position] = groupIntervals[min_position]+groupIntervals[combinedPosition]
                # after combining two intervals, we need to remove one of them
                groupIntervals.remove(groupIntervals[combinedPosition])
                groupNum = len(groupIntervals)
            return groupIntervals
            
    def __total_process_two(self,customer_comput,col):
        #要实现这样一种预分组方式
        #若，某一个值的counts占比>于5%，则它单独作为一组
        #若，某一个值的counts 占比 《5%，则 向下累加，直到累计到5%为止
        bad_num = customer_comput.groupby([col])[self.remove_col].sum()
        bad_num.sort_index(inplace = True)
        
        value_counts = customer_comput[col].value_counts()
        #value_counts.sort(inplace = True,ascending=False)
        value_counts.sort_index(inplace = True)
        value_count_index = value_counts.index.tolist()
     
        
        temp_list = []
        end_list = []
        value_list = []
        bad_list = []
        sum_temp = 0
        bad = 0
        #for i in list(range(mark_point+1,len(value_count_index))):
        for i in list(range(0,len(value_count_index))):
            sum_temp += value_counts.iloc[i]
            bad  += bad_num.iloc[i]
            temp_list =temp_list +  [value_count_index[i]]
            if sum_temp  >1000 and bad>0 :
                end_list.append(temp_list)
                bad_list.append(bad)
                value_list.append(sum_temp)
                temp_list = []
                sum_temp = 0
                bad = 0
        groupIntervals = pd.DataFrame({'bad': bad_list, 'total':value_list, col:end_list})    
        return(groupIntervals)
        
    def D_making_chi2_float(self,max_interval):
        D_cut_list_info = {}
        for x in self.float_df_need_cut_col:
            print(x)
            self.df_need_cut[x] = self.df_need_cut[x].apply(lambda x: np.nan if x == '' else x )
            df = self.df_need_cut[[self.remove_col, x ]]
            df.dropna(inplace = True)
            cut_temt = self.__ChiMerge_MaxInterval_float(self.__total_process_two(df,x), x ,max_interval) 
            cut_list = list(set([round(max(set(sub_list)),2) for sub_list in cut_temt]))
            cut_list.sort()
            D_cut_list_info[x] = cut_list
            print(x)
            print(cut_list)
        return D_cut_list_info
        
#############################################

class Test_too_close(Grouping):
    min_list = []
    stop_tag = 0
    times = 0
    def __init__(self,df_need_cut,df_cuted,D_equal_freq,remove_col):
        Grouping.__init__(self,df_need_cut,remove_col)# = 'cus_type'
#        self.df_need_cut = df_need_cut
#        float_df_need_cut_tmp = df_need_cut.columns.tolist()
#        float_df_need_cut_tmp.remove(remove_col)
#        self.float_df_need_cut_col = float_df_need_cut_tmp
#        self.remove_col = remove_col
        self.df_cuted = df_cuted
        self.D_equal_freq = D_equal_freq
    
    def __check_too_close(self,threshold):
        
        for x in self.float_df_need_cut_col:
            bad_rate_list = list(self.df_cuted.groupby(x+'_cut').apply(lambda surf:surf[self.remove_col].sum()/(len(surf)+1)).values)
            bad_rate_after = bad_rate_list[:-1]
            bad_rate_befor = bad_rate_list[1:]
            bad_rate_bite = list(map(lambda x:abs(x[0] - x[1]),zip(bad_rate_after,bad_rate_befor)))
            if bad_rate_bite ==[]:
                print(x)
                continue
            if min(bad_rate_bite)<threshold and len(self.D_equal_freq[x])>3:
                self.min_list.append(min(bad_rate_bite))
                try:
                    the_delet_x = self.D_equal_freq[x][bad_rate_bite.index(min(bad_rate_bite))]
                    self.D_equal_freq[x].remove(the_delet_x)
                except:
                    print(x)
                    print(bad_rate_bite)            
        if self.min_list !=[]:
            print('the lenth of min_list:',len(self.min_list))
            
            self.df_cuted = Grouping.gouping_function_float(self,self.D_equal_freq)
        else:
            self.stop_tag = 1
        
        

    def recursive_fun(self,threshold):
        self.__check_too_close(threshold)
        if self.min_list == []:
            print('初始数据中没有目标值太接近的分组。')
        else:
            while self.stop_tag == 0:
                self.times +=1
                print(self.times)
                print(self.min_list)
                self.min_list = []
                self.__check_too_close(threshold)
#                if  self.min_list !=[]:
#                    self.df_cuted = Grouping.gouping_function(self,self.D_equal_freq)

            print('Too closing checking done!')
        return self.df_cuted,self.D_equal_freq
        
############################################################################################## 
#考虑：这个需要继承 groupig 那个类  然后写一个 递归 来 完成 该功能 
class Test_wave(Grouping):
    stop_tag = 0
    not_ok_list = []
    def __init__(self,df_need_cut,float_cut_df_equal_1_put,D_equal_freq_put,remove_col):# = 'cus_type'
        Grouping.__init__(self,df_need_cut,remove_col)
        self.float_cut_df_equal_1 = float_cut_df_equal_1_put
        self.D_equal_freq = D_equal_freq_put
#
#        rate_sub_list = []
#        ok_list = []
#        not_ok_list = []
#        not_ok_D = {}
    #把输入的df（已经分段了的）
    def __make_list(self,x):# ,type_name = 'is_fraud'= 'cus_type'
        #for x in columns_col:
                #if D_equal_freq != [0]:
        bad_rate = self.float_cut_df_equal_1.groupby(x).apply(lambda surf:surf[self.remove_col].sum()/(len(surf)+1))
        bad_rate_befor = bad_rate.values[:-1]
        bad_rate_after = bad_rate.values[1:]
        rate_sub_list = list(map(lambda x :(x[0] - x[1]),zip(bad_rate_after,bad_rate_befor)))
        return  rate_sub_list
                  
    def cheak_tag(self,x):
        times_count_1 = 0
        small_one = 1
        stop_time = 0
        rate_sub_list = self.__make_list(x)
        print(rate_sub_list)
        while times_count_1 != len(rate_sub_list):
            if times_count_1 == 0:
                if rate_sub_list[times_count_1]<0:
                        small_one = 1
                else:
                        small_one = 0
            else:
                if small_one ==1 and rate_sub_list[times_count_1]<0 and stop_time ==0:
                    small_one = 1
                    stop_time = 0
                elif small_one == 0 and rate_sub_list[times_count_1]>0 and stop_time == 0:
                    small_one = 0
                    stop_time = 0
                elif small_one == 1 and rate_sub_list[times_count_1]>0 and stop_time ==0:
                    small_one = 0 
                    stop_time = 1
                elif small_one  ==  0 and rate_sub_list[times_count_1]<0 and stop_time == 0 :
                    small_one = 1
                    stop_time = 1
                elif small_one == 0 and rate_sub_list[times_count_1] <0 and stop_time ==1 :
                    break
                elif small_one == 1 and rate_sub_list[times_count_1] >0 and stop_time ==1:
                    break
                elif small_one == 1 and rate_sub_list[times_count_1] <0 and stop_time == 1:
                    small_one = 1
                    stop_time = 1
                    #应该可以直接continu
                elif small_one == 0 and  rate_sub_list[times_count_1] >0 and  stop_time == 1:
                    small_one = 0
                    stop_time = 1
                    #应该可以直接continu
            times_count_1 = times_count_1+1
            
        return times_count_1,len(rate_sub_list)

    
    def renew_dict(self,x,times_count_1):
        bad_rate = self.float_cut_df_equal_1.groupby(x).apply(lambda surf: surf[self.remove_col].sum()/(len(surf)+1))
        position = times_count_1 #not_ok_D[x]
        befor_rate = abs(bad_rate[position-1] - bad_rate[position])
        after_rate = abs(bad_rate[position+1] - bad_rate[position])
        if befor_rate < after_rate:
            self.D_equal_freq[x[:-4]].remove(self.D_equal_freq[x[:-4]][position-1])
        else:
            self.D_equal_freq[x[:-4]].remove(self.D_equal_freq[x[:-4]][position])
        return self.D_equal_freq
   
    def renew_all(self,cut_sub_name = '_cut'):

        for x in self.float_df_need_cut_col:
            x = x+cut_sub_name
            if len(self.float_cut_df_equal_1[x].unique())>3:
                print(x)
                times_count_1,len_of_rate_sub_list= self.cheak_tag(x)
                print(times_count_1,len_of_rate_sub_list)
                if times_count_1!= len_of_rate_sub_list:
                    self.renew_dict(x,times_count_1)
                    self.not_ok_list.append(x)
        print('*'*50)
        print('the lenth of the ok_list if {0}'.format(len(self.not_ok_list)))
        print('*'*50)
        if len(self.not_ok_list) == 0:
            print('here???')
            self.stop_tag = 1
        else:
            print(self.not_ok_list)
        print('ok')
        return    
                
    def wave_recursive(self):
        while self.stop_tag == 0:
            self.renew_all()
            self.not_ok_list = []
            self.float_cut_df_equal_1 = Grouping.gouping_function_float(self,self.D_equal_freq)
        return self.float_cut_df_equal_1,self.D_equal_freq
#################################
#画图 

def plot_bad_rate(iv_list,df_cuted_3,target,picture_name):
#用来画图的工具，用来看分箱的合理性，保证分组逾期率是最多只有一个拐点的，并且每组的逾期率不是特别接近
#这个需要再研究一下啦
#输入 iv_list： 要画图的列名列表
# df_cuted_3 ：已经分好箱的df
#target：目标变量
#picture_name：存好的 
    var_range_map = {}
    for x in iv_list :
        var_range_map[x] = df_cuted_3.groupby([x]).apply(lambda surf: surf[target].sum()/surf[target].count())
    # matrix sub image
    fig, ax = plt.subplots(nrows = 6, ncols = 10)
    fig.set_size_inches(160, 90)
    var_names = list(var_range_map.keys())
    var_num = len(var_names)
    vid = 0

    for row in ax:
        if vid == var_num:
            break
        for col in row:
            if vid == var_num:
                break
            
            name = var_names[vid]
            range_val = var_range_map[name]
            vid += 1

            if len(range_val) > 16:
                print ("discard id variable!")
                continue
            
            #for i in range(len(range_val)):
                #if range_val[i] != range_val[i]:
                    #range_val.remove(range_val[i])
                   # print ("remove nan value!")
                
            print (name)
            print (range_val)
            X = np.arange(len(range_val)) + 1
        
            col.bar(X, range_val, width=0.25, facecolor = 'lightskyblue', edgecolor = 'white')
            col.plot(X, range_val, 'r', lw=5, label=name)
            col.legend()
            
    # save
    img_name = picture_name + '.jpg'
    plt.savefig(img_name)
    plt.close('all')
    print ("save img " + img_name)

               