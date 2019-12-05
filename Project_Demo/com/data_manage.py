import codecs
import random

import numpy as np
import pandas as pd


from gensim.models import word2vec
from gensim.models.word2vec import LineSentence, Word2Vec

data_source = '/Users/feizhang/PycharmProjects/Project_Demo/datas/text.xlsx'

stop_words_path = '/Users/feizhang/PycharmProjects/Project_Demo/datas/stopwords.txt'

vip_words_path = '/Users/feizhang/PycharmProjects/Project_Demo/datas/VIP_20000.txt'

#把两个excel的数据合在一起，并且只保留text和seg列
def read_excel():
    df1 = pd.read_excel('/Users/feizhang/PycharmProjects/Project_Demo/datas/S.xlsx',usecols=[2,6])
    df_list1 = df1.values.tolist()
    df2 = pd.read_excel('/Users/feizhang/PycharmProjects/Project_Demo/datas/F.xlsx',usecols=[2,4])
    df_list2 = df2.values.tolist()
    data_list = []

    for i in range(1,len(df_list1)):
        if df_list1[i][0] is not None:
            data_list.append(df_list1[i])

    for j in range(1,len(df_list2)):
        if df_list2[j][0] != "NaN" or df_list2[j][0] != "nan":
            df_list2[j][1] = (df_list2[j][1])[0]
            data_list.append(df_list2[j])
    df_new = pd.DataFrame(data_list,columns=['txt','tag'])
    df_new.to_excel('/Users/feizhang/PycharmProjects/Project_Demo/datas/text.xlsx',index=False)


#jieba分词
import jieba


#设置停用词
def stopwords(stop_words_path):
    stopwords = [line.strip() for line in open(stop_words_path,encoding='utf-8').readlines()]
    return stopwords

stopword = stopwords()

# 分词
def fenci(sentence):
    #生成的是一行用空格隔开的字
    # cutted_sentence = ''
    # sentence_list = jieba.cut(sentence.strip())
    # for word in sentence_list:
    #     if word not in stopword:
    #         if word != '\t':
    #             cutted_sentence += word
    #             cutted_sentence += ' '
    #生成的是一个list
    cutted_sentence = []
    sentence_list = jieba.cut(sentence.strip())
    for word in sentence_list:
        if word not in stopword:
            if word != '\t':
                cutted_sentence.append(word)
    return cutted_sentence

#从VIP_20000.txt中建立字典
def vip_word_dic(vip_words_path):
    codecs_r = codecs.open(vip_words_path,'r',encoding='utf-8')
    vip_lines = codecs_r.readlines()
    vip_dict = {}
    for i in range(0,len(vip_lines)):
        vip_lines_list = vip_lines[i].replace('\n','').split('    ')
        key = vip_lines_list[0]
        # value = vip_lines_list[1:]
        value = []
        for j in range(1,len(vip_lines_list)):
            value.append(float(vip_lines_list[j]))
        vip_dict[key] = value
    # print(vip_dict)
    return vip_dict

vip_dic = vip_word_dic()

#计算每一句的向量，用平均向量来表示这句话的向量
def sentence_2_vec(cutted_sentence):
    cutted_sentence_list = []
    for i in range(0,len(cutted_sentence)):
        cutted_sentence_list = []
        if cutted_sentence[i] in vip_dic.keys():
            cutted_sentence_list.append(vip_dic.get(cutted_sentence[i]))
    if len(cutted_sentence_list) > 1:
        cutted_sentence_list_mean = (np.mean(cutted_sentence_list))
        # cutted_sentence_list_mean = (cutted_sentence_list_mean[0])
    elif len(cutted_sentence_list) == 0 :
        cutted_sentence_list_mean = ([0]*200)
    else:
        cutted_sentence_list_mean = (cutted_sentence_list[0])
    # print(cutted_sentence_list_mean)
    return cutted_sentence_list_mean


#生成样本矩阵与标签
def generate_tag_feature_list(data_source):
    # df1 = pd.read_excel('/Users/feizhang/PycharmProjects/Project_Demo/datas/text.xlsx', usecols=[0, 1])
    df1 = pd.read_excel(data_source, usecols=[0, 1])
    df_list1 = df1.values.tolist()

    generate_list = []
    for i in range(0, len(df_list1)):
        vec_list = []
        sentence = fenci(df_list1[i][0])
        sentence_vec = sentence_2_vec(sentence)
        # print(sentence_vec)
        vec_list.append(int(df_list1[i][1]))
        vec_list.append(sentence_vec)
        generate_list.append(vec_list)
    return generate_list


#拷贝样本
def copy_list(large_list,small_list):
    bei = int(len(large_list)/len(small_list))
    ge = len(large_list)%len(small_list)
    small_list_copy = []
    if bei > 1:
        for i in range(0,bei):
            list_copy = []
            list_copy = small_list.copy()
            for l in range(0,len(list_copy)):
                small_list_copy.append(list_copy[l])
        random_small_list_number = random.sample(range(0,len(small_list)),ge)
        for k in range(0,len(random_small_list_number)):
            small_list_copy.append(small_list[random_small_list_number[k]])
    else:
        random_small_list_number = random.sample(range(0,len(small_list)),ge)
        for j in range(0,len(random_small_list_number)):
            small_list_copy.append(small_list[j])
    return small_list_copy


# 样本平衡
def sample_balance(matrixs_list):
    list_2 = []
    list_1 = []
    list_0 = []
    for i in range(0,len(matrixs_list)):
        if int(matrixs_list[i][0]) == 2:
            list_2.append(matrixs_list[i])
        elif int(matrixs_list[i][0]) == 1:
            list_1.append(matrixs_list[i])
        elif int(matrixs_list[i][0]) == 0:
            list_0.append(matrixs_list[i])
    #因为肉眼可见2的编号会比较多，所以，就不用比较了，直接平衡
    list_1_copy = copy_list(list_2,list_1)
    list_0_copy = copy_list(list_2,list_1)

    list_2.extend(list_1_copy)
    list_2.extend(list_0_copy)

    return list_2


#把特征list和标签list拆分出来
def sperate_tag_feature(list):
    tag_list = []
    feature_list = []
    for i in range(0,len(list)):
        feature_list.append(list[i][1])
        tag_list.append(int(list[i][0]))
    return feature_list,tag_list


#随机百分之十样本用来测试，百分之九十来测试模型
def disperse_samples(list):
    ge2 = int(len(list) * 0.9)
    random_list_number = random.sample(range(0,len(list)), ge2)
    num_list = []
    train_list = []
    test_list = []

    for i in range(0,len(list)):
        if i not in random_list_number:
            test_list.append(list[i])

    for i in range(0,len(random_list_number)):
        train_list.append(list[random_list_number[i]])

    train_feature_list, train_tag_list = sperate_tag_feature(train_list)
    test_feature_list, test_tag_list = sperate_tag_feature(test_list)

    return (train_feature_list), (train_tag_list), (test_feature_list), (test_tag_list)


#数据处理流程
def data_processing():
    matrixs_list = generate_tag_feature_list()
    list = sample_balance(matrixs_list)
    train_feature_list, train_tag_list, test_feature_list, test_tag_list = disperse_samples(list)

    return train_feature_list, train_tag_list, test_feature_list, test_tag_list





if __name__ == '__main__':
    train_feature_list, train_tag_list, test_feature_list, test_tag_list = data_processing()