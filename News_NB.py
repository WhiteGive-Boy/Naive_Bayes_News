# -*- coding: UTF-8 -*-
import os
import random
import jieba
#from sklearn.naive_bayes import MultinomialNB
#import matplotlib.pyplot as plt

class classifier:
    def __init__(self):
        # Counts of feature/category combinations
        self.fc = {}
        # Counts of documents in each category
        self.cc = {}
        #self.getfeatures = getfeatures

    def incf(self, f, cat):
        self.fc.setdefault(f, {})
        self.fc[f].setdefault(cat, 0)

        self.fc[f][cat]=1+self.fc[f][cat]

    # 增加某一个分类的计数值:
    def incc(self, cat):
        self.cc.setdefault(cat,0)
        self.cc[cat] =self.cc[cat]+ 1

    # 计算某一个特征在某一个分类中出现的次数
    def fcount(self, f, cat):
        #print("fcountinput:",f, cat)
        if f in self.fc and cat in self.fc[f]:
            #print("self.fc[f][cat]:",self.fc[f][cat])
            return self.fc[f][cat]
        else:
            return 0.1

    # 属于某一个分类的文档总数
    def catcount(self, cat):
        if cat in self.cc:
            return self.cc[cat]
        return 0

    # 所有的文档总数
    def totalcount(self):
        return sum(self.cc.values())

    # 所有文档的种类
    def categories(self):
        return self.cc.keys()

    def train(self, features, cat):
        #print("6666666")
       # print(features)
        #print(cat)
        #int("6666666")

        # 针对该分类，为每个特征增加计数值
        for index in range(len(features)):
            #features[index]=str(index)+"_"+str(features[index])
            self.incf(features[index], cat)

        # 增加该分类的计数值
        self.incc(cat)

    def fprob(self, f, cat):
        if self.catcount(cat) == 0:
            return 0.0001

        # 特征在该分类中出现的次数 /
        # 该特征下文档的总数目
        #print("fcount", self.fcount(f, cat))
       # print("catcount", self.catcount(cat))
        return self.fcount(f, cat) / self.catcount(cat)
    def docprob(self, features, cat):

        # Multiply the probabilities of all the features together
        p = 1
        for f in features:
            p = p*self.fprob(f, cat)
            #print("fprob", self.fprob(f, cat))
        return p

    def prob(self, features, cat):
        catprob = self.catcount(cat) / self.totalcount()
        #print("catpro",catprob)
        docprob = self.docprob(features, cat)
        #print("docprob", docprob)
        return docprob * catprob

    def predict(self, features):
        #for index in range(len(features)):
            #features[index]=str(index)+"_"+str(features[index])
        max = 0.0
        best=list(self.categories())[0]
        #print(self.categories())
        for cat in self.categories():
            probs = self.prob(features, cat)
            #print("cat:", cat, "score:", probs)
            if probs > max:
                max = probs
                best = cat
        return best

def sampletrain(cl, traindata, traintarget):
    for left, right in zip(traindata, traintarget):
        #print("left",left)
        #print("right",right)
        c1.train(left, right)

# 手写拉普拉斯修正的朴素贝叶斯
import numpy as np
import pandas as pd



"""
函数说明:中文文本处理
Parameters:
    folder_path - 文本存放的路径
    test_size - 测试集占比，默认占所有数据集的百分之20
Returns:
    all_words_list - 按词频降序排序的训练集列表
    train_data_list - 训练集列表
    test_data_list - 测试集列表
    train_class_list - 训练集标签列表
    test_class_list - 测试集标签列表
"""
def TextProcessing(path, test_size=0.2):
#    folder_list = os.listdir(folder_path)  # 查看folder_path下的文件
    data_list = []  # 数据集数据
    class_list = []  # 数据集类别
    with open(path, 'r', encoding='utf-8') as f:  # 打开txt文件
        for line in f.readlines():
            line = line.strip().split("_!_")
            # print(line)
            if (len(line) >= 5):
                strr = line[3] + line[4]
            else:
                strr = line[3]
            word_cut = jieba.cut(strr, cut_all=False)  # 精简模式，返回一个可迭代的generator
            word_list = list(word_cut)  # generator转换为list
            data_list.append(word_list)
            class_list.append(line[2])

    data_class_list = list(zip(data_list, class_list))  # zip压缩合并，将数据与标签对应压缩
    random.shuffle(data_class_list)  # 将data_class_list乱序
    index = int(len(data_class_list) * test_size) + 1  # 训练集和测试集切分的索引值
    train_list = data_class_list[index:]  # 训练集
    test_list = data_class_list[:index]  # 测试集
    train_data_list, train_class_list = zip(*train_list)  # 训练集解压缩
    test_data_list, test_class_list = zip(*test_list)  # 测试集解压缩

    all_words_dict = {}  # 统计训练集词频
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    # 根据键的值倒序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)
    all_words_list, all_words_nums = zip(*all_words_tuple_list)  # 解压缩
    all_words_list = list(all_words_list)  # 转换成列表
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


"""
函数说明:读取文件里的内容，并去重
Parameters:
    words_file - 文件路径
Returns:
    words_set - 读取的内容的set集合
"""
def MakeWordsSet(words_file):
    words_set = set()  # 创建set集合
    with open(words_file, 'r', encoding='utf-8') as f:  # 打开文件
        for line in f.readlines():  # 一行一行读取
            word = line.strip()  # 去回车
            if len(word) > 0:  # 有文本，则添加到words_set中
                words_set.add(word)
    return words_set  # 返回处理结果


"""
函数说明:文本特征选取
Parameters:
    all_words_list - 训练集所有文本列表
    deleteN - 删除词频最高的deleteN个词
    stopwords_set - 指定的结束语
Returns:
    feature_words - 特征集
"""
def words_dict(all_words_list, deleteN, stopwords_set=set()):
    feature_words = []  # 特征列表
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000:  # feature_words的维度为1000
            break
            # 如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
        n += 1
    return feature_words


"""
函数说明:根据feature_words将文本向量化
Parameters:
    train_data_list - 训练集
    test_data_list - 测试集
    feature_words - 特征集
Returns:
    train_feature_list - 训练集向量化列表
    test_feature_list - 测试集向量化列表
"""
def TextFeatures(train_data_list, test_data_list, feature_words):
    def text_features(text, feature_words):  # 出现在特征集中，则置1
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features

    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    for features in train_feature_list:
        for index in range(len(features)):
            features[index]=str(index)+"_"+str(features[index])
    for features in test_feature_list:
        for index in range(len(features)):
            features[index]=str(index)+"_"+str(features[index])

    return train_feature_list, test_feature_list  # 返回结果


"""
函数说明:新闻分类器
Parameters:
    train_feature_list - 训练集向量化的特征文本
    test_feature_list - 测试集向量化的特征文本
    train_class_list - 训练集分类标签
    test_class_list - 测试集分类标签
Returns:
    test_accuracy - 分类器精度
"""
def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list,c1):
    classifier = sampletrain(c1,train_feature_list, train_class_list)
    acc=0
    for index in range(len(test_feature_list)):
        lable = c1.predict(test_feature_list[index])

        print("real lable:", test_class_list[index])
        print("predict lable:",lable)
        if(test_class_list[index]==lable):
            acc=acc+1
    return float(acc)/len(test_feature_list)


if __name__ == '__main__':
    # 文本预处理
    folder_path = "./toutiao.txt"  # 训练集存放地址
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path,test_size=0.2)
    # 生成stopwords_set
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)

    test_accuracy_list = []

    c1=classifier()

    feature_words = words_dict(all_words_list, 450, stopwords_set)
    print(feature_words)
    train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)

    acc = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list,c1)
    print(c1.cc)
    print(c1.fc)
    print("acc:",acc)
    #print("predict lable:",lable)