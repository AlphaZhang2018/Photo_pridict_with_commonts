from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
import numpy as np
from com.data_manage import data_processing

#模型文件路径
path = '/Users/feizhang/PycharmProjects/Project_Demo/models/'

def generate_models():
    train_feature_list, train_tag_list, test_feature_list, test_tag_list = data_processing()
    x = np.array(train_feature_list)
    y = np.array(train_tag_list)
    # x_test = test_feature_list
    # y_text = test_tag_list


    #Logistic
    Logistic_model = LogisticRegression()
    Logistic_model.fit(x,y)
    # logistic_pre_y = Logistic_model.predict(x_test)
    #保存模型
    model_file = path + "logistic_model.m"
    joblib.dump(Logistic_model, model_file)

    #朴素贝叶斯
    nb_model = GaussianNB()
    nb_model.fit(x,y)
    # nb_pre_y = nb_model.predict(x_test)
    # 保存模型
    model_file = path + "nb_model.m"
    joblib.dump(nb_model, model_file)

    #KNN
    Knn_model = KNeighborsClassifier()
    Knn_model.fit(x,y)
    # knn_pre_y = Knn_model.predict(x_test)
    # 保存模型
    model_file = path + "Knn_model.m"
    joblib.dump(Knn_model, model_file)

    #SVC
    svc_model = SVC()
    svc_model.fit(x,y)
    # svc_pre_y = svc_model.predict(x_test)
    # 保存模型
    model_file = path + "svc_model.m"
    joblib.dump(svc_model, model_file)

    #XGboost
    xgboost_model = XGBClassifier()
    xgboost_model.fit(x,y)
    # xgboost_pre_y = xgboost_model.predict(x_test)
    # 保存模型
    model_file = path + "xgboost_model.m"
    joblib.dump(xgboost_model, model_file)

    #随机森林
    rf_model = RandomForestClassifier()
    rf_model.fit(x,y)
    # rf_pre_y = rf_model.predict(x_test)
    # 保存模型
    model_file = path + "rf_model.m"
    joblib.dump(rf_model, model_file)

    #GDBT
    gdbt_model = GradientBoostingClassifier()
    gdbt_model.fit(x,y)
    # gdbt_pre_y = gdbt_model.predict(x_test)
    # 保存模型
    model_file = path + "gdbt_model.m"
    joblib.dump(gdbt_model, model_file)


result_file = '/Users/feizhang/PycharmProjects/Project_Demo/datas/result.txt'
def confusion_matrix(expect,predict,str1):
    tn, fp, fn, tp = metrics.confusion_matrix(expect, predict).ravel()
    print("tn, fp, fn, tp")
    print(tn, fp, fn, tp)
    Accuracy = str(round((tp + tn) / (tp + fp + fn + tn), 3))
    Recall = str(round((tp) / (tp + fn), 3))
    Precision = str(round((tp) / (tp + fp), 3))
    print("Accuracy: " + str(round((tp + tn) / (tp + fp + fn + tn), 3)))
    print("Recall: " + str(round((tp) / (tp + fn), 3)))
    print("Precision: " + str(round((tp) / (tp + fp), 3)))

    strs = []
    strs.append(str1 + '\n')
    strs.append("tn, fp, fn, tp"+ '\n')
    strs.append(str(tn) + "  " + str(fp) + "  " + str(fn) + "  " + str(tp)+ '\n')
    strs.append("Accuracy: " + str(round((tp + tn) / (tp + fp + fn + tn), 3))+ '\n')
    strs.append("Recall: " + str(round((tp) / (tp + fn), 3))+ '\n')
    strs.append("Precision: " + str(round((tp) / (tp + fp), 3))+ '\n')
    strs.append("\n")

    for i in range(0,len(strs)):
        write_2_txt(result_file,strs[i])


def write_2_txt(filename,str2):
    f = open(filename,'a')
    f.writelines(str(str2))
    f.close()

def predict_with_models():
    train_feature_list, train_tag_list, test_feature_list, test_tag_list = data_processing()
    # x = train_feature_list
    # y = train_tag_list
    x_test = test_feature_list
    y_text = test_tag_list

    model_file = path + "logistic_model.m"
    Logistic_model = joblib.load(model_file)
    Logistic_pre_y = Logistic_model.predict(x_test)
    confusion_matrix(y_text, Logistic_pre_y,"Logistic_model")

    model_file = path + "nb_model.m"
    nb_model = joblib.load(model_file)
    nb_pre_y = nb_model.predict(x_test)
    confusion_matrix(y_text, nb_pre_y, "nb_model")

    model_file = path + "Knn_model.m"
    Knn_model = joblib.load(model_file)
    Knn_pre_y = Knn_model.predict(x_test)
    confusion_matrix(y_text, Knn_pre_y, "Knn_model")

    model_file = path + "svc_model.m"
    svc_model = joblib.load(model_file)
    svc_pre_y = svc_model.predict(x_test)
    confusion_matrix(y_text, svc_pre_y, "svc_model")

    model_file = path + "xgboost_model.m"
    xgboost_model = joblib.load(model_file)
    xgboost_pre_y = xgboost_model.predict(x_test)
    confusion_matrix(y_text, xgboost_pre_y, "xgboost_model")

    model_file = path + "rf_model.m"
    rf_model = joblib.load(model_file)
    rf_pre_y = rf_model.predict(x_test)
    confusion_matrix(y_text, rf_pre_y, "rf_model")

    model_file = path + "gdbt_model.m"
    gdbt_model = joblib.load(model_file)
    gdbt_pre_y = gdbt_model.predict(x_test)
    confusion_matrix(y_text, gdbt_pre_y, "gdbt_model")


if __name__ == '__main__':
    generate_models()
    predict_with_models()
