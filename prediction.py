import pickle
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from src.io import *
from src.data_preprocessing import *
import scipy.stats
from src.classification import *

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h

np.random.seed(0);

data_folder = '../../data_v5'

#convert new medication code to old medication code
drugConvTable = loadData(data_folder + '/drugCode.csv')

phyExam1 = loadData(data_folder + '/physicalExam_1.csv')
phyExam2 = loadData(data_folder + '/physicalExam_2.csv')

phyExam = mergePhyExams(phyExam1,phyExam2)

disease_code = 'F00'
d_lv = 3 #merging based on the first three codes.
m_lv = 4 #merging based on a main ingredient

years = ['0yr','1yr','2yr','3yr','4yr']#'1yr',

for year in years:
    print("--------------{} prediction---------------".format(year))

    for idx_prescription in range(2): # all data (0) or data with prescription (1)

        bLoadCSV = False
        b_prepr = False #whether there is a preprocessed csv file or not.

        if not b_prepr:
            cur_data_folder = data_folder + '/' + year + '/'
            d_train = pickle.load(open( cur_data_folder+disease_code+".p", "rb" ) )
            cal_date(d_train,8,9,4,True)
            D_data = convertDrugCode(d_train,drugConvTable,[8,11],13)
        else:
            D_data = loadData(cur_data_folder +disease_code + '_c.csv')

        if idx_prescription:
            print('w/ prescripted drugs')
            D_data = findTagetCodeInEHR(D_data)
        else:
            print('w/o prescripted drugs')

        dcode_list = ['F00','F000','F001','F002','F009','G30','G300','G301','G308','G309'] #disease code list for AD

        if not bLoadCSV:
            print('data loading {}'.format(cur_data_folder+"all.p"))
            nonD_data = pickle.load(open( cur_data_folder+"all.p", "rb" ) )
            #cal_date(d_train,8,9,4,True)
        else:
            print('data loading {}'.format(cur_data_folder + "all.csv"))
            nonD_data = loadData(cur_data_folder +'all.csv')
        cal_date(nonD_data,4,5,4,False)

        print('normal data converting')
        nonD_data_t = convertDrugCode(nonD_data,drugConvTable,[8])
        nonD_data1,diseaseMap_t2,diseaseFreq_t2,drugMap_t2,drugFreq_t2 = extractData(nonD_data_t,dcode_list,d_lv,m_lv)


        disFreqCutOff = 5
        drugFreqCutOff = 5
        t2_cache = [diseaseMap_t2,drugMap_t2,diseaseFreq_t2,drugFreq_t2]

        print('computer disease & drug distribution for AD')
        prediction_yr = int(year[0]) # x year in advance
        npD_data = comDiseaseDrugDist(D_data,t2_cache,disFreqCutOff,drugFreqCutOff,prediction_yr,d_lv,m_lv)
        print('computer disease & drug distribution for normal')
        npNonD_data = comDiseaseDrugDist(nonD_data1,t2_cache,disFreqCutOff,drugFreqCutOff,prediction_yr,d_lv,m_lv)

        phyExamDic = {}
        years = {2004: [0, 142280], 2003: [142281, 216038], 2002: [261039, 374679], 2008: [374680, 585639], \
                 2005: [585640, 721114], 2007: [721115, 883943], 2006: [883944, 1058568], \
                 2010: [1058569, 1058569 + 228745], 2009: [1058569 + 228746, 1058569 + 440286]}
        for yr in range(2002,2011):
            tempData = {}
            [start, end] = years[yr]
            for dat in phyExam[start:end+1]:
                tempData[dat[1]] = dat
            phyExamDic[yr] = tempData

        D_data_allFeatures = []
        nonD_data_allFeatures = []
        phyExamLength = phyExam.shape[1]

        print('add physical records')
        #for idx,d_type in enumerate(data_type):
        D_data_allFeatures = addPhyExamFeatures(D_data,npD_data,phyExamDic,phyExamLength,5)
        nonD_data_allFeatures = addPhyExamFeatures(nonD_data1,npNonD_data,phyExamDic,phyExamLength,5)

        age = 13
        ageCol = 1

        print('filtering out people w/ 65')
        D_65 = D_data_allFeatures[D_data_allFeatures[:,ageCol]>age]
        nonD_65 = nonD_data_allFeatures[nonD_data_allFeatures[:,ageCol]>age]
        nonD_65_bal = nonD_65


        print(D_65.shape)
        print(nonD_65_bal.shape)

        #mean & std of the record period.
        D_period=D_65[:,3]
        nonD_period=nonD_65_bal[:,3]
        print('Look-back: D:{} ({}),normal:{}({})'.format(D_period.mean(),D_period.std(),nonD_period.mean(),nonD_period.std()))

        m, m_h1, m_h2 = mean_confidence_interval(D_period)
        print('Look-back D: {},{},{}'.format(m, m_h1, m_h2))
        m, m_h1, m_h2 = mean_confidence_interval(nonD_period)
        print('Look-back non-D: {},{},{}'.format(m, m_h1, m_h2))


        #remove the record period
        D_65_noP = np.delete(D_65,3,axis=1)
        nonD_65_noP = np.delete(nonD_65_bal,3,axis=1)
        #remove the income info
        D_65_noP = np.delete(D_65_noP,2,axis=1)
        nonD_65_noP = np.delete(nonD_65_noP,2,axis=1)


        #mean & std of the number of descriptors for each group
        AD_sum = np.sum(D_65_noP[:,2:-32],axis = 1)
        NP_sum = np.sum(nonD_65_noP[:,2:-32],axis = 1)

        print('Features: AD:{:.2f}({:.2f}),normal:{:.2f}({:.2f})'\
              .format(round(AD_sum.mean(),2)+34,round(AD_sum.std(),2),\
                      round(NP_sum.mean(),2)+34,round(NP_sum.std(),2)))

        m, m_h1, m_h2 = mean_confidence_interval(AD_sum)
        print('Features AD: {},{},{}'.format(m+34, m_h1+34, m_h2+34))
        m, m_h1, m_h2 = mean_confidence_interval(NP_sum)
        print('Features normal: {},{},{}'.format(m+34, m_h1+34, m_h2+34))


        #mean & std of age for each group
        print('Age: D m:{}({}),nonD m:{}({})'.format \
                  (D_65_noP.mean(axis=0)[1] * 5.0,D_65_noP.std(axis=0)[1] * 5.0,\
                   nonD_65_noP.mean(axis=0)[1] * 5.0,\
                   nonD_65_noP.std(axis=0)[1] * 5.0))

        m, m_h1, m_h2 = mean_confidence_interval(D_65_noP[:, 1])
        print('Age AD: {},{},{}'.format(m*5.0, m_h1*5.0, m_h2*5.0))
        m, m_h1, m_h2 = mean_confidence_interval(nonD_65_noP[:, 1])
        print('Age normal: {},{},{}'.format(m*5.0, m_h1*5.0, m_h2*5.0))


        #mean & std of income for each group
        print('income: D m:{}({}),nonD m:{}({})'.format
                  (D_65.mean(axis=0)[2] ,D_65.std(axis=0)[2] ,
                   nonD_65.mean(axis=0)[2] ,
                   nonD_65.std(axis=0)[2] ))

        m,m_h1,m_h2 = mean_confidence_interval(D_65[:,2])
        print('income AD: {},{},{}'.format(m,m_h1,m_h2))
        m,m_h1,m_h2 = mean_confidence_interval(nonD_65[:,2])
        print('income normal: {},{},{}'.format(m,m_h1,m_h2))

        # create X and y
        X = np.concatenate((D_65_noP, nonD_65_noP))

        # set 1 for PC, 0 for non-PC
        y = np.zeros((X.shape[0],), dtype=int)
        y[0:D_65_noP.shape[0]] = 1


        result_folder = cur_data_folder + 'results' + str(idx_prescription) + '/'

        pickle.dump(D_65, open(result_folder + "tX_" + disease_code + ".p", "wb"))
        pickle.dump(nonD_65, open(result_folder + "ty_" + disease_code + ".p", "wb"))

        pickle.dump(diseaseMap_t2, open(result_folder +"diseaseMap_"+disease_code+".p", "wb"))
        pickle.dump(drugMap_t2, open(result_folder +"drugMap_"+disease_code+".p", "wb"))
        pickle.dump(phyExam, open(result_folder +"phyExam"+".p", "wb"))


        n_iter = 100
        print('X={}'.format(X.shape))
        print('y={}'.format(y.shape))

        for method in ['SVM', 'RF', 'LR']:
            print('----{}----'.format(method))
            roc_label, roc_pred, roc_prob, clf = JP_classify(method, D_65_noP, nonD_65_noP, n_iter)

            pickle.dump(roc_label,
                        open(result_folder + 'roc_label_con_' + str(age) + '_' + disease_code + '_' + method + '.p', "wb"))
            pickle.dump(roc_pred,
                        open(result_folder + 'roc_pred_con_' + str(age) + '_' + disease_code + '_' + method + '.p', "wb"))
            pickle.dump(roc_prob,
                        open(result_folder + 'roc_prob_con_' + str(age) + "_" + disease_code + '_' + method + '.p', "wb"))
            pickle.dump(clf, open(result_folder + 'clf' + str(age) + "_" + disease_code + '_' + method + '.p', "wb"))





