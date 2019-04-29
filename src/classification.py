from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score,cross_val_predict,StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import numpy as np
import pickle
from sklearn.metrics import f1_score
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def getCoeff(X,clf,task):
    diseaseMap = pickle.load(open("diseaseMap_" + task + ".p", "rb"))
    drugMap = pickle.load(open("drugMap_" + task + ".p", "rb"))
    phyExam = pickle.load(open("phyExam.p", "rb"))

    if task == 't1':
        coeff = clf.coef_[0]
        features = np.empty(X.shape[1], dtype=object)
        for code in diseaseMap:
            # print(code,diseaseMap[code])
            features[0] = 'sex'
            features[1] = 'age_group'
            features[diseaseMap[code] + 2] = code
        for code in drugMap:
            features[len(diseaseMap) + 2 + drugMap[code]] = code
        for peIdx in range(phyExam.shape[1] - 2):
            features[len(diseaseMap) + 2 + len(drugMap) + peIdx] = 'pe' + str(peIdx)
    elif task == 't3':
        features = np.empty(X.shape[1], dtype=object)
        coeff = clf.best_estimator_.coef_[0]

        for code in diseaseMap:
            # print(code,diseaseMap[code])
            features[0] = 'sex'
            features[1] = 'age_group'
            features[2] = 'num_in_patient_visits'
            features[3] = 'num_in_patient_days'
            features[diseaseMap[code] + 4] = code
        for code in drugMap:
            features[len(diseaseMap) + 4 + drugMap[code]] = code
        for peIdx in range(phyExam.shape[1] - 3):
            features[len(diseaseMap) + 4 + len(drugMap) + peIdx] = 'pe' + str(peIdx)

    log_weights = np.transpose(coeff)
    abs_log_weights = np.absolute(log_weights)
    log_srtd_ix = np.argsort(abs_log_weights, axis=0)
    srtd_array = []
    for i in log_srtd_ix:
        line = [[features[i], log_weights[i]]]
        srtd_array = srtd_array + line
    srtd_array = np.array(srtd_array)

    pickle.dump(srtd_array, open("coeff_" + task + ".p", "wb"))

from sklearn.model_selection import ShuffleSplit
from sklearn.utils import resample

def JP_classify(method,X,y,n_fold):
    """classification

        Args:
            method: classification method (random forest, logistic regression, SVM)
            X_0: AD patient data
            X_1: normal people data
            n_iter: # of iteration for bootstrap

        Returns:

        """
    #from sklearn import cross_validation

    #bs = cross_validation.Bootstrap(n=len(X_1), n_bootstraps=n_iter, n_train=0.8, n_test=0.2, random_state=0)


    #n_fold = 5


    # inner_cv = StratifiedKFold(n_splits=n_fold, shuffle=False, random_state=234)
    # outer_cv = StratifiedKFold(n_splits=n_fold, shuffle=False, random_state=234)

    # avg_acc = []
    # avg_train_acc = []
    # avg_TP = []
    # avg_TN = []
    # avg_FP = []
    # avg_FN = []
    # avg_sen = []
    # avg_spec = []
    #
    # roc_label = []
    # roc_pred = []
    # roc_prob = []
    # outer_loop = 0
    # print('loading')
    # n_samples = len(X_1)
    # for idx in range(n_iter):
    #
    #     rand_idx = np.random.choice(X_1.shape[0], X_0.shape[0])
    #     X = np.concatenate((X_0, X_1[rand_idx]))
    #     y = np.zeros((X.shape[0],), dtype=int)
    #     y[:X_0.shape[0]] = 1
    #
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #
    #     sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    #     print(outer_loop)
    inner_cv = StratifiedKFold(n_splits=n_fold, shuffle=False, random_state=234)
    outer_cv = StratifiedKFold(n_splits=n_fold, shuffle=False, random_state=234)

    avg_acc = []
    avg_train_acc = []
    avg_TP = []
    avg_TN = []
    avg_FP = []
    avg_FN = []
    avg_sen = []
    avg_spec = []

    roc_label = []
    roc_pred = []
    roc_prob = []
    outer_loop = 0
    print('loading')
    # n_samples = len(X_1)
    for train_index, test_index in outer_cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(outer_loop, X_train.shape, X_test.shape)

        outer_loop+=1

        if method == "RF":
            params = {'randomforest__min_samples_leaf': np.arange(1, 51, 5),
                      'randomforest__n_estimators': np.arange(10, 100, 10)}
            # clf_m = RandomForestClassifier(random_state=0)

            pipe = Pipeline([
                ('featureExtract', VarianceThreshold()),
                ('scaling', StandardScaler()),
                ('randomforest', RandomForestClassifier(random_state=0))
            ])
        elif method == 'SVM':
            params = {'svm__alpha': np.logspace(-4, 7, 12)}
            # params = {'svm__alpha': np.logspace(-5, -3, 10),
            #           'kernel__gamma': np.logspace(-5, -3, 10)}
            # clf_m = RandomForestClassifier(random_state=0)

            pipe = Pipeline([
                ('featureExtract', VarianceThreshold()),
                ('scaling', StandardScaler()),
                ("svm", SGDClassifier(max_iter=1000, tol=1e-5,random_state=0))
            ])
        elif method == 'LR':
            params = {'lr__C': np.logspace(-3, 8, 12)}

            pipe = Pipeline([
                ('featureExtract', VarianceThreshold()),
                ('scaling', StandardScaler()),
                ('lr', linear_model.LogisticRegression(random_state=0))
            ])

        clf = GridSearchCV(estimator=pipe, param_grid=params, cv=inner_cv, scoring='f1', n_jobs=-1)
        # clf = GridSearchCV(estimator=pipe, param_grid=params, cv=sss, scoring='accuracy',n_jobs = -1)
        clf.fit(X_train, y_train)

        fs = clf.best_estimator_.named_steps['featureExtract']
        mask = fs.get_support()
        y_pred = clf.predict(X_test)

        if method == 'SVM':
            y_prob = clf.decision_function(X_test)
        else:
            y_prob = clf.predict_proba(X_test)


        y_pred_train = clf.predict(X_train)
        acc = accuracy_score(y_test, y_pred)
        train_acc = accuracy_score(y_train, y_pred_train)
        if method == 'SVM':
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        f1 = f1_score(y_test, y_pred, average='weighted')

        roc_label = np.append(roc_label, y_test)
        roc_pred = np.append(roc_pred, y_pred)
        if method == 'SVM':
            roc_prob = np.append(roc_prob, y_prob)
        else:
            roc_prob = np.append(roc_prob, y_prob[:, 1])

        TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()


        avg_TP = np.append(avg_TP, TP)
        avg_TN = np.append(avg_TN, TN)
        avg_FP = np.append(avg_FP, FP)
        avg_FN = np.append(avg_FN, FN)

        avg_acc = np.append(avg_acc, acc)
        avg_train_acc = np.append(avg_train_acc, train_acc)
        print(TP, FP, FN, TN)
        sen = TP / (TP + FN)
        spec = TN / (TN + FP)

        avg_sen = np.append(avg_sen, sen)
        avg_spec = np.append(avg_spec, spec)
        print('Accuracy:{},AUC:{},F1:{}'.format(acc, auc,f1))
        print('Train Accuracy:{}'.format(train_acc))
        print('Sensitivity:{},Specificity:{}'.format(sen, spec))

    print("Train Accuracy Avg: {}".format(np.mean(avg_train_acc)))
    print("Accuracy Avg: {} ({})".format(np.mean(avg_acc),np.std(avg_acc)))
    m, m_h1, m_h2 = mean_confidence_interval(avg_acc)
    print('Accuracy: {},{},{}'.format(m, m_h1, m_h2))


    print("Sensitivity Avg: {} ({})".format(np.mean(avg_sen),np.std(avg_sen)))
    m, m_h1, m_h2 = mean_confidence_interval(avg_sen)
    print('Sensitivity: {},{},{}'.format(m, m_h1, m_h2))

    print("Specificity Avg: {}({})".format(np.mean(avg_spec),np.std(avg_spec)))
    m, m_h1, m_h2 = mean_confidence_interval(avg_spec)
    print('Specificity: {},{},{}'.format(m , m_h1 , m_h2))

    if method == 'RF' or method == 'SVM':
        return roc_label,roc_pred,roc_prob,clf
    elif method == 'LR':
        return roc_label, roc_pred, roc_prob,clf


