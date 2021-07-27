import pandas as pd
from numpy import mean
from numpy import std
from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score



if __name__ == '__main__':
    df = pd.read_csv('\data\adni\...\input\Merged_clean1_310.csv')
    
    features = ['lHip','Entorhinal.bl','Hippocampus.bl','lMidTemGy','lInfTemGy',
    'lAmy','lFusGy','lSupTemGy','rAmy','lAcc','rPosCinGy','lLatVen','rHip','lAngGy',
    'rTemPo','lMidOccGy','MidTemp.bl','lEnt','rPrcGy','rInfTemGy','lTemPo','lCun',
    'rLinGy','lSupMarGy','AGE','lOccPo','lFroPo','rMidTemGy','rSupParLo','rSupMarGy',
    'rAngGy','lBasCbr+FobBr','rFroPo','lCau','rCbrWM','r3thVen','rCal+Cbr','lTem','rEnt','rCun']
    df_CA = df[(df['Classifier']=='AD') | (df['Classifier'] == 'CN')]
    '''le = LabelEncoder()
    le.fit(df_CA.loc[:,'PTGENDER'])
    df_CA.loc[:, 'PTGENDER'] = le.transform(df_CA.loc[:, 'PTGENDER'])'''
    df_f = df_CA[features].values
    df_t = df_CA.Classifier.values

    # Random Forest Models
    rf_model = RandomForestClassifier(n_estimators=250,random_state=10)
    scores_rf = cross_val_score(rf_model, df_f, df_t, cv=5, scoring = 'accuracy')
    print(f'score for {5} folds:', scores_rf)
    print('RF_GINI_mean', mean(scores_rf))
    print('RF_GINI_std', std(scores_rf))
    
    rf_model_en = RandomForestClassifier(n_estimators=250,random_state=10, criterion='entropy')
    scores_rf_en = cross_val_score(rf_model_en, df_f, df_t, cv=5, scoring = 'accuracy')
    print(f'score for {5} folds:', scores_rf_en)
    print('RF_Entropy_mean', mean(scores_rf_en))
    print('RF_Entropy_std', std(scores_rf_en))

    # XGBoost Classifier Models
    xg_model = XGBClassifier(n_estimators=200, learning_rate=0.5)
    scores_xg = cross_val_score(xg_model, df_f, df_t, cv=5, scoring = 'accuracy')
    print(f'score for {5} folds:', scores_xg)
    print('XG_Model_mean', mean(scores_xg))
    print('XG_Model_std', std(scores_xg))

    # Support Vector Machine Models
    svm_model = svm.SVC(kernel='rbf', C=30, gamma='auto')
    scores_svm = cross_val_score(svm_model, df_f, df_t, cv=5, scoring = 'accuracy')
    print(f'score for {5} folds:', scores_svm)
    print('SVM_Model_mean', mean(scores_svm))
    print('SVM_Model_std', std(scores_svm))

    # KNeighbour Classifier Models
    knn_model = KNeighborsClassifier()
    scores_knn = cross_val_score(knn_model, df_f, df_t, cv=5, scoring = 'accuracy')
    print(f'score for {5} folds:', scores_knn)
    print('KNN_Model_mean', mean(scores_knn))
    print('KNN_Model_std', std(scores_knn))

    # Stacking Models
    level0 = list()
    level0.append(('rf_model', RandomForestClassifier(n_estimators=250,random_state=10)))
    level0.append(('rf_model_en', RandomForestClassifier(n_estimators=250,random_state=10, criterion='entropy')))
    level0.append(('xg_model', XGBClassifier(n_estimators=200, learning_rate=0.5)))
    level0.append(('svm_model', svm.SVC(kernel='rbf', C=30, gamma='auto')))
    level1 = KNeighborsClassifier()
    model = StackingClassifier(estimators=level0, final_estimator=level1)
    scores_model = cross_val_score(model, df_f, df_t, cv=5, scoring = 'accuracy')
    print(f'score for {5} folds:', scores_model)
    print('Ensemble_Model_mean', mean(scores_model))
    print('Ensemble_Model_std', std(scores_model))

    
