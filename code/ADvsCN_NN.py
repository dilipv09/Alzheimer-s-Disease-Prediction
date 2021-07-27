import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    df = pd.read_csv('\data\adni\...\input\Merged_clean1_310.csv')
    features = ['lHip','Entorhinal.bl','Hippocampus.bl','lMidTemGy','lInfTemGy',
    'lAmy','lFusGy','lSupTemGy','rAmy','lAcc','rPosCinGy','lLatVen','rHip','lAngGy',
    'rTemPo','lMidOccGy','MidTemp.bl','lEnt','rPrcGy','rInfTemGy','lTemPo','lCun',
    'rLinGy','lSupMarGy','AGE','lOccPo','lFroPo','rMidTemGy','rSupParLo','rSupMarGy',
    'rAngGy','lBasCbr+FobBr','rFroPo','lCau','rCbrWM','r3thVen','rCal+Cbr','lTem','rEnt','rCun']
    df_CA = df[(df['Classifier']=='AD') | (df['Classifier'] == 'CN')]
    df_CA['Classifier'] = df_CA['Classifier'].map({'AD':1, 'CN':0})
    df_f = df_CA[features]
    df_f = StandardScaler().fit_transform(df_f)
    df_t = df_CA.Classifier
    X_train, X_test, y_train, y_test = train_test_split(df_f, df_t, test_size=0.2, random_state=1)

    model_multi_t = Sequential()
    model_multi_t.add(Dense(units=32, activation='relu', kernel_initializer = 'he_uniform', input_dim=40))
    model_multi_t.add(Dense(units=16, activation='relu'))
    model_multi_t.add(Dense(units=8, activation='relu'))
    model_multi_t.add(Dense(units=1, activation='sigmoid'))
    #opt_t = SGD(lr=0.01, momentum=0.9)
    model_multi_t.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    model_multi_t.fit(X_train, 
                y_train,
               epochs=50,
               validation_data=(X_test, y_test))



    
