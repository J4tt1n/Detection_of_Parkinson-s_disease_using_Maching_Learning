import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost
from xgboost import XGBClassifier

#for Gait dataset
#dfread= pd.read_csv('GaitDataset.csv')
#df = pd.DataFrame(dfread)
#X = df.drop(['Column1', 'Time(sec)Min', 'VGRF_left_s1Min', 'VGRF_left_s2Min', 'VGRF_left_s3Min', 'VGRF_left_s4Min', 'VGRF_left_s5Min', 'VGRF_left_s6Min', 'VGRF_left_s7Min', 'VGRF_left_s8Min', 'VGRF_right_s1Min', 'VGRF_right_s2Min', 'VGRF_right_s3Min', 'VGRF_right_s4Min', 'VGRF_right_s5Min', 'VGRF_right_s6Min', 'VGRF_left_s7Min', 'VGRF_right_s8Min','Total_force_leftMin', 'Status'], axis = 'columns')
#Y = df.Status

#Voice Dataset
dfread= pd.read_csv('UCI_Dataset_on_Voice.csv')
df = pd.DataFrame(dfread)
X = df.drop(['name', 'status'], axis = 'columns')
Y = df.status

accuracy =[]
i_value =[]
acc_avg=0

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)
#print(X_scaled)
for i in range(1, 101):   
    X_train, X_test, Y_train, Y_test  = train_test_split(X_scaled, Y, test_size=0.2)
    xg_reg = xgboost.XGBClassifier(eval_metric = 'mlogloss', use_label_encoder=False)
    xg_reg.fit(X_train, Y_train)
    accuracy.append(xg_reg.score(X_test, Y_test))
    acc_avg = acc_avg + xg_reg.score(X_test, Y_test)
    i_value.append(i)
plt.plot(i_value, accuracy)
print("Accuracy: ", acc_avg/100)

#Gait prediction
#dfpredict = pd.read_csv('Testing_values_gait.csv')
#dfprediction = pd.DataFrame(dfpredict)
#dfprediction = dfprediction.drop(['Column1', 'Time(sec)Min', 'VGRF_left_s1Min', 'VGRF_left_s2Min', 'VGRF_left_s3Min', 'VGRF_left_s4Min', 'VGRF_left_s5Min', 'VGRF_left_s6Min', 'VGRF_left_s7Min', 'VGRF_left_s8Min', 'VGRF_right_s1Min', 'VGRF_right_s2Min', 'VGRF_right_s3Min', 'VGRF_right_s4Min', 'VGRF_right_s5Min', 'VGRF_right_s6Min', 'VGRF_left_s7Min', 'VGRF_right_s8Min','Total_force_leftMin'], axis = 'columns')

#Voice prediction
#dfpredict = pd.read_csv('Testing_values_voice.csv')
#dfprediction = pd.DataFrame(dfpredict)
#dfprediction = dfprediction.drop(['name'], axis = 'columns')
#print(xg_reg.predict(dfprediction))


#implementing using Bagging along with XGBoost
#bag_i=[]
#bag_acc=[]
#bag_accu=0
#from sklearn.ensemble import BaggingClassifier
#for i in range(1, 31):
#    bag_model = BaggingClassifier(base_estimator= xgb.XGBClassifier(eval_metric = 'mlogloss', use_label_encoder=False), n_estimators=30, oob_score = True)
#    bag_model.fit(X_train, Y_train)
#    bag_i.append(i)
#    score = bag_model.score(X_test, Y_test)
#    bag_acc.append(score)
#    bag_accu = bag_accu + score
    #print(score)
    
#plt.plot(bag_i, bag_acc, color="red")
#print("After baggin: ", bag_accu/30)

    
    


