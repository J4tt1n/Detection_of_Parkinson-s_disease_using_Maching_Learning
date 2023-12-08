import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.svm import SVC  

#for Gait dataset
dfread= pd.read_csv('GaitDataset.csv')
df = pd.DataFrame(dfread)
X = df.drop(['Column1', 'Time(sec)Min', 'VGRF_left_s1Min', 'VGRF_left_s2Min', 'VGRF_left_s3Min', 'VGRF_left_s4Min', 'VGRF_left_s5Min', 'VGRF_left_s6Min', 'VGRF_left_s7Min', 'VGRF_left_s8Min', 'VGRF_right_s1Min', 'VGRF_right_s2Min', 'VGRF_right_s3Min', 'VGRF_right_s4Min', 'VGRF_right_s5Min', 'VGRF_right_s6Min', 'VGRF_left_s7Min', 'VGRF_right_s8Min','Total_force_leftMin', 'Status'], axis = 'columns')
Y = df.Status

#Voice dataset
dfread= pd.read_csv('UCI_Dataset_on_Voice.csv')
df = pd.DataFrame(dfread)
X = df.drop(['name', 'status'], axis = 'columns')
Y = df.status

accuracy =[]
c_value =[]
acc_sum=0

for i in range(1,100):
    X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size=0.2)
    model=SVC(C=i,gamma=i)
    model.fit(X_train, Y_train)
    c_value.append(i)
    accuracy.append(model.score(X_test, Y_test))
    acc_sum = acc_sum + model.score(X_test, Y_test)
plt.xlabel("C and gamma variation")
plt.ylabel("Accuracy")
plt.plot(c_value, accuracy)
print("Accuracy: ", acc_sum/100)

#Gait prediction
dfpredict = pd.read_csv('Testing_values_gait.csv')
dfprediction = pd.DataFrame(dfpredict)
dfprediction = dfprediction.drop(['Column1', 'Time(sec)Min', 'VGRF_left_s1Min', 'VGRF_left_s2Min', 'VGRF_left_s3Min', 'VGRF_left_s4Min', 'VGRF_left_s5Min', 'VGRF_left_s6Min', 'VGRF_left_s7Min', 'VGRF_left_s8Min', 'VGRF_right_s1Min', 'VGRF_right_s2Min', 'VGRF_right_s3Min', 'VGRF_right_s4Min', 'VGRF_right_s5Min', 'VGRF_right_s6Min', 'VGRF_left_s7Min', 'VGRF_right_s8Min','Total_force_leftMin'], axis = 'columns')


#Voice prediction
dfpredict = pd.read_csv('Testing_values_voice.csv')
dfprediction = pd.DataFrame(dfpredict)
dfprediction = dfprediction.drop(['name'], axis = 'columns')

print("Prediction: ",model.predict(dfprediction))


