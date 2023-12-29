# Early_Detection_of_Parkinson-s_disease_using_Maching_Learning
A project using machine learning to detect if a person has or will have Parkinson’s disease in future or not so that the person can take necessary precautions and delay its effect.
As we know that Parkinson's disease can not be cured but if it is found in early stages, the effects can be reduced. 
In this project, we use multiple machine learning algorithms to determine if the person might have Parkinson's disease in future using the Gait of the person and the Voice of the person.

We train multiple supervised models like Support Vector Machines, K-Nearest Neighbours and Random Forests using the Gait or Voice dataset extracted from Kaggle.

After training, we test the dataset by letting a subject walk for about 2 minutes while wearing 8 vertical ground reaction force sensors on the sole of each feet.

After recording the values, we build our data by calculating the minimum, maximum, skewness, standard deviation etc. of the force for each the sensors. 

This prepared data is then given to the trained models for testing the Parkinson's disease in the subjects.

The recorded accuracy for SVM has been 90%, for Random Forests has been 87% and for KNN has been 83%.
