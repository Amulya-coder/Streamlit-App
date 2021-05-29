import streamlit as st
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.set_option('deprecation.showPyplotGlobalUse', False)

from PIL import Image

##Set title
st.title('Total data science')

image=Image.open('DS.jpg')
st.image(image,use_column_width=True)

#sub title

st.write("""### A simple Data App with Streamlit """)


st.write("""### Let's create different classifiers and datasets
	""")


dataset_name=st.sidebar.selectbox('Select dataset',('Breast Cancer','Iris','Wine'))

classifier_name=st.sidebar.selectbox('Select Clasifier',('SVM','KNN'))

def get_dataset(name):
	data=None
	if name=='Iris':
		data=datasets.load_iris()
	elif name=='Wine':
		data=datasets.load_wine()
	else:
		data=datasets.load_breast_cancer()
	x=data.data
	y=data.target

	return x,y
x,y=get_dataset(dataset_name)
st.dataframe(x)
st.write('Shape of your dataset is:',x.shape)
st.write('Unique target variables:',len(np.unique(y)))


# Plotting boxplot
fig=plt.figure()
sns.boxplot(data=x, orient='h')
st.pyplot()


#Plotting Histogram
plt.hist(x)
st.pyplot()



#BUILDING OUR ALGORITHM

def add_parameter(name_of_clf):
	params=dict()
	if name_of_clf=='SVM':
		C=st.sidebar.slider('C',0.01,15.0)
		params['C']=C
	else:
		name_of_clf=='KNN'
		k=st.sidebar.slider('k',1,15)
		params['k']=k
	return params

params=add_parameter(classifier_name)	


# Accessing Our Classifier

def get_classifier(name_of_clf,params):
	clf=None
	if name_of_clf=='SVM':
		clf=SVC(C=params['C'])
	elif name_of_clf=='KNN':
		clf=KNeighborsClassifier(n_neighbors=params['k'])
	else:
		st.warning("You didn't select any option, please select any one algo")
	return clf

clf=get_classifier(classifier_name,params)

		
# Splitting our data into X train and Y train

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=10)

# fit
clf.fit(x_train,y_train) #80% for training

# Predictions
y_pred=clf.predict(x_test) #20% for training


st.write(y_pred)


accuracy=accuracy_score(y_test,y_pred)

st.write('classifier_name:',classifier_name)
st.write('Accuracy for your model is:',accuracy)
























































