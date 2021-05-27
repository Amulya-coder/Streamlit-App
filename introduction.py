import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

# set title
st.title("Our First Streamlit App")

from PIL import Image

st.subheader('This is a subheader')

image=Image.open("background.jpg")

st.image(image,use_column_width=True)

st.success("Congrat you run the app Successfully")
st.info("this is an information")

st.warning("Be cautious")
st.error("You run into an error")


import numpy as np
import pandas as pd

dataframe=np.random.rand(10,20)

st.dataframe(dataframe)

st.text("---"*100)

df=pd.DataFrame(np.random.rand(10,20),columns=('col %d' % i for i in range(20)))
st.dataframe(df.style.highlight_max(axis=1))

st.text("---"*100)

#Display Charts
chart_data=pd.DataFrame(np.random.randn(20,3), columns=['a','b','c'])

st.line_chart(chart_data)

st.text("---"*100)

st.area_chart(chart_data)


chart_data=pd.DataFrame(np.random.randn(50,3), columns=['a','b','c'])

st.bar_chart(chart_data)

#Adding histogram

import matplotlib.pyplot as plt

arr=np.random.normal(1,1,size=100)
plt.hist(arr,bins=20)

st.pyplot()

st.text("---"*100)

import plotly
import plotly.figure_factory as ff


#Adding distplot

x1=np.random.randn(200)-2
x2=np.random.randn(200)
x3=np.random.randn(200)-2

hist_data=[x1,x2,x3]
group_labels=['Group1','Group2','Group3']

fig=ff.create_distplot(hist_data,group_labels,bin_size=[.2,.25,.5])

st.plotly_chart(fig,use_container_width=True)

st.text("---"*100)

#Creating a map

df=pd.DataFrame(np.random.randn(100,2)/[50,50]+[37.76,-122.4],columns=['lat','lon'])

st.map(df)

st.text("---"*100)

#Creating buttons

if st.button("Say hello"):
	st.write('hello')
else:
	st.write('why helloe')

st.text("---"*100)

genre=st.radio("What was your favourite genre?",('Comedy','Drama','Documentary'))

st.text("---"*100)

age=st.slider('How old are you?',0,10,20)
st.write("Your age is",age)	

st.text("---"*100)

values=st.slider('Select a range of values',0,200,(15,80))
st.write('You selected a range of between:',values)

st.text("---"*100)

number=st.number_input('Input number')
st.write('The number you inputed is:',number)

st.text("---"*100)
st.text("---"*100)

#File Uploader

upload_file=st.file_uploader("Choose a csv file",type='csv')

if upload_file is not None:
	data=pd.read_csv(upload_file)
	st.write(data)
	st.success("successfully uploaded")
else:
	st.markdown("Please upload a CSV file")


Color picker

color=st.beta_color_picker("Choose your preferred color",'#00f900')
st.write('This your color',color)

#Side bar

st.text("---"*100)
st.text("---"*100)

add_sidebar=st.sidebar.selectbox("What is your favourite course?",("A course from TDS on building Data web app","Others"))

import time

my_bar=st.progress(0)
for percent_complete in range(100):
	time.sleep(0.1)
	my_bar.progress(percent_complete+1)

with st.spinner('wait for it...'):
	time.sleep(5)
st.success('successfully')

st.balloons()

st.text("---"*100)
st.text("---"*100)





