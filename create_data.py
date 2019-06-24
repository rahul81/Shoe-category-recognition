# importing required libraries

import numpy as np 
import pandas as pd 
import os
import cv2
from sklearn.utils import shuffle


# path to downloaded images folder
datadir = ".../data/View_1"

# importing the csv file into pandas dataframe

df = pd.read_csv(".../I_data.csv")


# selecting required columns and removing null values

id_class = df[['id','view_2','class']]
id_class=id_class.dropna()

ID = id_class['id']
label = id_class['class']
length = len(id_class['class'])
print(length)



#assigning integer values to all unique categories

label=label.map({'zipper':'0','backstrap':1,'slip_on':2,'lace_up':3,'buckle':4,'hook&look':5})

label=np.array(label,dtype=int)


labels=[]


count=0
c=0

# creating a 3D numpy array to store the images

data = np.zeros(shape=(0,0,0))
for i in ID:

	image_path = os.path.join(datadir,i)
	if os.path.exists(image_path):
		img = cv2.imread(os.path.join(datadir,i),cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img,(128,128))
		data = np.append(data,img)
		labels.append(label[c])
		c=c+1
		count=count+1
	else:
		print(i) 
		c=c+1



len_data= count
data = data.reshape(len_data,128,128) 

print(data.shape)

lab_count = len(labels)

print(lab_count)

# Converting the integer labels to one hot encoded arrays for training the CNN

labels = np.array(labels,dtype=int)
temp= np.zeros((lab_count,6))
temp[np.arange(lab_count),labels]=1
labels = temp
labels = np.array(labels,dtype=int)




x,y = shuffle(data,labels)

print(x)
print(x.shape)
print(y)
print(y.shape)


# saving the numpy files with image and label data

np.save('.../filename.npy',x)
np.save('.../filename.npy',y)


# To load
# data = np.load('filename.npy')






