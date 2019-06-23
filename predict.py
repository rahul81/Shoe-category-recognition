from tensorflow.python.keras.models import load_model
import numpy as np
from plot_images import plot_images,plot_example_errors,plot_conv_weights
import math
from sklearn.utils import shuffle
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix,accuracy_score
import seaborn as sb
import pandas as pd 
from matplotlib import style
style.use("ggplot")

# loading the trained model
model = load_model('.../filename.model')

# loading the numpy files with image and label data
x = np.load('.../Image_numpy_File.npy')
y = np.load('.../labels_numpy_files.npy')


x,y = shuffle(x,y)

X= x[:].reshape(-1,128,128,1)
x_test = X[:]
y_test = y[:]




img = x[80:89]
cls_true=y_test[80:89]

# plotting images with true classes
plot_images(images=img,cls_true=np.argmax(cls_true,axis=1))




prediction = model.predict(x=x_test)

cls_pred=np.argmax(prediction,axis=1)


test_labs = np.argmax(y_test,axis=1)


# plotting images with predicted labels
plot_images(images=img,cls_true=np.argmax(cls_true,axis=1),cls_pred=cls_pred[80:89])


# plotting misclassified images
plot_example_errors(cls_pred=cls_pred,image=x,y_true=y_test)



# getting weights of first convolutional layer
layer_conv1 = model.layers[0]

weights_conv1 = layer_conv1.get_weights()[0]

# plotting convolutional weights
plot_conv_weights(weights=weights_conv1)


# getting weights of second convolutional layer

layer_conv2  = model.layers[2]

weights_conv2 = layer_conv2.get_weights()[0]

# plotting convolutional weights

plot_conv_weights(weights=weights_conv2)


# Creating a confusion matrix 
cm =confusion_matrix(test_labs,cls_pred)

cm_df=pd.DataFrame(cm)


plt.matshow(cm)
plt.colorbar()
tick_marks = np.arange(6)
plt.xticks(tick_marks, range(6))
plt.yticks(tick_marks, range(6))
plt.title('Classification_Accuracy:{}'.format(accuracy_score(test_labs,cls_pred)))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


