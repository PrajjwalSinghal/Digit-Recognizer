import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import time
import os
from sklearn.preprocessing import OneHotEncoder
# To Change the Backend
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import History
import pickle
from keras.utils import to_categorical
# If using Keras Backend
#os.environ['KMP_DUPLICATE_LIB_OK']='True'


def reshapeAllImages(Dataset):
    
    if 'label' in Dataset.columns:
        y = Dataset['label']
        x = Dataset.drop('label', axis = 1)
    else:
        x = Dataset
    imagelist = []
    for index, row in x.iterrows():
        temp = np.array(row)
        temp = np.reshape(temp, (28,28))
        temp = np.array(temp, dtype = np.uint8)
        imagelist.append(temp)
        #plt.imshow(temp, cmap = 'gray')
        #plt.show()
    if 'label' in Dataset.columns:
        return np.array(imagelist), list(y)
    else:
        return np.array(imagelist)
'''   
def saveToDirectory(Images, Labels, parent_directory):
    if len(Images) != len(Labels):
        return 1
    CountList = [0]*10
    num_to_str = {
            0 : "zero",
            1 : "one",
            2 : "two",
            3 : "three",
            4 : "four",
            5 : "five",
            6 : "six",
            7 : "seven",
            8 : "eight",
            9 : "nine"}
    for i in range(0, len(Labels)):
        path = parent_directory + "/" + num_to_str[Labels[i]] + "/" + num_to_str[Labels[i]] + str(CountList[Labels[i]]) + ".png"
        CountList[Labels[i]] = CountList[Labels[i]] + 1
        img = Image.fromarray(Images[i])
        img.save(path)
        #print(Labels[i])
        #plt.imshow(Images[i], cmap = 'gray')
        #plt.show()
    return 0
'''
def trainCNNModel(X_train, y_train, X_test, y_test):
    # Initializing the network
    classifier = Sequential()
    
    #Step 1 - Adding a Convolutional Layer
    classifier.add(Conv2D(64, (3, 3) ,input_shape = (28,28, 1), activation = 'relu'))
    
    #Pooling the first Layer
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    
    #Step 2 : Adding another Convolutional Layer
    classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
    
    # Pooling the second layer
    #classifier.add(MaxPooling2D(pool_size = (2,2)))
    
    # Step 3 - Flatting the pixels
    classifier.add(Flatten())
    
    # Adding a ANN and making it Fully connected
    classifier.add(Dense(output_dim = 16, activation = 'relu'))
    
    # Adding a ANN and making it Fully connected
    classifier.add(Dense(output_dim = 16, activation = 'relu'))
    
    # Adding the output layer
    classifier.add(Dense(output_dim = 10, activation = 'softmax'))
    
    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    # Part 2 - Fitting the CNN 
    h = classifier.fit(X_train, y_train,validation_data = (X_test, y_test) ,epochs = 25)
    return classifier, h
    
def main():
    start = time.clock()
    dataset = pd.read_csv('train.csv')
    trainingDataset, testDataset = train_test_split(dataset, test_size = 0.20, random_state = 0)
    X_train, y_train = reshapeAllImages(trainingDataset)
    X_test, y_test = reshapeAllImages(testDataset)
    X_train = X_train.reshape(len(X_train), 28, 28, 1)
    X_test = X_test.reshape(len(X_test), 28, 28, 1)
    # One hot encoding the target column
    encoder = OneHotEncoder()
    y_train = encoder.fit_transform(np.array(y_train).reshape(-1,1)).toarray()
    y_test = encoder.fit_transform(np.array(y_test).reshape(-1,1)).toarray()
    y_train = np.array(y_train, dtype = np.uint8)
    y_test = np.array(y_test, dtype = np.uint8)
    # saveToDirectory(trainingImages, trainingLabels, "trainingImages")
    # saveToDirectory(testingImages, testingLabels, "testImages")
    classifier, h = trainCNNModel(X_train, y_train, X_test, y_test)
    h.history['val_acc']
    classifier.save('my_model.h5')
    with open('trainHistoryDict.pkl', 'wb') as file_pi:
        pickle.dump(h.history, file_pi)
    print(time.clock() - start)
    
    # PREDICTING THE RESULTS FROM TEST DATASET
    test_dataset = pd.read_csv('test.csv')
    test_dataset = reshapeAllImages(test_dataset)
    y_pred = classifier.predict(test_dataset.reshape(len(test_dataset), 28, 28, 1))
    labels = [i for i in range(1, len(test_dataset)+1)]
    submission = pd.DataFrame()
    submission['ImageId'] = labels
    y_pred = encoder.inverse_transform(y_pred)
    submission['Label'] = y_pred
    submission.to_csv('my_submission.csv')
    
main()