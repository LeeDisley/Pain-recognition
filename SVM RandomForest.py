# Code from Paul Van Gent available at http://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/ 

import cv2
import os
import glob
import random
import math
import numpy as np
import dlib
import itertools
#from sklearn import svm
from sklearn.ensemble import RandomForestRegressor

emotions = ['anger', 'disgust', 'happy', 'neutral', 'surprise', 'pain'] #emotion list
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') #landmarks data file
#clf = svm.SVC (kernel='linear', probability=True, tol=1e-3) # set the classifier as a support vector machine with polynomial kernel
clf = RandomForestRegressor (n_estimators=100, random_state=0)
data = {} #dictionary for all values
#data['landmarks_vectorised'] = []

def get_files(emotions): # function to get file list, shuffle and split 80/20
        files = glob.glob ('dataset//%s//*' %emotions)
        random.shuffle(files)   
        training = files[:int(len(files)*0.8)] #first 80% of file list
        prediction = files[-int(len(files)*0.2):] #last 20% of file list
        return training, prediction

def get_landmarks(image):
        detections = detector(image, 1)
        for k,d in enumerate(detections): #for all detected faces
                shape = predictor(image, d) #draw facial landmarks with predictor class
                xlist = []
                ylist = []
                for i in range (0,68): #store x and y corords in two list
                        xlist.append(float(shape.part(i).x))
                        ylist.append(float(shape.part(i).y))
                xmean = np.mean(xlist)
                ymean = np.mean(ylist)
                xcentral = [(x-xmean) for x in xlist]
                ycentral = [(y-ymean) for y in ylist]
                landmarks_vectorised=[]
                for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
                        landmarks_vectorised.append(w)
                        landmarks_vectorised.append(z)
                        meanp = np.asarray((ymean, xmean))
                        coornp = np.asarray((z,w))
                        dist = np.linalg.norm(coornp-meanp)
                        landmarks_vectorised.append(dist)
                        landmarks_vectorised.append((math.atan2(y, x)*360)/(math.pi*2))
                data['landmarks_vectorised'] = landmarks_vectorised
        if len(detections) < 1:
                data ['landmarks_vectorised'] = 'error'
                
def make_sets():
        training_data=[]
        training_labels=[]
        prediction_data=[]
        prediction_labels=[]
        for emotion in emotions:
                print(' Working on %s' %emotion)
                training, prediction = get_files(emotion)
                
                for item in training:
                        image = cv2.imread(item) #open image
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayyscale
                        clahe_image = clahe.apply(gray)
                        get_landmarks(clahe_image)
                        if data['landmarks_vectorised'] == 'error':
                                print('No Face Detected In This One')
                        else:
                                training_data.append(data['landmarks_vectorised'])
                                training_labels.append(emotions.index(emotion))
                        #training_data.append(data['landmarks_vectorised'])
                        #training_labels.append(emotions.index(emotion))
                                
                for item in prediction:
                        image = cv2.imread(item)
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        clahe_image = clahe.apply(gray)
                        get_landmarks(clahe_image)
                        if data['landmarks_vectorised']=='error':
                                print('No Face Detected On This One')
                        else:
                                prediction_data.append(data['landmarks_vectorised'])
                                prediction_labels.append(emotions.index(emotion))
                                
                        #prediction_data.append(data['landmarks_vectorised'])
                        #prediction_labels.append(emotions.index(emotion))
                                
        return training_data, training_labels, prediction_data, prediction_labels
        
accur_lin = []
for i in range (0,20):
                print ('Making sets %s' %i)
                training_data, training_labels, prediction_data, prediction_labels = make_sets()
                npar_train = np.array(training_data) #Turn training data into numpy array for training
                npar_trainlabs = np.array(training_labels)
                print('Training SVM linear %s' %i) #train SVM
                clf.fit(npar_train, npar_trainlabs)
                print ('Getting accuracies %s' %i)
                npar_pred = np.array(prediction_data)
                pred_lin = clf.score(npar_pred, prediction_labels)
                
                accur_lin.append(pred_lin) #store accuracy in a list
result = np.mean(accur_lin)
result2 = result * 100
print('mean value in linear svm: %s' %result2)       
