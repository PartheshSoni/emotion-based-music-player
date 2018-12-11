import numpy as np
import glob
import random
import cv2

fishface=cv2.face.FisherFaceRecognizer_create()
data={}

def update(emotions):
    run_recognizer(emotions)
    print("Saving model...")
    fishface.save("model.xml")
    print("Model saved!!")

def make_sets(emotions):
    training_data=[]
    training_label=[]

    for emotion in emotions:
        training=training=sorted(glob.glob("dataset/%s/*" %emotion))
        for item in training:
            image=cv2.imread(item)
            gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            training_data.append(gray)
            training_label.append(emotions.index(emotion))
    return training_data, training_label

def run_recognizer(emotions):
    training_data, training_label=make_sets(emotions)
    print("Training model...")
    print("The size of the dataset is "+str(len(training_data))+" images")
    fishface.train(training_data, np.asarray(training_label))

