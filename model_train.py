
from pycocotools.coco import COCO
import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

import cv2
import skimage
from skimage.feature import hog

from sklearn.model_selection import train_test_split

import os
import gzip
import pickle
from tqdm.auto import tqdm
from ast import literal_eval
from sklearn.neural_network import MLPClassifier

import wandb

wandb.init(project="household_classifier", entity="gwilson")

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
images = np.load("image.npy")
image_labels = np.load("image_category.npy")
labels = np.unique(image_labels)

def extract_HOG_features(data):
    num_samples = data.shape[0]
    hog_features = []
    for i in tqdm(range(num_samples)):
        img = data[i]
        feature = hog(img, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(3, 3))
        hog_features.append(feature)
        if i %20 == 0:
            print()
    return np.array(hog_features)

test_data = images[int(images.shape[0] * .75):]
train_data = images[:int(images.shape[0] * .75)]

test_labels = image_labels[int(image_labels.shape[0] * .75):]
train_labels = image_labels[:int(image_labels.shape[0] * .75)]

train_features = extract_HOG_features(train_data)
test_features = extract_HOG_features(test_data)

clf = MLPClassifier(solver='adam',
                    activation='relu',
                    alpha=.01,
                    hidden_layer_sizes=(512, 128, 5),
                    random_state=1,
                    max_iter=100,
                    verbose=True)

clf.fit(train_features, train_labels)

pred = clf.predict(train_features)
train_accuracy = np.mean(pred == train_labels)

pred = clf.predict(test_features)
proba = clf.predict_proba(test_features)
test_accuracy = np.mean(pred == test_labels)

print("Training accuracy: {}".format(train_accuracy))
print("Testing accuracy: {}".format(test_accuracy))

wandb.sklearn.plot_classifier(clf, train_data,
 test_data,
 train_labels,
  test_labels,
   pred,
    proba,
     labels,
      model_name="SVC", feature_names=None)
