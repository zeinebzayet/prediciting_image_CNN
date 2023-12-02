import numpy as np
import pandas as pd
import random
import tqdm as tqdm
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report
import cv2
from keras.models import Sequential
from keras.layers import BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Conv2D

####################
normal = r"CNN_algorithm\Dataset\NORMAL"
covid = r"CNN_algorithm\Dataset\COVID"
pneu = r"CNN_algorithm\Dataset\PNEUMONIA"
###############
print("Number of Images in Each Directory:")
print(f"Normal: {len(os.listdir(normal))}")
print(f"Covid: {len(os.listdir(covid))}")
print(f"Pneumonia: {len(os.listdir(pneu))}")
###############
x = []
y = []
dataset = []
img_size=256

def get_data(directory, dir_name):
    for i in tqdm.tqdm(os.listdir(directory)):
        full_path = os.path.join(directory, i)
        try:
            img = cv2.imread(full_path)
            img = cv2.resize(img, (256, 256))
        except:
            continue
        x.append(img)
        y.append(dir_name)
    return x, y
def pre_process():
    x, y = get_data(normal, "normal")
    print(len(x), len(y))
    x, y = get_data(covid, "covid")
    print(len(x), len(y))
    x, y = get_data(pneu, "pneumonia")
    print(len(x), len(y))
    ################
    x = np.array(x)  # array of images
    y = np.array(y)  # array of labels
    x.shape, y.shape
    ###############
    le = LabelEncoder()
    y = le.fit_transform(y)
    ################
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    ###################
    x_train = np.array(x_train) / 255.0
    x_test = np.array(x_test) / 255.0

    x_train = x_train.reshape(-1, img_size, img_size, 3)
    y_train = np.array(y_train)

    x_test = x_test.reshape(-1, img_size, img_size, 3)
    y_test = np.array(y_test)
    ######################"
    lb = LabelBinarizer()
    y_train_lb = lb.fit_transform(y_train)
    y_test_lb = lb.fit_transform(y_test)
    return x_test, y_test, y_test_lb, x_train, y_train, y_train_lb
#build model
def build_and_train_model(x_train, y_train_lb, x_test, y_test_lb,epochs=8, batch_size=32):
    model = Sequential()
    # Couche de convolution 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Couche de convolution 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Couche de convolution 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Couche de convolution 4
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Aplatir les données pour la couche dense
    model.add(Flatten())
    # Couche dense avec dropout pour éviter le surajustement
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    # Couche de sortie avec activation softmax pour la classification multi-classes
    model.add(Dense(3, activation='softmax'))

    # Compiler le modèle
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Afficher un résumé du modèle
    model.summary()
    checkpoint = ModelCheckpoint("custom_cnn_model.h5", monitor="val_accuracy", verbose=1, save_best_only=True,
                                 save_weights_only=False)
    earlystop = EarlyStopping(monitor="val_accuracy", patience=5, verbose=1)
    # Entraînement du modèle
    model.fit(x_train, y_train_lb, epochs=epochs, validation_data=(x_test, y_test_lb), batch_size=batch_size,
                        verbose=1, callbacks=[checkpoint, earlystop])
    return model
# Évaluation du modèle
def performance(model,x_test,y_test_lb):
    loss, accuracy = model.evaluate(x_test, y_test_lb)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")
# Prédictions
def predict(model,x_test,y_test):
    predict_x = model.predict(x_test)
    y_pred = np.argmax(predict_x, axis=1)
    print(classification_report(y_test, y_pred))

def classify_an_image(image_path, model):
    class_names = ['covid', 'normal', 'pneumonia']
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    img = np.reshape(img, [1, 256, 256, 3]) / 255.0
    class_probabilities = model.predict(img)
    predicted_class_id = np.argmax(class_probabilities)
    predicted_class = class_names[predicted_class_id]
    return predicted_class

#x_test, y_test, y_test_lb, x_train, y_train, y_train_lb=pre_process()
#model=build_and_train_model(x_train, y_train_lb, x_test, y_test_lb)
#performance(model,x_test,y_test_lb)
#predict(model,x_test,y_test)
#predicted_class = classify_an_image('./img.png', model)
#print("Predicted Class:", predicted_class)