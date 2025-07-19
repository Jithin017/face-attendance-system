import tensorflow as tf, os, numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from PIL import Image

def load_data(path='dataset'):
    X, y, labels = [], [], {}
    lid = 0
    for user in os.listdir(path):
        labels[lid] = user
        for img in os.listdir(f"{path}/{user}"):
            img_path = os.path.join(path, user, img)
            img_arr = Image.open(img_path).resize((100,100)).convert('L')
            X.append(np.array(img_arr)/255.0)
            y.append(lid)
        lid += 1
    return np.expand_dims(np.array(X), -1), np.array(y), labels

def train():
    X, y, labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(100,100,1)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(set(y)), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    model.save('pretrained_model/face_recognition_model.h5')
    with open('pretrained_model/labels.txt', 'w') as f:
        for k, v in labels.items():
            f.write(f"{k},{v}\n")

if __name__ == '__main__':
    train()

