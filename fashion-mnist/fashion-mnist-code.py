# Import packages for CNN model code
import tensorflow as tf
import os
import time
from sklearn.metrics import confusion_matrix
os.environ["TF_CPP_MIN_LOG_LEVEL"]= "2"
print("tensorflow Version is: " + str(tf.__version__))

import numpy as np
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K
print(os.environ['KERAS_BACKEND'])


#Fashion MNIST Dataset CNN model development: https://github.com/zalandoresearch/fashion-mnist
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import utils, losses, optimizers
import matplotlib.pyplot as plt

# Create current Run context to deliver logging
from azureml.core import Run
run = Run.get_context()

#no. of classes
num_classes = 10

# batch size and training iterations (epochs)
batch_size = 128
epochs = 10

#input image dimensions
img_rows,img_cols = 28,28


#data for train and testing
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, 'train set')
print(x_test.shape, 'test set')

# Define the text labels
fashion_mnist_labels = ["Top",          # index 0
                        "Trouser",      # index 1
                        "Jumper",       # index 2 
                        "Dress",        # index 3 
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6 
                        "Trainer",      # index 7 
                        "Bag",          # index 8 
                        "Ankle boot"]   # index 9

# Show sample image
img_index=100
label_index = y_train[img_index]
plt.figure(0)
plt.imshow(x_train[img_index])
print('Label Index: ' + str(label_index) + " Fashion Labels: " + (fashion_mnist_labels[label_index]))
plt.savefig('sampleimage.png')
run.log_image(name='Sample Data', plot=plt)
plt.close(0)

#type convert and scale the test and training data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#one-hot encoding
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test,  num_classes)

#formatting issues for depth of image (greyscale = 1) with different kernels (tensorflow, cntk, etc)
if K.image_data_format()== 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0],1,img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,1)
    x_test = x_test.reshape(x_test.shape[0],img_rows, img_cols,1)
    input_shape = (img_rows, img_cols,1)


# Create Keras CNN model architecture
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()


#compile - how to measure loss
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#train the model and return loss and accuracy for each epoch - history dictionary
start = time.time()
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
end = time.time()

#evaluate the model on the test data
score = model.evaluate(x_test, y_test, verbose=0)
print('Test Loss: ', score[0])
print('Test Accuracy: ', score[1])
print('Time to run: ', (end-start))

model.save('fmm.h5')

# Log loss, accuracy and time and send back to AML experiment
run.log('Test Loss: ', score[0])
run.log('Test Accuracy: ', score[1])
run.log('Time to run: ', (end-start))


# Create Training Vs Validation Accuracy graph
epoch_list = list(range(1, len(hist.history['accuracy']) + 1))
plt.figure(1)
plt.plot(epoch_list, hist.history['accuracy'], epoch_list, hist.history['val_accuracy'])
plt.legend(('Training Accuracy', "Validation Accuracy"))
plt.show()
plt.savefig('TrainingVsValidationAccuracy.png')
run.log_image(name='TrainingVsValidationAccuracy', plot=plt)
plt.close(1)

# Create Training Vs Validation Loss graph
epoch_list = list(range(1, len(hist.history['loss']) + 1))
plt.figure(2)
plt.plot(epoch_list, hist.history['loss'], epoch_list, hist.history['val_loss'])
plt.legend(('Training Loss', "Validation Loss"))
plt.show()
plt.savefig('TrainingVsValidationLoss.png')
run.log_image(name='TrainingVsValidationLoss', plot=plt)
plt.close(2)

# Create predictions on test data
predictions = model.predict(x_test)

# Plot a random sample of 10 test images, their predicted labels and ground truth
plt.figure(3)
figure = plt.figure(figsize=(20, 8))
for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):
    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    # Display each image
    ax.imshow(np.squeeze(x_test[index]))
    predict_index = np.argmax(predictions[index])
    true_index = np.argmax(y_test[index])
    # Set the title for each image
    ax.set_title("{} ({})".format(fashion_mnist_labels[predict_index], 
                                  fashion_mnist_labels[true_index]),
                                  color=("green" if predict_index == true_index else "red"))

# Output image for test dataset
plt.savefig('TestData.png')
run.log_image(name='TestData', plot=plt)
plt.close(3)

#Correlation Matrix
y_pred = model.predict(x_test)
print(y_pred.shape)
y_prediction = np.zeros(shape = (10000,10))

count = 0
for x in y_pred:
    index = np.argmax(x)
    y_prediction[count, index] = 1
    count = count + 1

# Create confusion matrix
print(fashion_mnist_labels)
cm = confusion_matrix(y_test.argmax(axis=1), y_prediction.argmax(axis=1))
print(cm)

plt.figure(4)
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)
res = ax.imshow(cm, interpolation='nearest')
plt.savefig('CorrelationMatrix.png')
run.log_image(name='CorrelationMatrix', plot=plt)
plt.close(4)
