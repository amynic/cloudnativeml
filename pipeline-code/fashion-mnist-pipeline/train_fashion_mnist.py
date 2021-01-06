# +
# Import packages for CNN model code
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
import argparse
from sklearn.metrics import confusion_matrix
os.environ["TF_CPP_MIN_LOG_LEVEL"]= "2"
print("tensorflow Version is: " + str(tf.__version__))

import numpy as np
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K
print(os.environ['KERAS_BACKEND'])

from azureml.core import Run, Dataset, Model
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.compute_target import ComputeTargetException
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.train.estimator import Estimator
import azureml.core
# -

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--training-folder", type=str, dest='training_folder', help='training data folder')
args = parser.parse_args()
training_folder = args.training_folder

# +
# Get the experiment run context
run = Run.get_context()

# load the prepared data file in the training folder
print("Loading Data...")
x_train = np.load(os.path.join(training_folder,'x_train.npy'))
y_train = np.load(os.path.join(training_folder,'y_train.npy'))
x_test = np.load(os.path.join(training_folder,'x_test.npy'))
y_test = np.load(os.path.join(training_folder,'y_test.npy'))

print(x_train.shape, 'train set')
print(x_test.shape, 'test set')

run.log('training-set-dim', x_train.shape)
run.log('test-set-dim', x_test.shape)
# -

# batch size and training iterations (epochs)
batch_size = 128
epochs = 10

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

# +
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
# -

#compile - how to measure loss
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#train the model and return loss and accuracy for each epoch - history dictionary
start = time.time()
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
end = time.time()

# +
#evaluate the model on the test data
score = model.evaluate(x_test, y_test, verbose=0)
print('Test Loss: ', score[0])
print('Test Accuracy: ', score[1])
print('Time to run: ', (end-start))

# Log loss, accuracy and time and send back to AML experiment
run.log('Test Loss: ', score[0])
run.log('Test Accuracy: ', score[1])
run.log('Time to run: ', (end-start))
# -

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

# Save the trained model in the outputs folder
print("Saving model...")
os.makedirs('outputs', exist_ok=True)
model.save('outputs/fashion_mnist_model.h5')
model_file = 'outputs/fashion_mnist_model.h5'

# Register the model
print('Registering model...')
Model.register(workspace=run.experiment.workspace,
               model_path = model_file,
               model_name = 'fashion_mnist_model',
               tags={'Training context':'Pipeline'},
               properties={'Test Loss': str(score[0]), 'Test Accuracy': str(score[1])})

run.complete()
