# Create current Run context to deliver logging
from azureml.core import Run, Dataset
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.compute_target import ComputeTargetException
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.train.estimator import Estimator
import azureml.core
import numpy as np
from keras import utils

# +
import argparse

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument('--prepped-data', type=str, dest='prepped_data', default='prepped_data', help='Folder for results')
args = parser.parse_args()
save_folder = args.prepped_data

# +
from azureml.core.authentication import ServicePrincipalAuthentication

sp = ServicePrincipalAuthentication(tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47", # tenantID
                                    service_principal_id="171de499-32e5-414a-bc2b-5f037676b2ff", # clientId
                                    service_principal_password="k-e9To~BUNKKziSgj7MWh8eRdsmHjYihaw") # clientSecret

# +
run = Run.get_context()

# check core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)

# Log into a workspace
ws = Workspace.get(name="cloudnativeml", auth=sp, subscription_id="a2a1fc9f-5671-4479-8922-ad16e34c0fdc")
print("Using workspace:",ws.name,"in region", ws.location)


# +
dataset_name = 'fashion_mnist_ds'

# Get a dataset by name
fashion_mnist_raw = Dataset.get_by_name(workspace=ws, name=dataset_name)
fashion_mnist_raw.download(target_path='.', overwrite=True)
(x_train, y_train), (x_test, y_test) = np.load('fashion-mnist-raw.npy', allow_pickle=True)

print(x_train.shape, 'train set')
print(x_test.shape, 'test set')

run.log('training-set-dim', x_train.shape)
run.log('test-set-dim', x_test.shape)
# -

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

import matplotlib.pyplot as plt
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

# +
#no. of classes
num_classes = 10

#one-hot encoding
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test,  num_classes)

# +
from keras import backend as K

#input image dimensions
img_rows,img_cols = 28,28

#formatting issues for depth of image (greyscale = 1) with different kernels (tensorflow, cntk, etc)
if K.image_data_format()== 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0],1,img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,1)
    x_test = x_test.reshape(x_test.shape[0],img_rows, img_cols,1)
    input_shape = (img_rows, img_cols,1)

# +
# Save the prepped data
print("Saving Data...")
os.makedirs(save_folder, exist_ok=True)

np.save(os.path.join(save_folder,'x_train.npy'),x_train)
np.save(os.path.join(save_folder,'y_train.npy'),y_train)
np.save(os.path.join(save_folder,'x_test.npy'),x_test)
np.save(os.path.join(save_folder,'y_test.npy'),y_test)
# -

run.complete()
