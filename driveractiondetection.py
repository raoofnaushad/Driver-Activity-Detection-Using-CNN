from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.preprocessing import image                  
from tqdm import tqdm
from PIL import ImageFile                            
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint  
from keras.models import load_model



# define function to load datasets
def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']), 10)
    return files, targets

# load train, test, and validation datasets
# train_files, train_targets = load_dataset('images/train')
test_files, test_targets = load_dataset('/home/accubits/Desktop/test1')
#valid_files, valid_targets = load_dataset('dogImages/valid')


# # load list of names
# names = [item[17:19] for item in sorted(glob("images/train/*/"))]

# ## Train test split
# train_files, valid_files, train_targets, valid_targets = train_test_split(train_files, train_targets, test_size=0.2, random_state=42)

# # print statistics about the dataset

# print('There are %s total images.\n' % len(np.hstack([train_files, valid_files, test_files])))
# print('There are %d training images.' % len(train_files))
# print('There are %d total training categories.' % len(names))
# print('There are %d validation images.' % len(valid_files))
# print('There are %d test images.'% len(test_files))


# df = pd.read_csv("driver_imgs_list.csv",header='infer')
# print(df['classname'].head(3))
# print(df.iloc[:,1].describe())
# print("\n Image Counts")
# print(df['classname'].value_counts(sort=False))

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
# train_tensors = paths_to_tensor(train_files).astype('float32')/255 - 0.5
# valid_tensors = paths_to_tensor(valid_files).astype('float32')/255 - 0.5
test_tensors = paths_to_tensor(test_files).astype('float32')/255 - 0.5


model = Sequential()

model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(224,224,3), kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=256, kernel_size=2, padding='same', activation='relu', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=512, kernel_size=2, padding='same', activation='relu', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(500, activation='relu', kernel_initializer='glorot_normal'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax', kernel_initializer='glorot_normal'))


# model.summary()

# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# epochs = 30 # epochs used is 30 and batch size is 40

# checkpointer = ModelCheckpoint(filepath='models/weights.best.from_scratch.hdf5', 
#                                verbose=1, save_best_only=True)

#################### Training
# model.fit(train_tensors, train_targets, 
#           validation_data=(valid_tensors, valid_targets),
#           epochs=epochs, batch_size=40, callbacks=[checkpointer], verbose=1)



model.load_weights('models/weights.best.from_scratch.hdf5')

test_files_final = [item_test[15:] for item_test in test_files]

predictions = [model.predict(np.expand_dims(tensor, axis=0))[0] for tensor in test_tensors]

subm = np.column_stack((np.asarray(test_files_final), np.asarray(predictions,dtype=np.float32)))

print(subm[0])

print("Finding the prediction of the file {}". format(subm[0][0]))
max = float("{:.8f}".format(float(subm[0][1])))
for i in range(1,11):
    if float("{:.8f}".format(float(subm[0][i]))) >= max:
        max = float("{:.8f}".format(float(subm[0][i])))
        category = i
print("Maximum value is observed for {} category, and value is {}".format(category, max))

if category == 1:
    print("Safe Driving")
elif category == 2:
    print("texting - right")
elif category == 3:
    print("talking on the phone - right")
elif category == 4:
    print("texting - left")
elif category == 5:
    print("talking on the phone - left")
elif category == 6:
    print("operating the radio")
elif category == 7:
    print("Drinking")
elif category == 8:
    print("Reaching behind")
elif category == 9:
    print("hair and makeup")
elif category == 10:
    print("talking to passenger")
 