from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2
from keras.constraints import max_norm
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import metrics

img_height, img_width = 256, 256

train_data_dir = "../data/PROCESSED_DATA/TRAIN"
validation_data_dir = "../data/PROCESSED_DATA/VALIDATION"
test_data_dir = "../data/TEST"

nb_train_samples = 281997
nb_validation_samples = 13712
epochs = 500
batch_size = 12
L2REG=0.003
FINAL_OUT_SIZE=300
W_INIT='he_normal'

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
    axis=1
else:
    input_shape = (img_width, img_height, 3)
    axis=3

def top_10_wrapper(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, 10)

def top_15_wrapper(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, 15)

def top_20_wrapper(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, 20)


model = load_model("best_wt_1k.hdf5", custom_objects={
'top_10_wrapper': top_10_wrapper,
'top_15_wrapper': top_15_wrapper,
'top_20_wrapper': top_20_wrapper
})

# fine tuning stuff
print(len(model.layers))
for layer in model.layers[:42]:
    layer.trainable = False

adam=Adam(lr=0.00025)



model.compile(
    loss='categorical_crossentropy',
    optimizer=adam,
    metrics=['accuracy', metrics.top_k_categorical_accuracy,
    top_10_wrapper, top_15_wrapper, top_20_wrapper])

train_datagen = ImageDataGenerator(
rescale=1./255,
shear_range=0.1,
zoom_range=0.1,
fill_mode='nearest')

test_datagen = ImageDataGenerator(
	    rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    )

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    )

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
	  patience=3, min_lr=0.0000008)
checkpt = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5")

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    use_multiprocessing=True,
    callbacks=[reduce_lr, checkpt],
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
