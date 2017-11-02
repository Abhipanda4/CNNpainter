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
from keras.utils import plot_model

img_height, img_width = 256, 256

test_data_dir= "../data/TEST"

nb_test_samples = 13712
batch_size = 16
FINAL_OUT_SIZE=300

# if K.image_data_format() == 'channels_first':
    # input_shape = (3, img_width, img_height)
    # axis=1
# else:
    # input_shape = (img_width, img_height, 3)
    # axis=3

model = load_model("weights.18-3.28.hdf5")
plot_model(model, to_file="./model.png", show_shapes=True)


# ---------------------------------------------------
# Uncomment the following lines to find test accuracy
# ---------------------------------------------------

# test_datagen = ImageDataGenerator(
	    # rescale=1./255)
# test_generator = test_datagen.flow_from_directory(
    # test_data_dir,
    # target_size=(img_width, img_height),
    # batch_size=batch_size,
    # )
# res = model.evaluate_generator(
        # test_generator,
        # steps=nb_test_samples // batch_size,
        # use_multiprocessing=True)
# print(res)
