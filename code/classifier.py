from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2
from keras.constraints import max_norm
from keras import backend as K
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import plot_model

img_height, img_width = 256, 256

train_data_dir = "../PROCESSED_DATA/TRAIN"
validation_data_dir = "../PROCESSED_DATA/VALIDATION"
# train_data_dir = "./test_train"
# validation_data_dir = "./test_validation"

nb_train_samples = 463094
nb_validation_samples = 92937
# nb_train_samples = 288
# nb_validation_samples = 48
epochs = 50
batch_size = 16
L2REG=0.001
W_INIT='he_normal'

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
    axis=1
else:
    input_shape = (img_width, img_height, 3)
    axis=3

def _design_model():
    model = Sequential()
    # add the first convolutional layer, so specify input shape
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=l2(L2REG), kernel_initializer=W_INIT, input_shape=input_shape))
    model.add(BatchNormalization(axis=axis))
    model.add(PReLU(alpha_initializer=W_INIT))

    # add the subsequent layers
    model.add(_intermediate_conv_layer(64, 1))
    model.add(BatchNormalization(axis=axis))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization(axis=axis))
    model.add(PReLU(alpha_initializer=W_INIT))

    model.add(_intermediate_conv_layer(128, 1))
    model.add(BatchNormalization(axis=axis))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_intermediate_conv_layer(128, 0))
    model.add(BatchNormalization(axis=axis))
    model.add(PReLU(alpha_initializer=W_INIT))

    model.add(_intermediate_conv_layer(256, 1))
    model.add(BatchNormalization(axis=axis))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_intermediate_conv_layer(256, 1))
    model.add(BatchNormalization(axis=axis))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_intermediate_conv_layer(256, 1))
    model.add(BatchNormalization(axis=axis))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_intermediate_conv_layer(512, 0))
    model.add(BatchNormalization(axis=axis))
    model.add(PReLU(alpha_initializer=W_INIT))

    model.add(_intermediate_conv_layer(512, 1))
    model.add(BatchNormalization(axis=axis))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_intermediate_conv_layer(512, 0))
    model.add(BatchNormalization(axis=axis))
    model.add(PReLU(alpha_initializer=W_INIT))

    model.add(_intermediate_conv_layer(512, 1))
    model.add(BatchNormalization(axis=axis))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_intermediate_conv_layer(512, 0))
    model.add(BatchNormalization(axis=axis))
    model.add(PReLU(alpha_initializer=W_INIT))

    model.add(Flatten())
    model.add(Dense(1024, kernel_regularizer=l2(L2REG)))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(Dropout(rate=0.4))
    model.add(Dense(2048, kernel_regularizer=l2(L2REG)))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(Dropout(rate=0.4))
    model.add(Dense(1584, kernel_regularizer=l2(L2REG), activation='softmax'))

    # SVG(model_to_dot(model).create(prog='dot', format='svg'))
    plot_model(model, show_shapes=True, to_file='model.png')

    return model


def _intermediate_conv_layer(nb_filter, padding):
    """
    Returns a 2D conv layer with stack_length=nb_filter and padding
    """
    if padding == 1:
        return Conv2D(
            filters=nb_filter,
            kernel_size=(3, 3),
            kernel_regularizer=l2(L2REG),
	    kernel_initializer=W_INIT,
            padding='same'
            )
    else:
        return Conv2D(
            filters=nb_filter,
            kernel_size=(2, 2),
            kernel_regularizer=l2(L2REG),
	    kernel_initializer=W_INIT,
            padding='valid',
            strides=(2, 2)
            )


def _train_model():
    """
    Function to actually run the model by feeding data into it
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
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

    model = _design_model()
    adam = Adam(lr=0.000074)
    model.compile(
            loss='categorical_crossentropy',
            optimizer=adam,
            metrics=['accuracy'])


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                  patience=5, min_lr=0.0000008)
    model.save_weights('weights.h5')
    model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            use_multiprocessing=True,
	    callbacks=[reduce_lr],
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size)


if __name__ == "__main__":
    # _train_model()
    _design_model();
