from tensorflow.keras import optimizers as opt
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger


optimizers_list = [ opt.Adadelta,
                    opt.Adagrad,
                    opt.Adam,
                    opt.Adamax,
                    opt.SGD,
                    opt.Nadam,
                    opt.RMSprop,]

activations_list = ["exponential","relu", "elu", "sigmoid", "tanh", "selu"]


def build_model_img(hp):
    activation = hp.Choice("activation", activations_list)
    layer_quant = hp.Int("layers", 1, 3)
    lr = hp.Float("lr", 1e-5, 0.1, step=10, sampling="log")
    optim = hp.Choice("optimizer", list(range(0, len(optimizers_list))), default=2)

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    for i in range(layer_quant):
        model.add(Dense(units=hp.Int(f"units_{i}", 32, 256, step=32), activation=activation))
        if hp.Boolean("dropout"):
            model.add(Dropout(0.2))

    model.add(Dense(1))
    model.compile(loss='mae', optimizer=optimizers_list[optim](learning_rate=lr))
    return model