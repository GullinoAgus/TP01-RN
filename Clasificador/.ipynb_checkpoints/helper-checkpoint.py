import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
from os.path import exists
from tensorflow.keras import optimizers as opt
from tensorflow.keras import initializers as winit
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, RocCurveDisplay

seed = hash("Los buenos caballos y los buenos estudiantes siempre terminan su carrera.") % 2**32

metrics=['accuracy', 'AUC']

optimizers_list = [ opt.Adadelta,
                    opt.Adagrad,
                    opt.Adam,
                    opt.Adamax,
                    opt.SGD,
                    opt.Nadam,
                    opt.RMSprop,]

activations_list = ["exponential","relu", "elu", "sigmoid", "tanh", "selu"]

weight_init_list = [ lambda: winit.GlorotNormal(seed=seed),
                    lambda: winit.GlorotUniform(seed=seed),
                    lambda: winit.RandomNormal(0, 1, seed=seed),
                    lambda: winit.RandomNormal(0, 0.001, seed=seed)]

weight_init_name_list = [ "GlorotNormal",
                    "GlorotUniform",
                    "Normal_0-1_",
                    "Normal_0-0.001_"]

def plot_decision_bounds(model, x_data, labels):
    label2txt = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    extractor = keras.Model(inputs=model.input,
                            outputs=model.layers[-2].output)
    bidim_data = extractor.predict(x_data)
    bidim_data_df = pd.DataFrame()
    bidim_data_df["x"] = bidim_data[...,0]
    bidim_data_df["y"] = bidim_data[...,1]
    bidim_data_df["label"] = labels

    extractor_softmax = keras.Model(inputs=model.layers[-1].input,
                            outputs=model.layers[-1].output)
    x1_min = np.min(bidim_data_df["x"])
    x1_max = np.max(bidim_data_df["x"])
    x2_min = np.min(bidim_data_df["y"])
    x2_max = np.max(bidim_data_df["y"])

    g = np.meshgrid(np.arange(x1_min,x1_max, 1), np.arange(x2_min,x2_max, 1))
    positions = np.append(g[0].reshape(-1,1),g[1].reshape(-1,1),axis=1)

    grid_values = np.argmax(extractor_softmax.predict(positions), (1))
    grid_values = np.reshape(grid_values, g[0].shape)

    plt.figure()
    cmap = cm.get_cmap('viridis', 10)
    
    contour_plot = plt.pcolormesh(g[0], g[1], grid_values, cmap=cmap)
    cbar = plt.colorbar()
    cbar.ax.set_yticklabels(['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
    plt.scatter(
        x="x", y="y",
        data=bidim_data_df,
        c=bidim_data_df["label"].values,
        edgecolors='black',
        s=20, 
        linewidths=.3,
    )

    plt.show()
    
def model_evaluation(y_true_oh, y_pred_oh):
    label2txt = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    y_true = np.argmax(y_true_oh, axis=-1)
    y_pred = np.argmax(y_pred_oh, axis=-1)
    print(classification_report(y_true, y_pred, target_names=label2txt))
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=label2txt, xticks_rotation='vertical')

    fig = plt.figure()
    ax = fig.add_subplot()
    fig.set_figwidth(8)
    fig.set_figheight(8)
    fig.set_tight_layout(True)
    for i in range(10):
        RocCurveDisplay.from_predictions(y_true_oh[:, i], y_pred_oh[:, i],
                                        name=f"ROC {label2txt[i]}",
                                        ax=ax)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("OvR ROC Curve for each class")
    plt.show()
    
def hyperparam_search(x_train, y_train, x_val, y_val, lr_list=[0.1, 0.01, 0.001], dr_list=[0.1, 0.2, 0.4, 0.6], bs_list=[32, 128, 256, 1024, 4098]):
    
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot()
    score = 0
    best_winit = 0
    for i, winit in enumerate(weight_init_list):
        winit_obj = winit()
        winit_name = weight_init_name_list[i]
        name = f"winit{winit_name}_actrelu_lr{0.01}_opNadam_bs{1024}"
        # Checkeo si ya se entreno un modelo con estos parametros para no volver a entrenar.
        if exists(f"./ModelsCP/{name}.hdf5"):
            print(f"El modelo {name} ya fue entrenado \n")
        else:
            dense_layers = [(512, 'relu'), (512, 'relu'), (10, 'softmax')]
            model = create_model_mnist(name, dense_layers, optimizer=opt.Nadam(0.01), weight_init=winit_obj)

            checkpoint_filepath = f"./ModelsCP/{name}.hdf5"
            earlyStopCB = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, start_from_epoch=5)
            LearningSchedulerCB = ReduceLROnPlateau(monitor='val_loss', factor=0.2, min_lr=1e-8, patience=5)
            CheckpointCB = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', mode='max', save_best_only=True)
            CSVLoggerCB = CSVLogger(f"./ModelsCP/{name}.csv")
            callbacks = [earlyStopCB, CheckpointCB, CSVLoggerCB, LearningSchedulerCB]

            history = model.fit(x=x_train, y=y_train, epochs=150, batch_size=1024, verbose=2, callbacks=callbacks,
                                validation_data=(x_val, y_val))

        # Compruebo el score del modelo y lo agrego a su respectivo grafico
        model_df = pd.read_csv(f"./ModelsCP/{name}.csv")
        max_acc_epoch = np.argmax(model_df["val_accuracy"])
        auxscore = model_df["val_accuracy"][max_acc_epoch]/(max_acc_epoch+1)
        max_acc = model_df["val_accuracy"][max_acc_epoch]
        # Chequeo el score.
        if auxscore > score and model_df["val_accuracy"][max_acc_epoch] > 0.83:
            score = auxscore
            best_winit = winit_name

        ax.plot(model_df["epoch"], model_df["val_accuracy"], label=f"{winit_name}, maxACC:{max_acc}")
    ax.legend()
    fig.suptitle(f"La mejor inicializacion de pesos es: {best_winit}")
    fig.savefig("./HyperparamSearch/WeightInit.png", bbox_inches='tight')
    plt.close(fig)
    
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot()
    score = 0
    best_act = ''
    for i, act in enumerate(activations_list):
        name = f"act{act}_lr{0.01}_opNadam_bs{1024}"
        # Checkeo si ya se entreno un modelo con estos parametros para no volver a entrenar.
        if exists(f"./ModelsCP/{name}.hdf5"):
            print(f"El modelo {name} ya fue entrenado \n")
        else:
            dense_layers = [(512, act), (512, act), (10, 'softmax')]
            model = create_model_mnist(name, dense_layers, optimizer=opt.Nadam(0.01))

            checkpoint_filepath = f"./ModelsCP/{name}.hdf5"
            earlyStopCB = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, start_from_epoch=5)
            LearningSchedulerCB = ReduceLROnPlateau(monitor='val_loss', factor=0.2, min_lr=1e-8, patience=5)
            CheckpointCB = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', mode='max', save_best_only=True)
            CSVLoggerCB = CSVLogger(f"./ModelsCP/{name}.csv")
            callbacks = [earlyStopCB, CheckpointCB, CSVLoggerCB, LearningSchedulerCB]


            history = model.fit(x=x_train, y=y_train, epochs=150, batch_size=1024, verbose=2, callbacks=callbacks,
                                validation_data=(x_val, y_val))

        # Compruebo el score del modelo y lo agrego a su respectivo grafico
        model_df = pd.read_csv(f"./ModelsCP/{name}.csv")
        max_acc_epoch = np.argmax(model_df["val_accuracy"])
        auxscore = model_df["val_accuracy"][max_acc_epoch]/(max_acc_epoch+1)
        max_acc = model_df["val_accuracy"][max_acc_epoch]
        # Chequeo el score.
        if auxscore > score and model_df["val_accuracy"][max_acc_epoch] > 0.83:
            score = auxscore
            best_act = act

        ax.plot(model_df["epoch"], model_df["val_accuracy"], label=f"{act}, maxACC:{max_acc}")
    ax.legend()
    fig.suptitle(f"La mejor activacion es: {best_act}")
    fig.savefig("./HyperparamSearch/Activation.png", bbox_inches='tight')
    plt.close(fig)
    dense_layers = [(512, best_act), (512, best_act), (10, 'softmax')]
    
    # Busco el mejor par optimizador-LR con modelos MLP de buen performance
    fig = plt.figure(figsize=(len(lr_list)*8, 10))
    # Preparo una variable de score para cada par. El score se mide como: val_acc_maxima/epoch_donde_ocurrio.
    # De esta manera se pesa el tiempo que tarda en llegar a un accuracy alto
    score = 0
    best_optlr_pair = ['opt', 0.1]
    for i, lr in enumerate(lr_list):
        ax = fig.add_subplot(len(lr_list), 1, i+1)
        ax.set_title(f"Learning Rate: {lr}")
        for opti in optimizers_list:
            opt_obj = opti(learning_rate=lr)
            name = f"act{best_act}_lr{lr}_op{opt_obj.name}"
            # Checkeo si ya se entreno un modelo con estos parametros para no volver a entrenar.
            if exists(f"./ModelsCP/{name}.hdf5"):
                print(f"El modelo {name} ya fue entrenado \n")
            else:
                model = create_model_mnist(name, dense_layers, optimizer=opt_obj)

                checkpoint_filepath = f"./ModelsCP/{name}.hdf5"
                earlyStopCB = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, start_from_epoch=5)
                # LearningSchedulerCB = ReduceLROnPlateau(monitor='val_loss', factor=0.2, min_lr=1e-8, patience=5)
                CheckpointCB = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', mode='max', save_best_only=True)
                CSVLoggerCB = CSVLogger(f"./ModelsCP/{name}.csv")
                callbacks = [earlyStopCB, CheckpointCB, CSVLoggerCB]


                history = model.fit(x=x_train, y=y_train, epochs=150, batch_size=128, verbose=2, callbacks=callbacks,
                                    validation_data=(x_val, y_val))
            # Compruebo el score del modelo y lo agrego a su respectivo grafico
            model_df = pd.read_csv(f"./ModelsCP/{name}.csv")
            max_acc_epoch = np.argmax(model_df["val_accuracy"])
            auxscore = model_df["val_accuracy"][max_acc_epoch]/(max_acc_epoch+1)
            # Chequeo el score.
            if auxscore > score and model_df["val_accuracy"][max_acc_epoch] > 0.83:
                score = auxscore
                best_optlr_pair[0] = opti
                best_optlr_pair[1] = lr
            ax.plot(model_df["epoch"], model_df["val_accuracy"], label=opt_obj.name)
            
        ax.legend()
    fig.suptitle(f"El mejor par LR-Opt es: {best_optlr_pair}")
    fig.savefig("./HyperparamSearch/LRandOpt.png", bbox_inches='tight')
    plt.close(fig)
    
    best_opt = best_optlr_pair[0]
    best_lr = best_optlr_pair[1]
    # Busco el mejor batch size.
    fig = plt.figure(figsize=(len(bs_list)*8, 10))
    score = 0
    best_bs = 0
    for i, bs in enumerate(bs_list):
            opt_obj = best_opt(learning_rate=best_lr)
            name = f"act{best_act}_lr{best_lr}_op{opt_obj.name}_bs{bs}"
            # Checkeo si ya se entreno un modelo con estos parametros para no volver a entrenar.
            if exists(f"./ModelsCP/{name}.hdf5"):
                print(f"El modelo {name} ya fue entrenado \n")
            else:
                model = create_model_mnist(name, dense_layers, optimizer=opt_obj)

                checkpoint_filepath = f"./ModelsCP/{name}.hdf5"
                earlyStopCB = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, start_from_epoch=5)
                LearningSchedulerCB = ReduceLROnPlateau(monitor='val_loss', factor=0.2, min_lr=1e-8, patience=5)
                CheckpointCB = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', mode='max', save_best_only=True)
                CSVLoggerCB = CSVLogger(f"./ModelsCP/{name}.csv")
                callbacks = [earlyStopCB, CheckpointCB, CSVLoggerCB, LearningSchedulerCB]


                history = model.fit(x=x_train, y=y_train, epochs=150, batch_size=bs, verbose=2, callbacks=callbacks,
                                    validation_data=(x_val, y_val))
                
            # Compruebo el score del modelo y lo agrego a su respectivo grafico
            model_df = pd.read_csv(f"./ModelsCP/{name}.csv")
            max_acc_epoch = np.argmax(model_df["val_accuracy"])
            auxscore = model_df["val_accuracy"][max_acc_epoch]/(max_acc_epoch+1)
            max_acc = model_df["val_accuracy"][max_acc_epoch]
            # Chequeo el score.
            if auxscore > score and model_df["val_accuracy"][max_acc_epoch] > 0.83:
                score = auxscore
                best_bs = bs
            ax = fig.add_subplot(len(bs_list), 1, i+1)
            ax.plot(model_df["epoch"], model_df["val_accuracy"])
            ax.set_title(f"BatchSize {bs} .Best Acc: {max_acc}")
    fig.suptitle(f"El mejor Batch Size es: {best_bs}")
    fig.savefig("./HyperparamSearch/BatchSize.png", bbox_inches='tight')
    plt.close(fig)
    
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot()
    score = 0
    best_dr = 0
    for i, dr in enumerate(dr_list):
        opt_obj = best_opt(learning_rate=best_lr)
        name = f"act{best_act}_lr{best_lr}_op{opt_obj.name}_bs{best_bs}_dr{dr}"
        # Checkeo si ya se entreno un modelo con estos parametros para no volver a entrenar.
        if exists(f"./ModelsCP/{name}.hdf5"):
            print(f"El modelo {name} ya fue entrenado \n")
        else:
            model = create_model_mnist(name, dense_layers, optimizer=opt_obj, dropout_rate=dr)

            checkpoint_filepath = f"./ModelsCP/{name}.hdf5"
            earlyStopCB = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, start_from_epoch=5)
            LearningSchedulerCB = ReduceLROnPlateau(monitor='val_loss', factor=0.2, min_lr=1e-8, patience=5)
            CheckpointCB = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', mode='max', save_best_only=True)
            CSVLoggerCB = CSVLogger(f"./ModelsCP/{name}.csv")
            callbacks = [earlyStopCB, CheckpointCB, CSVLoggerCB, LearningSchedulerCB]


            history = model.fit(x=x_train, y=y_train, epochs=150, batch_size=best_bs, verbose=2, callbacks=callbacks,
                                validation_data=(x_val, y_val))

        # Compruebo el score del modelo y lo agrego a su respectivo grafico
        model_df = pd.read_csv(f"./ModelsCP/{name}.csv")
        max_acc_epoch = np.argmax(model_df["val_accuracy"])
        auxscore = model_df["val_accuracy"][max_acc_epoch]/(max_acc_epoch+1)
        max_acc = model_df["val_accuracy"][max_acc_epoch]
        # Chequeo el score.
        if auxscore > score and model_df["val_accuracy"][max_acc_epoch] > 0.83:
            score = auxscore
            best_dr = dr
        ax.plot(model_df["epoch"], model_df["val_accuracy"], label=f"DR:{dr}, MaxACC{max_acc}")
    ax.legend()
    fig.suptitle(f"El mejor Droput Rate es: {best_dr}. Con un score {score}")
    fig.savefig("./HyperparamSearch/DropoutRate.png", bbox_inches='tight')
    plt.close(fig)
    
    fig = plt.figure(figsize=(8, 10))
    score = 0
    bn_on = False
    ax = fig.add_subplot(1, 1, 1)
    for bn in [True, False]:
        opt_obj = best_opt(learning_rate=best_lr)
        name = f"act{best_act}_lr{best_lr}_op{opt_obj.name}_bs{best_bs}_bn{bn}"
        # Checkeo si ya se entreno un modelo con estos parametros para no volver a entrenar.
        if exists(f"./ModelsCP/{name}.hdf5"):
            print(f"El modelo {name} ya fue entrenado \n")
        else:
            model = create_model_mnist(name, dense_layers, optimizer=opt_obj, dropout_rate=best_dr, batch_normalization=bn)

            checkpoint_filepath = f"./ModelsCP/{name}.hdf5"
            earlyStopCB = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, start_from_epoch=5)
            LearningSchedulerCB = ReduceLROnPlateau(monitor='val_loss', factor=0.2, min_lr=1e-8, patience=5)
            CheckpointCB = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', mode='max', save_best_only=True)
            CSVLoggerCB = CSVLogger(f"./ModelsCP/{name}.csv")
            callbacks = [earlyStopCB, CheckpointCB, CSVLoggerCB, LearningSchedulerCB]


            history = model.fit(x=x_train, y=y_train, epochs=150, batch_size=best_bs, verbose=2, callbacks=callbacks,
                                validation_data=(x_val, y_val))

        # Compruebo el score del modelo y lo agrego a su respectivo grafico
        model_df = pd.read_csv(f"./ModelsCP/{name}.csv")
        max_acc_epoch = np.argmax(model_df["val_accuracy"])
        auxscore = model_df["val_accuracy"][max_acc_epoch]/(max_acc_epoch+1)
        max_acc = model_df["val_accuracy"][max_acc_epoch]
        # Chequeo el score.
        if auxscore > score and model_df["val_accuracy"][max_acc_epoch] > 0.83:
            score = auxscore
            bn_on = bn

        ax.plot(model_df["epoch"], model_df["val_accuracy"], label=f"{bn}, MaxACC: {max_acc}")
        
    ax.legend()
    fig.suptitle(f"¿Sirve Batch Normalization?: {bn_on}. Con un score {score}")
    fig.savefig("./HyperparamSearch/BatchNormalization.png", bbox_inches='tight')
    plt.close(fig)
    
            
def create_model_mnist(name, dense_NN: list[tuple[int, str]], dropout_rate=0, batch_normalization=False, optimizer=opt.SGD(), weight_init="glorot_normal"):
    model = Sequential(name=name)
    model.add(Flatten(input_shape=(28,28)))
    
    for NN, activ in dense_NN:
        model.add(Dense(NN, activation=activ, kernel_initializer=weight_init))
        if dropout_rate and activ != 'softmax':
            model.add(Dropout(dropout_rate))
        if batch_normalization and activ != 'softmax':
            model.add(BatchNormalization())
            
    model.compile(optimizer, loss='categorical_crossentropy', metrics=metrics)
    return model