"""
Explainable AI

Project II

Journal Paper

Author: Saleh ValizadehSotubadi

Research Lab: Automation in Smart Manufacturing

MEEM Department at Michigan Technological University
"""

import numpy as np
import cv2
import os
import tensorflow as tf
import tensorflow.keras.backend
import matplotlib.pyplot as plt
from Pre_process import Import_Data
import pandas as pd 
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
from sklearn.metrics import accuracy_score
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
from Integrated_Gradients_tf import integrated_gradients, plot_img_attributions
from Grad_CAM_tf import make_gradcam_heatmap, save_and_display_gradcam, grad_cam_plus
from transfer_learning_set_weight import set_weights_transfer_learning_LSTM, set_weights_LSTM, set_weights_transfer_learning_Transformer

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Create a TensorFlow GPU device placement strategy
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

current_directory = os.getcwd()

saving_directory = 'XAI_Multi_Final_Results_Paper_II_New_Run_Just_for_images'

if saving_directory not in os.listdir(current_directory):
    os.mkdir(os.path.join(current_directory, saving_directory))

k = 0

save_directory = os.path.join(current_directory, saving_directory)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = tf.keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


def plotting(RNN):
    history_net_RNN = RNN.history
    RNNlosses = history_net_RNN['loss']
    RNNlossesVal = history_net_RNN['val_loss']
    RNNaccuracy = history_net_RNN['accuracy']
    RNNaccuracyVal = history_net_RNN['val_accuracy']
    plt.figure()
    plt.plot(RNNlosses, label = 'Train Loss')
    plt.plot(RNNlossesVal, label = 'Validation Loss')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    #plt.savefig('C:\\Users\\svalizad\\Desktop\\First Paper\\loss_function_7.png', dpi=500)
    plt.figure()
    plt.plot(RNNaccuracy, label = 'Train Accuracy')
    plt.plot(RNNaccuracyVal, label = 'Validation Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    #plt.savefig('C:\\Users\\svalizad\\Desktop\\First Paper\\accuracy_function_7.png', dpi=500)

def find_difference(y_pred, y_true):
    
    differences = np.where(y_pred != y_true)
    
    return differences

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return TP, FP, TN, FN


def MCC(TP, FP, TN, FN):
    
    mcc = ((TP*TN) - (FN * FP))/np.sqrt((TP + FN)*(TP + FP)*(TN + FN)*(TN + FP))
    
    return mcc

def normal_dist(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

def convert_to_excel(model):
    
    history_Net = model.history
    modellosses = history_Net['loss']
    modellossesVal = history_Net['val_loss']
    modelaccuracy = history_Net['accuracy']
    modelaccuracyVal = history_Net['val_accuracy']
    
    modellosses_pd = pd.DataFrame(modellosses)
    modellossesVal_pd = pd.DataFrame(modellossesVal)
    modelaccuracy_pd = pd.DataFrame(modelaccuracy)
    modelaccuracyVal_pd = pd.DataFrame(modelaccuracyVal)
    
    modellosses_pd.to_excel(os.path.join(save_directory,'Model_Loss3.xlsx'))
    modellossesVal_pd.to_excel(os.path.join(save_directory,'Model_Val_Loss3.xlsx'))
    modelaccuracy_pd.to_excel(os.path.join(save_directory,'Model_Accuracy3.xlsx'))
    modelaccuracyVal_pd.to_excel(os.path.join(save_directory,'Model_Val_Accuracy3.xlsx'))
    
    print("Saving Done! :)")

lr, b_1, b_2 = 0.0002 , 0.99, 0.95

lr_schedulor = tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_accuracy", 
                                                    factor = 0.99,
                                                    patience=1,
                                                    min_lr=0.0001)

early_schedulor = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                   min_delta=0,
                                                   patience=4
                                                   )

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='C:\\Users\\Saleh\\OneDrive\\Desktop\\First Paper\\best_model.h5',  # Path to save the model
                                                         save_best_only=True,  # Saves only the best model
                                                         monitor='val_accuracy',  # Monitors validation accuracy
                                                         mode='max',  # Mode for monitoring
                                                         verbose=0)


K_fold = 1

epoch, batch = 30 , 12

if K_fold == 0:

    total_ratio_for_test = 0.3
    
    train_rate = 1 - total_ratio_for_test
    
    validation_rate = 0.25
    
    test_rate = total_ratio_for_test - validation_rate
    
elif K_fold == 1:
    
    total_ratio_for_test = 0.20
    
    train_rate = 1 - total_ratio_for_test
    
    validation_rate = 0.05
    
    test_rate = total_ratio_for_test 

transfer, save_plots = True, True

transfer_learning = 1

transformer = 0

if transfer_learning == 0:
    transfer_learning = False
elif transfer_learning == 1:
    transfer_learning = True
    
if transfer == 0:
    transfer = False
elif transfer == 1:
    transfer = True
    
if save_plots == 0:
    save_plots = False
elif save_plots == 1:
    save_plots = True

machine_data, time_series, flank_image, rake_image, Labels = Import_Data('Train', transfer_learning)

img_size = flank_image.shape[1]

time_series_list = []

for i in range(time_series.shape[0]):
    
    machine_speed = machine_data[i,0] * np.ones(60,)
    
    machine_feed = machine_data[i,1] * np.ones(60,)
    
    machine_params = np.array([machine_speed, machine_feed])
    
    machine_params = np.transpose(machine_params)
    
    Time_Series = time_series[i]
    
    Time_Series = np.transpose(Time_Series)
    
    Time_Series_mixed = np.concatenate([Time_Series, machine_params], axis = 1)
    
    time_series_list.append(Time_Series_mixed)
    
time_series = np.array(time_series_list)

num_validation = int(time_series.shape[0] * validation_rate)

num_test = int(time_series.shape[0] * test_rate)

num_train = int(time_series.shape[0] * train_rate)

num_validation += num_train

Train_time_series, Train_flank_image, Train_rake_image, Train_Labels = time_series[:num_train], flank_image[:num_train], rake_image[:num_train], Labels[:num_train] 

if K_fold == 0:

    Val_time_series, Val_fkank_image, Val_rake_image, Val_Labels = time_series[num_train:num_validation], flank_image[num_train:num_validation], rake_image[num_train:num_validation], Labels[num_train:num_validation] 

    time_series_test, flank_image_test, rake_image_test, Labels_test = time_series[num_validation:], flank_image[num_validation:], rake_image[num_validation:], Labels[num_validation:]

    Train_Label, Val_Label = tf.keras.utils.to_categorical(Train_Labels), tf.keras.utils.to_categorical(Val_Labels)
elif K_fold == 1:
    time_series_test, flank_image_test, rake_image_test, Labels_test = time_series[num_train:], flank_image[num_train:], rake_image[num_train:], Labels[num_train:]

    Train_Label = tf.keras.utils.to_categorical(Train_Labels)

"""

if transfer_learning == True:

    VGG_features_CNN_I = tf.keras.applications.vgg19.VGG19(include_top=False, weights= "imagenet", input_shape=(224,224,3))
    VGG_features_CNN_I.trainable = False
    VGG_features_CNN_I._name = 'VGG_CNN_I'
    
    VGG_features_CNN_I.summary()
    
    VGG_features_CNN_II = tf.keras.applications.vgg19.VGG19(include_top=False, weights= "imagenet", input_shape=(224,224,3))
    VGG_features_CNN_II.trainable = False
    VGG_features_CNN_II._name = 'VGG_CNN_II'
    
    VGG_features_CNN_II.summary()
    
"""    
feature_neurons = 15

def design_network(name):
    
    if transfer_learning == True:
        
        VGG_features_CNN_I = tf.keras.applications.vgg19.VGG19(include_top=False, weights= "imagenet", input_shape=(224,224,3))
        VGG_features_CNN_I.trainable = False
        VGG_features_CNN_I._name = 'VGG_CNN_I'
        
        VGG_features_CNN_I.layers[-2].trainable = True
    
        #CNN_I_Inputs = tf.keras.layers.Input(shape=(224, 224, 3), name = "Image_Input_I")
        #feature_I_extractor = VGG_features_CNN_I(CNN_I_Inputs)
        
        for layer in VGG_features_CNN_I.layers:
            layer._name = layer._name + str('_flank')
        
        CNN_I_Inputs = VGG_features_CNN_I.input
        
        feature_I_extractor = VGG_features_CNN_I.output
        
        VGG_features_CNN_II = tf.keras.applications.vgg19.VGG19(include_top=False, weights= "imagenet", input_shape=(224,224,3))
        VGG_features_CNN_II.trainable = False
        VGG_features_CNN_II._name = 'VGG_CNN_II'
        
        VGG_features_CNN_II.layers[-2].trainable = True
        
        for layer in VGG_features_CNN_II.layers:
            layer._name = layer._name + str('_rake')
        
        #CNN_II_Inputs = tf.keras.layers.Input(shape=(224, 224, 3), name = "Image_Input_II")
        #feature_II_extractor = VGG_features_CNN_II(CNN_II_Inputs)
        
        CNN_II_Inputs = VGG_features_CNN_II.input
        
        feature_II_extractor = VGG_features_CNN_II.output
        
        
    elif transfer_learning == False:
        
        filter_I, kernel_size_I, stride_I = 64, 3, 1
        
        filter_II, kernel_size_II, stride_II = 128, 3, 1
        
        filter_III, kernel_size_III, stride_III = 128, 3, 1
        
        filter_IV, kernel_size_IV, stride_IV = 256, 3, 1
        
        CNN_I_Inputs = tf.keras.layers.Input(shape=(224, 224, 3), name = "Image_Input_I")
        CNN_I_I = tf.keras.layers.Conv2D(filter_I, kernel_size_I, stride_I, padding = "same", activation = "relu")(CNN_I_Inputs)
        CNN_I_II = tf.keras.layers.Conv2D(filter_I, kernel_size_I, stride_I, padding = "same", activation = "relu")(CNN_I_I)
        MaxPool_I_I = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(CNN_I_II)
        
        CNN_I_III = tf.keras.layers.Conv2D(filter_II, kernel_size_II, stride_II, padding = "same", activation = "relu")(MaxPool_I_I)
        CNN_I_IV = tf.keras.layers.Conv2D(filter_II, kernel_size_II, stride_II, padding = "same", activation = "relu")(CNN_I_III)
        MaxPool_I_II = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(CNN_I_IV)
        
        CNN_I_V = tf.keras.layers.Conv2D(filter_III, kernel_size_III, stride_III, padding = "same", activation = "relu")(MaxPool_I_II)
        CNN_I_VI = tf.keras.layers.Conv2D(filter_III, kernel_size_III, stride_III, padding = "same", activation = "relu")(CNN_I_V)
        MaxPool_I_III = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(CNN_I_VI)
        
        CNN_I_VII = tf.keras.layers.Conv2D(filter_IV, kernel_size_IV, stride_IV, padding = "same", activation = "relu")(MaxPool_I_III)
        CNN_I_VIII = tf.keras.layers.Conv2D(filter_IV, kernel_size_IV, stride_IV, padding = "same", activation = "relu", name = "CNN_output_I")(CNN_I_VII)
        feature_I_extractor = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2))(CNN_I_VIII)



        CNN_II_Inputs = tf.keras.layers.Input(shape=(224, 224, 3), name = "Image_Input_II")
        CNN_II_I = tf.keras.layers.Conv2D(filter_I, kernel_size_I, stride_I, padding = "same", activation = "relu")(CNN_II_Inputs)
        CNN_II_II = tf.keras.layers.Conv2D(filter_I, kernel_size_I, stride_I, padding = "same", activation = "relu")(CNN_II_I)
        MaxPool_II_I = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(CNN_II_II)
        
        CNN_II_III = tf.keras.layers.Conv2D(filter_II, kernel_size_II, stride_II, padding = "same", activation = "relu")(MaxPool_II_I)
        CNN_II_IV = tf.keras.layers.Conv2D(filter_II, kernel_size_II, stride_II, padding = "same", activation = "relu")(CNN_II_III)
        MaxPool_II_II = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(CNN_II_IV)
        
        CNN_II_V = tf.keras.layers.Conv2D(filter_III, kernel_size_III, stride_III, padding = "same", activation = "relu")(MaxPool_II_II)
        CNN_II_VI = tf.keras.layers.Conv2D(filter_III, kernel_size_III, stride_III, padding = "same", activation = "relu")(CNN_II_V)
        MaxPool_II_III = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(CNN_II_VI)
        
        CNN_II_VII = tf.keras.layers.Conv2D(filter_IV, kernel_size_IV, stride_IV, padding = "same", activation = "relu")(MaxPool_II_III)
        CNN_II_VIII = tf.keras.layers.Conv2D(filter_IV, kernel_size_IV, stride_IV, padding = "same", activation = "relu", name = "CNN_output_II")(CNN_II_VII)
        feature_II_extractor = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2))(CNN_II_VIII)
        
        
    flatten_I = tf.keras.layers.Flatten(name = "Flatten_I")(feature_I_extractor)
    class_1 = tf.keras.layers.Dense(feature_neurons , kernel_regularizer = tf.keras.regularizers.l2(0.05),  name = "Class_I")(flatten_I)
    class_1 = tf.keras.activations.relu(class_1)
    class_1 = tf.keras.layers.Dropout(0.25, name = "Dropout_I")(class_1)
    
    flatten_II = tf.keras.layers.Flatten(name = "Flatten_II")(feature_II_extractor)
    class_2 = tf.keras.layers.Dense(feature_neurons, kernel_regularizer = tf.keras.regularizers.l2(0.05), name = "Class_II")(flatten_II)
    class_2 = tf.keras.activations.relu(class_2)
    class_2 = tf.keras.layers.Dropout(0.25, name = "Dropout_II")(class_2)
    
    if transformer == 0:
    
        RNN_input = tf.keras.layers.Input(shape = (60, 8), name = "Time_Input")
        RNN_first = tf.keras.layers.LSTM(256, return_sequences=True, activation = "tanh", kernel_regularizer = tf.keras.regularizers.l2(0.01), name = "RNN_1")(RNN_input)
        #RNN_first = tf.keras.layers.Dropout(0.3, name = "Dropout_III_I")(RNN_first)
        RNN_second = tf.keras.layers.LSTM(512 , return_sequences=False, activation = "tanh", kernel_regularizer = tf.keras.regularizers.l2(0.01), name = "RNN_2")(RNN_first)
        RNN_second = tf.keras.layers.Dropout(0.25, name = "Dropout_III_II")(RNN_second)
        class_3 = tf.keras.layers.Dense(feature_neurons, kernel_regularizer = tf.keras.regularizers.l2(0.01), name = "RNN_III")(RNN_second)
        class_3 = tf.keras.activations.relu(class_3)
        class_3 = tf.keras.layers.Dropout(0.25, name = "Dropout_III_III")(class_3)
    elif transformer == 1:
        
        RNN_input = tf.keras.layers.Input(shape = (60, 8), name = "Time_Input")
        x = transformer_encoder(RNN_input, 256, 8, 4, 0.2)
        x = transformer_encoder(x, 256, 8, 4, 0.2)
        x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")(x)
        class_3 = tf.keras.layers.Dense(feature_neurons, kernel_regularizer = tf.keras.regularizers.l2(0.01), name = "RNN_III")(x)
        class_3 = tf.keras.activations.relu(class_3)
        class_3 = tf.keras.layers.Dropout(0.25, name = "Dropout_III_III")(class_3)
    
    
    concat = tf.keras.layers.concatenate([class_1, class_2, class_3], name = "Concat")
    
    if name == "Full_model":
        
        concat = tf.keras.layers.BatchNormalization(name = "Batch_Full")(concat)
        """
        classifier_Input = tf.keras.layers.Dense(8, kernel_regularizer = tf.keras.regularizers.l2(0.001), name = "CLassifier_Dense_I")(concat)
        classifier_one = tf.keras.activations.relu(classifier_Input)
        classifier_one = tf.keras.layers.Dropout(0.2, name = "Dropout_Classifier_I")(classifier_one)
        classifier_one = tf.keras.layers.BatchNormalization(name = "Batch_I")(classifier_one)
        
        classifier_two = tf.keras.layers.Dense(16, kernel_regularizer = tf.keras.regularizers.l2(0.001), name = "CLassifier_Dense_II")(classifier_one)
        classifier_two = tf.keras.activations.relu(classifier_two)
        classifier_two = tf.keras.layers.Dropout(0.2, name = "Dropout_Classifier_II")(classifier_two)
        classifier_two = tf.keras.layers.BatchNormalization(name = "Batch_II")(classifier_two)
        """
        classifier_output = tf.keras.layers.Dense(2, activation="softmax", name = "Classifier_III")(concat )
        
        
        Full_model = tf.keras.models.Model([CNN_I_Inputs, CNN_II_Inputs, RNN_input], classifier_output)
        
        return Full_model
    
    elif name == "CNN_I_submodel":
        
        concat = tf.keras.layers.BatchNormalization(name = "Batch_CNN_I")(class_1)
        """
        classifier_Input = tf.keras.layers.Dense(8, kernel_regularizer = tf.keras.regularizers.l2(0.001), name = "CLassifier_Dense_I")(class_1)
        classifier_one = tf.keras.activations.relu(classifier_Input)
        classifier_one = tf.keras.layers.Dropout(0.2, name = "Dropout_Classifier_I")(classifier_one)
        classifier_one = tf.keras.layers.BatchNormalization(name = "Batch_I")(classifier_one)
        
        classifier_two = tf.keras.layers.Dense(16, kernel_regularizer = tf.keras.regularizers.l2(0.001), name = "CLassifier_Dense_II")(classifier_one)
        classifier_two = tf.keras.activations.relu(classifier_two)
        classifier_two = tf.keras.layers.Dropout(0.2, name = "Dropout_Classifier_II")(classifier_two)
        classifier_two = tf.keras.layers.BatchNormalization(name = "Batch_II")(classifier_two)
        """
        classifier_output = tf.keras.layers.Dense(2, activation="softmax", name = "Classifier_III")(concat)
        
        CNN_I_submodel = tf.keras.models.Model(CNN_I_Inputs, classifier_output)
        
        return CNN_I_submodel
    
    elif name == "CNN_II_submodel":
        
        concat = tf.keras.layers.BatchNormalization(name = "Batch_CNN_II")(class_2)
        """
        classifier_Input = tf.keras.layers.Dense(8, kernel_regularizer = tf.keras.regularizers.l2(0.001), name = "CLassifier_Dense_I")(class_2)
        classifier_one = tf.keras.activations.relu(classifier_Input)
        classifier_one = tf.keras.layers.Dropout(0.2, name = "Dropout_Classifier_I")(classifier_one)
        classifier_one = tf.keras.layers.BatchNormalization(name = "Batch_I")(classifier_one)
        
        classifier_two = tf.keras.layers.Dense(16, kernel_regularizer = tf.keras.regularizers.l2(0.001), name = "CLassifier_Dense_II")(classifier_one)
        classifier_two = tf.keras.activations.relu(classifier_two)
        classifier_two = tf.keras.layers.Dropout(0.2, name = "Dropout_Classifier_II")(classifier_two)
        classifier_two = tf.keras.layers.BatchNormalization(name = "Batch_II")(classifier_two)
        """
        classifier_output = tf.keras.layers.Dense(2, activation="softmax", name = "Classifier_III")(concat)
        
        CNN_II_submodel = tf.keras.models.Model(CNN_II_Inputs, classifier_output)
        
        return CNN_II_submodel
    
    elif name == "RNN_submodel":
        
        concat = tf.keras.layers.BatchNormalization(name = "Batch_RNN")(class_3)
        """
        classifier_Input = tf.keras.layers.Dense(8, kernel_regularizer = tf.keras.regularizers.l2(0.001), name = "CLassifier_Dense_I")(class_3)
        classifier_one = tf.keras.activations.relu(classifier_Input)
        classifier_one = tf.keras.layers.Dropout(0.2, name = "Dropout_Classifier_I")(classifier_one)
        classifier_one = tf.keras.layers.BatchNormalization(name = "Batch_I")(classifier_one)
        
        classifier_two = tf.keras.layers.Dense(16, kernel_regularizer = tf.keras.regularizers.l2(0.001), name = "CLassifier_Dense_II")(classifier_one)
        classifier_two = tf.keras.activations.relu(classifier_two)
        classifier_two = tf.keras.layers.Dropout(0.2, name = "Dropout_Classifier_II")(classifier_two)
        classifier_two = tf.keras.layers.BatchNormalization(name = "Batch_II")(classifier_two)
        """
        classifier_output = tf.keras.layers.Dense(2, kernel_regularizer = tf.keras.regularizers.l2(0.01), activation="softmax", name = "Classifier_III")(concat)
        
        RNN_submodel = tf.keras.models.Model(RNN_input, classifier_output)
        
        return RNN_submodel
    
    elif name == "Classifier":
        
        concat_I = tf.keras.layers.Input(shape = (feature_neurons * 3, ), name = "Latent_Input")
        
        classifier_output = tf.keras.layers.Dense(2, activation="softmax", name = "Classifier_III")(concat_I)
        
        classifier_model = tf.keras.models.Model(concat_I, classifier_output)
        
        return classifier_model
    
    elif name == "Latent_Space":
        
        concat = tf.keras.layers.BatchNormalization(name = "Batch_Full")(concat)
        
        Latent_space = tf.keras.models.Model([CNN_I_Inputs, CNN_II_Inputs, RNN_input], concat)
        
        return Latent_space

###################################################################################################
Full_model = design_network("Full_model")

Full_model.summary()

RNN_submodel = design_network("RNN_submodel")

CNN_I_submodel = design_network("CNN_I_submodel")

CNN_II_submodel = design_network("CNN_II_submodel")

Classifier = design_network("Classifier")

Latent_Space = design_network("Latent_Space")

"""
if save_plots == True:
    tf.keras.utils.plot_model(Full_model, to_file = os.path.join(save_directory,"model_architecture.png"))
"""
Loss_function = tf.keras.losses.BinaryCrossentropy()

Model_optimizer = tf.keras.optimizers.Adam(learning_rate = lr, 
                                           beta_1 = b_1, 
                                           beta_2 = b_2)

Model_optimizer_II = tf.keras.optimizers.RMSprop(learning_rate = lr,
                                                 rho = 0.9,
                                                 momentum = 0.0,
                                                 epsilon = 1e-07)

tf.keras.backend.clear_session()

Full_model.compile(optimizer = Model_optimizer_II,
                   loss = Loss_function,
                   metrics = "accuracy")

tf.keras.backend.clear_session()

CNN_II_submodel.compile(optimizer = Model_optimizer_II,
                   loss = Loss_function,
                   metrics = "accuracy")

tf.keras.backend.clear_session()

CNN_I_submodel.compile(optimizer = Model_optimizer_II,
                   loss = Loss_function,
                   metrics = "accuracy")

tf.keras.backend.clear_session()

RNN_submodel.compile(optimizer = Model_optimizer_II,
                   loss = Loss_function,
                   metrics = "accuracy")

tf.keras.backend.clear_session()

Classifier.compile(optimizer = Model_optimizer_II,
                   loss = Loss_function,
                   metrics = "accuracy")

tf.keras.backend.clear_session()

Latent_Space.compile(optimizer = Model_optimizer_II,
                   loss = Loss_function,
                   metrics = "accuracy")


if K_fold == 0:

    history = Full_model .fit([Train_flank_image, Train_rake_image, Train_time_series], Train_Label,
                             epochs = epoch,
                             batch_size = batch,
                             validation_data=([Val_fkank_image, Val_rake_image, Val_time_series], Val_Label),
                             verbose='auto',
                             callbacks=[lr_schedulor, early_schedulor, checkpoint_callback])
elif K_fold == 1:
    
    print("K Fold Cross Validation Applied")
    
    k_folds = 4
    kf = KFold(n_splits=k_folds, shuffle=True)

# Iterate over the folds
    for train_index, test_index in kf.split(Train_flank_image):
        
        Train_flank_image_I, Train_rake_image_I, Train_time_series_I = Train_flank_image[train_index], Train_rake_image[train_index], Train_time_series[train_index]
        
        Train_Label_I = Train_Label[train_index]
        
        Val_fkank_image, Val_rake_image, Val_time_series = Train_flank_image[test_index], Train_rake_image[test_index], Train_time_series[test_index]
        
        Val_Label = Train_Label[test_index]
    
        history = Full_model .fit([Train_flank_image_I, Train_rake_image_I, Train_time_series_I], Train_Label_I,
                                 epochs = epoch,
                                 batch_size = batch,
                                 validation_data=([Val_fkank_image, Val_rake_image, Val_time_series], Val_Label),
                                 verbose='auto',
                                 callbacks=[lr_schedulor, early_schedulor])
    

Weights = []

for layer in Full_model.layers:
    Weights.append(layer.get_weights())
    
Weights_RNN = []    

for layer in RNN_submodel.layers:
    Weights_RNN.append(layer.get_weights())
    
Weights_CNN_I = []

for layer in CNN_I_submodel.layers:
    Weights_CNN_I.append(layer.get_weights())   
    
Weights_CNN_II = []

for layer in CNN_II_submodel.layers:
    Weights_CNN_II.append(layer.get_weights()) 
    
Weights_Latent = []

for layer in Latent_Space.layers:
    Weights_Latent.append(layer.get_weights())
    
Weights_Classifier = []

for layer in Classifier.layers:
    Weights_Classifier.append(layer.get_weights())


for i in range(5):
    
    if transfer_learning == True:
        
        if transformer == 0:
            set_weights_transfer_learning_LSTM(i, CNN_I_submodel, CNN_II_submodel, RNN_submodel, Latent_Space, Classifier, Weights)
        elif transformer == 1:
            set_weights_transfer_learning_Transformer(i, CNN_I_submodel, CNN_II_submodel, RNN_submodel, Latent_Space, Classifier, Weights)
    elif transfer_learning == False:
        
        set_weights_LSTM(i, CNN_I_submodel, CNN_II_submodel, RNN_submodel, Latent_Space, Classifier, Weights)


for i in range(len(Full_model.weights)):
    Full_model.weights[i]._handle_name = Full_model.weights[i].name + "_" + str(i)

plotting(history)


if save_plots == True:

    History = history.history
    
    Loss = History['loss']
    
    Validation_Loss = History['val_loss']
    
    Accuracy = History['accuracy']
    
    Validation_Accuracy = History['val_accuracy']
    
    Loss_pd = pd.DataFrame(Loss)
    
    Val_loss_pd = pd.DataFrame(Validation_Loss)
    
    Accuracy_pd = pd.DataFrame(Accuracy)
    
    Val_Accuracy_pd = pd.DataFrame(Validation_Accuracy)
    
    Loss_pd.to_excel(os.path.join(save_directory,"Loss_Info_7.xlsx"))
    
    Val_loss_pd.to_excel(os.path.join(save_directory,"Val_loss_Info_7.xlsx"))
    
    Accuracy_pd.to_excel(os.path.join(save_directory,"Accuracy_Info_7.xlsx"))
    
    Val_Accuracy_pd.to_excel(os.path.join(save_directory,"Val_accuracy_Info_7.xlsx"))
    
    """
    Full_model.save(os.path.join(save_directory,"Trained_model_Full_Good_Results.h5"))
    
    CNN_I_submodel.save(os.path.join(save_directory,"Trained_model_CNN_I_Good_Results.h5"))
    
    CNN_II_submodel.save(os.path.join(save_directory,"Trained_model_CNN_II_Good_Results.h5"))
    
    RNN_submodel.save(os.path.join(save_directory,"Trained_model_RNN_Good_Results.h5"))
    
    Latent_Space.save(os.path.join(save_directory,"Trained_model_Latent_Space_Good_Results.h5"))
    
    Classifier.save(os.path.join(save_directory,"Trained_model_Classifier_Good_Results.h5"))
    """

predictions_full = Full_model.predict([flank_image_test, rake_image_test, time_series_test])

prediction_full_label = predictions_full.argmax(axis=-1)

xticklabels = ["Unworn", "Worn"]
yticklabels = ["Unworn", "Worn"]

matrix = confusion_matrix(Labels_test, prediction_full_label)
plt.figure()
plt.title("Model Accuracy: %" + str(100*round(accuracy_score(Labels_test, prediction_full_label),4)))
sns.heatmap(matrix, linewidth=0.5, cmap="YlGnBu", annot=True, xticklabels= xticklabels, yticklabels=yticklabels)

if save_plots == True:
    matrix_pd = pd.DataFrame(matrix)
    matrix_pd.to_excel(os.path.join(save_directory,"Matrix_For_Test_data.xlsx"))
    plt.savefig(os.path.join(save_directory,"Confusion_Matrix_Fullmodel.png"), format="png", bbox_inches="tight", dpi=500)

fpr, tpr, thresholds = roc_curve(Labels_test, prediction_full_label)

ROC_full_model = []

ROC_full_model.append(tpr)
ROC_full_model.append(fpr)

ROC_full_model_pd = pd.DataFrame(np.array(ROC_full_model))

ROC_full_model_pd.to_excel(os.path.join(save_directory,"ROF_full_model.xlsx"))

TP, FP, TN, FN = perf_measure(Labels_test, prediction_full_label)

Mcc = MCC(TP, FP, TN, FN)

Sens = ((TP)/(TP + FN))*100

Spec = ((TN)/(FP + TN))*100

plt.figure()
plt.plot([0, 1], [0, 1], 'k--',  label = 'Trained Model Performance')
plt.plot([0, 0], [0, 1], 'g', label = 'Ideal Model Performance')
plt.plot([0, 1], [1, 1], 'g')
plt.plot(fpr,tpr, label = 'Trained Model Performance')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()

if save_plots == True:
    plt.savefig(os.path.join(save_directory,"ROC_Curve_Full_model.png"), format="png", bbox_inches="tight", dpi=500)


CNN_I_prediction_on_data_test = CNN_I_submodel.predict(flank_image_test)

CNN_II_prediction_on_data_test = CNN_II_submodel.predict(rake_image_test)

RNN_prediction_on_data_test = RNN_submodel.predict(time_series_test)

print("CNN_I Model Accuracy: %" + str(100*round(accuracy_score(Labels_test, CNN_I_prediction_on_data_test.argmax(axis=-1)),4)))

print("CNN_II Model Accuracy: %" + str(100*round(accuracy_score(Labels_test, CNN_II_prediction_on_data_test.argmax(axis=-1)),4)))

print("RNN Model Accuracy: %" + str(100*round(accuracy_score(Labels_test, RNN_prediction_on_data_test.argmax(axis=-1)),4)))


matrix_CNN_I = confusion_matrix(Labels_test, CNN_I_prediction_on_data_test.argmax(axis=-1))
plt.figure()
sns.heatmap(matrix_CNN_I, linewidth=0.5, cmap="YlGnBu", annot=True, xticklabels= xticklabels, yticklabels=yticklabels)

if save_plots == True:

    plt.savefig(os.path.join(save_directory,"Confusion_Matrix_CNN_I.png"), format="png", bbox_inches="tight", dpi=500)

fpr, tpr, thresholds = roc_curve(Labels_test, CNN_I_prediction_on_data_test.argmax(axis=-1))

ROC_CNN_I_model = []

ROC_CNN_I_model.append(tpr)
ROC_CNN_I_model.append(fpr)

ROC_CNN_I_model_pd = pd.DataFrame(np.array(ROC_CNN_I_model))

ROC_CNN_I_model_pd.to_excel(os.path.join(save_directory,"ROF_CNN_I_model.xlsx"))

TP, FP, TN, FN = perf_measure(Labels_test, CNN_I_prediction_on_data_test.argmax(axis=-1))

Mcc = MCC(TP, FP, TN, FN)

Sens = ((TP)/(TP + FN))*100

Spec = ((TN)/(FP + TN))*100

plt.figure()
plt.plot([0, 1], [0, 1], 'k--',  label = 'Trained Model Performance')
plt.plot([0, 0], [0, 1], 'g', label = 'Ideal Model Performance')
plt.plot([0, 1], [1, 1], 'g')
plt.plot(fpr,tpr, label = 'Trained Model Performance')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()

if save_plots == True:
    plt.savefig(os.path.join(save_directory,"ROC_Curve_CNN_I.png"), format="png", bbox_inches="tight", dpi=500)


matrix_CNN_II = confusion_matrix(Labels_test, CNN_II_prediction_on_data_test.argmax(axis=-1))
plt.figure()
sns.heatmap(matrix_CNN_II, linewidth=0.5, cmap="YlGnBu", annot=True, xticklabels= xticklabels, yticklabels=yticklabels)

if save_plots == True:

    plt.savefig(os.path.join(save_directory,"Confusion_Matrix_CNN_II.png"), format="png", bbox_inches="tight", dpi=500)

fpr, tpr, thresholds = roc_curve(Labels_test, CNN_II_prediction_on_data_test.argmax(axis=-1))

ROC_CNN_II_model = []

ROC_CNN_II_model.append(tpr)
ROC_CNN_II_model.append(fpr)

ROC_CNN_II_model_pd = pd.DataFrame(np.array(ROC_CNN_II_model))

ROC_CNN_II_model_pd.to_excel(os.path.join(save_directory,"ROF_CNN_II_model.xlsx"))

TP, FP, TN, FN = perf_measure(Labels_test, CNN_II_prediction_on_data_test.argmax(axis=-1))

Mcc = MCC(TP, FP, TN, FN)

Sens = ((TP)/(TP + FN))*100

Spec = ((TN)/(FP + TN))*100

plt.figure()
plt.plot([0, 1], [0, 1], 'k--',  label = 'Trained Model Performance')
plt.plot([0, 0], [0, 1], 'g', label = 'Ideal Model Performance')
plt.plot([0, 1], [1, 1], 'g')
plt.plot(fpr,tpr, label = 'Trained Model Performance')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()

if save_plots == True:
    plt.savefig(os.path.join(save_directory,"ROC_Curve_CNN_II.png"), format="png", bbox_inches="tight", dpi=500)

matrix_RNN = confusion_matrix(Labels_test, RNN_prediction_on_data_test.argmax(axis=-1))
plt.figure()
sns.heatmap(matrix_RNN, linewidth=0.5, cmap="YlGnBu", annot=True, xticklabels= xticklabels, yticklabels=yticklabels)

if save_plots == True:

    plt.savefig(os.path.join(save_directory,"Confusion_Matrix_RNN.png"), format="png", bbox_inches="tight", dpi=500)

fpr, tpr, thresholds = roc_curve(Labels_test, RNN_prediction_on_data_test.argmax(axis=-1))

ROC_RNN_model = []

ROC_RNN_model.append(tpr)
ROC_RNN_model.append(fpr)

ROC_RNN_model_pd = pd.DataFrame(np.array(ROC_RNN_model))

ROC_RNN_model_pd.to_excel(os.path.join(save_directory,"ROF_RNN_model.xlsx"))

TP, FP, TN, FN = perf_measure(Labels_test, RNN_prediction_on_data_test.argmax(axis=-1))

Mcc = MCC(TP, FP, TN, FN)

Sens = ((TP)/(TP + FN))*100

Spec = ((TN)/(FP + TN))*100

plt.figure()
plt.plot([0, 1], [0, 1], 'k--',  label = 'Trained Model Performance')
plt.plot([0, 0], [0, 1], 'g', label = 'Ideal Model Performance')
plt.plot([0, 1], [1, 1], 'g')
plt.plot(fpr,tpr, label = 'Trained Model Performance')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()

if save_plots == True:
    plt.savefig(os.path.join(save_directory,"ROC_Curve_RNN.png"), format="png", bbox_inches="tight", dpi=500)

if save_plots == True:
    matrix_pd_CNN_I = pd.DataFrame(matrix_CNN_I)
    matrix_pd_CNN_I.to_excel(os.path.join(save_directory,"Matrix_For_Test_data_CNN_I.xlsx")) 
    
    matrix_pd_CNN_II = pd.DataFrame(matrix_CNN_II)
    matrix_pd_CNN_II.to_excel(os.path.join(save_directory,"Matrix_For_Test_data_CNN_II.xlsx")) 
    
    matrix_pd_RNN = pd.DataFrame(matrix_RNN)
    matrix_pd_RNN.to_excel(os.path.join(save_directory,"Matrix_For_Test_data_RNN.xlsx")) 

#############################################################
"""
import shap

Full_Model_Explainder = shap.GradientExplainer(Full_model, [Val_fkank_image, Val_rake_image,Val_time_series])

Shap_Values = Full_Model_Explainder.shap_values([Val_fkank_image[:1], Val_rake_image[:1], Val_time_series[:1]])

RNN_Shap_Values = Shap_Values[0][2][0]

CNN_I_Shap_Values = Shap_Values[0][0][0]

CNN_II_Shap_Values = Shap_Values[0][1][0]

shap.summary_plot(RNN_Shap_Values, plot_type = 'bar')


RNN_Explainer = shap.GradientExplainer(RNN_submodel, Train_time_series)

RNN_sv = RNN_Explainer(Val_time_series)

RNN_Explainer = shap.DeepExplainer(RNN_submodel, Train_time_series)

shap_values_RNN = RNN_Explainer.shap_values(Train_time_series[:1])

CNN_I_Explainer = shap.GradientExplainer(CNN_I_submodel, Train_flank_image) 
"""

""" Data Collection for XAI purposes """

Full_model_prediction_on_data = Full_model.predict([Train_flank_image, Train_rake_image, Train_time_series])

CNN_I_prediction_on_data = CNN_I_submodel.predict(Train_flank_image)

CNN_II_prediction_on_data = CNN_II_submodel.predict(Train_rake_image)

RNN_prediction_on_data = RNN_submodel.predict(Train_time_series)

print("Model Accuracy: %" + str(100*round(accuracy_score(Train_Labels, Full_model_prediction_on_data.argmax(axis=-1)),4)))

print("CNN_I Model Accuracy: %" + str(100*round(accuracy_score(Train_Labels, CNN_I_prediction_on_data.argmax(axis=-1)),4)))

print("CNN_II Model Accuracy: %" + str(100*round(accuracy_score(Train_Labels, CNN_II_prediction_on_data.argmax(axis=-1)),4)))

print("RNN Model Accuracy: %" + str(100*round(accuracy_score(Train_Labels, RNN_prediction_on_data.argmax(axis=-1)),4)))

matrix_full = confusion_matrix(Train_Labels, Full_model_prediction_on_data.argmax(axis=-1))
plt.figure()
sns.heatmap(matrix_full, linewidth=0.5, cmap="YlGnBu", annot=True, xticklabels= xticklabels, yticklabels=yticklabels)
plt.show()

matrix_CNN_I = confusion_matrix(Train_Labels, CNN_I_prediction_on_data.argmax(axis=-1))
plt.figure()
sns.heatmap(matrix_CNN_I, linewidth=0.5, cmap="YlGnBu", annot=True, xticklabels= xticklabels, yticklabels=yticklabels)
plt.show()

matrix_CNN_II = confusion_matrix(Train_Labels, CNN_II_prediction_on_data.argmax(axis=-1))
plt.figure()
sns.heatmap(matrix_CNN_II, linewidth=0.5, cmap="YlGnBu", annot=True, xticklabels= xticklabels, yticklabels=yticklabels)
plt.show()

matrix_RNN = confusion_matrix(Train_Labels, RNN_prediction_on_data.argmax(axis=-1))
plt.figure()
sns.heatmap(matrix_RNN, linewidth=0.5, cmap="YlGnBu", annot=True, xticklabels= xticklabels, yticklabels=yticklabels)
plt.show()

if save_plots == True:
    matrix_full_pd = pd.DataFrame(matrix_full)
    matrix_full_pd.to_excel(os.path.join(save_directory,"Matrix_For_Train_data_CNN_I.xlsx")) 
    
    matrix_pd_CNN_I = pd.DataFrame(matrix_CNN_I)
    matrix_pd_CNN_I.to_excel(os.path.join(save_directory,"Matrix_For_Train_data_CNN_I.xlsx")) 
    
    matrix_pd_CNN_II = pd.DataFrame(matrix_CNN_II)
    matrix_pd_CNN_II.to_excel(os.path.join(save_directory,"Matrix_For_Train_data_CNN_II.xlsx")) 
    
    matrix_pd_RNN = pd.DataFrame(matrix_RNN)
    matrix_pd_RNN.to_excel(os.path.join(save_directory,"Matrix_For_Train_data_RNN.xlsx"))

""" Extract the cases for XAI """

Full_model_True_index = np.where(Train_Labels == Full_model_prediction_on_data.argmax(axis=-1))

Full_model_False_index = np.where(Train_Labels != Full_model_prediction_on_data.argmax(axis=-1))

CNN_I_model_True_index = np.where(Train_Labels == CNN_I_prediction_on_data.argmax(axis=-1))

CNN_I_model_False_index = np.where(Train_Labels != CNN_I_prediction_on_data.argmax(axis=-1))

CNN_II_model_True_index = np.where(Train_Labels == CNN_II_prediction_on_data.argmax(axis=-1))

CNN_II_model_False_index = np.where(Train_Labels != CNN_II_prediction_on_data.argmax(axis=-1))

RNN_model_True_index = np.where(Train_Labels == RNN_prediction_on_data.argmax(axis=-1))

RNN_model_False_index = np.where(Train_Labels != RNN_prediction_on_data.argmax(axis=-1))

different_cases = []

""" CNN_I """

case_I_CNN_I = np.in1d(Full_model_True_index ,  CNN_I_model_True_index)

case_I_CNN_I = np.where(case_I_CNN_I == True)

if np.any(case_I_CNN_I[0]) == True:
    different_cases.append(Full_model_True_index[0][case_I_CNN_I[0][0]]) 
else:
    print("No case_I_CNN_I")


case_II_CNN_I = np.in1d(Full_model_True_index ,  CNN_I_model_False_index)

case_II_CNN_I = np.where(case_II_CNN_I == True)

if np.any(case_II_CNN_I[0]) == True:
    different_cases.append(Full_model_True_index[0][case_II_CNN_I[0][0]])
else:
    print("No case_II_CNN_I")

case_III_CNN_I = np.in1d(Full_model_False_index ,  CNN_I_model_True_index)

case_III_CNN_I = np.where(case_III_CNN_I == True)

if np.any(case_III_CNN_I[0]) == True:
    different_cases.append(Full_model_False_index[0][case_III_CNN_I[0][0]])
else:
    print("No case_III_CNN_I")

case_IV_CNN_I = np.in1d(Full_model_False_index ,  CNN_I_model_False_index)

case_IV_CNN_I = np.where(case_IV_CNN_I == True)

if np.any(case_IV_CNN_I[0]) == True:
    different_cases.append(Full_model_False_index[0][case_IV_CNN_I[0][0]])
else:
    print("No case_IV_CNN_I")
    
""" CNN_II """

case_I_CNN_II = np.in1d(Full_model_True_index ,  CNN_II_model_True_index)

case_I_CNN_II = np.where(case_I_CNN_II == True)

if np.any(case_I_CNN_II[0]) == True:
    different_cases.append(Full_model_True_index[0][case_I_CNN_II[0][0]])
else:
    print("No case_I_CNN_II")

case_II_CNN_II = np.in1d(Full_model_True_index ,  CNN_II_model_False_index)

case_II_CNN_II = np.where(case_II_CNN_II == True)

if np.any(case_II_CNN_II[0]) == True:
    different_cases.append(Full_model_True_index[0][case_II_CNN_II[0][0]])
else:
    print("No case_II_CNN_II")

case_III_CNN_II = np.in1d(Full_model_False_index ,  CNN_II_model_True_index)

case_III_CNN_II = np.where(case_III_CNN_II == True)

if np.any(case_III_CNN_II[0]) == True:
    different_cases.append(Full_model_False_index[0][case_III_CNN_II[0][0]])
else:
    print("No case_III_CNN_II")

case_IV_CNN_II = np.in1d(Full_model_False_index ,  CNN_II_model_False_index)

case_IV_CNN_II = np.where(case_IV_CNN_II == True)

if np.any(case_IV_CNN_II[0]) == True:
    different_cases.append(Full_model_False_index[0][case_IV_CNN_II[0][0]])
else:
    print("No case_IV_CNN_II")
    
""" RNN """

case_I_RNN = np.in1d(Full_model_True_index ,  RNN_model_True_index)

case_I_RNN = np.where(case_I_RNN == True)

if np.any(case_I_RNN[0]) == True:
    different_cases.append(Full_model_True_index[0][case_I_RNN[0][0]])
else:
    print("No case_I_LSTM")

case_II_RNN = np.in1d(Full_model_True_index ,  RNN_model_False_index)

case_II_RNN = np.where(case_II_RNN == True)

if np.any(case_II_RNN[0]) == True:
    different_cases.append(Full_model_True_index[0][case_II_RNN[0][0]])
else:
    print("No case_II_LSTM")

case_III_RNN = np.in1d(Full_model_False_index ,  RNN_model_True_index)

case_III_RNN = np.where(case_III_RNN == True)

if np.any(case_III_RNN[0]) == True:
    different_cases.append(Full_model_False_index[0][case_III_RNN[0][0]])
else:
    print("No case_III_LSTM")

case_IV_RNN = np.in1d(Full_model_False_index ,  RNN_model_False_index)

case_IV_RNN = np.where(case_IV_RNN == True)

if np.any(case_IV_RNN[0]) == True:
    different_cases.append(Full_model_False_index[0][case_IV_RNN[0][0]])
else:
    print("No case_IV_LSTM")
    
#flank_show = []

#rake_show = []

XAI_flank_image = []

XAI_Rake_image = []

XAI_Time_series = []

XAI_labels = []

#flank_image_to_show = []

#rake_image_to_show = []

different_cases_poped = np.unique(different_cases)

predictions_full_model = []

predictions_CNN_I = []

predictions_CNN_II = []

predictions_RNN = []

true_labels_for_cases = []

for i in range(len(different_cases_poped)):
    
    index = different_cases_poped[i]
    
    predictions_full_model.append(Full_model_prediction_on_data[index])
    
    predictions_CNN_I.append(CNN_I_prediction_on_data[index])
    
    predictions_CNN_II.append(CNN_II_prediction_on_data[index])
    
    predictions_RNN.append(RNN_prediction_on_data[index])
    
    true_labels_for_cases.append(Train_Label[index])
    
predictions_full_model_np = np.array(predictions_full_model)

predictions_full_model_pd = pd.DataFrame(predictions_full_model_np)

predictions_full_model_pd.to_excel(os.path.join(save_directory,"predictions_full_model_pd.xlsx"))

predictions_CNN_I_np = np.array(predictions_CNN_I)

predictions_CNN_I_pd = pd.DataFrame(predictions_CNN_I_np)

predictions_CNN_I_pd.to_excel(os.path.join(save_directory,"predictions_CNN_I_pd.xlsx"))

predictions_CNN_II_np = np.array(predictions_CNN_II)

predictions_CNN_II_pd = pd.DataFrame(predictions_CNN_II_np)

predictions_CNN_II_pd.to_excel(os.path.join(save_directory,"predictions_CNN_II_pd.xlsx"))

predictions_RNN_np = np.array(predictions_RNN)

predictions_RNN_pd = pd.DataFrame(predictions_RNN_np)

predictions_RNN_pd.to_excel(os.path.join(save_directory,"predictions_RNN_pd.xlsx"))

true_labels_for_cases_np = np.array(true_labels_for_cases)

true_labels_for_cases_pd = pd.DataFrame(true_labels_for_cases_np)

true_labels_for_cases_pd.to_excel(os.path.join(save_directory,"true_labels_for_cases_pd.xlsx"))

from tensorflow.keras.applications.vgg16 import preprocess_input

for i in range(len(different_cases_poped)):
    
    index = different_cases_poped[i]
    
    #flank_image_to_show.append(Train_flank_image[index])
    
    #rake_image_to_show.append(Train_rake_image[index])
    
    #flank_show.append(Train_flank_image[index])
    
    #rake_show.append(Train_rake_image[index])
    
    XAI_flank_image.append((Train_flank_image[index]))
                           
    XAI_Rake_image.append((Train_rake_image[index]))
    
    XAI_Time_series.append(Train_time_series[index])
    
    XAI_labels.append(Train_Labels[index])
    
Val_flank_image = np.array(XAI_flank_image)

Val_Rake_image = np.array(XAI_Rake_image)

flank_show = np.copy(Val_flank_image)

rake_show = np.copy(Val_Rake_image)

Val_Time_series = np.array(XAI_Time_series)

val_labels = np.array(XAI_labels)

lower = int(img_size * 0.2)
upper = int(img_size * 0.8)

print("Shapley XAI!")

import shap

tf.keras.backend.set_learning_phase(0)

explainer_lstm = shap.GradientExplainer(RNN_submodel, Val_Time_series)

shap_values_lstm = explainer_lstm.shap_values(Val_Time_series)

shap_values_lstm = np.array(shap_values_lstm)

shap_values_lstm = np.nan_to_num(shap_values_lstm, nan=0)

# Visualize the waterfall plot
#shap.waterfall_plot(shap.Explanation(values=shap_values_lstm[0], data=Val_time_series), max_display=10)
k =  1

for i in range(len(different_cases_poped)):
    
    save_dir = os.path.join(save_directory, 'LSTM_XAI_SHAP_VAL')
    
    if 'LSTM_XAI_SHAP_VAL' not in os.listdir(save_directory):
        
        os.mkdir(os.path.join(save_directory, 'LSTM_XAI_SHAP_VAL')) 
        
        os.chdir(save_dir)
    
    index = different_cases_poped[i]
    
    label = np.argmax(RNN_prediction_on_data[index])
    
    plt.figure()
    plt.subplot(211)
    fig = shap.summary_plot(shap_values_lstm[label][i], plot_type = 'dot', 
                            feature_names = ['ACC_X', 'ACC_Y', 'ACC_Z', 'MEMS Micro', 'Elec Micro', 'Temp', 'Spindle Speed', 'Feed Rate'], 
                            show=False)
    plt.xlabel("Impact on model output")
    plt.subplot(212)
    fig = shap.summary_plot(shap_values_lstm[label][i] * 10, plot_type = 'bar', 
                            feature_names = ['ACC_X', 'ACC_Y', 'ACC_Z', 'MEMS Micro', 'Elec Micro', 'Temp', 'Spindle Speed', 'Feed Rate'],
                            show=False)
    plt.xlabel("Average impact on model output magnitude")
    plt.savefig(str(k) + '_LSTM.png')
    
    plt.figure()
    fig = shap.summary_plot(shap_values_lstm[label][i], plot_type = 'dot', 
                            feature_names = ['ACC_X', 'ACC_Y', 'ACC_Z', 'MEMS Micro', 'Elec Micro', 'Temp', 'Spindle Speed', 'Feed Rate'], 
                            show=False)
    plt.xlabel("Impact on model output")
    plt.savefig(str(k) + '_LSTM_dot.png')
    
    plt.figure()
    fig = shap.summary_plot(shap_values_lstm[label][i] * 10 , plot_type = 'bar', 
                            feature_names = ['ACC_X', 'ACC_Y', 'ACC_Z', 'MEMS Micro', 'Elec Micro', 'Temp', 'Spindle Speed', 'Feed Rate'],
                            show=False)
    plt.savefig(str(k) + '_LSTM_bar.png')
    plt.xlabel("Average impact on model output magnitude")
    
    k += 1
    

os.chdir(save_directory)

print("Shapley of the flank for the first image")

explainer_CNN_I = shap.GradientExplainer(CNN_I_submodel, Val_flank_image)

shap_values_cnn_I = explainer_CNN_I.shap_values(Val_flank_image)

k = 1

for i in range(len(different_cases_poped)):
    
    save_dir = os.path.join(save_directory,'Flank_XAI_SHAP_VAL')
    
    if 'Flank_XAI_SHAP_VAL' not in os.listdir(save_directory):
        
        os.mkdir(os.path.join(save_directory, 'Flank_XAI_SHAP_VAL')) 
        
        os.chdir(save_dir)
        
    index = different_cases_poped[i]
    
    label = np.argmax(CNN_I_prediction_on_data[index])
    
    shap_values_cnn_I_I_I = np.squeeze(shap_values_cnn_I[0][i])  # assuming batch size = 1
    shap_values_cnn_I_I_I /= np.max(np.abs(shap_values_cnn_I_I_I))
    
    shap_values_cnn_I_I_II = np.squeeze(shap_values_cnn_I[1][i])  # assuming batch size = 1
    shap_values_cnn_I_I_II /= np.max(np.abs(shap_values_cnn_I_I_II))
    
    #shap_values_cnn_I_I_I[lower:upper, lower:upper] /= 3
    #shap_values_cnn_I_I_II[lower:upper, lower:upper] /= 3
    
    shap_values_cnn_I_I_I[lower:upper, lower:upper] /= 1.5
    shap_values_cnn_I_I_II[lower:upper, lower:upper] /= 1.5
    
    if label == 0 :
        
        shap_show = shap_values_cnn_I_I_I
        
    elif label == 1:
        
        shap_show = shap_values_cnn_I_I_II
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(flank_show[i])
    plt.subplot(122)
    plt.imshow(shap_show * 3, cmap='jet', alpha=0.6)
    print(label)
    plt.savefig(os.path.join(save_dir,str(k)+"_two_pics_flank_shap.png"), format="png", bbox_inches="tight", dpi=250)
    
    plt.figure()
    plt.imshow(flank_show[i])
    plt.imshow(shap_show * 3, cmap='jet', alpha=0.6)
    plt.savefig(os.path.join(save_dir,str(k)+"_one_pic_flank_shap.png"), format="png", bbox_inches="tight", dpi=100)
    
    k += 1
    

"""
print("Shapley of the flank for the second image")

shap_values_cnn_I_II = shap_values_cnn_I[1]

shap_values_cnn_I_II_I = np.squeeze(shap_values_cnn_I_II[0])  # assuming batch size = 1
shap_values_cnn_I_II_I /= np.max(np.abs(shap_values_cnn_I_II_I))

shap_values_cnn_I_II_II = np.squeeze(shap_values_cnn_I_II[1])  # assuming batch size = 1
shap_values_cnn_I_II_II /= np.max(np.abs(shap_values_cnn_I_II_II))

shap_values_cnn_I_II_I[lower:upper, lower:upper] *= 3
shap_values_cnn_I_II_II[lower:upper, lower:upper] *= 3

plt.figure()
plt.subplot(121)
plt.imshow(shap_values_cnn_I_II_I * 1.5, cmap='jet', alpha=0.6)
plt.axis('off')
plt.subplot(122)
plt.imshow(shap_values_cnn_I_II_II * 1.5, cmap='jet', alpha=0.6)
plt.axis('off')
plt.show()

"""

print("Shapley of the rake for the first image")

explainer_CNN_II = shap.GradientExplainer(CNN_II_submodel, Val_Rake_image)

shap_values_cnn_II = explainer_CNN_II.shap_values(Val_Rake_image)

k = 1

for i in range(len(different_cases_poped)):
    
    save_dir = os.path.join(save_directory,'Rake_XAI_SHAP_VAL')
    
    if 'Rake_XAI_SHAP_VAL' not in os.listdir(save_directory):
        
        os.mkdir(os.path.join(save_directory, 'Rake_XAI_SHAP_VAL')) 
        
        os.chdir(save_dir)
    
    index = different_cases_poped[i]
    
    label = np.argmax(CNN_I_prediction_on_data[index])
    
    shap_values_cnn_II_I_I = np.squeeze(shap_values_cnn_II[0][i])  # assuming batch size = 1
    shap_values_cnn_II_I_I /= np.max(np.abs(shap_values_cnn_II_I_I))
    
    shap_values_cnn_II_I_II = np.squeeze(shap_values_cnn_II[1][i])  # assuming batch size = 1
    shap_values_cnn_II_I_II /= np.max(np.abs(shap_values_cnn_II_I_II))
    
    #shap_values_cnn_II_I_I[lower:upper, lower:upper] /= 3
    #shap_values_cnn_II_I_II[lower:upper, lower:upper] /= 3
    
    shap_values_cnn_II_I_I[lower:upper, lower:upper] /= 1.5
    shap_values_cnn_II_I_II[lower:upper, lower:upper] /= 1.5
    
    if label == 0 :
        
        shap_show = shap_values_cnn_II_I_I
        
    elif label == 1:
        
        shap_show = shap_values_cnn_II_I_II
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(rake_show[i])
    plt.subplot(122)
    plt.imshow(np.abs(shap_show) * 3, cmap='hot', alpha=0.6)
    print(label)
    plt.savefig(os.path.join(save_dir,str(k)+"_two_pics_rake_shap.png"), format="png", bbox_inches="tight", dpi=250)
    
    plt.figure()
    plt.imshow(rake_show[i])
    plt.imshow(np.abs(shap_show) * 3, cmap='hot', alpha=0.6)
    plt.savefig(os.path.join(save_dir,str(k)+"_one_pic_rake_shap.png"), format="png", bbox_inches="tight", dpi=100)
    
    k += 1

"""
print("Shapley of the rake for the second image")

shap_values_cnn_II_II = shap_values_cnn_II[1]

shap_values_cnn_II_II_I = np.squeeze(shap_values_cnn_II_II[0])  # assuming batch size = 1
shap_values_cnn_II_II_I /= np.max(np.abs(shap_values_cnn_II_II_I))

shap_values_cnn_II_II_II = np.squeeze(shap_values_cnn_II_II[1])  # assuming batch size = 1
shap_values_cnn_II_II_II /= np.max(np.abs(shap_values_cnn_II_II_II))

shap_values_cnn_II_II_I[lower:upper, lower:upper] *= 3
shap_values_cnn_II_II_II[lower:upper, lower:upper] *= 3

plt.figure()
plt.subplot(121)
plt.imshow(shap_values_cnn_II_II_I * 1.5, cmap='jet', alpha=0.6)
plt.axis('off')
plt.subplot(122)
plt.imshow(shap_values_cnn_II_II_II * 1.5, cmap='jet', alpha=0.6)
plt.axis('off')
plt.show()
"""

print("Shapley Gradient XAI for latent space")

latent_space = Latent_Space.predict([Val_flank_image, Val_Rake_image, Val_Time_series])

prediction_full_model = Full_model.predict([Val_flank_image, Val_Rake_image, Val_Time_series])

prediction_classifier = Classifier.predict(latent_space)

prediction_CNN_I = CNN_I_submodel.predict(Val_flank_image)

prediction_CNN_II = CNN_II_submodel.predict(Val_Rake_image)

prediction_LSTM = RNN_submodel.predict(Val_Time_series)

print("Real Labels:")
print(val_labels)
print("--------")

print("CNN_I predictions:")
print(prediction_CNN_I)
print("--------")

print("CNN_II predictions:")
print(prediction_CNN_II)
print("--------")

print("LSTM predictions:")
print(prediction_LSTM)
print("--------")

print("Full model predictions:")
print(prediction_full_model)
print("--------")

print("Classifier predictions:")
print(prediction_classifier)
print("--------")

latent_space_expanded = np.expand_dims(latent_space, axis = 1)

shapley_classifier = shap.GradientExplainer(Classifier, latent_space_expanded)

shap_values_latent_space = shapley_classifier.shap_values(latent_space)

shap.summary_plot(shap_values_latent_space[0], plot_type = 'bar', max_display=20)

k = 1

shap_latent_real = []

shap_latent_abs = []

for i in range(len(different_cases_poped)):
    
    index_two = val_labels[i]
    
    shap_latent_values_I = shap_values_latent_space[index_two][i] * 100
    
    shap_latent_real.append(shap_latent_values_I)
    
    shap_latent_values_I_abs = np.abs(shap_latent_values_I)
    
    shap_latent_abs.append(shap_latent_values_I)
    
    CNN_I_latent = shap_latent_values_I[0:feature_neurons]
    
    CNN_II_latent = shap_latent_values_I[feature_neurons:feature_neurons*2]
    
    LSTM_latent = shap_latent_values_I[feature_neurons*2:feature_neurons*3]
    
    CNN_I_latent_abs = shap_latent_values_I_abs[0:feature_neurons]
    
    CNN_II_latent_abs = shap_latent_values_I_abs[feature_neurons:feature_neurons*2]
    
    LSTM_latent_abs = shap_latent_values_I_abs[feature_neurons*2:feature_neurons*3]
    
    CNN_I_latent_mean_abs = np.mean(CNN_I_latent_abs)
    
    CNN_II_latent_mean_abs = np.mean(CNN_II_latent_abs)
    
    LSTM_latent_mean_abs = np.mean(LSTM_latent_abs)
    
    CNN_I_latent_mean = np.mean(CNN_I_latent)
    
    CNN_II_latent_mean = np.mean(CNN_II_latent)
    
    LSTM_latent_mean = np.mean(LSTM_latent)
    
    b_axes = np.array([CNN_I_latent_mean_abs, CNN_II_latent_mean_abs, LSTM_latent_mean_abs])
    
    CNN_I_latent_std = np.std(CNN_I_latent_abs)
    
    CNN_II_latent_std = np.std(CNN_II_latent_abs)
    
    LSTM_latent_std = np.std(LSTM_latent_abs)
    
    CNN_I_pdf = normal_dist(np.sort(CNN_I_latent) , CNN_I_latent_mean , CNN_I_latent_std)
    
    CNN_II_pdf = normal_dist(np.sort(CNN_II_latent) , CNN_II_latent_mean , CNN_II_latent_std)
    
    LSTM_pdf = normal_dist(np.sort(LSTM_latent) , LSTM_latent_mean , LSTM_latent_std)
    
    plt.figure()
    plt.plot(np.sort(CNN_I_latent), CNN_I_pdf)
    plt.plot(np.sort(CNN_II_latent), CNN_II_pdf)
    plt.plot(np.sort(LSTM_latent), LSTM_pdf)
    
    plt.figure()
    
    a_axes = np.array([1,2,3])
    
    plt.bar(a_axes, b_axes)
    
save_dir = os.path.join(save_directory, 'SHAP_Method_II_VAL')

if 'SHAP_Method_II_VAL' not in os.listdir(save_directory):
    
    os.mkdir(os.path.join(save_directory, 'SHAP_Method_II_VAL')) 
    
    os.chdir(save_dir)

shap_latent_real_pd = pd.DataFrame(np.array(shap_latent_real))

shap_latent_real_pd.to_excel(os.path.join(save_dir,"shap_latent_real_pd.xlsx"))

shap_latent_abs_pd = pd.DataFrame(np.array(shap_latent_abs))

shap_latent_abs_pd.to_excel(os.path.join(save_dir,"shap_latent_abs_pd.xlsx"))

print("Saliency Map!")

from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

replace2linear = ReplaceToLinear()

from tf_keras_vis.utils.scores import CategoricalScore

from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize

saliency_CNN_I = Saliency(CNN_I_submodel,
                    model_modifier=replace2linear)

saliency_CNN_II = Saliency(CNN_II_submodel,
                    model_modifier=replace2linear)

saliency_LSTM = Saliency(RNN_submodel,
                         model_modifier=replace2linear)


"""
saliency_map_CNN_I = saliency_CNN_I(score, [Val_flank_image,Val_Rake_image,Val_time_series])

saliency_map_CNN_I[0] = normalize(saliency_map_CNN_I[0])

saliency_map_CNN_I[1] = normalize(saliency_map_CNN_I[1])
"""

saliency_map_CNN_I_list = []

saliency_map_CNN_II_list = []


for i in range(len(different_cases_poped)):
    
    Val_flank_saliency = np.expand_dims(Val_flank_image[i], axis = 0)
    
    print(i)
    
    index = different_cases_poped[i]
    
    score_instance = np.argmax(CNN_I_prediction_on_data[index]) 
    
    score = CategoricalScore([score_instance])

    saliency_map_CNN_I_instance = saliency_CNN_I(score, 
                                        Val_flank_saliency,
                                        smooth_samples=100, # The number of calculating gradients iterations.
                                        smooth_noise=0.05)
    
    saliency_map_CNN_I_list.append(saliency_map_CNN_I_instance)
    
    

for i in range(len(different_cases_poped)):
    
    Val_rake_saliency = np.expand_dims(Val_Rake_image[i], axis = 0)
    
    print(i)
    
    index = different_cases_poped[i]
    
    score_instance = np.argmax(CNN_II_prediction_on_data[index]) 
    
    score = CategoricalScore([score_instance])

    saliency_map_CNN_II_instance = saliency_CNN_II(score, 
                                            Val_rake_saliency,
                                            smooth_samples=100, # The number of calculating gradients iterations.
                                            smooth_noise=0.05)
    
    saliency_map_CNN_II_list.append(saliency_map_CNN_II_instance)
    
saliency_map_CNN_I = normalize(np.array(saliency_map_CNN_I_list))

saliency_map_CNN_II = normalize(np.array(saliency_map_CNN_II_list))

saliency_map_CNN_I[lower:upper, lower:upper] *= 2
saliency_map_CNN_II[lower:upper, lower:upper] *= 2

saliency_map_CNN_I = np.squeeze(saliency_map_CNN_I)

saliency_map_CNN_II = np.squeeze(saliency_map_CNN_II)

num = saliency_map_CNN_I.shape[0]

saliency_map_CNN_I = saliency_map_CNN_I.reshape(num, 224, 224)

saliency_map_CNN_II = saliency_map_CNN_II.reshape(num, 224, 224)

k = 1

for i in range(len(different_cases_poped)):
    
    save_dir = os.path.join(save_directory, 'Flank_XAI_Saliency_VAL')
    
    if 'Flank_XAI_Saliency_VAL' not in os.listdir(save_directory):
        
        os.mkdir(os.path.join(save_directory, 'Flank_XAI_Saliency_VAL')) 
        
        os.chdir(save_dir)
    
    index = different_cases_poped[i]
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(flank_show[i])
    plt.subplot(122)
    plt.imshow(saliency_map_CNN_I[i], cmap='hot')
    plt.savefig(os.path.join(save_dir,str(k)+"_two_pics_flank_saliency.png"), format="png", bbox_inches="tight", dpi=250)
    
    k += 1

k = 1

for i in range(len(different_cases_poped)):
    
    save_dir = os.path.join(save_directory, 'Rake_XAI_Saliency_VAL')
    
    if 'Rake_XAI_Saliency_VAL' not in os.listdir(save_directory):
        
        os.mkdir(os.path.join(save_directory, 'Rake_XAI_Saliency_VAL')) 
        
        os.chdir(save_dir)
    
    index = different_cases_poped[i]
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(rake_show[i])
    plt.subplot(122)
    plt.imshow(saliency_map_CNN_II[i], cmap='hot')
    
    plt.savefig(os.path.join(save_dir,str(k)+"_two_pics_rake_saliency.png"), format="png", bbox_inches="tight", dpi=250)
    
    k += 1


""" XAI for the first part """

CNN_I_prediction_on_validation_data = CNN_I_submodel.predict(Val_fkank_image)

CNN_II_prediction_on_validation_data = CNN_II_submodel.predict(Val_rake_image)

RNN_prediction_on_validation_data = RNN_submodel.predict(Val_time_series)

Full_model_prediction_on_validation_data = Full_model.predict([Val_fkank_image, Val_rake_image, Val_time_series])

print("Full Model Accuracy: %" + str(100*round(accuracy_score(Val_Label.argmax(axis=-1), Full_model_prediction_on_validation_data.argmax(axis=-1)),4)))

print("CNN_I Model Accuracy: %" + str(100*round(accuracy_score(Val_Label.argmax(axis=-1), CNN_I_prediction_on_validation_data.argmax(axis=-1)),4)))

print("CNN_II Model Accuracy: %" + str(100*round(accuracy_score(Val_Label.argmax(axis=-1), CNN_II_prediction_on_validation_data.argmax(axis=-1)),4)))

print("RNN Model Accuracy: %" + str(100*round(accuracy_score(Val_Label.argmax(axis=-1), RNN_prediction_on_validation_data.argmax(axis=-1)),4)))

indeces = np.random.choice(Val_fkank_image.shape[0], 4)

Data_val_XAI_flank = []

labels_of_flank_val = []

Data_val_XAI_rake = []

labels_of_rake_val = []

Data_val_XAI_time = []

labels_of_time_val = []

labels_of_val_full_model = []

real_labels_val = []

for i in range(indeces.shape[0]):
    
    index = indeces[i]
    
    Data_val_XAI_flank.append(Val_fkank_image[index])
    
    labels_of_flank_val.append(CNN_I_prediction_on_validation_data[index])
    
    Data_val_XAI_rake.append(Val_rake_image[index])
    
    labels_of_rake_val.append(CNN_II_prediction_on_validation_data[index])
    
    Data_val_XAI_time.append(Val_time_series[index])
    
    labels_of_time_val.append(RNN_prediction_on_validation_data[index])
    
    labels_of_val_full_model.append(Full_model_prediction_on_validation_data[index])
    
    real_labels_val.append(Val_Label[index])
    
Data_val_XAI_flank_np = np.array(Data_val_XAI_flank)

flank_val_show_XAI = np.copy(Data_val_XAI_flank_np)

labels_of_flank_val_np = np.array(labels_of_flank_val)

labels_of_flank_val_pd = pd.DataFrame(labels_of_flank_val_np)

labels_of_flank_val_pd.to_excel(os.path.join(save_directory, "labels_of_flank_val_pd_case_I.xlsx"))

Data_val_XAI_rake_np = np.array(Data_val_XAI_rake)

rake_val_show_XAI = np.copy(Data_val_XAI_rake_np)

labels_of_rake_val_np = np.array(labels_of_rake_val)

labels_of_rake_val_pd = pd.DataFrame(labels_of_rake_val_np)

labels_of_rake_val_pd.to_excel(os.path.join(save_directory, "labels_of_rake_val_pd_case_I.xlsx"))

Data_val_XAI_time_np = np.array(Data_val_XAI_time)

labels_of_time_val_np = np.array(labels_of_time_val)

labels_of_time_val_pd = pd.DataFrame(labels_of_time_val_np)

labels_of_time_val_pd.to_excel(os.path.join(save_directory, "labels_of_time_val_pd_case_I.xlsx"))

labels_of_val_full_model_np = np.array(labels_of_val_full_model)

labels_of_val_full_model_pd = pd.DataFrame(labels_of_val_full_model_np)

labels_of_val_full_model_pd.to_excel(os.path.join(save_directory, "labels_of_val_full_model_pd_case_I.xlsx"))

real_labels_val_np = np.array(real_labels_val)

real_labels_val_pd = pd.DataFrame(real_labels_val_np)

real_labels_val_pd.to_excel(os.path.join(save_directory, "real_labels_val_pd_case_I.xlsx"))

""" XAI RNN """

explainer_lstm_val = shap.GradientExplainer(RNN_submodel, Data_val_XAI_time_np)

shap_values_lstm_val = explainer_lstm_val.shap_values(Data_val_XAI_time_np)

shap_values_lstm_val = np.array(shap_values_lstm_val)

shap_values_lstm_val = np.nan_to_num(shap_values_lstm_val, nan=0)

# Visualize the waterfall plot
#shap.waterfall_plot(shap.Explanation(values=shap_values_lstm[0], data=Val_time_series), max_display=10)
k =  1

for i in range(indeces.shape[0]):
    
    save_dir = os.path.join(save_directory, 'LSTM_XAI_SHAP_TEST')
    
    if 'LSTM_XAI_SHAP_TEST' not in os.listdir(save_directory):
        
        os.mkdir(os.path.join(save_directory, 'LSTM_XAI_SHAP_TEST')) 
        
        os.chdir(save_dir)
    
    os.chdir(save_dir)
    
    index = indeces[i]
    
    label = np.argmax(RNN_prediction_on_validation_data[index])
    
    plt.figure()
    plt.subplot(211)
    fig = shap.summary_plot(shap_values_lstm_val[label][i], plot_type = 'dot', 
                            feature_names = ['ACC_X', 'ACC_Y', 'ACC_Z', 'MEMS Micro', 'Elec Micro', 'Temp', 'Spindle Speed', 'Feed Rate'], 
                            show=False)
    plt.xlabel("Impact on model output")
    plt.subplot(212)
    fig = shap.summary_plot(shap_values_lstm_val[label][i] * 10, plot_type = 'bar', 
                            feature_names = ['ACC_X', 'ACC_Y', 'ACC_Z', 'MEMS Micro', 'Elec Micro', 'Temp', 'Spindle Speed', 'Feed Rate'],
                            show=False)
    plt.xlabel("Average impact on model output magnitude")
    plt.savefig(str(k) + '_LSTM_case_I.png')
    
    plt.figure()
    fig = shap.summary_plot(shap_values_lstm_val[label][i], plot_type = 'dot', 
                            feature_names = ['ACC_X', 'ACC_Y', 'ACC_Z', 'MEMS Micro', 'Elec Micro', 'Temp', 'Spindle Speed', 'Feed Rate'], 
                            show=False)
    plt.xlabel("Impact on model output")
    plt.savefig(str(k) + '_LSTM_dot_case_I.png')
    
    plt.figure()
    fig = shap.summary_plot(shap_values_lstm_val[label][i] * 10 , plot_type = 'bar', 
                            feature_names = ['ACC_X', 'ACC_Y', 'ACC_Z', 'MEMS Micro', 'Elec Micro', 'Temp', 'Spindle Speed', 'Feed Rate'],
                            show=False)
    plt.savefig(str(k) + '_LSTM_bar_case_I.png')
    plt.xlabel("Average impact on model output magnitude")
    
    k += 1
    
os.chdir(save_directory)

""" XAI RNN """

explainer_lstm_val = shap.GradientExplainer(RNN_submodel, time_series_test)

shap_values_lstm_val = explainer_lstm_val.shap_values(time_series_test)

shap_values_lstm_val = np.array(shap_values_lstm_val)

shap_values_lstm_val = np.nan_to_num(shap_values_lstm_val, nan=0)

# Visualize the waterfall plot
#shap.waterfall_plot(shap.Explanation(values=shap_values_lstm[0], data=Val_time_series), max_display=10)
k =  1

mean_shap_lstm = np.zeros([Labels_test.shape[0],8])

for i in range(Labels_test.shape[0]):
    
    pred_label = np.argmax(RNN_prediction_on_data_test[i])
    
    shap_lstm_instance = shap_values_lstm_val[pred_label][i]
    
    shap_lstm_instance_mean = np.mean(shap_lstm_instance, 0)
    
    mean_shap_lstm[i,:] = shap_lstm_instance_mean
    
    save_dir = os.path.join(save_directory, 'LSTM_XAI_SHAP_FULL')
    
    if 'LSTM_XAI_SHAP_FULL' not in os.listdir(save_directory):
        
        os.mkdir(os.path.join(save_directory, 'LSTM_XAI_SHAP_FULL')) 
        
        os.chdir(save_dir)
    
    os.chdir(save_dir)
    
    mean_shap_lstm_pd = pd.DataFrame(mean_shap_lstm)
    
    mean_shap_lstm_pd.to_excel(os.path.join(save_dir,"LSTM_XAI_SHAP_FULL.xlsx"))
    
os.chdir(save_directory)
    
""" XAI Flank """

explainer_CNN_I_val = shap.GradientExplainer(CNN_I_submodel, Data_val_XAI_flank_np)

shap_values_cnn_I_val = explainer_CNN_I_val.shap_values(Data_val_XAI_flank_np)

k = 1

for i in range(indeces.shape[0]):
    
    save_dir = os.path.join(save_directory, 'Flank_XAI_SHAP_TEST')
    
    if 'Flank_XAI_SHAP_TEST' not in os.listdir(save_directory):
        
        os.mkdir(os.path.join(save_directory, 'Flank_XAI_SHAP_TEST')) 
        
        os.chdir(save_dir)
    
    index = indeces[i]
    
    label = np.argmax(CNN_I_prediction_on_validation_data[index])
    
    shap_values_cnn_I_I_I = np.squeeze(shap_values_cnn_I_val[0][i])  # assuming batch size = 1
    shap_values_cnn_I_I_I /= np.max(np.abs(shap_values_cnn_I_I_I))
    
    shap_values_cnn_I_I_II = np.squeeze(shap_values_cnn_I_val[1][i])  # assuming batch size = 1
    shap_values_cnn_I_I_II /= np.max(np.abs(shap_values_cnn_I_I_II))
    
    #shap_values_cnn_I_I_I[lower:upper, lower:upper] /= 3
    #shap_values_cnn_I_I_II[lower:upper, lower:upper] /= 3
    
    #shap_values_cnn_I_I_I[lower:upper, lower:upper] *= 1.5
    #shap_values_cnn_I_I_II[lower:upper, lower:upper] *= 1.5
    
    if label == 0 :
        
        shap_show = shap_values_cnn_I_I_I
        
    elif label == 1:
        
        shap_show = shap_values_cnn_I_I_II
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(flank_val_show_XAI[i])
    plt.subplot(122)
    plt.imshow(shap_show * 3, cmap='jet', alpha=0.6)
    print(label)
    plt.savefig(os.path.join(save_dir,str(k)+"_two_pics_flank_shap_case_I.png"), format="png", bbox_inches="tight", dpi=250)
    
    plt.figure()
    plt.imshow(flank_val_show_XAI[i])
    plt.imshow(shap_show * 3, cmap='jet', alpha=0.6)
    plt.savefig(os.path.join(save_dir,str(k)+"_one_pic_flank_shap_case_I.png"), format="png", bbox_inches="tight", dpi=100)
    
    k += 1
    
""" XAI Rake """

explainer_CNN_II_val = shap.GradientExplainer(CNN_II_submodel, Data_val_XAI_rake_np)

shap_values_cnn_II_val = explainer_CNN_II_val.shap_values(Data_val_XAI_rake_np)

k = 1

for i in range(indeces.shape[0]):
    
    save_dir = os.path.join(save_directory, 'Rake_XAI_SHAP_TEST')
    
    if 'Rake_XAI_SHAP_TEST' not in os.listdir(save_directory):
        
        os.mkdir(os.path.join(save_directory, 'Rake_XAI_SHAP_TEST')) 
        
        os.chdir(save_dir)
    
    index = indeces[i]
    
    label = np.argmax(CNN_II_prediction_on_validation_data[index])
    
    shap_values_cnn_II_I_I = np.squeeze(shap_values_cnn_II_val[0][i])  # assuming batch size = 1
    shap_values_cnn_II_I_I /= np.max(np.abs(shap_values_cnn_II_I_I))
    
    shap_values_cnn_II_I_II = np.squeeze(shap_values_cnn_II_val[1][i])  # assuming batch size = 1
    shap_values_cnn_II_I_II /= np.max(np.abs(shap_values_cnn_II_I_II))
    
    #shap_values_cnn_II_I_I[lower:upper, lower:upper] /= 3
    #shap_values_cnn_II_I_II[lower:upper, lower:upper] /= 3
    
    #shap_values_cnn_II_I_I[lower:upper, lower:upper] *= 1.5
    #shap_values_cnn_II_I_II[lower:upper, lower:upper] *= 1.5
    
    if label == 0 :
        
        shap_show = shap_values_cnn_II_I_I
        
    elif label == 1:
        
        shap_show = shap_values_cnn_II_I_II
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(rake_val_show_XAI[i])
    plt.subplot(122)
    plt.imshow(np.abs(shap_show) * 3, cmap='hot', alpha=0.6)
    print(label)
    plt.savefig(os.path.join(save_dir,str(k)+"_two_pics_rake_shap_case_I.png"), format="png", bbox_inches="tight", dpi=250)
    
    plt.figure()
    plt.imshow(rake_val_show_XAI[i])
    plt.imshow(np.abs(shap_show) * 3, cmap='hot', alpha=0.6)
    plt.savefig(os.path.join(save_dir,str(k)+"_one_pic_rake_shap_case_I.png"), format="png", bbox_inches="tight", dpi=100)
    
    k += 1

""" XAI Saliency Map """

saliency_map_CNN_I_list_val = []

saliency_map_CNN_II_list_val = []


for i in range(indeces.shape[0]):
    
    Val_flank_saliency = np.expand_dims(Data_val_XAI_flank_np[i], axis = 0)
    
    print(i)
    
    index = indeces[i]
    
    score_instance = np.argmax(CNN_I_prediction_on_validation_data[index]) 
    
    score = CategoricalScore([score_instance])

    saliency_map_CNN_I_instance = saliency_CNN_I(score, 
                                        Val_flank_saliency,
                                        smooth_samples=100, # The number of calculating gradients iterations.
                                        smooth_noise=0.05)
    
    saliency_map_CNN_I_list_val.append(saliency_map_CNN_I_instance)
    
    

for i in range(indeces.shape[0]):
    
    Val_rake_saliency = np.expand_dims(Data_val_XAI_rake_np[i], axis = 0)
    
    print(i)
    
    index = indeces[i]
    
    score_instance = np.argmax(CNN_II_prediction_on_validation_data[index]) 
    
    score = CategoricalScore([score_instance])

    saliency_map_CNN_II_instance = saliency_CNN_II(score, 
                                            Val_rake_saliency,
                                            smooth_samples=100, # The number of calculating gradients iterations.
                                            smooth_noise=0.05)
    
    saliency_map_CNN_II_list_val.append(saliency_map_CNN_II_instance)
    
saliency_map_CNN_I = normalize(np.array(saliency_map_CNN_I_list_val))

saliency_map_CNN_II = normalize(np.array(saliency_map_CNN_II_list_val))

saliency_map_CNN_I[lower:upper, lower:upper] *= 2
saliency_map_CNN_II[lower:upper, lower:upper] *= 2

saliency_map_CNN_I = np.squeeze(saliency_map_CNN_I)

saliency_map_CNN_II = np.squeeze(saliency_map_CNN_II)

saliency_map_CNN_I = saliency_map_CNN_I.reshape(4, 224, 224)

saliency_map_CNN_II = saliency_map_CNN_II.reshape(4, 224, 224)

k = 1

for i in range(indeces.shape[0]):
    
    save_dir = os.path.join(save_directory, 'Flank_XAI_Saliency_TEST')
    
    if 'Flank_XAI_Saliency_TEST' not in os.listdir(save_directory):
        
        os.mkdir(os.path.join(save_directory, 'Flank_XAI_Saliency_TEST')) 
        
        os.chdir(save_dir)
    
    index = indeces[i]
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(flank_val_show_XAI[i])
    plt.subplot(122)
    plt.imshow(saliency_map_CNN_I[i], cmap='hot')
    plt.savefig(os.path.join(save_dir,str(k)+"_two_pics_flank_saliency_Case_I.png"), format="png", bbox_inches="tight", dpi=250)
    
    k += 1

k = 1

for i in range(indeces.shape[0]):
    
    save_dir = os.path.join(save_directory, 'Rake_XAI_Saliency_TEST')
    
    if 'Rake_XAI_Saliency_TEST' not in os.listdir(save_directory):
        
        os.mkdir(os.path.join(save_directory, 'Rake_XAI_Saliency_TEST')) 
        
        os.chdir(save_dir)
    
    index = indeces[i]
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(rake_val_show_XAI[i])
    plt.subplot(122)
    plt.imshow(saliency_map_CNN_II[i], cmap='hot')
    
    plt.savefig(os.path.join(save_dir,str(k)+"_two_pics_rake_saliency_Case_I.png"), format="png", bbox_inches="tight", dpi=250)
    
    k += 1
    
" Shapley Gradient XAI for latent space "

latent_space_val = Latent_Space.predict([Data_val_XAI_flank_np, Data_val_XAI_rake_np, Data_val_XAI_time_np])

prediction_full_model = Full_model.predict([Data_val_XAI_flank_np, Data_val_XAI_rake_np, Data_val_XAI_time_np])

prediction_classifier = Classifier.predict(latent_space_val)

prediction_CNN_I = CNN_I_submodel.predict(Data_val_XAI_flank_np)

prediction_CNN_II = CNN_II_submodel.predict(Data_val_XAI_rake_np)

prediction_LSTM = RNN_submodel.predict(Data_val_XAI_time_np)

print("Real Labels:")
print(val_labels)
print("--------")

print("CNN_I predictions:")
print(prediction_CNN_I)
print("--------")

print("CNN_II predictions:")
print(prediction_CNN_II)
print("--------")

print("LSTM predictions:")
print(prediction_LSTM)
print("--------")

print("Full model predictions:")
print(prediction_full_model)
print("--------")

print("Classifier predictions:")
print(prediction_classifier)
print("--------")

latent_space_expanded_val = np.expand_dims(latent_space_val, axis = 1)

shapley_classifier = shap.GradientExplainer(Classifier, latent_space_expanded_val)

shap_values_latent_space = shapley_classifier.shap_values(latent_space_val)

shap.summary_plot(shap_values_latent_space[0], plot_type = 'bar', max_display=20)

k = 1

shap_latent_real_val = []

shap_latent_abs_val = []

for i in range(len(different_cases_poped)):
    
    index_two = val_labels[i]
    
    shap_latent_values_I = shap_values_latent_space[index_two][i] * 10
    
    shap_latent_real_val.append(shap_latent_values_I)
    
    shap_latent_values_I_abs = np.abs(shap_latent_values_I)
    
    shap_latent_abs_val.append(shap_latent_values_I)
    
    CNN_I_latent = shap_latent_values_I[0:feature_neurons]
    
    CNN_II_latent = shap_latent_values_I[feature_neurons:feature_neurons*2]
    
    LSTM_latent = shap_latent_values_I[feature_neurons*2:feature_neurons*3]
    
    CNN_I_latent_abs = shap_latent_values_I_abs[0:feature_neurons]
    
    CNN_II_latent_abs = shap_latent_values_I_abs[feature_neurons:feature_neurons*2]
    
    LSTM_latent_abs = shap_latent_values_I_abs[feature_neurons*2:feature_neurons*3]
    
    CNN_I_latent_mean_abs = np.mean(CNN_I_latent_abs)
    
    CNN_II_latent_mean_abs = np.mean(CNN_II_latent_abs)
    
    LSTM_latent_mean_abs = np.mean(LSTM_latent_abs)
    
    CNN_I_latent_mean = np.mean(CNN_I_latent)
    
    CNN_II_latent_mean = np.mean(CNN_II_latent)
    
    LSTM_latent_mean = np.mean(LSTM_latent)
    
    b_axes = np.array([CNN_I_latent_mean_abs, CNN_II_latent_mean_abs, LSTM_latent_mean_abs])
    
    CNN_I_latent_std = np.std(CNN_I_latent_abs)
    
    CNN_II_latent_std = np.std(CNN_II_latent_abs)
    
    LSTM_latent_std = np.std(LSTM_latent_abs)
    
    CNN_I_pdf = normal_dist(np.sort(CNN_I_latent) , CNN_I_latent_mean , CNN_I_latent_std)
    
    CNN_II_pdf = normal_dist(np.sort(CNN_II_latent) , CNN_II_latent_mean , CNN_II_latent_std)
    
    LSTM_pdf = normal_dist(np.sort(LSTM_latent) , LSTM_latent_mean , LSTM_latent_std)
    
    plt.figure()
    plt.plot(np.sort(CNN_I_latent), CNN_I_pdf)
    plt.plot(np.sort(CNN_II_latent), CNN_II_pdf)
    plt.plot(np.sort(LSTM_latent), LSTM_pdf)
    
    plt.figure()
    
    a_axes = np.array([1,2,3])
    
    plt.bar(a_axes, b_axes)
    
    
save_dir = os.path.join(save_directory, 'SHAP_Method_II_TEST')

if 'SHAP_Method_II_TEST' not in os.listdir(save_directory):
    
    os.mkdir(os.path.join(save_directory, 'SHAP_Method_II_TEST')) 
    
    os.chdir(save_dir)
    
shap_latent_real_val_pd = pd.DataFrame(np.array(shap_latent_real_val))

shap_latent_real_val_pd.to_excel(os.path.join(save_dir,"shap_latent_real_val_pd.xlsx"))

shap_latent_abs_val_pd = pd.DataFrame(np.array(shap_latent_abs_val))

shap_latent_abs_val_pd.to_excel(os.path.join(save_dir,"shap_latent_abs_val_pd.xlsx"))

" Shapley Gradient XAI for latent space "

latent_space_test_full = Latent_Space.predict([flank_image_test, rake_image_test, time_series_test])

prediction_classifier_test_full = Classifier.predict(latent_space_test_full)

latent_space_test_full_expanded = np.expand_dims(latent_space_test_full, axis = 1)

shapley_classifier = shap.GradientExplainer(Classifier, latent_space_test_full_expanded)

shap_values_latent_space_test_full = shapley_classifier.shap_values(latent_space_test_full)

shap.summary_plot(shap_values_latent_space_test_full[0], plot_type = 'bar', max_display=20)

k = 1

shap_latent_real_val = []

shap_latent_abs_val = []

for i in range(len(prediction_classifier_test_full)):
    
    index_two = np.argmax(prediction_classifier_test_full[i])
    
    shap_latent_values_I = shap_values_latent_space_test_full[index_two][i] * 10
    
    shap_latent_real_val.append(shap_latent_values_I)
    
    shap_latent_values_I_abs = np.abs(shap_latent_values_I)
    
    shap_latent_abs_val.append(shap_latent_values_I)
    
    CNN_I_latent = shap_latent_values_I[0:feature_neurons]
    
    CNN_II_latent = shap_latent_values_I[feature_neurons:feature_neurons*2]
    
    LSTM_latent = shap_latent_values_I[feature_neurons*2:feature_neurons*3]
    
    CNN_I_latent_abs = shap_latent_values_I_abs[0:feature_neurons]
    
    CNN_II_latent_abs = shap_latent_values_I_abs[feature_neurons:feature_neurons*2]
    
    LSTM_latent_abs = shap_latent_values_I_abs[feature_neurons*2:feature_neurons*3]
    
    CNN_I_latent_mean_abs = np.mean(CNN_I_latent_abs)
    
    CNN_II_latent_mean_abs = np.mean(CNN_II_latent_abs)
    
    LSTM_latent_mean_abs = np.mean(LSTM_latent_abs)
    
    CNN_I_latent_mean = np.mean(CNN_I_latent)
    
    CNN_II_latent_mean = np.mean(CNN_II_latent)
    
    LSTM_latent_mean = np.mean(LSTM_latent)
    
    b_axes = np.array([CNN_I_latent_mean_abs, CNN_II_latent_mean_abs, LSTM_latent_mean_abs])
    
    CNN_I_latent_std = np.std(CNN_I_latent_abs)
    
    CNN_II_latent_std = np.std(CNN_II_latent_abs)
    
    LSTM_latent_std = np.std(LSTM_latent_abs)
    
    CNN_I_pdf = normal_dist(np.sort(CNN_I_latent) , CNN_I_latent_mean , CNN_I_latent_std)
    
    CNN_II_pdf = normal_dist(np.sort(CNN_II_latent) , CNN_II_latent_mean , CNN_II_latent_std)
    
    LSTM_pdf = normal_dist(np.sort(LSTM_latent) , LSTM_latent_mean , LSTM_latent_std)
    
    plt.figure()
    plt.plot(np.sort(CNN_I_latent), CNN_I_pdf)
    plt.plot(np.sort(CNN_II_latent), CNN_II_pdf)
    plt.plot(np.sort(LSTM_latent), LSTM_pdf)
    
    plt.figure()
    
    a_axes = np.array([1,2,3])
    
    plt.bar(a_axes, b_axes)

save_dir = os.path.join(save_directory, 'SHAP_Method_II_Full')

if 'SHAP_Method_II_Full' not in os.listdir(save_directory):
    
    os.mkdir(os.path.join(save_directory, 'SHAP_Method_II_Full')) 
    
    os.chdir(save_dir)
    
shap_latent_real_val_pd = pd.DataFrame(np.array(shap_latent_real_val))

shap_latent_real_val_pd.to_excel(os.path.join(save_dir,"shap_latent_real_val_pd.xlsx"))

shap_latent_abs_val_pd = pd.DataFrame(np.array(shap_latent_abs_val))

shap_latent_abs_val_pd.to_excel(os.path.join(save_dir,"shap_latent_abs_val_pd.xlsx"))

print("Integrated Gradients!")

baseline = tf.zeros(shape=(224,224,3))

k = 1

for i in range(indeces.shape[0]):
    
    img = Data_val_XAI_flank_np[i]
    
    save_dir = os.path.join(save_directory, 'CNN_I_XAI_IG_VAL')
    
    index = indeces[i]
    
    target_class_idx = np.argmax(Val_Label[index])
    
    if 'CNN_I_XAI_IG_VAL' not in os.listdir(save_directory):
        
        os.mkdir(os.path.join(save_directory, 'CNN_I_XAI_IG_VAL')) 
        
        os.chdir(save_dir)
    
    save_Dir = os.path.join(save_dir, str(k))
    
    plot_img_attributions(CNN_I_submodel,
                              baseline,
                              img,
                              target_class_idx,
                              save_Dir,
                              m_steps=50,
                              cmap=None,
                              overlay_alpha=0.5)
    
    k += 1

k = 1
    
for i in range(indeces.shape[0]):
    
    img = Data_val_XAI_rake_np[i]
    
    save_dir = os.path.join(save_directory, 'CNN_II_XAI_IG_VAL')
    
    index = indeces[i]
    
    target_class_idx = np.argmax(Val_Label[index])
    
    if 'CNN_II_XAI_IG_VAL' not in os.listdir(save_directory):
        
        os.mkdir(os.path.join(save_directory, 'CNN_II_XAI_IG_VAL')) 
        
        os.chdir(save_dir)
        
    save_Dir = os.path.join(save_dir, str(k))    
    
    plot_img_attributions(CNN_II_submodel,
                              baseline,
                              img,
                              target_class_idx,
                              save_Dir,
                              m_steps=50,
                              cmap=None,
                              overlay_alpha=0.5)
    
    k += 1
    
print("Grad Cam!")

k = 1

for i in range(indeces.shape[0]):
    
    img = Data_val_XAI_flank_np[i]
    
    save_dir = os.path.join(save_directory, 'CNN_I_XAI_GC_VAL')
    
    index = indeces[i]
    
    target_class_idx = np.argmax(Val_Label[index])
    
    if 'CNN_I_XAI_GC_VAL' not in os.listdir(save_directory):
        
        os.mkdir(os.path.join(save_directory, 'CNN_I_XAI_GC_VAL')) 
        
        os.chdir(save_dir)
    
    CNN_I_Layer = "CNN_output_I" if transfer_learning == False else "block5_conv4_flank"
    
    heatmap = make_gradcam_heatmap(img, CNN_I_submodel, CNN_I_Layer)
    
    save_and_display_gradcam(img, heatmap, k, cam_path = save_dir, alpha=0.2)
    
    k += 1
    
k = 1

for i in range(indeces.shape[0]):
    
    img = Data_val_XAI_rake_np[i]
    
    save_dir = os.path.join(save_directory, 'CNN_II_XAI_GC_VAL')
    
    index = indeces[i]
    
    target_class_idx = np.argmax(Val_Label[index])
    
    if 'CNN_II_XAI_GC_VAL' not in os.listdir(save_directory):
        
        os.mkdir(os.path.join(save_directory, 'CNN_II_XAI_GC_VAL')) 
        
        os.chdir(save_dir)

    CNN_II_Layer = "CNN_output_II" if transfer_learning == False else "block5_conv4_rake"
    
    heatmap = make_gradcam_heatmap(img, CNN_II_submodel, CNN_II_Layer)
    
    save_and_display_gradcam(img, heatmap, k, cam_path=save_dir, alpha=0.2)
    
    k += 1
    


print("Integrated Gradients!")

baseline = tf.zeros(shape=(224,224,3))

k = 1

for i in range(indeces.shape[0]):
    
    img = Val_flank_image[i]
    
    save_dir = os.path.join(save_directory, 'CNN_I_XAI_IG_Test')
    
    index = different_cases_poped[i]
    
    target_class_idx = np.argmax(Val_Label[index])
    
    if 'CNN_I_XAI_IG_Test' not in os.listdir(save_directory):
        
        os.mkdir(os.path.join(save_directory, 'CNN_I_XAI_IG_Test')) 
        
        os.chdir(save_dir)
    
    save_Dir = os.path.join(save_dir, str(k))
    
    plot_img_attributions(CNN_I_submodel,
                              baseline,
                              img,
                              target_class_idx,
                              save_Dir,
                              m_steps=50,
                              cmap=None,
                              overlay_alpha=0.5)
    
    k += 1

k = 1
    
for i in range(indeces.shape[0]):
    
    img = Val_Rake_image[i]
    
    save_dir = os.path.join(save_directory, 'CNN_II_XAI_IG_Test')
    
    index = different_cases_poped[i]
    
    target_class_idx = np.argmax(Val_Label[index])
    
    if 'CNN_II_XAI_IG_Test' not in os.listdir(save_directory):
        
        os.mkdir(os.path.join(save_directory, 'CNN_II_XAI_IG_Test')) 
        
        os.chdir(save_dir)
        
    save_Dir = os.path.join(save_dir, str(k))    
    
    plot_img_attributions(CNN_II_submodel,
                              baseline,
                              img,
                              target_class_idx,
                              save_Dir,
                              m_steps=50,
                              cmap=None,
                              overlay_alpha=0.5)
    
    k += 1
    
print("Grad Cam!")

k = 1

for i in range(indeces.shape[0]):
    
    img = Val_flank_image[i]
    
    save_dir = os.path.join(save_directory, 'CNN_I_XAI_GC_Test')
    
    index = different_cases_poped[i]
    
    target_class_idx = np.argmax(Val_Label[index])
    
    if 'CNN_I_XAI_GC_Test' not in os.listdir(save_directory):
        
        os.mkdir(os.path.join(save_directory, 'CNN_I_XAI_GC_Test')) 
        
        os.chdir(save_dir)
    
    CNN_I_Layer = "CNN_output_I" if transfer_learning == False else "block5_conv4_flank"
    
    heatmap = make_gradcam_heatmap(img, CNN_I_submodel, CNN_I_Layer)
    
    save_and_display_gradcam(img, heatmap, k, cam_path = save_dir, alpha=0.2)
    
    k += 1
    
k = 1

for i in range(indeces.shape[0]):
    
    img = Val_Rake_image[i]
    
    save_dir = os.path.join(save_directory, 'CNN_II_XAI_GC_Test')
    
    index = different_cases_poped[i]
    
    target_class_idx = np.argmax(Val_Label[index])
    
    if 'CNN_II_XAI_GC_Test' not in os.listdir(save_directory):
        
        os.mkdir(os.path.join(save_directory, 'CNN_II_XAI_GC_Test')) 
        
        os.chdir(save_dir)

    CNN_II_Layer = "CNN_output_II" if transfer_learning == False else "block5_conv4_rake"
    
    heatmap = make_gradcam_heatmap(img, CNN_II_submodel, CNN_II_Layer)
    
    save_and_display_gradcam(img, heatmap, k, cam_path=save_dir, alpha=0.2)
    
    k += 1
    
"Full Images"

