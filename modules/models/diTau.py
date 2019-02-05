from keras.layers import Dense, Dropout, Flatten,Concatenate, Convolution2D, LSTM,merge, Convolution1D, Conv2D, GRU, SpatialDropout1D, Conv1D, Lambda
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add, Multiply
from buildingBlocks import block_deepFlavourConvolutions, block_deepFlavourDense, block_SchwartzImage, block_deepFlavourBTVConvolutions

#############
### DiTau ###
#############
def model_diTauReference(inputs, num_classes, num_regclasses, datasets = ['global','cpf','npf','sv'], removedVars = None, multi_gpu=1,  **kwargs):
    kernel_initializer = 'he_normal'
    kernel_initializer_fc = 'lecun_uniform'

    globalvars = BatchNormalization(momentum=0.6,name='globals_input_batchnorm') (inputs[0])
    cpf    =     BatchNormalization(momentum=0.6,name='cpf_input_batchnorm')     (inputs[1])
    npf    =     BatchNormalization(momentum=0.6,name='npf_input_batchnorm')     (inputs[2])
    vtx    =     BatchNormalization(momentum=0.6,name='vtx_input_batchnorm')     (inputs[3])

    flattenLayers = []
    flattenLayers.append(globalvars)

    for ds, x in zip(datasets[1:],[cpf,npf,vtx]):
        x = Conv1D(filters=32, kernel_size=(1,), strides=(1,), padding='same',
                                kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv1'%ds,
                                activation = 'relu')(x)
        x = SpatialDropout1D(rate=0.1)(x)
        x = Conv1D(filters=32, kernel_size=(1,), strides=(1,), padding='same',
                             kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv2'%ds,
                             activation = 'relu')(x)
        x = SpatialDropout1D(rate=0.1)(x)
        x = GRU(50,go_backwards=True,implementation=2,name='%s_gru'%ds)(x)
        x = Dropout(rate=0.1)(x)
        flattenLayers.append(x)

    concat = Concatenate()(flattenLayers)

    dense = Dense(200, activation='relu',name='dense_1',kernel_initializer=kernel_initializer_fc,trainable=True)(concat)
    dropout = Dropout(rate=0.1, name='dense_dropout_1')(dense)
    dense = Dense(100, activation='relu',name='dense_2',kernel_initializer=kernel_initializer_fc,trainable=True)(dropout)
    dropout = Dropout(rate=0.1, name='dense_dropout_2')(dense)
    dense = Dense(100, activation='relu',name='dense_3',kernel_initializer=kernel_initializer_fc,trainable=True)(dropout)
    dropout = Dropout(rate=0.1, name='dense_dropout_3')(dense)

    output = Dense(num_classes, activation='softmax', name='ID_pred', kernel_initializer=kernel_initializer_fc)(dropout)

    model = Model(inputs=inputs, outputs=[output])
    if multi_gpu > 1:
        from multi_gpu_model import multi_gpu_model
        model = multi_gpu_model(model, gpus=multi_gpu)

    return model
