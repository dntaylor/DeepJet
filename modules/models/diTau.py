from keras.layers import Dense, Dropout, Flatten,Concatenate, Convolution2D, LSTM,merge, Convolution1D, Conv2D, GRU, SpatialDropout1D, Conv1D, Lambda
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add, Multiply
from buildingBlocks import block_deepFlavourConvolutions, block_deepFlavourDense, block_SchwartzImage, block_deepFlavourBTVConvolutions

##############
### blocks ###
##############
def block_convolution(x,label,dropoutRate=0.1,active=True,batchnorm=False,batchmomentum=0.6,pattern=[64,32,32,8]):
    '''
    deep Flavour convolution part.
    '''
    if active:
        for i,p in enumerate(pattern):
            x  = Convolution1D(p, 1, kernel_initializer='lecun_uniform',  activation='relu', name='{}_conv{}'.format(label,i))(x)
            if i<len(pattern)-1:
                if batchnorm:
                    x = BatchNormalization(momentum=batchmomentum ,name='{}_conv_batchnorm{}'.format(label,i))(x)
                x = Dropout(dropoutRate,name='{}_conv_dropout{}'.format(label,i))(x)
    else:
        x = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(x)

    return x

def block_dense(x,dropoutRate,depth=8,width=100,active=True,batchnorm=False,batchmomentum=0.6):
    if active:
        for i in range(depth):
            x=  Dense(width, activation='relu',kernel_initializer='lecun_uniform', name='df_dense{}'.format(i))(x)
            if batchnorm:
                x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm{}'.format(i))(x)
            x = Dropout(dropoutRate,name='df_dense_dropout{}'.format(i))(x)
    else:
        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_dense_off')(x)

    return x

##############
### models ###
##############
def model_diTauReference(inputs, num_classes, num_regclasses, datasets = ['global','cpf','npf','sv'], removedVars = None, multi_gpu=1, dropoutRate=0.1, momentum=0.6, batchnorm=True, depth=8, width=100, **kwargs):
    kernel_initializer = 'he_normal'
    kernel_initializer_fc = 'lecun_uniform'

    if batchnorm:
        xs = [BatchNormalization(momentum=momentum, name='{}_input_batchnorm'.format(datasets[i]))(inputs[i]) for i in range(len(inputs))]
    else:
        xs = inputs

    flatten = [xs[0]]
    for i in range(1,len(datasets)):
        x = block_convolution(xs[i],datasets[i],dropoutRate=dropoutRate,batchnorm=batchnorm,batmomentum=momenutm)
        x = LSTM(150,go_backwards=True,implementation=2, name='{}_lstm'.format(datasets[i]))(x)
        if batchnorm:
            x = BatchNormalization(momentum=momentum,name='{}_lstm_batchnorm')(x)
        x = Dropout(dropoutRate,name='{}_lstm_dropout'.format(label))(x)
        flatten += [x]

    x = Concatenate()(flatten)
    
    x = block_dense(x,dropoutRate,depth=depth,width=width,active=True,batchnorm=batchnorm,batchmomentum=momentum)
    
    output = Dense(num_classes, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)

    model = Model(inputs=inputs, outputs=[output])
    if multi_gpu > 1:
        from multi_gpu_model import multi_gpu_model
        model = multi_gpu_model(model, gpus=multi_gpu)

    return model

def model_diTauDense(inputs, num_classes, num_regclasses, datasets = ['global'], removedVars = None, multi_gpu=1, dropoutRate=0.1, momentum=0.6, batchnorm=True, depth=8, width=100, **kwargs):
    kernel_initializer = 'he_normal'
    kernel_initializer_fc = 'lecun_uniform'

    if batchnorm:
        xs = [BatchNormalization(momentum=momentum, name='{}_input_batchnorm'.format(datasets[i]))(inputs[i]) for i in range(len(inputs))]
    else:
        xs = inputs

    if len(xs)>1:
        flatten = [xs[0]]
        for i in range(1,len(xs)):
            x = Flatten(name='{}_flatten'.format(datasets[i]))(xs[i])
            flatten += [x]
        x = Concatenate()(flatten)
    else:
        x = xs[0]
     
    x = block_dense(x,dropoutRate,depth=depth,width=width,active=True,batchnorm=batchnorm,batchmomentum=momentum)
    
    output = Dense(num_classes, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)

    model = Model(inputs=inputs, outputs=[output])
    if multi_gpu > 1:
        from multi_gpu_model import multi_gpu_model
        model = multi_gpu_model(model, gpus=multi_gpu)

    return model
