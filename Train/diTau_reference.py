import os
os.environ['DECORRELATE'] = "False"
from DeepJetCore.training.training_base import training_base
from Losses import loss_NLL, loss_meansquared   
from DeepJetCore.modeltools import fixLayersContaining,printLayerInfosAndWeights

import tensorflow as tf
from keras import backend as k

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
k.tensorflow_backend.set_session(tf.Session(config=config))



train=training_base(testrun=False,renewtokens=False)

if not train.modelSet():
    from models import model_diTauReference as trainingModel

    train.setModel(trainingModel)
    
    train.compileModel(learningrate=0.001,
                       loss=['categorical_crossentropy'],
                       metrics=['accuracy'],
                       loss_weights=[1.])

from keras.utils import plot_model
plot_model(train.keras_model, to_file=train.outputDir+'model.eps')

model, history = train.trainModel(nepochs=500,
                                  batchsize=10000,
                                  stop_patience=100,
                                  lr_factor=0.8,
                                  lr_patience=10,
                                  lr_epsilon=0.0001,
                                  lr_cooldown=8,
                                  lr_minimum=0.000001,
                                  maxqsize=5,
                                  verbose=1,
                                  )

