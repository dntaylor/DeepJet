from __future__ import print_function
import os
import sys
import operator
import pickle
import argparse
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import ROOT

ROOT.gROOT.SetBatch()

import tdrstyle
tdrstyle.setTDRStyle()

ROOT.gStyle.SetPaintTextFormat('4.1f');
ROOT.gStyle.SetPalette(ROOT.kBird)


def parse_command_line(argv):
    parser = argparse.ArgumentParser(description='Dump contents of tfile')

    parser.add_argument('dataCollection', type=str)
    parser.add_argument('predictionDir', type=str)

    return parser.parse_args(argv)

def plot_confusion_matrix(y_test, y_pred, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    args = parse_command_line(argv)

    print('This currently runs as part of predict step, will do nothing')
    return 0

    # dcData[6] holds the TrainData class
    dcData = []
    with open(args.dataCollection,'rb') as f:
        while True:
            try:
                o = pickle.load(f)
            except:
                break
            dcData += [o]
    
    treePredName = 'tree'
    treeBaseName = dcData[6].treename
    truthMap = dcData[6].reducedtruthmap
    truthOrder = dcData[6].reducedtruthclasses
    n = len(truthMap)
    
    tfileMapFile = '{}/tree_association.txt'.format(args.predictionDir)
    tfileMap = {}
    with open(tfileMapFile) as f:
        for line in f.readlines():
            tin, tpred = line.rstrip().split(' ')
            tfileMap[tpred] = tin
    
    tfilePreds = {}
    tfileBases = {}
    treePreds = {}
    treeBases = {}
    for tp,tb in tfileMap.iteritems():
        tfilePreds[tp] = ROOT.TFile.Open(tp)
        tfileBases[tb] = ROOT.TFile.Open(tb)
        treePreds[tp] = tfilePreds[tp].Get(treePredName)
        treeBases[tb] = tfileBases[tb].Get(treeBaseName)
        treeBases[tb].AddFriend(treePreds[tp])
    
    
    truthPredCount = {t: {p: 0 for p in truthMap} for t in truthMap}
    truthCount = {t:0 for t in truthMap}
    truthArray = None
    predArray = None
    for tb,tree in treeBases.iteritems():
        for row in tree:
            probs = {t: getattr(row,'prob_{}'.format(t)) for t in truthMap}
            truths = {t: any([getattr(row,x) for x in truthMap[t]]) for t in truthMap}
            if not any(truths.keys()): continue
            pred = max(probs.iteritems(), key=operator.itemgetter(1))[0]
            truth = max(truths.iteritems(), key=operator.itemgetter(1))[0]
            truthPredCount[truth][pred] += 1
            truthCount[truth] += 1
            tarray = np.zeros(n)[:,np.newaxis]
            tarray[truthOrder.index(truth)] = 1
            parray = np.array([probs[t] for t in truthOrder])[:,np.newaxis]
            if truthArray is not None:
                truthArray = np.concatenate([truthArray,tarray],axis=1)
                predArray  = np.concatenate([predArray, parray],axis=1)
            else:
                truthArray = tarray
                predArray  = parray
    
    truthArray = truthArray.T
    predArray = predArray.T
    
    labels = {
        'isJet'     : r'jet',
        'isLight'   : r'udsg',
        'isB'       : r'b',
        'isC'       : r'c',
        'isTauTau'  : r'$\tau\tau$',
        'isTauHTauH': r'$\tau_{h}\tau_{h}$',
        'isTauHTauM': r'$\tau_{\mu}\tau_{h}$',
        'isTauHTauE': r'$\tau_{e}\tau_{h}$',
        'isTau'     : r'$\tau$',
        'isTauH'    : r'$\tau_{h}$',
        'isTauM'    : r'$\tau_{\mu}$',
        'isTauE'    : r'$\tau_{e}$',
    }
    
    f = plt.figure()
    plot_confusion_matrix(
        truthArray.argmax(axis=1),
        predArray.argmax(axis=1),
        [labels[t] for t in truthOrder],
        normalize=True,
    )
    f.savefig('{}/confusion.png'.format(args.predictionDir))

    return 0

if __name__ == "__main__":
    status = main()
    sys.exit(status)
