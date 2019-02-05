import os
import sys
import operator
import pickle
import argparse

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


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    args = parse_command_line(argv)

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
    
    for tb,tree in treeBases.iteritems():
        for row in tree:
            probs = {t: getattr(row,'prob_{}'.format(t)) for t in truthMap}
            truths = {t: any([getattr(row,x) for x in truthMap[t]]) for t in truthMap}
            pred = max(probs.iteritems(), key=operator.itemgetter(1))[0]
            truth = max(truths.iteritems(), key=operator.itemgetter(1))[0]
            truthPredCount[truth][pred] += 1
            truthCount[truth] += 1
    
    
    labels = {
        'isJet'     : 'jet',
        'isLight'   : 'udsg',
        'isB'       : 'b'
        'isC'       : 'c'
        'isTauTau'  : '#tau#tau',
        'isTauHTauH': '#tau_{h}#tau_{h}',
        'isTauHTauM': '#tau_{#mu}#tau_{h}',
        'isTauHTauE': '#tau_{e}#tau_{h}',
        'isTau'     : '#tau',
        'isTauH'    : '#tau_{h}',
        'isTauM'    : '#tau_{#mu}',
        'isTauE'    : '#tau_{e}',
    }
    
    n = len(truthMap)
    hist = ROOT.TH2D('confusion','confusion',n,-0.5,n-0.5,n,-0.5,n-0.5)
    for t in range(n):
        for p in range(n):
            truth = truthOrder[t]
            pred = truthOrder[p]
            hist.SetBinContent(hist.FindBin(t,p),float(truthPredCount[truth][pred])/truthCount[truth]*100)
    
    canvas = ROOT.TCanvas('c','c',50,50,650,600)
    canvas.SetRightMargin(0.18)
    
    hist.Draw('colz text')
    for b, label in enumerate(truthOrder):
        hist.GetXaxis().SetBinLabel(b+1,labels[label])
        hist.GetYaxis().SetBinLabel(b+1,labels[label])
    hist.GetXaxis().SetTitle('Truth')
    hist.GetYaxis().SetTitle('Prediction')
    hist.GetZaxis().SetTitle('% Predicted')
    hist.GetZaxis().SetRangeUser(0,100)
    canvas.Print('{}/confusion.png'.format(args.predictionDir))

    return 0

if __name__ == "__main__":
    status = main()
    sys.exit(status)
