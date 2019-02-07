from __future__ import print_function

from DeepJetCore.TrainData import TrainData, fileTimeOut
import numpy

class TrainData_DiTau(TrainData):
    
    def __init__(self):
        '''
        This class is meant as a base class for the FatJet studies
        You will not need to edit it for trying out things
        '''
        TrainData.__init__(self)
        
        #define truth:
        self.treename = "deepntuplizerCA8/tree"
        self.undefTruth=['isUndefined']
        self.truthclasses=['isB','isBB',#'isGBB',
                           #'isLeptonicB','isLeptonicB_C',
                           'isC','isCC',#'isGCC',
                           'isUD','isS','isG',
                           'isTauHTauH','isTauHTauM','isTauHTauE',
                           #'isTauMTauM','isTauMTauE','isTauETauE',
                           #'isTauH','isTauM','isTauE',
                           ]

        self.registerBranches(self.truthclasses)
        self.registerBranches(self.undefTruth)

        self.referenceclass='isTauHTauH' # 'flatten' or class name
        #self.referenceclass='flatten'
        #self.referenceclass='lowest'
        self.weightbranchX='jet_pt'
        self.weightbranchY='jet_eta'
        #self.weightbranchY='jet_mass'

        self.registerBranches([self.weightbranchX,self.weightbranchY])

        self.weight_binX = numpy.array([
                10,25,30,35,40,45,50,60,75,100,
                125,150,175,200,250,300,400,500,
                600,2000],dtype=float)

        self.weight_binY = numpy.array(
            [-2.5,-2.,-1.5,-1.,-0.5,0.5,1,1.5,2.,2.5],
            dtype=float
            )

        #self.weight_binY = numpy.array([
        #        10,30,40,50,75,100,
        #        125,150,175,200,250,300,400,500,
        #        600,800,1000,1500,2000],dtype=float)


        self.weight=True
        self.remove=False
    
        
class TrainData_DiTau_glb_cpf_npf_sv(TrainData_DiTau):
    
    def __init__(self):
        TrainData_DiTau.__init__(self)
        
        self.addBranches(['jet_pt', 'jet_eta',
                          'nCpfcand','nNpfcand',
                          'nsv','npv',
                          #'electrons_number', # int in current implementation
                          #'muons_number',     # int in current implementation
                          'TagVarCSVTrk_trackDeltaR',
                          'TagVarCSVTrk_trackJetDistVal',
                          'TagVarCSVTrk_trackPtRatio',
                          'TagVarCSVTrk_trackPtRel',
                          'TagVarCSVTrk_trackSip2dSig',
                          'TagVarCSVTrk_trackSip3dSig',
                          'TagVarCSV_flightDistance2dSig',
                          'TagVarCSV_flightDistance2dVal',
                          'TagVarCSV_flightDistance3dSig',
                          'TagVarCSV_flightDistance3dVal',
                          'TagVarCSV_trackSumJetEtRatio',
                          'TagVarCSV_trackSumJetDeltaR',
                          'TagVarCSV_vertexCategory',
                          'TagVarCSV_vertexMass',
                          'TagVarCSV_vertexEnergyRatio',
                          'TagVarCSV_vertexNTracks',
                          'TagVarCSV_trackSip2dValAboveCharm',
                          'TagVarCSV_trackSip2dSigAboveCharm',
                          'TagVarCSV_trackSip3dValAboveCharm',
                          'TagVarCSV_trackSip3dSigAboveCharm',
                          'TagVarCSV_jetNSecondaryVertices',
                          'TagVarCSV_jetNSelectedTracks',
                          'TagVarCSV_jetNTracksEtaRel'])

        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackPtRatio',
                          'Cpfcan_BtagPf_trackPPar',
                          'Cpfcan_BtagPf_trackDeltaR',
                          'Cpfcan_BtagPf_trackPParRatio',
                          'Cpfcan_BtagPf_trackSip2dVal',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dVal',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          'Cpfcan_BtagPf_trackJetDistSig',

                          'Cpfcan_ptrel',
                          'Cpfcan_deltaR',
                          'Cpfcan_dz',
                          'Cpfcan_erel',
                          'Cpfcan_etarel',
                          'Cpfcan_phirel',
                          'Cpfcan_drminsv',
                          'Cpfcan_fromPV',
                          'Cpfcan_VTX_ass',
                          'Cpfcan_puppiw',
                          'Cpfcan_chi2',
                          'Cpfcan_quality',
                          'Cpfcan_isMu',
                          'Cpfcan_isEl',
                              ],
                             25)


        self.addBranches(['Npfcan_ptrel',
                          'Npfcan_erel',
                          'Npfcan_etarel',
                          'Npfcan_phirel',
                          'Npfcan_deltaR',
                          'Npfcan_isGamma',
                          'Npfcan_HadFrac',
                          'Npfcan_drminsv',
                          'Npfcan_puppiw'
                          ],
                         25)

        self.addBranches(['sv_pt',
                          'sv_deltaR',
                          'sv_mass',
                          'sv_ntracks',
                          'sv_chi2',
                          'sv_normchi2',
                          'sv_dxy',
                          'sv_dxysig',
                          'sv_d3d',
                          'sv_d3dsig',
                          'sv_costhetasvpv',
                          'sv_enratio',
                          ],
                          4)

        #self.addBranches(['jet_corr_pt'])

        #self.registerBranches(['gen_pt_WithNu'])

        #self.regressiontargetclasses=['uncPt','Pt']

        
    #this function describes how the branches are converted
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        import ROOT
        
        fileTimeOut(filename, 60) #give eos 1 minutes to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)

        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                         self.branches[1],
                                         self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                         self.branches[2],
                                         self.branchcutoffs[2],self.nsamples)
        
        x_sv  = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                         self.branches[3],
                                         self.branchcutoffs[3],self.nsamples)

        
        Tuple = self.readTreeFromRootToTuple(filename)

        undef=Tuple['isUndefined']
        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            notremoves -= undef
            

        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            #print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)

        truthtuple = Tuple[self.truthclasses]
        alltruth = self.reduceTruth(truthtuple)

        # remove extra jets so that there are the same number of jets in each 
        # reduced truth category
        # This seems to work, but the numbers still aren't equal...
        # likely this is coming from the difference between processed jets and the reweighted/removed jets
        # need to grab the remove probs and correct
        #truthcounts = weighter.totalcounts
        #countmap = dict(zip(self.truthclasses,truthcounts))
        ##print(countmap)
        #if hasattr(self,'reducedtruthmap'):
        #    sums = {truth:0 for truth in self.reducedtruthclasses}
        #    for t in self.truthclasses:
        #        for truth in self.reducedtruthclasses:
        #            if t in self.reducedtruthmap[truth]:
        #                sums[truth] += countmap[t]
        #    ref = sums[self.reducedreferenceclass]
        #    keepfracs = {truth: float(ref)/sums[truth] for truth in self.reducedtruthclasses}
        #    for i,row in enumerate(iter(alltruth)):
        #        for t,truth in enumerate(self.reducedtruthclasses):
        #            if row[t]==1:
        #                if self.remove:
        #                    rand = numpy.random.ranf()
        #                    if rand>keepfracs[truth]:
        #                        notremoves[i] = 0
        #                elif self.weight:
        #                    weights[i] = weights[i]*keepfracs[truth]

        # scale down by number of classes in a reduced class
        if hasattr(self,'reducedtruthmap'):
            for i,row in enumerate(iter(alltruth)):
                for t,truth in enumerate(self.reducedtruthclasses):
                    if row[t]==1:
                        weights[i] = weights[i]*1./len(self.reducedtruthmap[truth])

        # pt cut
        #pt = Tuple['jet_pt']
        #weights   = weights[ pt > 30]
        #x_global  = x_global[pt > 30]
        #x_cpf     = x_cpf[   pt > 30]
        #x_npf     = x_npf[   pt > 30]
        #x_sv      = x_sv[    pt > 30]
        #alltruth  = alltruth[pt > 30]

        if self.remove:
            weights   = weights[ notremoves > 0]
            x_global  = x_global[notremoves > 0]
            x_cpf     = x_cpf[   notremoves > 0]
            x_npf     = x_npf[   notremoves > 0]
            x_sv      = x_sv[    notremoves > 0]
            alltruth  = alltruth[notremoves > 0]

        if self.weight:
            x_global  = x_global[weights > 0]
            x_cpf     = x_cpf[   weights > 0]
            x_npf     = x_npf[   weights > 0]
            x_sv      = x_sv[    weights > 0]
            alltruth  = alltruth[weights > 0]
            weights   = weights[ weights > 0]

        #weights   = weights[ numpy.any(alltruth, axis=1)]
        #x_global  = x_global[numpy.any(alltruth, axis=1)]
        #x_cpf     = x_cpf[   numpy.any(alltruth, axis=1)]
        #x_npf     = x_npf[   numpy.any(alltruth, axis=1)]
        #x_sv      = x_sv[    numpy.any(alltruth, axis=1)]
        #alltruth  = alltruth[numpy.any(alltruth, axis=1)]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp

        if weights.ndim>1:
            weights = weights.reshape(weights.shape[0])

        self.w=[weights]
        self.x=[x_global,x_cpf,x_npf,x_sv]
        self.y=[alltruth]


    def reduceTruth(self,tuple_in):
        import numpy
        if tuple_in is not None:
            if hasattr(self,'reducedtruthmap'):
                truths = []
                for truth in self.reducedtruthclasses:
                    ts = [tuple_in[t] for t in self.reducedtruthmap[truth]]
                    truths.append(ts[0])
                    for t in ts[1:]:
                        truths[-1] = truths[-1]+t
                return numpy.vstack(truths).transpose()


class TrainData_DiTau_glb_cpf_npf_sv_3cat_signal(TrainData_DiTau_glb_cpf_npf_sv):
    def __init__(self):
        TrainData_DiTau_glb_cpf_npf_sv.__init__(self)
        self.reducedtruthclasses=['isTauHTauH','isTauHTauM','isTauHTauE']
        self.reducedtruthmap = {
            'isTauHTauH': ['isTauHTauH'],
            'isTauHTauM': ['isTauHTauM'],
            'isTauHTauE': ['isTauHTauE'],
        }
        self.reducedreferenceclass='isTauHTauH'


class TrainData_DiTau_glb_cpf_npf_sv_2cat(TrainData_DiTau_glb_cpf_npf_sv):
    def __init__(self):
        TrainData_DiTau_glb_cpf_npf_sv.__init__(self)
        self.reducedtruthclasses=['isJet','isTauTau']
        self.reducedtruthmap = {
            'isJet'   : ['isB','isBB','isC','isCC','isUD','isS','isG'],
            'isTauTau': ['isTauHTauH','isTauHTauM','isTauHTauE'],
        }
        self.reducedreferenceclass='isTauTau'

class TrainData_DiTau_glb_cpf_npf_sv_4cat(TrainData_DiTau_glb_cpf_npf_sv):
    def __init__(self):
        TrainData_DiTau_glb_cpf_npf_sv.__init__(self)
        self.reducedtruthclasses=['isB','isC','isLight','isTauTau']
        self.reducedtruthmap = {
            'isB'     : ['isB','isBB',],
            'isC'     : ['isC','isCC',],
            'isLight' : ['isUD','isS','isG'],
            'isTauTau': ['isTauHTauH','isTauHTauM','isTauHTauE'],
        }
        self.reducedreferenceclass='isTauTau'

