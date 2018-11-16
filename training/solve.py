from datetime import datetime

import caffe
import numpy as np

from . import surgery, score

# import setproctitle
# setproctitle.setproctitle(os.path.basename(os.getcwd()))

# weights = '../vgg16fc.caffemodel'
# weights = '../vgg/16-layer/VGG_ILSVRC_16_layers.caffemodel'

print('>>>', datetime.now(), 'Begin program')

caffe.set_device(0)
caffe.set_mode_gpu()

print('>>>', datetime.now(), 'Begin load solver')

solver = caffe.SGDSolver('/tsi/doctorants/avirzi/Deep_Vessels_Cross_Validation/cnn_models/solver.prototxt')

print('>>>', datetime.now(), 'Begin transfer sugery')
# #----------------------------------------------------
# # train from solverstate i.e. resume training from previous snapshot
# solver.restore('/work/xu/neobrain/axialSnaps/snapshot_iter_10000.solverstate')
# #----------------------------------------------------

# ---------------------------------------------------
# train from vgg
base_net = caffe.Net(
    '/tsi/doctorants/avirzi/Deep_Vessels_Cross_Validation/cnn_models/vgg_16/VGG_ILSVRC_16_layers_deploy.prototxt',
    '/tsi/doctorants/avirzi/Deep_Vessels_Cross_Validation/cnn_models/vgg_16/VGG_ILSVRC_16_layers.caffemodel',
    caffe.TEST)
surgery.transplant(solver.net, base_net)
del base_net
# surgeries
interp_layers = [k for k in list(solver.net.params.keys()) if 'up' in k]
surgery.interp(solver.net, interp_layers)
# ----------------------------------------------------

# scoring
val = np.loadtxt('/tsi/doctorants/avirzi/Deep_Vessels_Cross_Validation/Data/Training/WorkingMix2/val_5.txt', dtype=str)

print('>>>', datetime.now(), 'Begin training')
# score.seg_tests(solver, False, val, layer='score')

# for _ in range(int(sys.argv[2])):
for x in range(0, 20):
    print('>>>', datetime.now(), 'begin training')
    solver.step(5000)
    print('>>>', datetime.now(), 'begin testing')
    score.seg_tests(solver, False, val, layer='score')
