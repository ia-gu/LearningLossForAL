''' Configuration File.
'''

##
# Learning Loss for Active Learning
NUM_TRAIN = 50000
NUM_VAL   = 50000 - NUM_TRAIN
BATCH     = 20
SUBSET    = 10000
ADDENDUM  = 1000
NAME = 'base/'

MARGIN = 1.0
WEIGHT = 1.0

TRIALS = 1
CYCLES = 10

EPOCH = 200
LR = 0.01
MILESTONES = [160]
EPOCHL = 120 # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model

MOMENTUM = 0.9
WDECAY = 5e-4
