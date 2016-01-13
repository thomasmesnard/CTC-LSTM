#!/usr/bin/env python

import theano
import numpy
from theano import tensor

from blocks.bricks import Linear, Tanh, Rectifier
from blocks.bricks.conv import Convolutional, MaxPooling
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import SimpleRecurrent, LSTM
from blocks.initialization import IsotropicGaussian, Constant

from blocks.algorithms import (GradientDescent, Scale, AdaDelta, RemoveNotFinite, RMSProp, BasicMomentum,
                               StepClipping, CompositeRule, Momentum)
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.main_loop import MainLoop

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT, BIAS
from blocks.graph import ComputationGraph, apply_dropout, apply_noise

from blocks.extensions import ProgressBar
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.extensions import FinishAfter, Printing

from blocks.extras.extensions.plot import Plot


from ctc import CTC
from timit import setup_datastream

from edit_distance import batch_edit_distance
from ext_param_info import ParamInfo


# ==========================================================================================
#                                     THE HYPERPARAMETERS
# ==========================================================================================

# Stop after this many epochs
n_epochs = 10000
# How often (number of batches) to print / plot
monitor_freq = 50           

sort_batch_count = 50
batch_size = 100

# The convolutionnal layers. Parameters:
#   nfilter         the number of filters
#   filter_size     the size of the filters (number of timesteps)
#   stride          the stride on which to apply the filter (non-1 stride are not optimized, runs very slowly with current Theano)
#   pool_stride     the block size for max pooling
#   normalize       do we normalize the values before applying the activation function?
#   activation      a brick for the activation function
#   dropout         dropout applied after activation function
#   skip            do we introduce skip connections from previous layer(s) to next layer(s) ?
convs = [
    {'nfilter':     20,
     'filter_size': 200,
     'stride':      1,
     'pool_stride': 10,
     'normalize':   True,
     'activation':  Rectifier(name='a0'),
     'dropout':     0.0,
     'skip':        ['min', 'max', 'subsample']},
    {'nfilter':     20,
     'filter_size': 200,
     'stride':      1,
     'pool_stride': 10,
     'normalize':   True,
     'activation':  Rectifier(name='a1'),
     'dropout':     0.0,
     'skip':        ['max']},
    {'nfilter':     20,
     'filter_size': 30,
     'stride':      1,
     'pool_stride': 2,
     'normalize':   True,
     'activation':  Rectifier(name='a2'),
     'dropout':     0.0,
     'skip':        ['max']},
    {'nfilter':     100,
     'filter_size': 20,
     'stride':      1,
     'pool_stride': 2,
     'normalize':   True,
     'activation':  Rectifier(name='a3'),
     'dropout':     0.0,
     'skip':        []},
]

# recurrent layers. Parameters:
#   type        type of the layer (simple, lstm, blstm)
#   dim         size of the state
#   normalize   do we normalize the values after the RNN ?
#   dropout     dropout after the RNN
#   skip        do we introduce skip connections from previous layer(s) to next layer(s) ?
recs = [
    {'type':        'blstm',
     'dim':         50,
     'normalize':   False,
     'dropout':     0.0,
     'skip':        True},
    {'type':        'blstm',
     'dim':         50,
     'normalize':   False,
     'dropout':     0.0,
     'skip':        True},
]

# do we normalize the activations just before the softmax layer ?
normalize_out = True

# regularization : noise on the weights
weight_noise = 0.01

# regularization : L2 penalization
l2_output_bias = 0.
l2_output_weight = 0.
l2_all_bias = 0.0
l2_all_weight = 0.

# number of phonemes in timit, a constant
num_output_classes = 61 


# the step rule (uncomment your favorite choice)
step_rule = CompositeRule([AdaDelta(), RemoveNotFinite()])
#step_rule = CompositeRule([Momentum(learning_rate=0.00001, momentum=0.99), RemoveNotFinite()])
#step_rule = CompositeRule([Momentum(learning_rate=0.001, momentum=0.9), RemoveNotFinite()])
#step_rule = CompositeRule([AdaDelta(), Scale(0.01), RemoveNotFinite()])
#step_rule = CompositeRule([RMSProp(learning_rate=0.1, decay_rate=0.95),
#                           RemoveNotFinite()])
#step_rule = CompositeRule([RMSProp(learning_rate=0.0001, decay_rate=0.95),
#                           BasicMomentum(momentum=0.9),
#                           RemoveNotFinite()])

# How the weights are initialized
weights_init = IsotropicGaussian(0.01)
biases_init = Constant(0.001)


# ==========================================================================================
#                                          THE MODEL
# ==========================================================================================

print('Building model ...')


#       THEANO INPUT VARIABLES
inputt = tensor.matrix('input')
input_mask = tensor.matrix('input_mask')
y = tensor.lmatrix('output').T
y_mask = tensor.matrix('output_mask').T
y_len = y_mask.sum(axis=0)
L = y.shape[0]
B = y.shape[1]
# inputt : B x T
# input_mask : B x T
# y : L x B
# y_mask : L x B

#       NORMALIZE THE INPUTS
inputt = inputt / (inputt**2).mean()

dropout_locs = []

#       CONVOLUTION LAYERS
conv_in = inputt[:, None, :, None]
conv_in_channels = 1
conv_in_mask = input_mask

cb = []
for i, p in enumerate(convs):
    # Convolution bricks
    conv = Convolutional(filter_size=(p['filter_size'],1),
    #                   step=(p['stride'],1),
                       num_filters=p['nfilter'],
                       num_channels=conv_in_channels,
                       batch_size=batch_size,
                       border_mode='valid',
                       tied_biases=True,
                       name='conv%d'%i)
    cb.append(conv)
    maxpool = MaxPooling(pooling_size=(p['pool_stride'], 1), name='mp%d'%i)

    conv_out = conv.apply(conv_in)[:, :, ::p['stride'], :]
    conv_out = maxpool.apply(conv_out)
    if p['normalize']:
        conv_out_mean = conv_out.mean(axis=2).mean(axis=0)
        conv_out_var = ((conv_out - conv_out_mean[None, :, None, :])**2).mean(axis=2).mean(axis=0).sqrt()
        conv_out = (conv_out - conv_out_mean[None, :, None, :]) / conv_out_var[None, :, None, :]
    if p['activation'] is not None:
        conv_out = p['activation'].apply(conv_out)
    if p['dropout'] > 0:
        b = [p['activation'] if p['activation'] is not None else conv]
        dropout_locs.append((VariableFilter(bricks=b, name='output'), p['dropout']))
    if p['skip'] is not None and len(p['skip'])>0:
        maxpooladd = MaxPooling(pooling_size=(p['stride']*p['pool_stride'], 1), name='Mp%d'%i)
        skip = []
        if 'max' in p['skip']:
            skip.append(maxpooladd.apply(conv_in)[:, :, :conv_out.shape[2], :])
        if 'min' in p['skip']:
            skip.append(maxpooladd.apply(-conv_in)[:, :, :conv_out.shape[2], :])
        if 'subsample' in p['skip']:
            skip.append(conv_in[:, :, ::(p['stride']*p['pool_stride']), :][:, :, :conv_out.shape[2], :])
        conv_out = tensor.concatenate([conv_out] + skip, axis=1)
        conv_out_channels = p['nfilter'] + len(p['skip']) * conv_in_channels
    else:
        conv_out_channels = p['nfilter']
    conv_out_mask = conv_in_mask[:, ::(p['stride']*p['pool_stride'])][:, :conv_out.shape[2]]

    conv_in = conv_out
    conv_in_channels = conv_out_channels
    conv_in_mask = conv_out_mask

#       RECURRENT LAYERS
rec_mask = conv_out_mask.dimshuffle(1, 0)
rec_in = conv_out[:, :, :, 0].dimshuffle(2, 0, 1)
rec_in_dim = conv_out_channels

rb = []
for i, p in enumerate(recs):
    # RNN bricks
    if p['type'] == 'lstm':
        pre_rec = Linear(input_dim=rec_in_dim, output_dim=4*p['dim'], name='rnn_linear%d'%i)
        rec = LSTM(activation=Tanh(), dim=p['dim'], name="rnn%d"%i)
        rb = rb + [pre_rec, rec]

        rnn_in = pre_rec.apply(rec_in)

        rec_out, _ = rec.apply(inputs=rnn_in, mask=rec_mask)
        dropout_b = [rec]
        rec_out_dim = p['dim']
    elif p['type'] == 'simple':
        pre_rec = Linear(input_dim=rec_in_dim, output_dim=p['dim'], name='rnn_linear%d'%i)
        rec = SimpleRecurrent(activation=Tanh(), dim=p['dim'], name="rnn%d"%i)
        rb = rb + [pre_rec, rec]

        rnn_in = pre_rec.apply(rec_in)

        rec_out = rec.apply(inputs=rnn_in, mask=rec_mask)
        dropout_b = [rec]
        rec_out_dim = p['dim']
    elif p['type'] == 'blstm':
        pre_frec = Linear(input_dim=rec_in_dim, output_dim=4*p['dim'], name='frnn_linear%d'%i)
        pre_brec = Linear(input_dim=rec_in_dim, output_dim=4*p['dim'], name='brnn_linear%d'%i)
        frec = LSTM(activation=Tanh(), dim=p['dim'], name="frnn%d"%i)
        brec = LSTM(activation=Tanh(), dim=p['dim'], name="brnn%d"%i)
        rb = rb + [pre_frec, pre_brec, frec, brec]

        frnn_in = pre_frec.apply(rec_in)
        frnn_out, _ = frec.apply(inputs=frnn_in, mask=rec_mask)
        brnn_in = pre_brec.apply(rec_in)
        brnn_out, _ = brec.apply(inputs=brnn_in, mask=rec_mask)

        rec_out = tensor.concatenate([frnn_out, brnn_out], axis=2)
        dropout_b = [frec, brec]
        rec_out_dim = 2*p['dim']
    else:
        assert False

    if p['normalize']:
        rec_out_mean = rec_out.mean(axis=1).mean(axis=0)
        rec_out_var = ((rec_out - rec_out_mean[None, None, :])**2).mean(axis=1).mean(axis=0).sqrt()
        rec_out = (rec_out - rec_out_mean[None, None, :]) / rec_out_var[None, None, :]
    if p['dropout'] > 0:
        dropout_locs.append((VariableFilter(bricks=dropout_b, name='output'), p['dropout']))

    if p['skip']:
        rec_out = tensor.concatenate([rec_in, rec_out], axis=2)
        rec_out_dim = rec_in_dim + rec_out_dim

    rec_in = rec_out
    rec_in_dim = rec_out_dim

#       LINEAR FOR THE OUTPUT
rec_to_o = Linear(name='rec_to_o',
                  input_dim=rec_out_dim,
                  output_dim=num_output_classes + 1)
y_hat_pre = rec_to_o.apply(rec_out)
# y_hat_pre : T x B x C+1

if normalize_out:
    y_hat_pre_mean = y_hat_pre.mean(axis=1).mean(axis=0)
    y_hat_pre_var = ((y_hat_pre - y_hat_pre_mean[None, None, :])**2).mean(axis=1).mean(axis=0).sqrt()
    y_hat_pre = (y_hat_pre - y_hat_pre_mean[None, None, :]) / y_hat_pre_var[None, None, :]

# y_hat : T x B x C+1
y_hat = tensor.nnet.softmax(
    y_hat_pre.reshape((-1, num_output_classes + 1))
).reshape((y_hat_pre.shape[0], y_hat_pre.shape[1], -1))
y_hat.name = 'y_hat'

y_hat_mask = rec_mask

#       CTC COST AND ERROR MEASURE
cost = CTC().apply_log_domain(y, y_hat, y_len, y_hat_mask).mean()
cost.name = 'CTC'

dl, dl_length = CTC().best_path_decoding(y_hat, y_hat_mask)
dl = dl[:L, :]
dl_length = tensor.minimum(dl_length, L)

edit_distances = batch_edit_distance(dl.T.astype('int32'), dl_length.astype('int32'),
                                     y.T.astype('int32'), y_len.astype('int32'))
edit_distance = edit_distances.mean()
edit_distance.name = 'edit_distance'
errors_per_char = (edit_distances / y_len).mean()
errors_per_char.name = 'errors_per_char'

is_error = tensor.neq(dl, y) * tensor.lt(tensor.arange(L)[:,None], y_len[None,:])
is_error = tensor.switch(is_error.sum(axis=0), tensor.ones((B,)), tensor.neq(y_len, dl_length))

error_rate = is_error.mean()
error_rate.name = 'error_rate'

#       REGULARIZATION
cg = ComputationGraph([cost, error_rate])
if weight_noise > 0:
    noise_vars = VariableFilter(roles=[WEIGHT])(cg)
    cg = apply_noise(cg, noise_vars, weight_noise)
for vfilter, p in dropout_locs:
    cg = apply_dropout(cg, vfilter(cg), p)
[cost_reg, error_rate_reg] = cg.outputs

ctc_reg = cost_reg + 1e-24
ctc_reg.name = 'CTC'

if l2_output_bias > 0:
    cost_reg += l2_output_bias * sum(x.norm(2) for x in VariableFilter(roles=[BIAS], bricks=[rec_to_o])(cg))
if l2_output_weight > 0:
    cost_reg += l2_output_weight * sum(x.norm(2) for x in VariableFilter(roles=[WEIGHT], bricks=[rec_to_o])(cg))
if l2_all_bias > 0:
    cost_reg += l2_all_bias * sum(x.norm(2) for x in VariableFilter(roles=[BIAS])(cg))
if l2_all_weight > 0:
    cost_reg += l2_all_weight * sum(x.norm(2) for x in VariableFilter(roles=[WEIGHT])(cg))
cost_reg.name = 'cost'


#       INITIALIZATION
for brick in [rec_to_o] + cb + rb:
    brick.weights_init = weights_init
    brick.biases_init = biases_init
    brick.initialize()


# ==========================================================================================
#                                     THE INFRASTRUCTURE
# ==========================================================================================

#       SET UP THE DATASTREAM

print('Bulding DataStream ...')
ds, stream = setup_datastream('/home/lx.nobackup/datasets/timit/readable',
                              batch_size=batch_size,
                              sort_batch_count=sort_batch_count)
valid_ds, valid_stream = setup_datastream('/home/lx.nobackup/datasets/timit/readable',
                                          batch_size=batch_size,
                                          sort_batch_count=sort_batch_count,
                                          valid=True)


#       SET UP THE BLOCKS ALGORITHM WITH EXTENSIONS

print('Bulding training process...')
algorithm = GradientDescent(cost=cost_reg,
                            parameters=ComputationGraph(cost).parameters,
                            step_rule=step_rule)

monitor_cost = TrainingDataMonitoring([ctc_reg, cost_reg, error_rate_reg],
                                      prefix="train",
                                      every_n_batches=monitor_freq,
                                      after_epoch=False)

monitor_valid = DataStreamMonitoring([cost, error_rate, edit_distance, errors_per_char],
                                     data_stream=valid_stream,
                                     prefix="valid",
                                     after_epoch=True)

plot = Plot(document='CTC_timit_%s%s%s%s_%s'%
                        (repr([p['nfilter'] for p in convs]),
                         repr([p['filter_size'] for p in convs]),
                         repr([p['stride'] for p in convs]),
                         repr([p['pool_stride'] for p in convs]),
                         repr([p['dim'] for p in recs])),
            channels=[['train_cost', 'train_CTC', 'valid_CTC'], 
                      ['train_error_rate', 'valid_error_rate'],
                      ['valid_edit_distance'],
                      ['valid_errors_per_char']],
            every_n_batches=monitor_freq,
            after_epoch=True)

model = Model(cost)
main_loop = MainLoop(data_stream=stream, algorithm=algorithm,
                     extensions=[
                                 ProgressBar(),

                                 monitor_cost, monitor_valid,

                                 plot,
                                 Printing(every_n_batches=monitor_freq, after_epoch=True),
                                 ParamInfo(Model([cost]), every_n_batches=monitor_freq),

                                 FinishAfter(after_n_epochs=n_epochs),
                                ],
                     model=model)


#       NOW WE FINALLY CAN TRAIN OUR MODEL

print('Starting training ...')
main_loop.run()


# vim: set sts=4 ts=4 sw=4 tw=0 et:
