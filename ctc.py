import numpy

import theano
from theano import tensor, scan

from blocks.bricks import Brick

# T: INPUT_SEQUENCE_LENGTH
# B: BATCH_SIZE
# L: OUTPUT_SEQUENCE_LENGTH
# C: NUM_CLASSES
class CTC(Brick):
    def apply(self, l, probs, l_len=None, probs_mask=None):
        """
        Numeration:
            Characters 0 to C-1 are true characters
            Character C is the blank character
        Inputs:
            l : L x B : the sequence labelling
            probs : T x B x C+1 : the probabilities output by the RNN
            l_len : B : the length of each labelling sequence
            probs_mask : T x B
        Output: the B probabilities of the labelling sequences
        Steps:
            - Calculate y' the labelling sequence with blanks
            - Calculate the recurrence relationship for the alphas
            - Calculate the sequence of the alphas
            - Return the probability found at the end of that sequence
        """
        T = probs.shape[0]
        B = probs.shape[1]
        C = probs.shape[2]-1
        L = l.shape[0]
        S = 2*L+1
        
        # l_blk = l with interleaved blanks
        l_blk = C * tensor.ones((S, B), dtype='int32')
        l_blk = tensor.set_subtensor(l_blk[1::2,:], l)
        l_blk = l_blk.T     # now l_blk is B x S

        # dimension of alpha (corresponds to alpha hat in the paper) :
        #   T x B x S
        # dimension of c :
        #   T x B
        # first value of alpha (size B x S)
        alpha0 = tensor.concatenate([   tensor.ones((B, 1)),
                                        tensor.zeros((B, S-1))
                                    ], axis=1)
        c0 = tensor.ones((B,))

        # recursion
        l_blk_2 = tensor.concatenate([-tensor.ones((B,2)), l_blk[:,:-2]], axis=1)
        l_case2 = tensor.neq(l_blk, C) * tensor.neq(l_blk, l_blk_2)
        # l_case2 is B x S

        def recursion(p, p_mask, prev_alpha, prev_c):
            # p is B x C+1
            # prev_alpha is B x S 
            prev_alpha_1 = tensor.concatenate([tensor.zeros((B,1)),prev_alpha[:,:-1]], axis=1)
            prev_alpha_2 = tensor.concatenate([tensor.zeros((B,2)),prev_alpha[:,:-2]], axis=1)

            alpha_bar = prev_alpha + prev_alpha_1
            alpha_bar = tensor.switch(l_case2, alpha_bar + prev_alpha_2, alpha_bar)
            next_alpha = alpha_bar * p[tensor.arange(B)[:,None].repeat(S,axis=1).flatten(), l_blk.flatten()].reshape((B,S))
            next_alpha = tensor.switch(p_mask[:,None], next_alpha, prev_alpha)
            next_alpha = next_alpha * tensor.lt(tensor.arange(S)[None,:], (2*l_len+1)[:, None])
            next_c = next_alpha.sum(axis=1)
            
            return next_alpha / next_c[:, None], next_c

        # apply the recursion with scan
        [alpha, c], _ = scan(fn=recursion,
                             sequences=[probs, probs_mask],
                             outputs_info=[alpha0, c0])

        # c = theano.printing.Print('c')(c)
        last_alpha = alpha[-1]
        # last_alpha = theano.printing.Print('a-1')(last_alpha)

        prob = tensor.log(c).sum(axis=0) + tensor.log(last_alpha[tensor.arange(B), 2*l_len.astype('int32')-1]
                                                      + last_alpha[tensor.arange(B), 2*l_len.astype('int32')]
                                                      + 1e-30)

        # return the log probability of the labellings
        return -prob

    def apply_log_domain(self, l, probs, l_len=None, probs_mask=None):
        # Does the same computation as apply, but alpha is in the log domain
        # This avoids numerical underflow issues that were not corrected in the previous version.

        def _log(a):
            return tensor.log(tensor.clip(a, 1e-12, 1e12))

        def _log_add(a, b):
            maximum = tensor.maximum(a, b)
            return (maximum + tensor.log1p(tensor.exp(a + b - 2 * maximum)))

        def _log_mul(a, b):
            return a + b

        # See comments above
        B = probs.shape[1]
        C = probs.shape[2]-1
        L = l.shape[0]
        S = 2*L+1
        
        l_blk = C * tensor.ones((S, B), dtype='int32')
        l_blk = tensor.set_subtensor(l_blk[1::2,:], l)
        l_blk = l_blk.T     # now l_blk is B x S

        alpha0 = tensor.concatenate([   tensor.ones((B, 1)),
                                        tensor.zeros((B, S-1))
                                    ], axis=1)
        alpha0 = _log(alpha0)

        l_blk_2 = tensor.concatenate([-tensor.ones((B,2)), l_blk[:,:-2]], axis=1)
        l_case2 = tensor.neq(l_blk, C) * tensor.neq(l_blk, l_blk_2)

        def recursion(p, p_mask, prev_alpha):
            prev_alpha_1 = tensor.concatenate([tensor.zeros((B,1)),prev_alpha[:,:-1]], axis=1)
            prev_alpha_2 = tensor.concatenate([tensor.zeros((B,2)),prev_alpha[:,:-2]], axis=1)

            alpha_bar1 = tensor.set_subtensor(prev_alpha[:,1:], _log_add(prev_alpha[:,1:],prev_alpha[:,:-1]))
            alpha_bar2 = tensor.set_subtensor(alpha_bar1[:,2:], _log_add(alpha_bar1[:,2:],prev_alpha[:,:-2]))

            alpha_bar = tensor.switch(l_case2, alpha_bar2, alpha_bar1)

            probs = _log(p[tensor.arange(B)[:,None].repeat(S,axis=1).flatten(), l_blk.flatten()].reshape((B,S)))
            next_alpha = _log_mul(alpha_bar, probs)
            next_alpha = tensor.switch(p_mask[:,None], next_alpha, prev_alpha)
            
            return next_alpha

        alpha, _ = scan(fn=recursion,
                             sequences=[probs, probs_mask],
                             outputs_info=[alpha0])

        last_alpha = alpha[-1]
        # last_alpha = theano.printing.Print('a-1')(last_alpha)

        prob = _log_add(last_alpha[tensor.arange(B), 2*l_len.astype('int32')-1],
                        last_alpha[tensor.arange(B), 2*l_len.astype('int32')])

        # return the negative log probability of the labellings
        return -prob

    
    def best_path_decoding(self, probs, probs_mask=None):
        # probs is T x B x C+1
        T = probs.shape[0]
        B = probs.shape[1]
        C = probs.shape[2]-1

        maxprob = probs.argmax(axis=2)
        is_double = tensor.eq(maxprob[:-1], maxprob[1:])
        maxprob = tensor.switch(tensor.concatenate([tensor.zeros((1,B)), is_double]),
                                C*tensor.ones_like(maxprob), maxprob)
        # maxprob = theano.printing.Print('maxprob')(maxprob.T).T

        # returns two values :
        # label : (T x) T x B
        # label_length : (T x) B
        def recursion(maxp, p_mask, label_length, label):
            nonzero = p_mask * tensor.neq(maxp, C)
            nonzero_id = nonzero.nonzero()[0]

            new_label = tensor.set_subtensor(label[label_length[nonzero_id], nonzero_id], maxp[nonzero_id])
            new_label_length = tensor.switch(nonzero, label_length + numpy.int32(1), label_length)

            return new_label_length, new_label
            
        [label_length, label], _ = scan(fn=recursion,
                                        sequences=[maxprob, probs_mask],
                                        outputs_info=[tensor.zeros((B,),dtype='int32'),-tensor.ones((T,B))])

        return label[-1], label_length[-1]

    def prefix_search(self, probs, probs_mask=None):
        # Hard one...
        pass
        
        
 
# vim: set sts=4 ts=4 sw=4 sw=4 tw=0 et:
