import numpy
import theano
from theano import tensor

@theano.compile.ops.as_op(itypes=[tensor.imatrix, tensor.ivector, tensor.imatrix, tensor.ivector],
                          otypes=[tensor.ivector])
def batch_edit_distance(a, a_len, b, b_len):
    B = a.shape[0]
    assert b.shape[0] == B

    for i in range(B):
        print "A:", a[i, :a_len[i]]
        print "B:", b[i, :b_len[i]]

    q = max(a.shape[1], b.shape[1]) * numpy.ones((B, a.shape[1]+1, b.shape[1]+1), dtype='int32')
    q[:, 0, 0] = 0

    for i in range(a.shape[1]+1):
        for j in range(b.shape[1]+1):
            if i > 0:
                q[:, i, j] = numpy.minimum(q[:, i, j], q[:, i-1, j]+1)
            if j > 0:
                q[:, i, j] = numpy.minimum(q[:, i, j], q[:, i, j-1]+1)
            if i > 0 and j > 0:
                q[:, i, j] = numpy.minimum(q[:, i, j], q[:, i-1, j-1]+numpy.not_equal(a[:, i-1], b[:, j-1]))
    return q[numpy.arange(B), a_len, b_len]

# vim: set sts=4 ts=4 sw=4 tw=0 et :
