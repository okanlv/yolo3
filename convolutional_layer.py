import numpy as np
from torch import nn

def make_convolutional_layer(modules, batch, h, w, c, n, groups, size, stride, padding, activation, batch_normalize, binary, xnor, params):
    l = {
        'type': 'CONVOLUTIONAL',
        'groups': groups,
        'h': h,
        'w': w,
        'c': c,
        'n': n,
        'binary': binary,
        'xnor': xnor,
        'batch': batch,
        'stride': stride,
        'size': size,
        'pad': padding,
        'batch_normalize': batch_normalize,
        'nweights': c/groups*n*size*size,
        'nbiases': n,
        'scale': np.sqrt(2/(size*size*c/groups))
    }

    modules.add_module('conv_%d' % params['index'], nn.Conv2d(in_channels=c,
                                                              out_channels=n,
                                                              kernel_size=size,
                                                              stride=stride,
                                                              padding=padding,
                                                              bias=not batch_normalize))
    if batch_normalize:
        modules.add_module('batch_norm_%d' % params['index'], nn.BatchNorm2d(n))
    if activation == 'leaky':
        modules.add_module('leaky_%d' % params['index'], nn.LeakyReLU(0.1))
    elif activation == 'logistic':
        modules.add_module('logistic_%d' % params['index'], nn.Sigmoid())


    # for(i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_normal();
    # int out_w = convolutional_out_width(l);
    # int out_h = convolutional_out_height(l);
    # l.out_h = out_h;
    # l.out_w = out_w;
    # l.out_c = n;
    # l.outputs = l.out_h * l.out_w * l.out_c;
    # l.inputs = l.w * l.h * l.c;

    # fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);

    return l;