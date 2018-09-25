import sys
from convolutional_layer import *


def parse_convolutional(modules, options, params):
    n = int(options['filters']) if 'filters' in options else 1
    size = int(options['size']) if 'size' in options else 1
    stride = int(options['stride']) if 'stride' in options else 1
    pad = int(options['pad']) if 'pad' in options else 0
    padding = int(options['padding']) if 'padding' in options else 0
    groups = int(options['groups']) if 'groups' in options else 1
    if pad:
        padding = size // 2

    activation = options['activation'] if 'activation' in options else 'logistic'

    h = params['h']
    w = params['w']
    c = params['c']
    batch = params['batch']

    if not (h and w and c):
        sys.exit("Layer before convolutional layer must output image.")

    batch_normalize = int(options['batch_normalize']) if 'batch_normalize' in options else 0
    binary = int(options['binary']) if 'binary' in options else 0
    xnor = int(options['xnor']) if 'xnor' in options else 0

    layer = make_convolutional_layer(modules,batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize, binary, xnor, params);

    options['flipped'] = int(options['flipped']) if 'flipped' in options else 0
    options['dot'] = float(options['dot']) if 'dot' in options else float(0)

    return layer


def get_base_args(net_config):
    return {
        'w': net_config['width'],
        'h': net_config['height'],
        'size': net_config['width'],
        'min': net_config['min_crop'],
        'max': net_config['max_crop'],
        'angle': net_config['angle'],
        'aspect': net_config['aspect'],
        'exposure': net_config['exposure'],
        'center': net_config['center'],
        'saturation': net_config['saturation'],
        'hue': net_config['hue']
    }