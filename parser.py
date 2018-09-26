import sys
from convolutional_layer import *


def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')] #get rid of comments and blank lines
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    net = module_defs[0]
    if not (net['type'] == 'net' or net['type'] == 'network'):
        sys.exit("First section must be [net] or [network]")

    net['batch'] = int(net['batch']) if 'batch' in net else 1
    net['learning_rate'] = float(net['learning_rate']) if 'learning_rate' in net else .001
    net['momentum'] = float(net['momentum']) if 'momentum' in net else .9
    net['decay'] = float(net['decay']) if 'decay' in net else .0001
    net['subdivisions'] = int(net['subdivisions']) if 'subdivisions' in net else 1
    net['time_steps'] = int(net['time_steps']) if 'time_steps' in net else 1
    net['notruth'] = int(net['notruth']) if 'notruth' in net else 0
    net['batch'] = int(net['batch'] / net['subdivisions'])
    net['batch'] = int(net['batch'] / net['time_steps'])
    net['random'] = int(net['random']) if 'random' in net else 0
    net['adam'] = int(net['adam']) if 'adam' in net else 0

    if net['adam']:
        net['B1'] = float(net['B1']) if 'B1' in net else .9
        net['B2'] = float(net['B2']) if 'B2' in net else .999
        net['eps'] = float(net['eps']) if 'eps' in net else .0000001

    net['height'] = int(net['height']) if 'height' in net else 0
    net['width'] = int(net['width']) if 'width' in net else 0
    net['channels'] = int(net['channels']) if 'channels' in net else 0
    net['inputs'] = int(net['inputs']) if 'inputs' in net else int(net['height'] * net['width'] * net['channels'])
    net['max_crop'] = int(net['max_crop']) if 'max_crop' in net else int(2 * net['width'])
    net['min_crop'] = int(net['min_crop']) if 'min_crop' in net else net['width']

    net['max_ratio'] = float(net['max_ratio']) if 'max_ratio' in net else net['max_crop'] / net['width']
    net['min_ratio'] = float(net['min_ratio']) if 'min_ratio' in net else net['min_crop'] / net['width']

    net['center'] = int(net['center']) if 'center' in net else 0
    net['clip'] = float(net['clip']) if 'clip' in net else float(0)

    net['angle'] = float(net['angle']) if 'angle' in net else float(0)
    net['aspect'] = float(net['aspect']) if 'aspect' in net else float(1)
    net['saturation'] = float(net['saturation']) if 'saturation' in net else float(1)
    net['exposure'] = float(net['exposure']) if 'exposure' in net else float(1)
    net['hue'] = float(net['hue']) if 'hue' in net else float(0)

    if not net['inputs'] and not (net['height'] and net['width'] and net['channels']):
        sys.exit("No input parameters supplied")

    net.setdefault('policy', 'constant')
    net['burn_in'] = int(net['burn_in']) if 'burn_in' in net else 0
    net['power'] = float(net['power']) if 'power' in net else float(4)

    if net['policy'] == 'step':
        net['step'] = int(net['step']) if 'step' in net else 1
        net['scale'] = float(net['scale']) if 'scale' in net else float(1)
    elif net['policy'] == 'steps':
        if not net['steps'] or not net['scales']:
            sys.exit("STEPS policy must have steps and scales in cfg file")

        net['steps'] = [int(step) for step in net['steps'].split(',')]
        net['scales'] = [float(scale) for scale in net['scales'].split(',')]

        assert len(net['steps']) == len(net['scales'])

        net['n'] = len(net['steps'])
    elif net['policy'] == 'exp':
        net['gamma'] = float(net['gamma']) if 'gamma' in net else float(1)
    elif net['policy'] == 'sigmoid':
        net['gamma'] = float(net['gamma']) if 'gamma' in net else float(1)
        net['step'] = int(net['step']) if 'step' in net else 1
    elif net['policy'] == 'poly' or net['policy'] == 'random':
        print("policy '{}' not implemented in darknet".format(net['policy']))

    net['max_batches'] = int(net['max_batches']) if 'max_batches' in net else 0
    net['seen'] = 0

    return module_defs


def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    options['classes'] = int(options['classes'])
    return options


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