from collections import defaultdict
import sys

import torch.nn as nn

from utils.utils import *
from parser import *


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    # params = {
    #     'h': hyperparams['height'],
    #     'w': hyperparams['width'],
    #     'c': hyperparams['channels'],
    #     'inputs': hyperparams['inputs'],
    #     'batch': hyperparams['batch'],
    #     'time_steps': hyperparams['time_steps']
    # }

    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        # params['index'] = i
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            # parse_convolutional(modules, module_def, params)
            module_def['batch_normalize'] = int(module_def['batch_normalize']) if 'batch_normalize' in module_def else 0
            module_def['filters'] = int(module_def['filters']) if 'filters' in module_def else 1
            module_def['size'] = int(module_def['size']) if 'size' in module_def else 1
            module_def['stride'] = int(module_def['stride']) if 'stride' in module_def else 1
            module_def['pad'] = int(module_def['pad']) if 'pad' in module_def else 0
            module_def['padding'] = int(module_def['padding']) if 'padding' in module_def else 0
            module_def['groups'] = int(module_def['groups']) if 'groups' in module_def else 1  # TODO
            module_def.setdefault('activation', 'logistic')
            if module_def['pad']:
                module_def['padding'] = module_def['size'] // 2

            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=module_def['filters'],
                                                        kernel_size=module_def['size'],
                                                        stride=module_def['stride'],
                                                        padding=module_def['padding'],
                                                        bias=not module_def['batch_normalize']))
            if module_def['batch_normalize']:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(module_def['filters']))
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))
            elif module_def['activation'] == 'logistic':
                modules.add_module('logistic_%d' % i, nn.Sigmoid())
            filters = module_def['filters']

        elif module_def['type'] == 'upsample':
            upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())

        elif module_def['type'] == 'yolo':
            module_def['anchor_idxs'] = [int(x) for x in module_def['mask'].split(',')]
            # Extract anchors
            anchors = [float(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            module_def['anchors'] = [anchors[i] for i in module_def['anchor_idxs']]

            module_def['classes'] = int(module_def['classes']) if 'classes' in module_def else 20
            module_def['total'] = int(module_def['num']) if 'num' in module_def else 1
            module_def['max_boxes'] = int(module_def['max']) if 'max' in module_def else 90
            module_def['jitter'] = float(module_def['jitter']) if 'jitter' in module_def else .2
            module_def['ignore_thresh'] = float(module_def['ignore_thresh']) if 'ignore_thresh' in module_def else .5
            module_def['truth_thresh'] = float(module_def['truth_thresh']) if 'truth_thresh' in module_def else float(1)
            module_def['random'] = int(module_def['random']) if 'random' in module_def else 0
            module_def['truths'] = 90*(4 + 1)
            module_def.setdefault('map', '')  # TODO

            # Define detection layer
            yolo_layer = YOLOLayer(module_def['anchors'], module_def['classes'], hyperparams['height'],
                                   module_def['anchor_idxs'])
            modules.add_module('yolo_%d' % i, yolo_layer)

        # Register module list and number of output filters
        module_def['learning_rate_scale'] = float(module_def['learning_rate_scale'])\
            if 'learning_rate_scale' in module_def else float(1)
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):

    def __init__(self, anchors, nC, img_dim, anchor_idxs):
        super(YOLOLayer, self).__init__()

        anchors = [(a_w, a_h) for a_w, a_h in anchors]  # (pixels)
        nA = len(anchors)

        self.anchors = anchors
        self.nA = nA  # number of anchors (3)
        self.nC = nC  # number of classes (80)
        self.bbox_attrs = 5 + nC
        self.img_dim = img_dim  # from hyperparams in cfg file, NOT from parser

        if anchor_idxs[0] == (nA * 2):  # 6
            stride = 32
        elif anchor_idxs[0] == nA:  # 3
            stride = 16
        else:
            stride = 8

        # Build anchor grids
        nG = int(self.img_dim / stride)
        self.scaled_anchors = torch.FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, nA, 1, 1))

    def forward(self, p, targets=None, requestPrecision=False, epoch=None):
        FT = torch.cuda.FloatTensor if p.is_cuda else torch.FloatTensor

        bs = p.shape[0]  # batch size
        nG = p.shape[2]  # number of grid points
        stride = self.img_dim / nG

        self.grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).float()
        self.grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).float()

        if p.is_cuda and not self.grid_x.is_cuda:
            self.grid_x, self.grid_y = self.grid_x.cuda(), self.grid_y.cuda()
            self.anchor_w, self.anchor_h = self.anchor_w.cuda(), self.anchor_h.cuda()

        # p.view(12, 255, 13, 13) -- > (12, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        # Get outputs, every grid has x, y, w, h for each scale, total of bs x nA x grid x grid = bs x 3 x 13 x 13
        x = torch.sigmoid(p[..., 0])  # Center x
        y = torch.sigmoid(p[..., 1])  # Center y
        w = p[..., 2]  # Width
        h = p[..., 3]  # Height
        width = torch.exp(w.data) * self.anchor_w
        height = torch.exp(h.data) * self.anchor_h

        # Add offset and scale with anchors (in grid space, i.e. 0-13)
        pred_boxes = FT(bs, self.nA, nG, nG, 4)
        pred_conf = p[..., 4]  # Conf
        pred_cls = p[..., 5:]  # Class

        # Training
        if targets is not None:
            BCEWithLogitsLoss1 = nn.BCEWithLogitsLoss(size_average=False)
            BCEWithLogitsLoss2 = nn.BCEWithLogitsLoss(size_average=True)
            MSELoss = nn.MSELoss(size_average=False)  # version 0.4.0
            CrossEntropyLoss = nn.CrossEntropyLoss()

            if requestPrecision:
                gx = self.grid_x[:, :, :nG, :nG]
                gy = self.grid_y[:, :, :nG, :nG]
                pred_boxes[..., 0] = x.data + gx - width / 2  # x coordinate of left  -->  x.data - width / 2 gives
                pred_boxes[..., 1] = y.data + gy - height / 2  # y coordinate of top
                pred_boxes[..., 2] = x.data + gx + width / 2  # x coordinate of right
                pred_boxes[..., 3] = y.data + gy + height / 2  # y coordinate of bot

            tx, ty, tw, th, mask, tcls, TP, FP, FN, TC = \
                build_targets(pred_boxes, pred_conf, pred_cls, targets, self.scaled_anchors, self.nA, self.nC, nG,
                              requestPrecision)
            tcls = tcls[mask]
            if x.is_cuda:
                tx, ty, tw, th, mask, tcls = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda(), mask.cuda(), tcls.cuda()

            # Mask outputs to ignore non-existing objects (but keep confidence predictions)
            nM = mask.sum().float()
            nGT = sum([len(x) for x in targets])
            if nM > 0:
                lx = 5 * MSELoss(x[mask], tx[mask]) / nM
                ly = 5 * MSELoss(y[mask], ty[mask]) / nM
                lw = 5 * MSELoss(w[mask], tw[mask]) / nM
                lh = 5 * MSELoss(h[mask], th[mask]) / nM
                lconf = BCEWithLogitsLoss1(pred_conf[mask], mask[mask].float()) / nM

                lcls = CrossEntropyLoss(pred_cls[mask], torch.argmax(tcls, 1))
                # lcls = nM * BCEWithLogitsLoss2(pred_cls[mask], tcls.float())
            else:
                lx, ly, lw, lh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0]), FT([0]), FT([0])

            # loss term from no objects
            lconf += BCEWithLogitsLoss2(pred_conf[~mask], mask[~mask].float())

            loss = lx + ly + lw + lh + lconf + lcls
            i = torch.sigmoid(pred_conf[~mask]) > 0.99
            FPe = torch.zeros(self.nC)
            if i.sum() > 0:
                FP_classes = torch.argmax(pred_cls[~mask][i], 1)
                for c in FP_classes:
                    FPe[c] += 1

            return loss, loss.item(), lx.item(), ly.item(), lw.item(), lh.item(), lconf.item(), lcls.item(), \
                   nGT, TP, FP, FPe, FN, TC

        else:
            pred_boxes[..., 0] = x.data + self.grid_x
            pred_boxes[..., 1] = y.data + self.grid_y
            pred_boxes[..., 2] = width
            pred_boxes[..., 3] = height

            # If not in training phase return predictions
            output = torch.cat((pred_boxes.view(bs, -1, 4) * stride,
                                torch.sigmoid(pred_conf.view(bs, -1, 1)), pred_cls.view(bs, -1, self.nC)), -1)
            return output.data


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, backbone_config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        # 1st section contains training def, discarded
        # last 3 layers are specific to the classification, discarded before initializing detection modules
        self.backbone_module_defs = parse_model_config(backbone_config_path)
        self.seen = 0  # Model is not trained for detection yet, so no images are seen

        for i in range(1, len(self.backbone_module_defs) - 3):
            if self.module_defs[i] == self.backbone_module_defs[i]:
                continue
            else:
                sys.exit("Backbone configuration '{}' is not compatible with yolov3 configuration '{}'"
                         .format(backbone_config_path, config_path))

        self.module_defs[0]['height'] = img_size  # input image size is changed
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.loss_names = ['loss', 'x', 'y', 'w', 'h', 'conf', 'cls', 'nGT', 'TP', 'FP', 'FPe', 'FN', 'TC']

    def forward(self, x, targets=None, requestPrecision=False, epoch=None):
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample']:
                x = module(x)
            elif module_def['type'] == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo':
                # Train phase: get loss
                if is_training:
                    x, *losses = module[0](x, targets, requestPrecision, epoch)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module(x)
                output.append(x)
            layer_outputs.append(x)

        if is_training:
            self.losses['nGT'] /= 3
            self.losses['TC'] /= 3
            metrics = torch.zeros(4, len(self.losses['FPe']))  # TP, FP, FN, target_count

            ui = np.unique(self.losses['TC'])[1:]
            for i in ui:
                j = self.losses['TC'] == float(i)
                metrics[0, i] = (self.losses['TP'][j] > 0).sum().float()  # TP
                metrics[1, i] = (self.losses['FP'][j] > 0).sum().float()  # FP
                metrics[2, i] = (self.losses['FN'][j] == 3).sum().float()  # FN
            metrics[3] = metrics.sum(0)
            metrics[1] += self.losses['FPe']

            self.losses['TP'] = metrics[0].sum()
            self.losses['FP'] = metrics[1].sum()
            self.losses['FN'] = metrics[2].sum()
            self.losses['TC'] = 0
            self.losses['metrics'] = metrics

        return sum(output) if is_training else torch.cat(output, 1)


def load_weights(self, weights_path):
    """Parses and loads the weights stored in 'weights_path'"""

    print("Loading weights from '{}'...".format(weights_path))

    # Open the weights file
    fp = open(weights_path, 'rb')
    header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

    # Needed to write header when saving weights
    self.header_info = header

    self.seen = header[3]
    weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
    fp.close()

    ptr = 0
    # conv_number = 0
    for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
        if module_def['type'] == 'convolutional':
            # conv_number += 1
            conv_layer = module[0]
            if module_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w
            # print(i, module_def['type'], num_w, conv_number)
    if len(weights) - ptr == 0:
        print("yolov3 weights are loaded from '{}' successfully".format(weights_path))
    else:
        sys.exit("yolov3 weights '{}' are corrupted".format(weights_path))


def load_backbone_weights(self, weights_path):
    """Parses and loads the weights stored in 'weights_path'"""

    print("Loading weights from '{}'...".format(weights_path))

    # Open the weights file
    fp = open(weights_path, 'rb')
    header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

    # Needed to write header when saving weights
    self.header_info = header

    weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
    fp.close()

    ptr = 0
    backbone_len = len(self.backbone_module_defs)-4
    # conv_number = 0
    for i, (module_def, module) in enumerate(zip(self.module_defs[:backbone_len], self.module_list[:backbone_len])):
        if module_def['type'] == 'convolutional':
            # conv_number += 1
            conv_layer = module[0]
            if module_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w
            # print(i, module_def['type'], num_w, conv_number)

    if len(weights) - ptr == 0:
        print("Backbone weights are loaded from '{}' successfully".format(weights_path))
    else:
        sys.exit("Backbone weights '{}' are corrupted".format(weights_path))


"""
    @:param path    - path of the new weights file
    @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
"""


def save_weights(self, path, cutoff=-1):
    fp = open(path, 'wb')
    self.header_info[3] = self.seen
    self.header_info.tofile(fp)

    # Iterate through layers
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            # If batch norm, load bn first
            if module_def['batch_normalize']:
                bn_layer = module[1]
                bn_layer.bias.data.cpu().numpy().tofile(fp)
                bn_layer.weight.data.cpu().numpy().tofile(fp)
                bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                bn_layer.running_var.data.cpu().numpy().tofile(fp)
            # Load conv bias
            else:
                conv_layer.bias.data.cpu().numpy().tofile(fp)
            # Load conv weights
            conv_layer.weight.data.cpu().numpy().tofile(fp)

    fp.close()
