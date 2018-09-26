import argparse
import time
import json

from models import *
from utils.datasets import *
from utils.utils import *
from parser import *

from torch.optim.lr_scheduler import LambdaLR

parser = argparse.ArgumentParser()
parser.add_argument('-epochs', type=int, default=160, help='number of epochs')
parser.add_argument('-batch_size', type=int, default=1, help='size of each image batch')
parser.add_argument('-datacfg', type=str, default='cfg/coco.data', help='data config file path')
parser.add_argument('-cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
parser.add_argument('-backbone_cfg', type=str, default='cfg/darknet53_448.cfg', help='cfg file path')
parser.add_argument('-img_size', type=int, default=32 * 13, help='size of each image dimension')
parser.add_argument('-resume', default=False, help='resume training flag')

parser.add_argument('-prefix', default='', help='prefix')
parser.add_argument('-thresh', default=.5, help='thresh')
parser.add_argument('-hier_thresh', default=.5, help='hier_thresh')
parser.add_argument('-cam_index', default=0, help='cam_index')
parser.add_argument('-frame_skip', default=0, help='frame_skip')
parser.add_argument('-avg', default=3, help='avg')

parser.add_argument('-gpu_index', default=0, help='gpu_index')
parser.add_argument('-gpu_list', default='', help='gpu_list')
parser.add_argument('-nogpu', default=0, help='do not use gpu if 1')

parser.add_argument('-clear', default=0, help='clear')
parser.add_argument('-fullscreen', default=0, help='fullscreen')
parser.add_argument('-width', default=0, help='width')
parser.add_argument('-height', default=0, help='height')
parser.add_argument('-fps', default=0, help='fps')

parser.add_argument('-weights', default='', help='weights')
parser.add_argument('-outfile', default='', help='outfile')

opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
device = 'cpu'

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if cuda:
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True


def main(opt):

    if opt.nogpu:
        opt.gpu_index = -1

    if not torch.cuda.is_available():
        opt.gpu_index = -1

    gpu_index = opt.gpu_index
    prefix = opt.prefix
    thresh = opt.thresh
    hier_thresh = opt.hier_thresh
    cam_index = opt.cam_index
    frame_skip = opt.frame_skip
    avg = opt.avg

    gpus = 0
    gpu = 0
    ngpus = 0

    if opt.gpu_list:
        gpus = [int(gpu) for gpu in opt.gpu_list.split(',')]
        ngpus = len(gpus)
    else:
        gpu = gpu_index
        gpus = gpu
        ngpus = 1

    clear = opt.clear
    fullscreen = opt.fullscreen
    width = opt.width
    height = opt.height
    fps = opt.fps

    datacfg = opt.datacfg
    cfgfile = opt.cfg
    weightfile = opt.weights
    # filename = opt.filename ?? what is this, detector.c
    # os.makedirs('checkpoints', exist_ok=True)

    # Configure run
    options = parse_data_config(opt.datacfg)
    options.setdefault('train', 'data/train.list')
    options.setdefault('backup', 'backup/')
    train_images = options['train']
    backup_directory = options['backup']
    os.makedirs(backup_directory, exist_ok=True)

    # char *base = basecfg(cfgfile);
    # printf("%s\n", base);
    print('data_config', json.dumps(options, indent=4))
    avg_loss = -1

    # TODO modify model loader nets[i] = load_network(cfgfile, weightfile, clear); in detector.c line 26
    # Initialize model
    model = Darknet(opt.cfg, opt.backbone_cfg, opt.img_size)

    net = model.hyperparams
    net['batch'] = opt.batch_size
    net['height'] = opt.img_size
    net['width'] = opt.img_size

    # net['learning_rate'] *= opt.ngpus  # TODO
    print('net', json.dumps(net, indent=4))

    # load the classifier weights
    backbone_weights_path = 'weights/darknet53.conv.74'
    load_backbone_weights(model, backbone_weights_path)

    # total number of images the networks saw during the training
    # should be run after the weights are loaded
    if opt.clear:
        net['seen'] = 0

    # imgs = net['batch'] *  net['subdivisions'] * ngpus TODO
    imgs = net['batch'] * net['subdivisions']
    print("Learning Rate: {}, Momentum: {}, Decay: {}\n".format(net['learning_rate'], net['momentum'], net['decay']))

    # extract the last layer information
    l = model.module_defs[-1]
    classes = l['classes']
    jitter = l['jitter']

    # Get dataloader
    dataloader = load_images_and_labels(train_images, batch_size=net['batch'], img_size=net['height'], augment=True,
                                        randomize=l['random'])
    args = get_base_args(net)
    args['n'] = imgs  # total number of images per batch
    args['m'] = dataloader.nF  # total number of training images
    args['classes'] = classes
    args['jitter'] = jitter
    args['num_boxes'] = l['max_boxes']
    # args.d = & buffer TODO
    args['type'] = 'DETECTION_DATA'
    args['threads'] = 64

    ###########################################################################

    # reload saved optimizer state

    # best_loss = float('inf')
    # start_epoch = 0
    # if opt.resume:
    #     checkpoint = torch.load('checkpoints/latest.pt', map_location='cpu')
    #
    #     model.load_state_dict(checkpoint['model'])
    #     if torch.cuda.device_count() > 1:
    #         print('Using ', torch.cuda.device_count(), ' GPUs')
    #         model = nn.DataParallel(model)
    #     model.to(device).train()
    #
    #     optimizer = torch.optim.Adam(model.parameters())
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #
    #     start_epoch = checkpoint['epoch'] + 1
    #     best_loss = checkpoint['best_loss']
    #
    #     del checkpoint  # current, saved
    # else:
    #     if torch.cuda.device_count() > 1:
    #         print('Using ', torch.cuda.device_count(), ' GPUs')
    #         model = nn.DataParallel(model)
    #     model.to(device).train()
    #
    #     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=5e-4)

    if torch.cuda.device_count() > 1:
        print('Using ', torch.cuda.device_count(), ' GPUs')
        model = nn.DataParallel(model)
    model.to(device).train()

    if net['adam']:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=net['learning_rate'],
                                     betas=(net['B1'], net['B2']),
                                     eps=net['eps'],
                                     weight_decay=net['decay'])
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=net['learning_rate'],
                                    momentum=net['momentum'],
                                    weight_decay=net['decay'])

    def get_current_rate(batch_num):
        # global net

        if batch_num < net['burn_in']:
            return (batch_num / net['burn_in']) ** net['power']

        if net['policy'] == 'constant':
            return 1
        elif net['policy'] == 'step':
            return net['scale'] ** (batch_num / net['step'])
        elif net['policy'] == 'steps':
            rate = 1
            for step in net['steps']:
                if batch_num < step:
                    return rate
                rate *= step
            return rate
        elif net['policy'] == 'exp':
            return net['gamma'] ** batch_num
        elif net['policy'] == 'poly':
            return (1 - batch_num / net['max_batches']) ** net['power']
        elif net['policy'] == 'random':
            return random() ** net['power']
        elif net['policy'] == 'sigmoid':
            return 1 / (1 + np.exp(net['gamma'] * (batch_num - net['step'])))
        else:
            print("{} policy is not implemented. Using constant policy".format(net['policy']))
            net['policy'] = 'constant'
            return 1

    scheduler = LambdaLR(optimizer, lr_lambda=get_current_rate)

    # modelinfo(model)
    # t0, t1 = time.time(), time.time()
    # print('%10s' * 16 % (
    #     'Epoch', 'Batch', 'x', 'y', 'w', 'h', 'conf', 'cls', 'total', 'P', 'R', 'nGT', 'TP', 'FP', 'FN', 'time'))

    ###################################################################################################################

    # count = 0
    # while get_current_batch(net) < net['max_batches']:  # TODO yolo3 get rand imgs frm imlist,no epoch
    #     # TODO define epoch = max_batches * batch_size / total_number_of_images
    #     count += 1
    #     if l['random'] and count % 10 == 0:  # TODO DONE
    #         print("Resizing\n")
    #         dim = random.randint(10, 19) * 32
    #         if get_current_batch(net)+200 > net['max_batches']:
    #             dim = 608
    #         print(dim)
    #         args['w'] = dim
    #         args['h'] = dim
    #
    #     time_now = time.time()
    #     loss = 0
    #
    #     net['seen'] += net['batch_size']
    #     loss = train_network(net, train)
    #
    #     if (net['seen'] / net['batch_size']) % net['subdivisions'] == 0:
    #         update_network(net)
    #
    #     if avg_loss < 0:
    #         avg_loss = loss
    #
    #     avg_loss = avg_loss*.9 + loss*.1
    #     i = get_current_batch(net)
    #     print("{}: {}, {} avg, {} rate, {} seconds, {} images\n"
    #           .format(i, loss, avg_loss, get_current_rate(i), time.time() - time_now, i * imgs))
    #     if i % 100 == 0:
    #         buff = backup_directory + 'model_' + str(i) + '.pt'
    #         print("Saving weights to {}".format(buff))
    #         # Save latest checkpoint
    #         checkpoint = {'current_batch': i,
    #                       'loss': loss,
    #                       'model': model.state_dict(),
    #                       'optimizer': optimizer.state_dict()}
    #         torch.save(checkpoint, buff)
    #         del checkpoint
    # # print("{}".format(i))
    # # Save the final checkpoint
    # buff = backup_directory + 'final_model_' + str(i) + '.pt'
    # print("Saving weights to {}".format(buff))
    # checkpoint = {'current_batch': i,
    #               'loss': loss,
    #               'model': model.state_dict(),
    #               'optimizer': optimizer.state_dict()}
    # torch.save(checkpoint, buff)
    # del checkpoint

    ###################################################################################################################
    count = 0
    optimizer.zero_grad()
    # following condition is not checked at every batch unlike darknet
    while get_current_batch(net) < net['max_batches']:
        for epoch in range(opt.epochs):
            # ui = -1  # ??
            # rloss = defaultdict(float)  # running loss ??
            # metrics = torch.zeros(4, classes)  # ??

            for i, (images, targets) in enumerate(dataloader):
                count += 1
                scheduler.step(count)
                time_now = time.time()
                loss = 0
                net['seen'] += net['batch'] # TODO change with total number of images in this batch

                nGT = sum([len(x) for x in targets]) # ??
                if nGT < 1: # ??
                    continue # ??

                loss = model(images.to(device), targets, requestPrecision=True, epoch=epoch)
                loss.backward()
                # loss = train_network(net, train)

                if (net['seen'] / net['batch']) % net['subdivisions'] == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                if avg_loss < 0:
                    avg_loss = loss

                avg_loss = avg_loss*.9 + loss*.1
                i = get_current_batch(net)
                print("{:.0f}: loss {:.5f}, avg loss {:.5f} , learning rate {:.15f} , {:.2f} seconds, {:.0f} images"
                      .format(i, loss.item(), avg_loss.item(), get_current_rate(i), time.time() - time_now, i * imgs))
                if i % 100 == 0:
                    buff = backup_directory + 'model_' + str(i) + '.pt'
                    print("Saving weights to {}".format(buff))
                    # Save latest checkpoint
                    checkpoint = {'current_batch': i,
                                  'loss': loss,
                                  'model': model.state_dict(),
                                  'optimizer': optimizer.state_dict()}
                    torch.save(checkpoint, buff)
                    del checkpoint

                # TODO check below section
                # ui += 1
                # metrics += model.losses['metrics']
                # for key, val in model.losses.items():
                #     rloss[key] = (rloss[key] * ui + val) / (ui + 1)

                # Precision
                # precision = metrics[0] / (metrics[0] + metrics[1] + 1e-16)
                # k = (metrics[0] + metrics[1]) > 0
                # if k.sum() > 0:
                #     mean_precision = precision[k].mean()
                # else:
                #     mean_precision = 0

                # # Recall
                # recall = metrics[0] / (metrics[0] + metrics[2] + 1e-16)
                # k = (metrics[0] + metrics[2]) > 0
                # if k.sum() > 0:
                #     mean_recall = recall[k].mean()
                # else:
                #     mean_recall = 0

                # s = ('%10s%10s' + '%10.3g' * 14) % (
                #     '%g/%g' % (epoch, opt.epochs - 1), '%g/%g' % (i, len(dataloader) - 1), rloss['x'],
                #     rloss['y'], rloss['w'], rloss['h'], rloss['conf'], rloss['cls'],
                #     rloss['loss'], mean_precision, mean_recall, model.losses['nGT'], model.losses['TP'],
                #     model.losses['FP'], model.losses['FN'], time.time() - t1)
                # t1 = time.time()
                # print(s)
                # TODO check above section
    # print("{}".format(i))
    # Save the final checkpoint
    buff = backup_directory + 'final_model_' + str(i) + '.pt'
    print("Saving weights to {}".format(buff))
    checkpoint = {'current_batch': i,
                  'loss': loss,
                  'model': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, buff)
    del checkpoint


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main(opt)
    torch.cuda.empty_cache()
