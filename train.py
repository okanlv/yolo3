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
parser.add_argument('-nogpu', default=0, help='avg')

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
    os.makedirs('checkpoints', exist_ok=True)

    # Configure run
    data_config = parse_data_config(opt.datacfg)
    data_config.setdefault('train', 'data/train.list')
    data_config.setdefault('backup', '/backup/')

    train_images = data_config['train']
    backup_directory = data_config['backup']

    print('data_config', json.dumps(data_config, indent=4))



    ## TODO modify model loader nets[i] = load_network(cfgfile, weightfile, clear); in detector.c line 26

    num_classes = data_config['classes']
    if platform == 'darwin':  # MacOS (local)
        train_path = data_config['train']
    else:  # linux (cloud, i.e. gcp)
        train_path = './coco/trainvalno5k.part'

    # Initialize model
    model = Darknet(opt.cfg, opt.backbone_cfg, opt.img_size)

    net = model.hyperparams
    net['batch'] = opt.batch_size
    net['height'] = opt.img_size
    net['width'] = opt.img_size

    if opt.clear:
        net['seen'] = 0

    # net['learning_rate'] *= opt.ngpus  # TODO
    print('net', json.dumps(net, indent=4))

    backbone_weights_path = 'weights/darknet53.conv.74'
    if backbone_weights_path.endswith('.74'):  # saved in darknet format
        load_backbone_weights(model, backbone_weights_path)



    # imgs = net['batch'] *  net['subdivisions'] * ngpus TODO
    imgs = net['batch'] * net['subdivisions']

    print("Learning Rate: {}, Momentum: {}, Decay: {}\n".format(net['learning_rate'], net['momentum'], net['decay']));

    l = model.module_defs[-1]

    # Get dataloader
    dataloader = load_images_and_labels(train_path, batch_size=opt.batch_size, img_size=opt.img_size, augment=True,
                                        randomize=l['random'])

    ###########################################################################

    classes = l['classes']
    jitter = l['jitter']

    args = get_base_args(net)
    args['n'] = imgs  # total number of images per batch
    args['m'] = dataloader.nF  # total number of images
    args['classes'] = classes
    args['jitter'] = jitter
    args['num_boxes'] = l['max_boxes']
    # args.d = & buffer TODO
    args['type'] = 'DETECTION_DATA'
    args['threads'] = 64


    # reload saved optimizer state
    start_epoch = 0
    best_loss = float('inf')
    if opt.resume:
        checkpoint = torch.load('checkpoints/latest.pt', map_location='cpu')

        model.load_state_dict(checkpoint['model'])
        if torch.cuda.device_count() > 1:
            print('Using ', torch.cuda.device_count(), ' GPUs')
            model = nn.DataParallel(model)
        model.to(device).train()

        # # Transfer learning
        # for i, (name, p) in enumerate(model.named_parameters()):
        #     #name = name.replace('module_list.', '')
        #     #print('%4g %70s %9s %12g %20s %12g %12g' % (
        #     #    i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
        #     if p.shape[0] != 650:  # not YOLO layer
        #         p.requires_grad = False

        # Set optimizer
        # optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9, weight_decay=5e-4, nesterov=True)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer'])

        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']

        del checkpoint  # current, saved
    else:
        if torch.cuda.device_count() > 1:
            print('Using ', torch.cuda.device_count(), ' GPUs')
            model = nn.DataParallel(model)
        model.to(device).train()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=5e-4)

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
    avg_loss = -1
    t0, t1 = time.time(), time.time()
    print('%10s' * 16 % (
        'Epoch', 'Batch', 'x', 'y', 'w', 'h', 'conf', 'cls', 'total', 'P', 'R', 'nGT', 'TP', 'FP', 'FN', 'time'))

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
    for epoch in range(opt.epochs):
        epoch += start_epoch

        # Multi-Scale Training
        # img_size = random.choice(range(10, 20)) * 32
        # dataloader = load_images_and_labels(train_path, batch_size=opt.batch_size, img_size=img_size, augment=True)
        # print('Running this epoch with image size %g' % img_size)

        # Update scheduler
        # if epoch % 25 == 0:
        #     scheduler.last_epoch = -1  # for cosine annealing, restart every 25 epochs
        # scheduler.step()
        # if epoch <= 100:
        # for g in optimizer.param_groups:
        # g['lr'] = 0.0005 * (0.992 ** epoch)  # 1/10 th every 250 epochs
        # g['lr'] = 0.001 * (0.9773 ** epoch)  # 1/10 th every 100 epochs
        # g['lr'] = 0.0005 * (0.955 ** epoch)  # 1/10 th every 50 epochs
        # g['lr'] = 0.0005 * (0.926 ** epoch)  # 1/10 th every 30 epochs

        ui = -1
        rloss = defaultdict(float)  # running loss
        metrics = torch.zeros(4, num_classes)
        for i, (imgs, targets) in enumerate(dataloader):

            nGT = sum([len(x) for x in targets])
            if nGT < 1:
                continue

            loss = model(imgs.to(device), targets, requestPrecision=True, epoch=epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ui += 1
            metrics += model.losses['metrics']
            for key, val in model.losses.items():
                rloss[key] = (rloss[key] * ui + val) / (ui + 1)

            # Precision
            precision = metrics[0] / (metrics[0] + metrics[1] + 1e-16)
            k = (metrics[0] + metrics[1]) > 0
            if k.sum() > 0:
                mean_precision = precision[k].mean()
            else:
                mean_precision = 0

            # Recall
            recall = metrics[0] / (metrics[0] + metrics[2] + 1e-16)
            k = (metrics[0] + metrics[2]) > 0
            if k.sum() > 0:
                mean_recall = recall[k].mean()
            else:
                mean_recall = 0

            s = ('%10s%10s' + '%10.3g' * 14) % (
                '%g/%g' % (epoch, opt.epochs - 1), '%g/%g' % (i, len(dataloader) - 1), rloss['x'],
                rloss['y'], rloss['w'], rloss['h'], rloss['conf'], rloss['cls'],
                rloss['loss'], mean_precision, mean_recall, model.losses['nGT'], model.losses['TP'],
                model.losses['FP'], model.losses['FN'], time.time() - t1)
            t1 = time.time()
            print(s)

            # if i == 1:
            #    return

        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '\n')

        # Update best loss
        loss_per_target = rloss['loss'] / rloss['nGT']
        if loss_per_target < best_loss:
            best_loss = loss_per_target

        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      'best_loss': best_loss,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, 'checkpoints/latest.pt')

        # Save best checkpoint
        if best_loss == loss_per_target:
            os.system('cp checkpoints/latest.pt checkpoints/best.pt')

        # Save backup checkpoint
        if (epoch > 0) & (epoch % 5 == 0):
            os.system('cp checkpoints/latest.pt checkpoints/backup' + str(epoch) + '.pt')

    # Save final model
    dt = time.time() - t0
    print('Finished %g epochs in %.2fs (%.2fs/epoch)' % (epoch, dt, dt / (epoch + 1)))


if __name__ == '__main__':
    torch.cuda.empty_cache()

    opt.h = opt.height
    opt.w = opt.width

    if opt.nogpu:
        opt.gpu_index = -1

    opt.gpu = 0

    if opt.gpu_list:
        opt.gpus = [int(gpu) for gpu in opt.gpu_list.split(',')]
        opt.ngpus = len(opt.gpus)
    else:
        opt.gpu = opt.gpu_index
        opt.gpus = opt.gpu
        opt.ngpus = 1

    main(opt)
    torch.cuda.empty_cache()
