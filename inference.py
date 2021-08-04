import argparse
import os
import numpy as np
import cv2

from modeling_infer.sync_batchnorm.replicate import patch_replication_callback
from modeling_infer.deeplab import *
from utils.visualize import Visualize as Vs
from torchvision import transforms
from PIL import Image


def gn(planes):
	return nn.GroupNorm(16, planes)
def syncbn(planes):
	return nn.BatchNorm2d(planes)
def bn(planes):
	return nn.BatchNorm2d(planes)
def syncabn(devices):
	return False
	def _syncabn(planes):
		return InplaceABNSync(planes, devices)
	return _syncabn
def abn(planes):
	return InPlaceABN(planes)

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.vs = Vs(args.dataset)
        # Define network
        self.nclass = 2
        model = DeepLab(num_classes=self.nclass,
                freeze_bn=args.freeze_bn)

        self.model = model
        self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
        patch_replication_callback(self.model)
        self.model = self.model.cuda()

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            # if args.ft:
            #     args.start_epoch = 0
            # else:
            #     args.start_epoch = checkpoint['epoch']

            #self.model.module.load_state_dict(checkpoint['state_dict'])
            pretrained_dict = checkpoint['state_dict']
            model_dict = {}
            state_dict = self.model.module.state_dict()
            for k, v in pretrained_dict.items():
                if k in state_dict:
                    model_dict[k] = v
                else:
                    print(f'{k} is not in model_dict')
            state_dict.update(model_dict)
            self.model.module.load_state_dict(state_dict)

            print("=> loaded checkpoint '{}'"
                  .format(args.resume))


    def predict(self):
        self.model.eval()

        normalize = transforms.Normalize(mean=[0.279, 0.293, 0.290],
                                         std=[0.197, 0.198, 0.201])
        # normalize = transforms.Normalize(mean=[0.290, 0.293, 0.279],
        #                                  std=[0.201, 0.198, 0.197])
        resize_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        source = 'samples'
        files = os.listdir(source)
        files.sort()
        for img_name_index in range(len(files)):
            img_name = files[img_name_index]
            img_path = os.path.join(source, img_name)
            img = Image.open(img_path).convert('RGB')
            input = resize_transform(img)
            # input = input[:,480:,:]
            input = input.unsqueeze(0).cuda()
            with torch.no_grad():
                output = self.model(input)
            pred = output.data.cpu().numpy()

            cls1 = pred[:, 0, :, :]
            cls2 = pred[:, 1, :, :]
            # cls3 = pred[:, 2, :, :]
            # cls = cls2 + cls3
            # pred = (cls1 > 3).astype(np.float)
            pred = np.argmax(pred, axis=1)
            binary = (pred[0]*255).astype(np.uint8)

            contours, hierchary = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            final = cv2.drawContours(binary, contours, -1, 0, 3)
            num, labels = cv2.connectedComponents(final)
            area_label = 0
            area = 0
            for li in range(1, num):
                area_temp = np.sum(labels == li)
                if area_temp > area:
                    area = area_temp
                    area_label = li
            empty_img = np.zeros_like(binary)
            empty_img[labels == area_label] = 255
            # empty_img[666:,5:1275] = 255

            cv2.imwrite('labels/' + img_name, empty_img)
            pred = (empty_img.astype(np.int64) / 255)[None]
            self.vs.predict_id(pred, [img_name], 'results')
            self.vs.predict_color(pred, input.cpu().numpy(), [img_name], 'results')


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='bdd',
                        #choices=['pascal', 'coco', 'cityscapes', 'bdd'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=1,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--sync-bn', default=False, action='store_true',
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # model
    parser.add_argument('--model', type=str, default='deeplabv3+',
			choices=['deeplabv3+', 'deeplabv3', 'fpn'])
    # Normalizations
    parser.add_argument('--norm', type=str, default='gn',
			choices=['gn', 'bn', 'abn', 'ign'],
			help='normalization methods')

    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default='weights/checkpoint_convert.pth.tar',
                        help='put the path to resuming file if needed')
    parser.add_argument('--decoder', type=str, default=None,
			help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    # test option
    parser.add_argument('--test', action='store_true', default=True,
			help='do not generate exp, nor train.')

    # additional option
    parser.add_argument('--labels', type=str, default=None)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.batch_size is None:
        args.batch_size = 1

    if args.test_batch_size is None:
        args.test_batch_size = 1

    if args.checkname is None:
        args.checkname = args.model+'-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    trainer.predict()



if __name__ == "__main__":
   main()
