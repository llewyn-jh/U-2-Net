import warnings
warnings.filterwarnings("ignore")

import torch
from torch.autograd import Variable
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim as optim

import glob
import os
from tqdm import tqdm

from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	return loss0, loss1, loss2, loss3, loss4, loss5, loss6

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train U^2 net to remove background.')

    parser.add_argument('--dataset',
                        required=True,
                        metavar="/path/to/dataset/",
                        help='Directory of image and label dataset')
    parser.add_argument('--saved_model',
                        required=False,
                        metavar="/path/to/saved_model dir",
                        help="Path to load a saved model .tar file")
    parser.add_argument('--save_model',
                        required=True,
                        metavar="/path/to/save dir",
                        help="Path to save a trained model")
    parser.add_argument('--save_pt',
                        default=2000,
                        type=int,
                        help="save a checkpoint per save point")
    parser.add_argument('--epoch',
                        default=100000,
                        type=int,
                        help="Set epoch for training model")
    parser.add_argument('--batch_size',
                        default=16,
                        type=int,
                        help="Set batch size for training model")
    parser.add_argument('--workers',
                        default=1,
                        type=int,
                        help="Set the number of cpu cores for loading dataset")

    args = parser.parse_args()

    # Validate arguments
    print("---")
    print("Train configuration")
    print("Dataset: ", args.dataset)
    print("Saved model: ", args.saved_model)
    print("Save model: ", args.save_model)
    print("Save point: ", args.save_pt)
    print("Epoch: ", args.epoch)
    print("Batch size: ", args.batch_size)
    print("Workers: ", args.workers)

    # Set the directory of training datasets
    data_dir = args.dataset
    tra_image_dir = os.path.join(data_dir, 'image/')
    tra_label_dir = os.path.join(data_dir, 'target/')

    image_ext = '.jpg'
    label_ext = '.png'
    saved_model_ext = '.tar'

    if args.saved_model:
        saved_model_paths = glob.glob(args.saved_model + '*' + saved_model_ext)
        sorted_model_paths = sorted(saved_model_paths, key=lambda path: -os.stat(path).st_mtime)
        last_modified_model_path = sorted_model_paths[0]

    epoch_num = args.epoch

    tra_img_name_list = glob.glob(tra_image_dir + '*' + image_ext)

    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split(os.sep)[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1,len(bbb)):
            imidx = imidx + "." + bbb[i]

        tra_lbl_name_list.append(tra_label_dir + imidx + label_ext)

    # Validate train datasets
    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")

    # Data augmentation
    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)]))
    # Data iterator
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # Initialize network and optimizer
    net = U2NET(3, 1)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    if args.saved_model:
        print("Loading the last saved model ...")
        print("Loaded the model: %s" % last_modified_model_path)
        checkpoint = torch.load(last_modified_model_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        device = torch.device("cuda")
        net.to(device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        saved_epoch = checkpoint['epoch']
        print('Retraining from %d' % saved_epoch)
        loss = checkpoint['loss']
    else:
        saved_epoch = 0

    # Set a distributed train with multi GPU
    net = torch.nn.DataParallel(net)
    net.cuda()

    # Train
    print("---start retraining from %d" % saved_epoch)
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = args.save_pt # save the model every set iterations

    # Tensorboard
    writer = SummaryWriter()

    for epoch in range(saved_epoch, epoch_num):
        net.train()

        for data in tqdm(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs = data['image']
            labels = data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            inputs_v = Variable(inputs.cuda(), requires_grad=False)
            labels_v = Variable(labels.cuda(), requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss0, loss1, loss2, loss3, loss4, loss5, loss6 = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
            loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

            loss.backward()
            optimizer.step()

            # tensorboard
            writer.add_scalar('Loss/d0', loss0, epoch)
            writer.add_scalar('Loss/d1', loss1, epoch)
            writer.add_scalar('Loss/d2', loss2, epoch)
            writer.add_scalar('Loss/d3', loss3, epoch)
            writer.add_scalar('Loss/d4', loss4, epoch)
            writer.add_scalar('Loss/d5', loss5, epoch)
            writer.add_scalar('Loss/d6', loss6, epoch)
            writer.add_scalar('Loss/target', loss/7, epoch)

            # print statistics
            running_loss += loss.data
            running_tar_loss += loss0.data

            # save checkpoint
            if ite_num % save_frq == 0:

                print("epoch: %3d/%3d, train loss: %3f, target loss: %3f " % (epoch + 1, epoch_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

                save_path = args.save_model + "u2net_bce_itr_%d_train_%3f_tar_%3f.tar" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val)

                torch.save({'epoch': epoch,
                            'model_state_dict': net.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'loss': loss},
                            save_path)

                filst = sorted(glob.glob('./saved_models/u2net/*.tar'), key=os.path.getctime)
                if len(filst) > 1:
                    os.remove(filst[0])

                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss, loss0, loss1, loss2, loss3, loss4, loss5, loss6

    writer.close()
    print("Training is finished")
