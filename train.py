from model import *
from dataset import *

import torch
import torch.nn as nn

from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from statistics import mean


class Train:
    def __init__(self, args):
        self.mode = args.mode
        self.train_continue = args.train_continue

        self.scope = args.scope
        self.dir_checkpoint = args.dir_checkpoint
        self.dir_log = args.dir_log

        self.dir_data = args.dir_data
        self.dir_result = args.dir_result

        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size

        self.lr = args.lr

        self.ny_in = args.ny_in
        self.nx_in = args.nx_in
        self.nch_in = args.nch_in

        self.ny_load = args.ny_load
        self.nx_load = args.nx_load
        self.nch_load = args.nch_load

        self.ny_out = args.ny_out
        self.nx_out = args.nx_out
        self.nch_out = args.nch_out

        self.nch_ker = args.nch_ker

        self.data_type = args.data_type
        self.norm = args.norm

        self.gpu_ids = args.gpu_ids

        self.num_freq_disp = args.num_freq_disp
        self.num_freq_save = args.num_freq_save

        self.name_data = args.name_data

        if self.gpu_ids and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % self.gpu_ids[0])
            torch.cuda.set_device(self.gpu_ids[0])
        else:
            self.device = torch.device("cpu")

    def save(self, dir_chck, net, optim, epoch):
        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
                    '%s/model_epoch%04d.pth' % (dir_chck, epoch))

    def load(self, dir_chck, net, optim=[], epoch=[], mode='train'):
        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

        print('Loaded %dth network' % epoch)

        if mode == 'train':
            net.load_state_dict(dict_net['net'])
            optim.load_state_dict(dict_net['optim'])

            return net, optim, epoch

        elif mode == 'test':
            net.load_state_dict(dict_net['net'])

            return net, epoch

    def train(self):
        mode = self.mode

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr = self.lr

        batch_size = self.batch_size
        device = self.device

        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        norm = self.norm
        name_data = self.name_data

        num_freq_disp = self.num_freq_disp
        num_freq_save = self.num_freq_save

        ny_in = self.ny_in
        nx_in = self.nx_in

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_data_train = os.path.join(self.dir_data, name_data, 'train')
        dir_log = os.path.join(self.dir_log, self.scope, name_data)

        transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
        # transform_train = transforms.Compose([ToTensor(), Normalize(mean=0.5, std=0.5)])
        transform_inv = transforms.Compose([ToNumpy(), Denormalize()])

        dataset_train = datasets.MNIST(root='.', train=True, download=True, transform=transform_train)
        # dataset_train = Dataset(dir_data_train, data_type=self.data_type, nch=self.nch_in, transform=transform_train)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

        num_train = len(loader_train.dataset)
        num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))

        ## setup network
        net = CLS(nch_in, nch_out, nch_ker, norm).to(device)

        init_net(net, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        ## setup loss & optimization
        # fn = nn.NLLLoss().to(device)
        fn = nn.CrossEntropyLoss().to(device)

        params = net.parameters()

        optim = torch.optim.Adam(params, lr=lr)

        ## load from checkpoints
        st_epoch = 0

        if train_continue == 'on':
            net, optim, st_epoch = self.load(dir_chck, net, optim, mode=mode)

        ## setup tensorboard
        writer_train = SummaryWriter(log_dir=dir_log)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            ## training phase
            net.train()

            loss_train = []
            pred_train = []

            # for i, data in enumerate(loader_train, 1):
            for i, (input, label) in enumerate(loader_train, 1):
                def should(freq):
                    return freq > 0 and (i % freq == 0 or i == num_batch_train)

                input = input.to(device)
                label = label.to(device)
                # input = data['input'].to(device)
                # label = data['label'].to(device)

                output = net(input)
                pred = output.max(1, keepdim=True)[1]

                # backward netD
                optim.zero_grad()

                loss = fn(output, label)
                loss.backward()

                optim.step()

                # get losses
                loss_train += [loss.item()]
                pred_train += [pred.eq(label.view_as(pred)).sum().item() / label.shape[0]]

                print('TRAIN: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f ACC: %.4f' %
                      (epoch, i, num_batch_train, mean(loss_train), 100 * mean(pred_train)))

                if should(num_freq_disp):
                    ## show output
                    input = transform_inv(input)
                    writer_train.add_images('input', input, num_batch_train * (epoch - 1) + i, dataformats='NHWC')

            writer_train.add_scalar('loss', mean(loss_train), epoch)

            ## save
            if (epoch % num_freq_save) == 0:
                self.save(dir_chck, net, optim, epoch)

        writer_train.close()

    def test(self):
        mode = self.mode

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr = self.lr

        batch_size = self.batch_size
        device = self.device

        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        norm = self.norm
        name_data = self.name_data

        num_freq_disp = self.num_freq_disp
        num_freq_save = self.num_freq_save

        ny_in = self.ny_in
        nx_in = self.nx_in

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_data_test = os.path.join(self.dir_data, name_data, 'test')

        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
        # transform_test = transforms.Compose([ToTensor(), Normalize(mean=0.5, std=0.5)])
        transform_inv = transforms.Compose([ToNumpy(), Denormalize()])

        dataset_test = datasets.MNIST(root='.', train=False, download=True, transform=transform_test)
        # dataset_test = Dataset(dir_data_test, data_type=self.data_type, nch=self.nch_in, transform=transform_test)
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

        num_test = len(loader_test.dataset)
        num_batch_test = int((num_test / batch_size) + ((num_test % batch_size) != 0))

        ## setup network
        net = CLS(nch_in, nch_out, nch_ker, norm).to(device)

        init_net(net, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        ## setup loss & optimization
        # fn = nn.NLLLoss().to(device)
        fn = nn.CrossEntropyLoss().to(device)

        ## load from checkpoints
        st_epoch = 0

        net, st_epoch = self.load(dir_chck, net, mode=mode)

        ## test phase
        with torch.no_grad():
            net.eval()

            loss_test = []
            pred_test = []

            # for i, data in enumerate(loader_test, 1):
            for i, (input, label) in enumerate(loader_test, 1):

                input = input.to(device)
                label = label.to(device)
                # input = data['input'].to(device)
                # label = data['label'].to(device)

                output = net(input)
                pred = output.max(1, keepdim=True)[1]

                loss = fn(output, label)

                # get losses
                loss_test += [loss.item()]
                pred_test += [pred.eq(label.view_as(pred)).sum().item()/label.shape[0]]

                print('TEST: BATCH %04d/%04d: LOSS: %.4f ACC: %.4f' % (i, num_batch_test, mean(loss_test), 100 * mean(pred_test)))

            print('TEST: AVERAGE LOSS: %.6f' % (mean(loss_test)))
            print('TEST: AVERAGE ACC: %.6f' % (100 * mean(pred_test)))
