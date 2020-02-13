import os
import numpy as np
import torch
from skimage import transform
import matplotlib.pyplot as plt


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_type='float32', nch=1, transform=[]):
        self.data_dir = data_dir
        self.transform = transform
        self.nch = nch
        self.data_type = data_type

        lst_data = os.listdir(data_dir)

        lst_input = [f for f in lst_data if f.startswith('input')]
        lst_label = [f for f in lst_data if f.startswith('label')]

        lst_input.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        lst_label.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        self.lst_input = lst_input
        self.lst_label = lst_label

    def __getitem__(self, index):
        input = plt.imread(os.path.join(self.data_dir, self.lst_input[index]))[:, :, :self.nch]
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))

        if input.dtype == np.uint8:
            input = input / 255.0

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.lst_label)


class ToTensor(object):
    def __call__(self, data):
        input, label = data['input'], data['label']

        if input.ndim == 3:
            input = input.transpose((2, 0, 1)).astype(np.float32)
        elif input.ndim == 4:
            input = input.transpose((0, 3, 1, 2)).astype(np.float32)

        data = {'input': torch.from_numpy(input), 'label': torch.from_numpy(label)}
        return data


class Normalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        input, label = data['input'], data['label']

        input = (input - self.mean)/self.std

        data = {'input': input, 'label': label}
        return data


class RandomFlip(object):
    def __call__(self, data):
        input, label = data['input'], data['label']

        if np.random.rand() > 0.5:
            input = np.fliplr(input)

        data = {'input': input, 'label': label}
        return data


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, data):
        input, label = data['input'], data['label']

        ch, h, w = input.shape

        if isinstance(self.output_size, int):
          if h > w:
            new_h, new_w = self.output_size * h / w, self.output_size
          else:
            new_h, new_w = self.output_size, self.output_size * w / h
        else:
          new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        input = transform.resize(input, (ch, new_h, new_w))

        data = {'input': input, 'label': label}
        return data


class CenterCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        input, label = data['input'], data['label']

        h, w = input.shape[:2]

        new_h, new_w = self.output_size

        top = int(abs(h - new_h) / 2)
        left = int(abs(w - new_w) / 2)

        input = input[top: top + new_h, left: left + new_w]

        data = {'input': input, 'label': label}
        return data


class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        input, label = data['input'], data['label']
        h, w = input.shape[:2]

        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        input = input[top: top + new_h, left: left + new_w]

        data = {'input': input, 'label': label}
        return data


class ToNumpy(object):
    def __call__(self, data):
        if data.ndim == 3:
            data = data.to('cpu').detach().numpy().transpose((1, 2, 0))
        elif data.ndim == 4:
            data = data.to('cpu').detach().numpy().transpose((0, 2, 3, 1))

        return data


class Denormalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data = self.std * data + self.mean
        return data
