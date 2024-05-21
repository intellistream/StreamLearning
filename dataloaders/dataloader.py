from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import pickle
from hashlib import md5
import json
from pathlib import Path
import torch.utils.data as data
from .utils import download_url, check_integrity
import random
import torchvision.datasets as datasets
from operator import itemgetter
import yaml

class iDataset(data.Dataset):
    
    def __init__(self, root,
                train=True, transform=None,
                download_flag=False, lab=True, swap_dset = None, cur_run=None,
                tasks=None, seed=-1, rand_split=False, validation=False, kfolds=5):

        # process rest of args
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.validation = validation
        self.seed = seed
        self.t = -1
        self.tasks = tasks
        self.download_flag = download_flag

        # load dataset
        self.load()
        self.num_classes = len(np.unique(self.targets))

        # remap labels to match task order
        c = 0
        self.class_mapping = {}
        self.class_mapping[-1] = -1
        for task in self.tasks:
            for k in task:
                self.class_mapping[k] = c
                c += 1

        # targets as numpy.array
        self.data = np.asarray(self.data)
        self.targets = np.asarray(self.targets)

        # if validation
        if self.validation:
            
            # shuffle
            state = np.random.get_state()
            np.random.seed(self.seed)
            randomize = np.random.permutation(len(self.targets))
            self.data = self.data[randomize]
            self.targets = self.targets[randomize]
            np.random.set_state(state)

            # sample
            n_data = len(self.targets)
            if self.train:
                self.data = self.data[:int(0.8*n_data)]
                self.targets = self.targets[:int(0.8*n_data)]
            else:
                self.data = self.data[int(0.8*n_data):]
                self.targets = self.targets[int(0.8*n_data):]

            # train set
            if self.train:
                self.data = self.data[:int(0.8*n_data)]
                self.targets = self.targets[:int(0.8*n_data)]
                self.archive = []
                domain_i = 0
                for task in self.tasks:
                    if True:
                        locs = np.isin(self.targets, task).nonzero()[0]
                        self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))

            # val set
            else:
                self.archive = []
                domain_i = 0
                for task in self.tasks:
                    if True:
                        locs = np.isin(self.targets, task).nonzero()[0]
                        self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))

        # else
        else:
            self.archive = []
            domain_i = 0
            for task in self.tasks:
                if True:
                    locs = np.isin(self.targets, task).nonzero()[0]
                    self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))

        if self.train:
            self.coreset = (np.zeros(0, dtype=self.data.dtype), np.zeros(0, dtype=self.targets.dtype))

    def __getitem__(self, index, simple = False):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        ori_img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        ori_img = Image.fromarray(ori_img)

        if self.transform is not None:
            img = self.transform(ori_img)
        return img, self.class_mapping[target], self.t

    def load_dataset(self, t, train=True):
        
        if train:
            self.data, self.targets = self.archive[t] 
        else:
            self.data    = np.concatenate([self.archive[s][0] for s in range(t+1)], axis=0)
            self.targets = np.concatenate([self.archive[s][1] for s in range(t+1)], axis=0)
        self.t = t

    def update_coreset(self, coreset_size, seen):
        num_data_per = coreset_size // len(seen)
        remainder = coreset_size % len(seen)
        data = []
        targets = []
        
        # random coreset management; latest classes take memory remainder
        # coreset selection without affecting RNG state
        state = np.random.get_state()
        np.random.seed(self.seed)
        for k in reversed(seen):
            mapped_targets = [self.class_mapping[self.targets[i]] for i in range(len(self.targets))]
            locs = (mapped_targets == k).nonzero()[0]
            if (remainder > 0) and (len(locs) > num_data_per):
                num_data_k = num_data_per + 1
                remainder -= 1
            else:
                num_data_k = min(len(locs), num_data_per)
            locs_chosen = locs[np.random.choice(len(locs), num_data_k, replace=False)]
            data.append([self.data[loc] for loc in locs_chosen])
            targets.append([self.targets[loc] for loc in locs_chosen])
        self.coreset = (np.concatenate(list(reversed(data)), axis=0), np.concatenate(list(reversed(targets)), axis=0))
        np.random.set_state(state)

    def load(self):
        pass

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class iCIFAR10(iDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the iDataset Dataset.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    im_size=32
    nch=3

    def load(self):

        # download dataset
        if self.download_flag:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train or self.validation:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        self.course_targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
                if 'coarse_labels' in entry:
                    self.course_targets.extend(entry['coarse_labels'])
                
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

class iCIFAR100(iCIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the iCIFAR10 Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    im_size=32
    nch=3

class iIMAGENET_R(iDataset):
    
    base_folder = 'imagenet-r'
    im_size=224
    nch=3
    def load(self):

        # load splits from config file
        if self.train or self.validation:
            data_config = yaml.load(open('dataloaders/splits/imagenet-r_train.yaml', 'r'), Loader=yaml.Loader)
        else:
            data_config = yaml.load(open('dataloaders/splits/imagenet-r_test.yaml', 'r'), Loader=yaml.Loader)
        self.data = data_config['data']
        self.targets = data_config['targets']

    def __getitem__(self, index, simple = False):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        img_path, target = self.data[index], self.targets[index]
        img = jpg_image_to_array(img_path)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.class_mapping[target], self.t

    def parse_archives(self) -> None:
        if not check_integrity(os.path.join(self.root, META_FILE)):
            parse_devkit_archive(self.root)

        if not os.path.isdir(self.split_folder):
            if self.split == 'train':
                parse_train_archive(self.root)
            elif self.split == 'val':
                parse_val_archive(self.root)

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

class iDOMAIN_NET(iIMAGENET_R):
    base_folder = 'DomainNet'
    im_size=224
    nch=3
    def load(self):
        
        # load splits from config file
        if self.train or self.validation:
            data_config = yaml.load(open('dataloaders/splits/domainnet_train.yaml', 'r'), Loader=yaml.Loader)
        else:
            data_config = yaml.load(open('dataloaders/splits/domainnet_test.yaml', 'r'), Loader=yaml.Loader)
        self.data = data_config['data']
        self.targets = data_config['targets']


class iMINI_IMAGENET(iDataset):
    base_folder = 'mini_imagenet'

    def load(self):
        TEST_SPLIT = 1/6
        train_in = open("./data/mini_imagenet/mini-imagenet-cache-train.pkl", "rb")
        train = pickle.load(train_in)
        train_x = train["image_data"].reshape([64, 600, 84, 84, 3])
        val_in = open("./data/mini_imagenet/mini-imagenet-cache-val.pkl", "rb")
        val = pickle.load(val_in)
        val_x = val['image_data'].reshape([16, 600, 84, 84, 3])
        test_in = open("./data/mini_imagenet/mini-imagenet-cache-test.pkl", "rb")
        test = pickle.load(test_in)
        test_x = test['image_data'].reshape([20, 600, 84, 84, 3])
        all_data = np.vstack((train_x, val_x, test_x))
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        for i in range(len(all_data)):
            cur_x = all_data[i]
            cur_y = np.ones((600,)) * i
            # rdm_x, rdm_y = self.shuffle_data(cur_x, cur_y)
            x_test = cur_x[: int(600 * TEST_SPLIT)]
            y_test = cur_y[: int(600 * TEST_SPLIT)]
            x_train = cur_x[int(600 * TEST_SPLIT):]
            y_train = cur_y[int(600 * TEST_SPLIT):]
            train_data.append(x_train)
            train_label.append(y_train)
            test_data.append(x_test)
            test_label.append(y_test)
        train_data = np.concatenate(train_data)
        train_label = np.concatenate(train_label)
        test_data = np.concatenate(test_data)
        test_label = np.concatenate(test_label)

        if self.train or self.validation:
            self.data = train_data
            self.targets = train_label
            print(self.data.shape)
        else:
            self.data = test_data
            self.targets = test_label

class iTINY_IMAGENET(iDataset):
    base_folder = 'tiny_imagenet'

    def load(self):
        train_in = open("./data/tiny_imagenet/train.pkl", "rb")
        train = pickle.load(train_in)
        train_x = train["data"].reshape([200, 500, 64, 64, 3])
        test_in = open("./data/tiny_imagenet/val.pkl", "rb")
        test = pickle.load(test_in)
        test_x = test['data'].reshape([200, 50, 64, 64, 3])

        train_data = []
        train_label = []
        test_data = []
        test_label = []
        if self.train or self.validation:
            for i in range(len(train_x)):
                x_train = train_x[i]
                y_train = np.ones((500,)) * i
                train_data.append(x_train)
                train_label.append(y_train)
            train_data = np.concatenate(train_data)
            train_label = np.concatenate(train_label)
            self.data = train_data
            print(self.data.shape)
            self.targets = train_label
        else:
            for i in range(len(test_x)):
                x_test = test_x[i]
                y_test = np.ones((50,)) * i
                test_data.append(x_test)
                test_label.append(y_test)
            test_data = np.concatenate(test_data)
            test_label = np.concatenate(test_label)
            self.data = test_data
            self.targets = test_label

class iCORE50(iDataset):
    def __init__(self, root, train=True, transform=None, download_flag=False, lab=True, swap_dset=None, cur_run=None,
                 tasks=None, seed=-1, rand_split=False, validation=False, kfolds=5):

        # process rest of args
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.validation = validation
        self.seed = seed
        self.t = -1
        self.tasks = []
        self.download_flag = download_flag
        self.scenario = 'nc'
        self.cur_run = cur_run
        # load dataset
        self.load()
        self.archive = []

        if self.train or self.validation:
            for task in range(9):
                self.archive.append(self.traindata_setup(task))
        else:
            self.testdata_setup(self.cur_run) # update self.archive inside

        # remap labels to match task order
        c = 0
        self.class_mapping = {}
        self.class_mapping[-1] = -1
        for task_idx in range(len(self.archive)):
            task = list(set(self.archive[task_idx][1]))
            self.tasks.append(task)
            for k in task:
                self.class_mapping[k] = c
                c += 1
        print(self.tasks)

        if self.train:
            self.coreset = (np.zeros(0), np.zeros(0))

    def load(self):

        if self.train or self.validation:
            s = 'Train Set'
        else:
            s = 'Test Set'
        print(f"[** {s}] loading paths & LUP & labels...")
        with open(os.path.join(self.root, 'core50/paths.pkl'), 'rb') as f:
            self.paths = pickle.load(f)

        with open(os.path.join(self.root, 'core50/LUP.pkl'), 'rb') as f:
            self.LUP = pickle.load(f)

        with open(os.path.join(self.root, 'core50/labels.pkl'), 'rb') as f:
            self.labels = pickle.load(f)

    def testdata_setup(self, cur_run):
        filename = os.path.join(self.root, 'core50', 'test', f'run{self.cur_run}.pkl')
        if os.path.exists(filename):
            print(f'==> loading from: {filename}')
            test_in = open(filename, "rb")
            self.archive = pickle.load(test_in)
            return
        else:
            print('==> loading from each image path...')
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
        test_idx_list = self.LUP[self.scenario][cur_run][-1]

        #test paths
        test_paths = []
        for idx in test_idx_list:
            test_paths.append(os.path.join(self.root, 'core50', self.paths[idx]))

        # test imgs
        test_data = self.get_batch_from_paths(test_paths)
        test_label = np.asarray(self.labels[self.scenario][cur_run][-1])

        task_labels = self.labels[self.scenario][cur_run][:-1]
        for labels in task_labels:
            labels = list(set(labels))
            x_test, y_test = load_task_with_labels(test_data, test_label, labels)
            self.archive.append((x_test, y_test))
        save_list_as_pickle(self.archive, filename)


    def traindata_setup(self, cur_task):
        cur_run = self.cur_run
        filename = os.path.join(self.root, 'core50', 'train', f'run{self.cur_run}task{cur_task}.pkl')
        if os.path.exists(filename):
            print(f'==> loading from: {filename}')
            train_in = open(filename, "rb")
            train = pickle.load(train_in)
            return (train['data'], train['target'])
        else:
            print('==> loading from each image path...')
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
        train_idx_list = self.LUP[self.scenario][cur_run][cur_task]
        # Getting the actual paths
        train_paths = []
        for idx in train_idx_list:
            train_paths.append(os.path.join(self.root, 'core50', self.paths[idx]))
        # loading imgs
        train_x = self.get_batch_from_paths(train_paths)
        train_y = self.labels[self.scenario][cur_run][cur_task]
        train_y = np.asarray(train_y)
        print(f'For task {cur_task}: {len(list(set(train_y)))}')
        save_dict_as_pickle(train_x, train_y, filename)
        return (train_x, train_y)

    def load_dataset(self, t, train=True):

        if train:
            self.data, self.targets = self.archive[t][0], self.archive[t][1]
        else:
            self.data = np.concatenate([self.archive[s][0] for s in range(t + 1)], axis=0)
            self.targets = np.concatenate([self.archive[s][1] for s in range(t + 1)], axis=0)
        self.t = t

    @staticmethod
    def get_batch_from_paths(paths, compress=False, snap_dir='',
                             on_the_fly=True, verbose=False):
        """ Given a number of abs. paths it returns the numpy array
        of all the images. """

        # If we do not process data on the fly we check if the same train
        # filelist has been already processed and saved. If so, we load it
        # directly. In either case we end up returning x and y, as the full
        # training set and respective labels.
        num_imgs = len(paths)
        hexdigest = md5(''.join(paths).encode('utf-8')).hexdigest()
        loaded = False
        x = None
        file_path = None

        if compress:
            file_path = snap_dir + hexdigest + ".npz"
            if os.path.exists(file_path) and not on_the_fly:
                loaded = True
                with open(file_path, 'rb') as f:
                    npzfile = np.load(f)
                    x, y = npzfile['x']
        else:
            x_file_path = snap_dir + hexdigest + "_x.bin"
            if os.path.exists(x_file_path) and not on_the_fly:
                loaded = True
                with open(x_file_path, 'rb') as f:
                    x = np.fromfile(f, dtype=np.uint8) \
                        .reshape(num_imgs, 128, 128, 3)

        # Here we actually load the images.
        if not loaded:
            # Pre-allocate numpy arrays
            x = np.zeros((num_imgs, 128, 128, 3), dtype=np.uint8)

            for i, path in enumerate(paths):
                if verbose:
                    print("\r" + path + " processed: " + str(i + 1), end='')
                x[i] = np.array(Image.open(path))

            if verbose:
                print()

            if not on_the_fly:
                # Then we save x
                if compress:
                    with open(file_path, 'wb') as g:
                        np.savez_compressed(g, x=x)
                else:
                    x.tofile(snap_dir + hexdigest + "_x.bin")

        assert (x is not None), 'Problems loading data. x is None!'

        return x


class iSTREAM51(iDataset):
    def __init__(self, root,
                 train=True, transform=None,
                 download_flag=False, lab=True, swap_dset=None, cur_run=None,
                 tasks=None, seed=-1, rand_split=False, validation=False, kfolds=5):

        # process rest of args
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.validation = validation
        self.seed = seed
        self.t = -1
        self.tasks = tasks
        self.download_flag = download_flag

        # load dataset
        self.load()
        self.num_classes = len(np.unique(self.targets))

        # remap labels to match task order
        c = 0
        self.class_mapping = {}
        self.class_mapping[-1] = -1
        for task in self.tasks:
            for k in task:
                self.class_mapping[k] = c
                c += 1

        # targets as numpy.array
        # self.data = np.asarray(self.data)
        self.targets = np.asarray(self.targets)

        # if validation
        if self.validation:

            # shuffle
            state = np.random.get_state()
            np.random.seed(self.seed)
            randomize = np.random.permutation(len(self.targets))
            self.data = self.data[randomize]
            self.targets = self.targets[randomize]
            np.random.set_state(state)

            # sample
            n_data = len(self.targets)
            if self.train:
                self.data = self.data[:int(0.8 * n_data)]
                self.targets = self.targets[:int(0.8 * n_data)]
            else:
                self.data = self.data[int(0.8 * n_data):]
                self.targets = self.targets[int(0.8 * n_data):]
            # train set
            if self.train:
                self.data = self.data[:int(0.8 * n_data)]
                self.targets = self.targets[:int(0.8 * n_data)]
                self.archive = []
                domain_i = 0
                for task in self.tasks:
                    if True:
                        locs = np.isin(self.targets, task).nonzero()[0]
                        self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))
            # val set
            else:
                self.archive = []
                domain_i = 0
                for task in self.tasks:
                    if True:
                        locs = np.isin(self.targets, task).nonzero()[0]
                        self.archive.append((self.data[locs].copy(), self.targets[locs].copy()))
        else:
            self.archive = []
            domain_i = 0
            for task in self.tasks:
                locs = np.isin(self.targets, task).nonzero()[0]
                locs = locs.tolist()
                selected_data = [self.data[i] for i in locs]
                selected_targets = [self.targets[i] for i in locs]
                self.archive.append((selected_data, selected_targets))
        if self.train:
            self.coreset = (np.zeros(0), np.zeros(0))

    def load(self):
        # root = './data/stream51/Stream-51'
        if self.train or self.validation:
            data_list = json.load(open(os.path.join(self.root,'stream51/Stream-51', 'Stream-51_meta_train.json')))
        else:
            data_list = json.load(open(os.path.join(self.root,'stream51/Stream-51', 'Stream-51_meta_test.json')))
            # Filter out novelty detection in the test dataset
            ind = [i for i in range(len(data_list)) if data_list[i][0] < 51]
            data_list = [data_list[i] for i in ind]

        samples = make_dataset(data_list, 'class_instance', seed=self.seed)

        self.loader = default_loader

        self.data = samples
        self.targets = [s[0] for s in samples]

        self.transform = self.transform

        self.bbox_crop = True
        self.ratio = 1.10

    def load_dataset(self, t, train=True):

        if train:
            self.data, self.targets = self.archive[t][0][:], self.archive[t][1][:]
        else:
            num_items = sum(len(self.archive[s][0]) for s in range(t + 1))
            # self.data = np.concatenate([self.archive[s][0] for s in range(t + 1)], axis=0)
            self.data = np.empty((num_items,), dtype=object)
            index = 0
            for s in range(t + 1):
                for item in self.archive[s][0]:
                    self.data[index] = item
                    index += 1
            self.targets = np.concatenate([self.archive[s][1] for s in range(t + 1)], axis=0)
        self.t = t

    def __getitem__(self, index, simple=False):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        fpath, target = self.data[index][-1], self.targets[index]
        sample = self.loader(os.path.join(self.root, 'stream51/Stream-51', fpath))
        if self.bbox_crop:
            bbox = self.data[index][-2]
            cw = bbox[0] - bbox[1];
            ch = bbox[2] - bbox[3];
            center = [int(bbox[1] + cw / 2), int(bbox[3] + ch / 2)]
            bbox = [min([int(center[0] + (cw * self.ratio / 2)), sample.size[0]]),
                    max([int(center[0] - (cw * self.ratio / 2)), 0]),
                    min([int(center[1] + (ch * self.ratio / 2)), sample.size[1]]),
                    max([int(center[1] - (ch * self.ratio / 2)), 0])]
            sample = sample.crop((bbox[1],
                                  bbox[3],
                                  bbox[0],
                                  bbox[2]))

        if self.transform is not None:
            img = self.transform(sample)
        return img, self.class_mapping[target], self.t

class iCLEAR10(iDataset):
    def __init__(self, root,
                 train=True, transform=None,
                 download_flag=False, lab=True, swap_dset=None, cur_run=None,
                 tasks=None, seed=-1, rand_split=False, validation=False, kfolds=5):

        # process rest of args
        self.root = Path(os.path.expanduser(root))
        self.root = self.root / 'clear10'
        self.transform = transform
        self.train = train  # training set or test set
        self.validation = validation
        self.seed = seed
        self.t = -1
        self.tasks = tasks
        self.download_flag = download_flag

        # load dataset
        self.load()
        self.num_classes = len(np.unique(self.targets))

        # remap labels to match task order
        c = 0
        self.class_mapping = {}
        self.class_mapping[-1] = -1
        for task in self.tasks:
            for k in task:
                self.class_mapping[k] = c
                c += 1

        # targets as numpy.array
        # self.data = np.asarray(self.data)
        self.targets = np.asarray(self.targets)


        self.archive = []
        for task in self.tasks:
            locs = np.isin(self.targets, task).nonzero()[0]
            locs = locs.tolist()
            selected_data = [self.data[i] for i in locs]
            selected_targets = [self.targets[i] for i in locs]
            self.archive.append((selected_data, selected_targets))
        if self.train:
            self.coreset = (np.zeros(0), np.zeros(0))

    def load(self):
        if self.train:
            splits = ['train']
        else:
            splits = ['test']
        for split in splits:
            train_folder_path = self.root / split
            if not train_folder_path.exists():
                print(f"{train_folder_path} does not exist. ")
                return False

            self.labeled_metadata = _load_json(train_folder_path / "labeled_metadata.json")
            if '0' in self.labeled_metadata:
                del self.labeled_metadata['0']

            class_names_file = train_folder_path / "class_names.txt"
            self.class_names = class_names_file.read_text().split("\n")

            self.data = []
            self.targets = []
            self._paths_and_targets = []

            for class_idx, class_name in enumerate(self.class_names):
                for bucket, data in self.labeled_metadata.items():
                    temp_data = []
                    temp_targets = []

                    metadata_path = train_folder_path / data[class_name]
                    if not metadata_path.exists():
                        print(f"{metadata_path} does not exist.")
                        return False
                    metadata = _load_json(metadata_path)

                    for v in metadata.values():
                        f_path = os.path.join(split, v["IMG_PATH"])
                        temp_data.append(f_path)
                        temp_targets.append(class_idx)

                    combined = list(zip(temp_data, temp_targets))
                    random.shuffle(combined)
                    temp_data, temp_targets = zip(*combined)

                    self.data.extend(temp_data)
                    self.targets.extend(temp_targets)

            # Check whether all labeled images exist
            for img_path in self.data:
                path = self.root / img_path
                if not os.path.exists(path):
                    print(f"{path} does not exist.")

        self.loader = default_loader

        self.transform = self.transform

    def load_dataset(self, t, train=True):

        if train:
            self.data, self.targets = self.archive[t][0][:], self.archive[t][1][:]
        else:
            num_items = sum(len(self.archive[s][0]) for s in range(t + 1))
            # self.data = np.concatenate([self.archive[s][0] for s in range(t + 1)], axis=0)
            self.data = np.empty((num_items,), dtype=object)
            index = 0
            for s in range(t + 1):
                for item in self.archive[s][0]:
                    self.data[index] = item
                    index += 1
            self.targets = np.concatenate([self.archive[s][1] for s in range(t + 1)], axis=0)
        self.t = t

    def __getitem__(self, index, simple=False):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        fpath, target = self.data[index], self.targets[index]
        sample = self.loader(str(self.root / fpath))

        if self.transform is not None:
            img = self.transform(sample)

        return img, self.class_mapping[target], self.t

class iCLEAR100(iDataset):
    def __init__(self, root,
                 train=True, transform=None,
                 download_flag=False, lab=True, swap_dset=None, cur_run=None,
                 tasks=None, seed=-1, rand_split=False, validation=False, kfolds=5):

        # process rest of args
        self.root = Path(os.path.expanduser(root))
        self.root = self.root / 'clear100'
        self.transform = transform
        self.train = train  # training set or test set
        self.validation = validation
        self.seed = seed
        self.t = -1
        self.tasks = tasks
        self.download_flag = download_flag

        # load dataset
        self.load()
        self.num_classes = len(np.unique(self.targets))

        # remap labels to match task order
        c = 0
        self.class_mapping = {}
        self.class_mapping[-1] = -1
        for task in self.tasks:
            for k in task:
                self.class_mapping[k] = c
                c += 1

        # targets as numpy.array
        # self.data = np.asarray(self.data)
        self.targets = np.asarray(self.targets)


        self.archive = []
        for task in self.tasks:
            locs = np.isin(self.targets, task).nonzero()[0]
            locs = locs.tolist()
            selected_data = [self.data[i] for i in locs]
            selected_targets = [self.targets[i] for i in locs]
            self.archive.append((selected_data, selected_targets))
        if self.train:
            self.coreset = (np.zeros(0), np.zeros(0))

    def load(self):
        if self.train:
            splits = ['train']
        else:
            splits = ['test']
        for split in splits:
            train_folder_path = self.root / split
            if not train_folder_path.exists():
                print(f"{train_folder_path} does not exist. ")
                return False

            self.labeled_metadata = _load_json(train_folder_path / "labeled_metadata.json")

            class_names_file = train_folder_path / "class_names.txt"
            self.class_names = class_names_file.read_text().split("\n")

            self.data = []
            self.targets = []
            self._paths_and_targets = []

            for class_idx, class_name in enumerate(self.class_names):
                for bucket, data in self.labeled_metadata.items():
                    temp_data = []
                    temp_targets = []

                    metadata_path = train_folder_path / data[class_name]
                    if not metadata_path.exists():
                        print(f"{metadata_path} does not exist.")
                        return False
                    metadata = _load_json(metadata_path)

                    for v in metadata.values():
                        f_path = os.path.join(split, v["IMG_PATH"])
                        temp_data.append(f_path)
                        temp_targets.append(class_idx)

                    combined = list(zip(temp_data, temp_targets))
                    random.shuffle(combined)
                    temp_data, temp_targets = zip(*combined)

                    self.data.extend(temp_data)
                    self.targets.extend(temp_targets)

            # Check whether all labeled images exist
            for img_path in self.data:
                path = self.root / img_path
                if not os.path.exists(path):
                    print(f"{path} does not exist.")

        self.loader = default_loader

        self.transform = self.transform

    def load_dataset(self, t, train=True):

        if train:
            self.data, self.targets = self.archive[t][0][:], self.archive[t][1][:]
        else:
            num_items = sum(len(self.archive[s][0]) for s in range(t + 1))
            # self.data = np.concatenate([self.archive[s][0] for s in range(t + 1)], axis=0)
            self.data = np.empty((num_items,), dtype=object)
            index = 0
            for s in range(t + 1):
                for item in self.archive[s][0]:
                    self.data[index] = item
                    index += 1
            self.targets = np.concatenate([self.archive[s][1] for s in range(t + 1)], axis=0)
        self.t = t

    def __getitem__(self, index, simple=False):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        fpath, target = self.data[index], self.targets[index]
        sample = self.loader(str(self.root / fpath))

        if self.transform is not None:
            img = self.transform(sample)

        return img, self.class_mapping[target], self.t

def _load_json(json_location):
    with open(json_location, "r") as f:
        obj = json.load(f)
    return obj

def instance_ordering(data_list, seed):
    # organize data by video
    total_videos = 0
    new_data_list = []
    temp_video = []
    for x in data_list:
        if x[3] == 0:
            new_data_list.append(temp_video)
            total_videos += 1
            temp_video = [x]
        else:
            temp_video.append(x)
    new_data_list.append(temp_video)
    new_data_list = new_data_list[1:]
    # shuffle videos
    random.shuffle(new_data_list)
    # reorganize by clip
    data_list = []
    for v in new_data_list:
        for x in v:
            data_list.append(x)
    return data_list # , new_data_list

def class_ordering(data_list, class_type, seed):
    # organize by class
    new_data_list = []
    # class_vids = []
    # class_vids_len = []
    # hist_data = {}
    for class_id in range(data_list[-1][0] + 1):
        class_data_list = [x for x in data_list if x[0] == class_id]
        if class_type == 'class_iid':
            # shuffle all class data
            random.seed(seed)
            random.shuffle(class_data_list)
        else:
            # shuffle clips within class
            class_data_list = instance_ordering(class_data_list, seed)
        new_data_list.append(class_data_list)
    # shuffle classes
    # random.seed(seed)
    random.shuffle(new_data_list)
    # reorganize by class
    data_list = []
    for v in new_data_list:
        for x in v:
            data_list.append(x)
    return data_list

def make_dataset(data_list, ordering='class_instance', seed=666):
    """
    data_list
    for train: [class_id, clip_num, video_num, frame_num, img_shape, bbox, file_loc]
    for test: [class_id, img_shape, bbox, file_loc]
    """
    if not ordering or len(data_list[0]) == 4:  # cannot order the test set
        return data_list
    if ordering not in ['iid', 'class_iid', 'instance', 'class_instance']:
        raise ValueError('dataset ordering must be one of: "iid", "class_iid", "instance", or "class_instance"')
    if ordering == 'iid':
        # shuffle all data
        random.seed(seed)
        random.shuffle(data_list)
        return data_list
    elif ordering == 'instance':
        return instance_ordering(data_list, seed)
    elif 'class' in ordering:
        return class_ordering(data_list, ordering, seed)

def save_list_as_pickle(list_data_labels, filename):
    with open(filename, 'wb') as f:
        pickle.dump(list_data_labels, f)

def save_dict_as_pickle(data, labels, filename):
    with open(filename, 'wb') as f:
        pickle.dump({"data": data, "target": labels}, f)
    print(f'==> save at {filename}')

def load_task_with_labels(x, y, labels):
    tmp = []
    for i in labels:
        tmp.append((np.where(y == i)[0]))
    idx = np.concatenate(tmp, axis=None)
    return x[idx], y[idx]

def jpg_image_to_array(image_path):
    """
    Loads JPEG image into 3D Numpy array of shape
    (width, height, channels)
    """
    with Image.open(image_path) as image:
        image = image.convert('RGB')
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3))
    return im_arr

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)


import torch
from torch.utils.data.sampler import Sampler
class SeqSampler(Sampler):
    def __init__(self, dataset_name, dataset, imbalanced=True, blend_ratio=0.5, n_concurrent_classes=2,
                  train_samples_ratio=0.5):
        """data_source is a Subset"""
        self.dataset_name = dataset_name
        self.num_samples = len(dataset)
        self.blend_ratio = blend_ratio
        self.n_concurrent_classes = n_concurrent_classes
        self.imbalanced = imbalanced
        self.train_samples_ratio = train_samples_ratio
        self.total_sample_num = int(self.num_samples * train_samples_ratio)

        # Configure the correct train_subset and val_subset
        if torch.is_tensor(dataset.targets):
            self.labels = dataset.targets.detach().cpu().numpy()
        else:  # targets in cifar10 and cifar100 is a list
            self.labels = np.array(dataset.targets)
        self.classes = list(set(self.labels))
        self.n_classes = len(self.classes)

    def __iter__(self):
        """Sequential sampler"""
        sample_idx = {}  # 存储每个类别的样本索引列表
        for label_idx, label in enumerate(self.labels):
            if label not in sample_idx:
                sample_idx[label] = []
            sample_idx[label].append(label_idx)

        # Configure blending class
        if self.blend_ratio > 0.0:
            for c in range(len(self.classes)):
                # Blend examples from the previous class if not the first
                if c > 0:
                    curr_class = self.classes[c]
                    prev_class = self.classes[c-1]
                    blendable_sample_num =  int(min(len(sample_idx[curr_class]),
                                                    len(sample_idx[prev_class])) * self.blend_ratio / 2)
                    # Generate a gradual blend probability
                    blend_prob = np.linspace(0.5, 0.2, blendable_sample_num)
                    assert blend_prob.size == blendable_sample_num, \
                        f"Unmatched sample and probability count: " \
                        f"blend_prob size = {blend_prob.size}, blendable_sample_num = {blendable_sample_num}"

                    # Exchange with the samples from the end of the previous
                    # class if satisfying the probability, which decays
                    # gradually
                    for ind in range(blendable_sample_num):
                        if random.random() < 0.5:
                        # if random.random() < blend_prob[ind]:
                            tmp = sample_idx[prev_class][ind-blendable_sample_num] # -20,-19,-18,...,-1
                            sample_idx[prev_class][ind-blendable_sample_num] = sample_idx[curr_class][ind]
                            sample_idx[curr_class][ind] = tmp

        final_idx = [idx for class_label in sample_idx.keys() for idx in sample_idx[class_label]]

        # Update total sample num
        self.total_sample_num = len(final_idx)

        return iter(final_idx)

    def __len__(self):
        return self.total_sample_num

