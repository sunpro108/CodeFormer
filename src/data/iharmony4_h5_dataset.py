from typing import Any
import random

import h5py 
from PIL import Image
from torch.utils.data import Dataset

from src.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from src.utils.registry import DATASET_REGISTRY
from src.data.transforms import Compose, ToTensor, Normalize

Dict_Iharmony4_Count = {
    'test':{
        'Total':(0,7404),
        'HAdobe5k': (0, 2160),
        'HCOCO': (2160, 6443),
        'HFlickr': (6443, 7271),
        'Hday2night': (7271,7404),
    },
    'train':{
        'Total': (0, 65742),
        'HAdobe5k': (0, 19437),
        'HCOCO': (19437, 57982),
        'HFlickr': (57982, 65431),
        'Hday2night': (65431, 65742),
    }
}


@DATASET_REGISTRY.register()
class IH5Dataset(Dataset):
    def __init__(self, opt):
    # def __init__(self, archive, transform, mode='train', subset='HAdobe5k', arch_mode=True) -> None:
        super().__init__()
        self.opt = opt
        self.logger = get_root_logger()
        self.archive = opt['archive']
        # self.transform = transform # todo
        mode = opt.get('mode','train') # todo
        subset = opt.get('subset','HAdobe5k') # todo
        arch_mode = opt.get('arch_mode', True) # todo
        if arch_mode: # use total archive file
            self.index_start = Dict_Iharmony4_Count[mode][subset][0]
            self.index_end = Dict_Iharmony4_Count[mode][subset][1]
        else: # use single subdata archive file
            self.index_start = 0
            self.index_end = Dict_Iharmony4_Count[mode][subset][1] - Dict_Iharmony4_Count[mode][subset][0]
        self._len_dataset = self.index_end - self.index_start
        with open(f'data/iharmony4/IHD_{mode}.txt','r') as f:
            self.list_names = f.readlines()
            self.list_names = [x.strip() for x in self.list_names]

        # self.archive = h5py.File(archive, 'r')
        self.comp = None
        self.real = None
        self.mask = None
        self.transform = Compose([
            ToTensor(),
        ])

    def __getitem__(self, index) -> Any:
        if self.comp is None:
            # print('load h5 data')
            self.dataset = h5py.File(self.archive, 'r')
            self.comp = self.dataset['comp'][self.index_start : self.index_end]
            self.real = self.dataset['real'][self.index_start : self.index_end]
            self.mask = self.dataset['mask'][self.index_start : self.index_end]
            self.list_names = self.list_names[self.index_start : self.index_end]
            self.dataset.close()
        x = random.randint(0, 192)
        y = random.randint(0, 192)
        comp = self.comp[index][x:x+64, y:y+64]
        real = self.real[index][x:x+64, y:y+64]
        mask = self.mask[index][x:x+64, y:y+64]
        img_path = self.list_names[index]
        # print(comp.shape)
        # comp = Image.fromarray(comp)
        # real = Image.fromarray(real)
        # mask = Image.fromarray(mask, mode='1')
        comp, real, mask = self.transform(comp, real, mask)
        comp  = self._compose(comp, mask, real)
        return dict(zip(['in','gt','mask','lq_path'],(comp, real, mask, img_path)))
        # return comp, real

    def __len__(self):
        return self._len_dataset

    @staticmethod
    def _compose(fore, mask, back):
        return fore * mask + back * (1 - mask)

