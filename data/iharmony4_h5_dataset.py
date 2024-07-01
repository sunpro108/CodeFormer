from typing import Any
import random

import h5py 
from PIL import Image
from torch.utils.data import Dataset


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


class IH5Dataset(Dataset):
    def __init__(self, archive, transform, mode='train', subset='HAdobe5k', arch_mode=True) -> None:
        super().__init__()
        self.archive = archive
        self.transform = transform
        if arch_mode: # use total archive file
            self.index_start = Dict_Iharmony4_Count[mode][subset][0]
            self.index_end = Dict_Iharmony4_Count[mode][subset][1]
        else: # use single subdata archive file
            self.index_start = 0
            self.index_end = Dict_Iharmony4_Count[mode][subset][1] - Dict_Iharmony4_Count[mode][subset][0]
        self._len_dataset = self.index_end - self.index_start
        with open(f'datasets/ihm4/IHD_{mode}.txt','r') as f:
            self.list_names = f.readlines()
            self.list_names = [x.strip() for x in self.list_names]

        # self.archive = h5py.File(archive, 'r')
        self.comp = None
        self.real = None
        self.mask = None

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
        print(comp.shape)
        # comp = Image.fromarray(comp)
        # real = Image.fromarray(real)
        # mask = Image.fromarray(mask, mode='1')
        comp, real, mask = self.transform(comp, real, mask)
        comp  = self._compose(comp, mask, real)
        # return dict(zip(['comp','real','mask','img_path'],(comp, real, mask, img_path)))
        return comp, real

    def __len__(self):
        return self._len_dataset

    def _compose(self, fore, mask, back):
        return fore * mask + back * (1 - mask)

