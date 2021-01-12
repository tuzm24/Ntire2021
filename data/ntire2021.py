import os
from data import srdata
from help_func import myUtil
import glob

class Ntire2021(srdata.SRData):
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(Ntire2021, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr = sorted(
            myUtil.xgetFileList(self.dir_hr, self.ext[0])
        )
        names_lr = [[]]

        for f in names_hr:
            filename, _ = os.path.splitext(myUtil.getBaseDirnameAndBasename(f))

            names_lr[0].append(os.path.join(
                self.dir_lr, '{}{}'.format(
                    filename, self.ext[1]
                )
            ))
        if not self.train and not self.args.test_only:
            stride = len(names_hr)//self.args.valid_num
            names_hr = names_hr[::stride]
            names_lr = [n[::stride] for n in names_lr]



        # names_hr = names_hr[self.begin - 1:self.end]
        # names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = dir_data
        dirs = myUtil.getDirlist(self.apath)
        if self.train:
            self.dir_hr = myUtil.filteringPath(dirs, 'train_sharp')
            self.dir_lr = myUtil.filteringPath(dirs, 'train_blur')
        if not self.train or self.dir_hr is None:
            self.dir_hr = myUtil.filteringPath(dirs, 'val_sharp')
            self.dir_lr = myUtil.filteringPath(dirs, 'val_blur')
        if '_jpeg' in os.path.basename(self.dir_lr):
            self.ext = ('.png', '.jpg')
        else:
            self.ext = ('.png', '.png')
        self.dir_hr = myUtil.intoRealFolder(self.dir_hr)
        self.dir_lr = myUtil.intoRealFolder(self.dir_lr)


