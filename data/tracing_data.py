from data import srdata
import glob
from help_func.CompArea import PictureFormat
import random
from help_func.help_func import *
import torch


class tracing_data(srdata.SRData):
    ppsfilename = 'PPSParam.npz'
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
        self.end -=self.begin
        self.begin = 1
        super(tracing_data, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )
        self.pps_data  = self._scanPPSData()
        self.blocks_ext = '.npy'
        self.blocks = {}
        self._scanblockData()
        self.image_dic = {}

    def _scanPPSData(self):
        named_ppsFile = []
        for f in self.images_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            named_ppsFile.append(os.path.join(
                self.dir_lr, '{}/{}/{}'.format(
                    'BLOCK', filename, self.ppsfilename
                )
            ))
        return named_ppsFile


    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.data_types]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.data_types):
                names_lr[si].append(os.path.join(
                    self.dir_lr, '{}/{}{}'.format(
                        s, filename, self.ext[1]
                    )
                ))

        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]
        return names_hr, names_lr

    def _scanblockData(self):
        for f in self.images_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            block_dir = os.path.join(self.dir_lr,
                                     'BLOCK', filename)
            for file in myUtil.getFileList(block_dir, self.blocks_ext):
                key, _ = os.path.splitext(os.path.basename(file))
                if key in self.blocks:
                    self.blocks[key].append(file)
                else:
                    self.blocks[key] = [file]

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data)
        self.dir_hr = os.path.join(self.apath, PictureFormat.INDEX_DIC[PictureFormat.ORIGINAL])
        self.dir_lr = os.path.join(self.apath)
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.npz', '.npz')




    def read_npz_file(self, file):
        def UpSamplingChroma(UVPic):
            return UVPic.repeat(2, axis=0).repeat(2, axis=1)
        if self.args.image_pin_memory:
            if file not in self.image_dic:
                self.image_dic[file] = np.load(file)
            f = self.image_dic[file]
        else:
            f = np.load(file)
        return np.stack((f['Y'], UpSamplingChroma(f['Cb']), UpSamplingChroma(f['Cr'])), axis=2)

    def read_npz_split_yuv(self, file):
        if self.args.image_pin_memory:
            if file not in self.image_dic:
                self.image_dic[file] = np.load(file)
            f = self.image_dic[file]
        else:
            f = np.load(file)
        return [f['Y'][:,:,None], np.stack((f['Cb'], f['Cr']), axis=2)]

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        lr = []
        if self.args.sc:
            hr = self.read_npz_split_yuv(f_hr)
            for flr in self.images_lr:
                lr.append(self.read_npz_split_yuv(flr[idx]))

        else:
            hr = self.read_npz_file(f_hr)
            for flr in self.images_lr:
                lr.append(self.read_npz_file(flr[idx]))
        return lr, hr, filename

    def get_patch(self, lr, hr):
        def _get_patch(*args, ih, iw, patch_size=96):
            # ih, iw = args[0].shape[:2]


            ip = patch_size

            ix = random.randrange(0, iw - ip + 1)
            iy = random.randrange(0, ih - ip + 1)
            if self.args.sc:
                ix >>= 1
                ix <<= 1
                iy >>= 1
                iy <<= 1
                ret = [
                    [a[iy//i:iy//i + ip//i, ix//i:ix//i + ip//i] for i, a in enumerate(args[0], 1)],
                    [[a[iy//i:iy//i + ip//i, ix//i:ix//i + ip//i]  for i, a in enumerate(arg, 1)] for arg in args[1]]
                ]
            else:
                ret = [
                    args[0][iy:iy + ip, ix:ix + ip, :],
                    [a[iy:iy + ip, ix:ix + ip, :] for a in args[1]]
                ]
            return ret[0], ret[1], (iy, ip, ix, ip)
        if self.args.sc:
            tpy, tpx = hr[0].shape[:2]
        else:
            tpy, tpx = hr.shape[:2]
        ty, tx = 0, 0
        if self.train:
            hr, lr, (ty, tpy, tx, tpx) = _get_patch(
                hr, lr, ih=tpy, iw=tpx,
                patch_size=self.args.patch_size
            )
        return lr, hr, (ty, tpy, tx, tpx)


    def getBlock2d(self, y, dy, x, dx, idx, name, value_idx = 0):
        if self.args.sc:
            block2d = np.full((dy//2, dx//2), np.nan)
        else:
            block2d = np.zeros((dy, dx))
        dy = y + dy
        dx = x + dx
        # 0:xpos, 1:ypos, 2:width, 3:height
        block = np.load(self.blocks[name][idx]).T
        filtered = block[:, ~np.any([(block[3] + block[1]) <= y,
                                     (block[0] + block[2]) <= x,
                                     block[1]>=dy,
                                     block[0]>=dx
                                     ], axis=0)]
        filtered[2, filtered[0]<x] = (filtered[0, filtered[0]<x] + filtered[2, filtered[0]<x]) - x
        filtered[3, filtered[1]<y] = (filtered[1, filtered[1]<y] + filtered[3, filtered[1]<y]) - y
        filtered[0, filtered[0]<x] = x
        filtered[1, filtered[1]<y] = y
        filtered[2, (filtered[0] + filtered[2]) > dx] = dx - filtered[0, (filtered[0]+filtered[2] > dx)]
        filtered[3, (filtered[1] + filtered[3]) > dy] = dy - filtered[1, (filtered[1] + filtered[3]) > dy]
        filtered[0,:] -= x
        filtered[1,:] -= y
        if self.args.sc:
            filtered[0:4, :] //= 2
        for _xp, _yp, _w, _h, *values in filtered.T:
            block2d[_yp:_yp+_h, _xp:_xp+_w] = values[value_idx]

        return block2d

    def getBlockScalar(self, idx, name, value_idx=0):
        block = np.load(self.blocks[name][idx]).T
        return block[4+value_idx].mean(axis=0)

    def getBlock2dFromScalar(self, value):
        return np.full((self.args.patch_size,self.args.patch_size), value)

    def np2tensor(self, args, rgb_range=1023):
        def _np2Tensor(img):
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
            tensor = torch.from_numpy(np_transpose).float()
            tensor.mul_(1 / rgb_range)
            return tensor

        if self.args.sc:
            return [[_np2Tensor(a) for a in arg] for arg in args]
        else:
            return [_np2Tensor(a) for a in args]


    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        if self.train and self.args.more_noisy:
            mse = -1
            tlr, thr, tpos = None, None, None
            for _ in range(self.args.more_noisy):
                _lr, _hr, _pos = self.get_patch(lr, hr)
                temp =np.mean(np.square(_lr[0].astype('int32') - _hr[0]))
                if temp>mse:
                    mse = temp
                    tlr, thr, tpos = _lr, _hr, _pos
            lr, hr, pos = tlr, thr, tpos
        else:
            lr, hr, pos = self.get_patch(lr, hr)
        # self.getBlock2d(*pos, idx, BlockType.Chroma_IntraMode)
        return self.np2tensor(lr), self.np2tensor([hr])[0], filename
from data import srdata
import glob
from help_func.CompArea import PictureFormat
import random
from help_func.help_func import *
import torch


class tracing_data(srdata.SRData):
    ppsfilename = 'PPSParam.npz'
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
        self.end -=self.begin
        self.begin = 1
        super(tracing_data, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )
        self.pps_data  = self._scanPPSData()
        self.blocks_ext = '.npy'
        self.blocks = {}
        self._scanblockData()
        self.image_dic = {}

    def _scanPPSData(self):
        named_ppsFile = []
        for f in self.images_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            named_ppsFile.append(os.path.join(
                self.dir_lr, '{}/{}/{}'.format(
                    'BLOCK', filename, self.ppsfilename
                )
            ))
        return named_ppsFile


    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.data_types]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.data_types):
                names_lr[si].append(os.path.join(
                    self.dir_lr, '{}/{}{}'.format(
                        s, filename, self.ext[1]
                    )
                ))

        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]
        return names_hr, names_lr

    def _scanblockData(self):
        for f in self.images_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            block_dir = os.path.join(self.dir_lr,
                                     'BLOCK', filename)
            for file in myUtil.getFileList(block_dir, self.blocks_ext):
                key, _ = os.path.splitext(os.path.basename(file))
                if key in self.blocks:
                    self.blocks[key].append(file)
                else:
                    self.blocks[key] = [file]

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data)
        self.dir_hr = os.path.join(self.apath, PictureFormat.INDEX_DIC[PictureFormat.ORIGINAL])
        self.dir_lr = os.path.join(self.apath)
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.npz', '.npz')




    def read_npz_file(self, file):
        def UpSamplingChroma(UVPic):
            return UVPic.repeat(2, axis=0).repeat(2, axis=1)
        if self.args.image_pin_memory:
            if file not in self.image_dic:
                self.image_dic[file] = np.load(file)
            f = self.image_dic[file]
        else:
            f = np.load(file)
        return np.stack((f['Y'], UpSamplingChroma(f['Cb']), UpSamplingChroma(f['Cr'])), axis=2)

    def read_npz_split_yuv(self, file):
        if self.args.image_pin_memory:
            if file not in self.image_dic:
                self.image_dic[file] = np.load(file)
            f = self.image_dic[file]
        else:
            f = np.load(file)
        return [f['Y'][:,:,None], np.stack((f['Cb'], f['Cr']), axis=2)]

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        lr = []
        if self.args.sc:
            hr = self.read_npz_split_yuv(f_hr)
            for flr in self.images_lr:
                lr.append(self.read_npz_split_yuv(flr[idx]))

        else:
            hr = self.read_npz_file(f_hr)
            for flr in self.images_lr:
                lr.append(self.read_npz_file(flr[idx]))
        return lr, hr, filename

    def get_patch(self, lr, hr):
        def _get_patch(*args, ih, iw, patch_size=96):
            # ih, iw = args[0].shape[:2]


            ip = patch_size

            ix = random.randrange(0, iw - ip + 1)
            iy = random.randrange(0, ih - ip + 1)
            if self.args.sc:
                ix >>= 1
                ix <<= 1
                iy >>= 1
                iy <<= 1
                ret = [
                    [a[iy//i:iy//i + ip//i, ix//i:ix//i + ip//i] for i, a in enumerate(args[0], 1)],
                    [[a[iy//i:iy//i + ip//i, ix//i:ix//i + ip//i]  for i, a in enumerate(arg, 1)] for arg in args[1]]
                ]
            else:
                ret = [
                    args[0][iy:iy + ip, ix:ix + ip, :],
                    [a[iy:iy + ip, ix:ix + ip, :] for a in args[1]]
                ]
            return ret[0], ret[1], (iy, ip, ix, ip)
        if self.args.sc:
            tpy, tpx = hr[0].shape[:2]
        else:
            tpy, tpx = hr.shape[:2]
        ty, tx = 0, 0
        if self.train:
            hr, lr, (ty, tpy, tx, tpx) = _get_patch(
                hr, lr, ih=tpy, iw=tpx,
                patch_size=self.args.patch_size
            )
        return lr, hr, (ty, tpy, tx, tpx)


    def getBlock2d(self, y, dy, x, dx, idx, name, value_idx = 0):
        if self.args.sc:
            block2d = np.full((dy//2, dx//2), np.nan)
        else:
            block2d = np.zeros((dy, dx))
        dy = y + dy
        dx = x + dx
        # 0:xpos, 1:ypos, 2:width, 3:height
        block = np.load(self.blocks[name][idx]).T
        filtered = block[:, ~np.any([(block[3] + block[1]) <= y,
                                     (block[0] + block[2]) <= x,
                                     block[1]>=dy,
                                     block[0]>=dx
                                     ], axis=0)]
        filtered[2, filtered[0]<x] = (filtered[0, filtered[0]<x] + filtered[2, filtered[0]<x]) - x
        filtered[3, filtered[1]<y] = (filtered[1, filtered[1]<y] + filtered[3, filtered[1]<y]) - y
        filtered[0, filtered[0]<x] = x
        filtered[1, filtered[1]<y] = y
        filtered[2, (filtered[0] + filtered[2]) > dx] = dx - filtered[0, (filtered[0]+filtered[2] > dx)]
        filtered[3, (filtered[1] + filtered[3]) > dy] = dy - filtered[1, (filtered[1] + filtered[3]) > dy]
        filtered[0,:] -= x
        filtered[1,:] -= y
        if self.args.sc:
            filtered[0:4, :] //= 2
        for _xp, _yp, _w, _h, *values in filtered.T:
            block2d[_yp:_yp+_h, _xp:_xp+_w] = values[value_idx]

        return block2d

    def getBlockScalar(self, idx, name, value_idx=0):
        block = np.load(self.blocks[name][idx]).T
        return block[4+value_idx].mean(axis=0)

    def getBlock2dFromScalar(self, value):
        return np.full((self.args.patch_size,self.args.patch_size), value)

    def np2tensor(self, args, rgb_range=1023):
        def _np2Tensor(img):
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
            tensor = torch.from_numpy(np_transpose).float()
            tensor.mul_(1 / rgb_range)
            return tensor

        if self.args.sc:
            return [[_np2Tensor(a) for a in arg] for arg in args]
        else:
            return [_np2Tensor(a) for a in args]


    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        if self.train and self.args.more_noisy:
            mse = -1
            tlr, thr, tpos = None, None, None
            for _ in range(self.args.more_noisy):
                _lr, _hr, _pos = self.get_patch(lr, hr)
                temp =np.mean(np.square(_lr[0].astype('int32') - _hr[0]))
                if temp>mse:
                    mse = temp
                    tlr, thr, tpos = _lr, _hr, _pos
            lr, hr, pos = tlr, thr, tpos
        else:
            lr, hr, pos = self.get_patch(lr, hr)
        # self.getBlock2d(*pos, idx, BlockType.Chroma_IntraMode)
        return self.np2tensor(lr), self.np2tensor([hr])[0], filename
