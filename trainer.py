import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm
from help_torch import rgb_to_ycbcr
from help_torch import ycbcr_to_rgb



class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.upsampler = [torch.nn.Upsample(scale_factor=s, mode='bicubic') for s in args.scale]
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8


    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)

        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            if self.args.grid_batch:
                b, n, c, h, w = lr.shape
                _, _, _, h2, w2 = hr.shape
                lr = lr.view(b * n, c, h, w)
                hr = hr.view(b * n, c, h2, w2)
            if self.args.jpeg_yuv_domain:
                hr = rgb_to_ycbcr(hr)
                lr[:,:3,...] = rgb_to_ycbcr(lr[:,:3,...])
            timer_data.hold()
            timer_model.tic()
            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            if self.args.grid_batch:
                sr = sr[:, :, self.args.grid_batch:-self.args.grid_batch,
                             self.args.grid_batch:-self.args.grid_batch]
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def train_class(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)

        print(len(self.loader_train.dataset))
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale)+1)
        )
        self.model.eval()

        timer_test = utility.timer()
        perlist = []
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    if self.args.grid_batch:
                        b, n, c, h, w = lr.shape
                        _, _, _, h2, w2 = hr.shape
                        lr = lr.view(b*n, c, h, w)
                        hr = hr.view(b*n, c, h2, w2)
                        lr_list = [lr[i:i+200] for i in range(0, b*n, 200)]
                        sr_list = []
                        for _lr in lr_list:
                            sr_list.append(self.model(_lr, idx_scale)[:, :, self.args.grid_batch:-self.args.grid_batch,
                             self.args.grid_batch:-self.args.grid_batch])
                        sr = torch.cat(sr_list, dim=0)
                        lr = lr[:, :, self.args.grid_batch:-self.args.grid_batch,
                             self.args.grid_batch:-self.args.grid_batch]
                    else:
                        if self.args.jpeg_yuv_domain:
                            lr[:,:3,...] = rgb_to_ycbcr(lr[:,:3,...])

                        sr = self.model(lr, idx_scale)

                    if self.args.jpeg_yuv_domain:
                        sr[:,:3,...] = ycbcr_to_rgb(sr[:,:3,...])
                        lr = rgb_to_ycbcr(lr[:,:3,...])
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]

                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if lr.size(1)>3:
                        lr = lr[:,:3,...]
                    if self.args.save_gt:
                        save_list.extend([lr, hr])
                    if self.args.scale[idx_scale]>1:
                        lr = self.upsampler[idx_scale](lr)
                    self.ckp.log[-1, idx_data, -1] += utility.calc_psnr(
                        lr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                self.ckp.log[-1, idx_data, -1] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} (Blur : {:.3f}) @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[0][idx_data, -1],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

                self.ckp.writeCSVFile(best[0][idx_data, idx_scale], self.ckp.log[-1, idx_data, idx_scale],
                                      best[0][idx_data, -1], best[1][idx_data, idx_scale] + 1)

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')
        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

