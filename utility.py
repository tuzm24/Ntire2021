import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import pandas as pd
from help_func.help_func import myUtil
import shutil


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        self.now = now
        self.dfcol = ['date', 'model_name', 'lr', 'batch_size', 'training_data', 'patch_size', 'epochs', 'base_psnr',
                      'best_psnr', 'latest_psnr']
        args.save = args.model + '_' + os.path.basename(args.dir_data)  + now
        self.name = args.save
        self.df = self.initCSVFile()
        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                self.name = self.args.load
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt')) else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    @staticmethod
    def GCExperience():
        if not os.path.exists(os.path.join('..', 'experiment')):
            return
        dir_list = myUtil.getDirlist(os.path.join('..', 'experiment'))
        for dir in dir_list:
            try:
                if not os.listdir(os.path.join(dir, 'model')):
                    shutil.rmtree(dir)
            except Exception as e:
                print(e)
        return

    def initCSVFile(self):
        filename = os.path.join('..', 'experiment', './result.csv')
        if not os.path.exists(filename):
            df = pd.DataFrame(columns=self.dfcol)
        else:
            df = pd.read_csv(filename, index_col=0)
        return df

    def writeCSVFile(self, best, latest, anc, epoch):
        data = {'date': str(self.now),
                'model_name': str(self.args.model),
                'lr': str(self.args.lr),
                'batch_size': str(self.args.batch_size),
                'training_data': str(self.args.dir_data),
                'patch_size': str(self.args.patch_size),
                'epochs': str(int(epoch)),
                'base_psnr': str(float(anc)),
                'best_psnr': str(float(best)),
                'latest_psnr': str(float(latest))}
        self.df.loc[self.name] = data
        self.df.to_csv('./result.csv')

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        log_with_date = str(datetime.datetime.now().strftime('[%Y-%m-%d-%H-%M-%S]  ')) + log
        self.log_file.write(log_with_date + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())

        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]

        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            # filename = self.get_path(
            #     'results-{}'.format(dataset.dataset.name),
            #     '{}_x{}_'.format(filename, scale)
            # )
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}'.format(filename)
            )

            os.makedirs(os.path.dirname(filename), exist_ok=True)

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    # valid = diff[..., shave:-shave, shave:-shave]
    mse = diff.pow(2).mean()

    return -10 * math.log10(mse)


def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''

    # optimizer

    class LearningRateWarmUP(object):
        def __init__(self, optimizer, warmup_iteration, target_lr, after_scheduler=None):
            self.optimizer = optimizer
            self.warmup_iteration = warmup_iteration
            self.target_lr = target_lr
            self.after_scheduler = after_scheduler
            self.last_epoch = 0

        def warmup_learning_rate(self, cur_iteration):
            warmup_lr = self.target_lr * float(cur_iteration + 1) / float(self.warmup_iteration)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr

        def step(self):
            if self.last_epoch < (self.warmup_iteration + 1):
                self.warmup_learning_rate(self.last_epoch)
            else:
                self.after_scheduler.step(self.last_epoch - self.warmup_iteration)
            self.last_epoch += 1

        def get_lr(self):
            if self.last_epoch < (self.warmup_iteration + 1):
                return [self.target_lr * float(self.last_epoch + 1) / float(self.warmup_iteration)]
            return self.after_scheduler.get_lr()

    trainable = filter(lambda x: x.requires_grad, target.parameters())

    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler

    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    if args.lr_warm_up:
        scheduler_class = LearningRateWarmUP
    else:
        scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)

    if args.lr_cosine:
        kwargs_scheduler = {'warmup_iteration': args.lr_warm_up, 'target_lr': args.lr,
                            'after_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)}

    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

