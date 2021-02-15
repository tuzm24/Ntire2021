import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

if __name__ == '__main__':
    utility.checkpoint.GCExperience()
    checkpoint = utility.checkpoint(args)
torch.manual_seed(args.seed)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            if not args.test_only:
                print('Train loader len : ', len(t.loader_train.dataset))
                print('Test loader len : ', len(t.loader_test[0].dataset))
            while not t.terminate():
                t.train()
                t.test()
            checkpoint.done()

if __name__ == '__main__':
    main()
