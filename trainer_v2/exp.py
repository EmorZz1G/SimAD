import torch
import os

class Trainer:
    def __init__(self, config, model, get_loader_func):
        self.config = config
        args = config
        self.model = model

        config.data_path = os.path.join(config.data_pth, config.data_name)

        self.train_loader = get_loader_func(mode='train', index=config.index, batch_size=config.batch_size, data_path=config.data_path, dataset=config.data_name, dist=config.distributed, win_size=config.win_size, step=config.step)
        # self.val_loader = get_loader_func(mode='val', index=config.index, batch_size=config.batch_size, data_path=config.data_path, dataset=config.data_name, dist=config.distributed, win_size=config.win_size, step=config.step)
        self.test_loader = get_loader_func(mode='thres', index=config.index, batch_size=config.batch_size, data_path=config.data_path, dataset=config.data_name, dist=config.distributed, win_size=config.win_size, step=config.step)

        ngpus_per_node = torch.cuda.device_count()
        logs_pth = config.logs_pth
        logs_pth = os.path.join(logs_pth, config.data_name)
        save_pth = os.path.join(logs_pth, config.save_pth, config.data_name)

        if config.data_name in ['UCR', 'UCR_AUG']:
            self.best_pth = os.path.join(logs_pth, str(config.index), config.model_id)#, 'best_checkpoint.pth')
        else:
            self.best_pth = os.path.join(logs_pth, config.model_id)#, 'best_checkpoint.pth')

        self.logs_pth = logs_pth
        self.save_pth = save_pth
        os.makedirs(logs_pth, exist_ok=True)
        os.makedirs(save_pth, exist_ok=True)


        if args.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                # betas=(0.9, 0.999),
                # weight_decay=1e-4,
            )
        elif args.optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                # [{'params': model.module.t_fcn.parameters(), 'params': model.module.t_classifier.parameters()}],
                lr=args.lr,
                # betas=(0.9, 0.999),
                # weight_decay=1e-4,
            )

        if args.gpu==-1:
            device = torch.device("cpu")
        elif args.gpu:
            device = torch.device("cuda:{}".format(args.gpu))
        else:
            device = torch.device("cuda")

        self.device = device
        if args.resume==1:
            self.model = torch.load(os.path.join(self.save_pth, 'checkout.pth'), map_location=device)

    
    def train(self):
        pass

    def test(self):
        pass

        