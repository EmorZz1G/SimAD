from collections import defaultdict, namedtuple
import time
import datetime
import os
import json
class Object:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def dict_to_object(dict):
    return Object(**dict)
import torch
import pandas

class MyLogger:

    def __init__(self, config, model) -> None:
        self.exp_config = config
        self.train_datas = defaultdict(list)
        self.test_datas = defaultdict(list)
        self.log_id = time.strftime('%Y_%m_%d_%H:%M:%S')
        self.save_pth = config.logs_pth
        self.save_pth = os.path.join(self.save_pth, config.data_name)
        if config.data_name in ['UCR', 'UCR_AUG']:
            self.save_pth = os.path.join(self.save_pth, str(config.index))

        self.save_pth = os.path.join(self.save_pth, self.log_id)

        os.makedirs(self.save_pth, exist_ok=True)
        self.model = model
        self.cnt = 0


    def save_all(self):
        self.write_exp_config()
        self.write_train_datas()
        self.write_test_datas()

        


    def write_exp_config(self):
        exp = self.exp_config
        exp_str = json.dumps(exp.__dict__)
        f_pth = os.path.join(self.save_pth, 'exp_config.txt')
        with open(f_pth, 'w') as f:
            f.write(exp_str)
            f.close()



    @property
    def get_id(self):
        return self.log_id


    def write_train_datas(self):
        data = pandas.DataFrame(self.train_datas)
        fpth = os.path.join(self.save_pth, f'{self.cnt}_train_log.csv')
        data.to_csv(fpth)


    def write_test_datas(self):
        data = pandas.DataFrame(self.test_datas)
        fpth = os.path.join(self.save_pth, f'{self.cnt}_test_log.csv')
        data.to_csv(fpth)

    def add_iter_value(self, kv: dict, train=1, metric='', max_=1):
        self.cnt += 1
        # {it: 100, sim: 0.4}
        data = self.train_datas if train else self.test_datas
        for k,v in kv.items():
            data[k].append(v)

            if train == 0 and k == metric:
                if v >= max(data[k]) and max_ == 1:
                    torch.save(self.model, os.path.join(self.save_pth, 'best_checkout.pth'))
                elif v <= min(data[k]) and max_ == 0:
                    torch.save(self.model, os.path.join(self.save_pth, 'best_checkout.pth'))




    def print_datas(self):
        print('==Train==')
        ks = self.train_datas.keys()
        for k in ks:
            print(k, end='\t')
        print()
        for i in range(self.train_datas[k].__len__()):
            for k in ks:
                print(self.train_datas[k][i], end='\t')
            print()

        print('==Test==')
        ks = self.test_datas.keys()
        for k in ks:
            print(k, end='\t')
        print()
        for i in range(self.test_datas[k].__len__()):
            for k in ks:
                print(self.test_datas[k][i], end='\t')
            print()
            



if __name__ == '__main__':
    config = {'ok':1, 'logs_pth': r'/share/home/202220143416/TF_AD_models_logs/logs/', 'index': 1, 'data_name': 'UCR', 'blblb': 'dfdfd'}
    config = json.dumps(config)
    config = json.loads(config)
    config = dict_to_object(config)
    log = MyLogger(config, config)
    log.add_iter_value({'ep':3, 'sim':0.3, 'sim2':0.3})
    log.add_iter_value({'ep':3, 'sim':0.3, 'sim2':0.3})

    log.print_datas()
    log.write_exp_config()
    log.write_train_datas()
    print(log.get_id)
