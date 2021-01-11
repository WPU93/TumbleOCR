from easydict import EasyDict as edict
import yaml
Config = edict()
Config.NAME = "sar-config"

#-----Global-----
Config.Global = edict()
Config.Global.gpu = None
Config.Global.aug = True
Config.Global.dict_path = "dict/dict.txt"
Config.Global.start_epoch = 0
Config.Global.epochs = 100
Config.Global.resume = True
Config.Global.pretrained = ""
Config.Global.checkpoints = '../save_models/'
Config.Global.print_feq = 10
Config.Global.val_iter = 10000
Config.Global.save_iter = 10000
Config.Global.out_seq_len = 30
Config.Global.model_name = ""

#-----Model-----
Config.Model = edict()
Config.Model.arch = SAR
Config.Model.backbone = 
Config.Model.ctc = True
Config.Model.featrue_layers = 512
Config.Model.hidden_dim = 512
Config.Model.hidden_dim_de = 512
Config.Model.num_layers = 2
Config.Model.embedding_size = 512

#-----Optimizer-----
Config.Optimizer = edict()
Config.Optimizer.type = "Adam"
Config.Optimizer.lr = 0.001
Config.Optimizer.warmup = -1
Config.Optimizer.warmup_lr = 1e-6
Config.Optimizer.scheduler = None # [None,cosine,step]
Config.Optimizer.weight_decay = 0.0 # the model maybe under-fitting, 0.0 gives much better results.

#-----Train-----
Config.Train = edict()
Config.Train.image_shape = [48,200]
Config.Train.aug = True # [rand_aug,text_aug]
Config.Train.batch_size = 320
Config.Train.workers = 32
Config.Train.drop_last = True
Config.Train.shuffle = True
Config.Train.balanced = True
# Config.Train.train_path = "/data/remote/ocr_data/OCR_GT/train_art_baidu_mtwi_rects_mine_url.txt"
Config.Train.train_path = ""

#-----Val-----
Config.Val = edict()
Config.Val.image_shape = [48,200]
Config.Val.aug = False
Config.Val.batch_size = 320
Config.Val.workers = 32
Config.Val.drop_last = True
Config.Val.shuffle = True
Config.Val.val_path = "/data/remote/ocr_data/OCR_GT/openReal/MTWI2018_test_url.txt"
wdict ={}

for k, v in Config.items():
    if isinstance(v, edict):
        wdict[k] = dict(v)

print(wdict)

fp = open('rec_sar_train_config.yaml', 'w')
fp.write(yaml.dump(wdict))
fp.close()