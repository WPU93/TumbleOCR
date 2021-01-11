import os
import numpy as np
from PIL import Image
import urllib
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from .data_utils import get_vocabulary
#from data_utils import get_vocabulary
from tumbocr.utils.transforms.text_aug import text_aug
traindata_list = ["MTWI2018","mySynth","Baidu_recognition_data"]

class ocrDataset(Dataset):
    def __init__(self,data_path,dict_path,maxlen,imgH,imgW,image_transform=None,is_training=True,text_aug=False):
        self.is_training = is_training
        self.text_aug = text_aug
        self.size = 0
        self.maxlen = maxlen
        self.imgH = imgH
        self.imgW = imgW
        self.image_transform = image_transform
        self.dict_path = dict_path #字典txt
        self.data_path = data_path #图片和标签txt
        self.char2id = {}
        self.id2char = {}
        self.images_list = []
        self.texts_list = []
        self.labels_list = []
        self.length_list = []
        if not os.path.isfile(self.dict_path):
            print(self.dict_path + "does not exist")
        if not os.path.isfile(self.data_path):
            print(self.data_path + "does not exist")
        self.char2id,self.id2char = get_vocabulary(self.dict_path)#获取字典
        with open(self.data_path,"r",encoding="utf-8-sig")as rf:
            lines = rf.readlines()[:]
            for iter,line in enumerate(lines):
                length = []
                label = np.zeros([self.maxlen],np.int)#用0填充label
                try:
                    image_path,text = line.strip().split("\t")
                except:
                    continue
                if "SynthChinese360w" in image_path:
                    continue
                self.images_list.append(image_path)
                self.texts_list.append(text)
                if len(text) >= self.maxlen:
                    text = text[:self.maxlen-1]
                for i,char in enumerate(text):
                    if char in self.char2id:
                        label[i] = self.char2id[char]
                    else:
                        label[i] = self.char2id["UNK"]
                label[len(text)] = self.char2id["EOS"]
                self.labels_list.append(label)
                self.length_list.append(len(text))
            self.size = len(self.images_list)
        print("Dataset init!")
    def __len__(self):
        return self.size
    def get_path(self, index):
        image_path = self.images_list[index]
        return image_path 
    def __getitem__(self, index):
        image_path = self.images_list[index]
        label = self.labels_list[index]
        length = self.length_list[index]
        text = self.texts_list[index]
        try:
             image = Image.open(urllib.request.urlopen(image_path)).convert('RGB')
        except IOError:
            print('Corrupted image for '+image_path)
            return self[index + 1]
        w,h = image.size
        if h > 3*w:
            if self.is_training:
                return self[index + 1]
            else:
                image = image.transpose(Image.ROTATE_90)
                w,h = h,w
        if self.text_aug:
            image = Image.fromarray(text_aug(np.array(image)))
        new_width = int(w/h*self.imgH)
        if new_width < self.imgW:#保持纵横比,用0填充
            resize_img = image.resize((new_width,self.imgH))
            new_img = Image.new('RGB', (self.imgW,self.imgH), (0,0,0))
            new_img.paste(resize_img,(0,0))
        else:
            new_img = image.resize((self.imgW,self.imgH))
        
        if self.image_transform:
            new_img = self.image_transform(new_img)
        else:
            new_img = transforms.ToTensor()(new_img)
        return (new_img,label,length,text)


if __name__=="__main__":
    image_transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    ]
    )
    maxlen, imgH, imgW = 30, 48, 160
    data_path = "/data/remote/ocr_data/OCR_GT/SynthChinese360w.txt"
    dict_path = "../dict/dict.txt"
    train_dataset = ocrDataset(data_path,dict_path,maxlen,imgH,imgW,image_transform)
    print("num of data:",len(train_dataset))
    trainset_dataloader = DataLoader(dataset=train_dataset,
                                     batch_size=512,
                                     shuffle=False,
                                     num_workers=32)

    for i_batch, (image,label,length,text) in enumerate(trainset_dataloader):
        if i_batch%100 == 0:
            print(i_batch/len(trainset_dataloader))

