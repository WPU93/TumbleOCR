import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from data_utils import get_vocabulary
import urllib
from utils.transforms.text_aug import text_aug
class ocrDataset(Dataset):
    def __init__(self,data_path,dict_path,maxlen,imgH,imgW,image_transform=None):
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
        self.count = 0
        if not os.path.isfile(self.dict_path):
            print(self.dict_path + "does not exist")
        if not os.path.isfile(self.data_path):
            print(self.data_path + "does not exist")
        self.char2id,self.id2char = get_vocabulary(self.dict_path)#获取字典
        with open(self.data_path,"r",encoding="utf-8-sig")as rf:
            lines = rf.readlines()[:10000]
            for iter,line in enumerate(lines):
                length = []
                label = np.zeros([self.maxlen],np.int)#用0填充label
                try:
                    image_path,text = line.strip().split("\t")
                except:
                    print(line)
                if "mySynth" in image_path:
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
    def __len__(self):
        return self.size
    def __getitem__(self, index):
        #wf = open("tmp.txt","a")
        image_path = self.images_list[index]
        label = self.labels_list[index]
        length = self.length_list[index]
        text = self.texts_list[index]
        try:
            #image = Image.open(image_path).convert("RGB")
             image = Image.open(urllib.request.urlopen(image_path)).convert('RGB')
        except IOError:
            print('Corrupted image for '+image_path)
            return self[index + 1]
        w,h = image.size
        if h > 2*w:
            print('image h>2w:'+image_path)
            return self[index + 1]
        # try:
        if True:
            image = Image.fromarray(text_aug(np.array(image)))
        # except:
            # print('text_aug error:'+image_path)
            # return self[index + 1]
        
        new_width = int(w/h*self.imgH)
        if new_width < self.imgW:#保持纵横比,用0填充
            resize_img = image.resize((new_width,self.imgH))
            new_img = Image.new('RGB', (self.imgW,self.imgH), (0,0,0))
            new_img.paste(resize_img,(0,0))
        else:
            new_img = image.resize((self.imgW,self.imgH))
        #new_img = new_img.convert("L")#转灰度图
        if self.image_transform:
            new_img = self.image_transform(new_img)
        else:
            new_img = transforms.ToTensor()(new_img)
        return (new_img,label,length,text)


#def collate_fn(batch):
#    batch.sort(key=lambda x: len(x[1]), reverse=True)
#    img, label = zip(*batch)
#    pad_label = []
#    lens = []
#    max_len = len(label[0])
#    for i in range(len(label)):
#        temp_label = [0] * max_len
#        temp_label[:len(label[i])] = label[i]
#        pad_label.append(temp_label)
#        lens.append(len(label[i]))
#    return img, pad_label, lens




if __name__=="__main__":
    image_transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    ]
    )
    maxlen, imgH, imgW = 30, 48, 160
    data_path = "/data/remote/ocr_data/OCR_GT/sampler.txt"
    dict_path = "../dict/dict.txt"
    train_dataset = ocrDataset(data_path,dict_path,maxlen,imgH,imgW,image_transform)
    print("num of data:",len(train_dataset))
    trainset_dataloader = DataLoader(dataset=train_dataset,
                                     batch_size=128,
                                     shuffle=False,
                                     num_workers=4)

    for i_batch, (image,label,length,text) in enumerate(trainset_dataloader):
         
        if i_batch%10 == 1:
            print(i_batch/len(trainset_dataloader))

