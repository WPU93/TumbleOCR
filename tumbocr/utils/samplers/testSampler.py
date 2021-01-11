import sys
sys.path.append("../..")

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from data.load_data_url import ocrDataset
from torch.utils.data import Dataset, DataLoader
import ClassAwareSampler
from config import get_args
from utils.samplers.dataSampler import recSampler

if __name__ == "__main__":
    args = get_args(sys.argv[1:])

    normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    image_transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        normalize,
    ])

    train_dataset = ocrDataset("/data/remote/ocr_data/OCR_GT/train_art_baidu_mtwi_rects_mine_url.txt","../../dict/dict.txt",
                    args.out_seq_len,args.height,args.width,
                    image_transform)

    print("num of data:",len(train_dataset))
    trainset_dataloader = DataLoader(dataset=train_dataset,
            batch_size=64,
            sampler=dataSampler(train_dataset),
            num_workers=4)
    for i_batch, (image,label,x2,x3) in enumerate(trainset_dataloader):
        #print(image.shape,label,x2,x3)
        if (i_batch+1)%100==0:
            break
        

