import cv2, torch, torch.nn as nn
import numpy as np, os
import kornia as k
class BairDataset(torch.utils.data.Dataset):

    def __init__(self, transforms, datakeys, opt, train=True, debug=False):
        if train:
            mode = "train"
        else:
            mode = "test"
        self.data_path = opt['data_path']
        self.mode = mode
        self.seq_length = opt['sequence_length']
        self.do_aug = opt['aug']

        print(f"Setup dataloder {mode}")
        self.videos = []
        videos = os.listdir(self.data_path + mode + '/')
        for vid in videos:
            if vid.isnumeric():
                self.videos.append(mode + '/' + vid + '/')
            else:
                subvideos = os.listdir(self.data_path + mode + '/' + vid + '/')
                for svid in subvideos:
                    self.videos.append(mode + '/' + vid + '/' + svid + '/')
        self.length = len(self.videos)

        self.aug = torch.nn.Sequential(
                    k.Resize(size=(opt['img_size'], opt['img_size'])),
                    #k.augmentation.Normalize(0.5, 0.5)
        )

    def __len__(self):
        return self.length

    def load_img(self, video, frame):
        try:
            filename = str(self.data_path + video + f'{frame}.png')
            img = cv2.imread(filename)

            return k.image_to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) /255.0
        except:
            raise RuntimeError(f'Could not deliver file {filename}')

    def __getitem__(self, idx):

        video  = self.videos[idx]
        frames = np.arange(0, 30)

        ## Load sequence
        start = 0 if self.mode == 'test' else np.random.randint(0, 30 - self.seq_length + 1)
        seq = torch.stack([self.load_img(video, frames[start + i]) for i in range(self.seq_length)], dim=0)
        #print(torch.min(self.load_img(video, frames[0])), torch.max(self.load_img(video, frames[0])))
        #exit()
        return {'images': self.aug(seq)}
