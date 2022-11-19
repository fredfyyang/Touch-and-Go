import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
# from data.image_folder import make_dataset
# from PIL import Image


class TouchAndGoDataset(BaseDataset):

    def __init__(self, opt):
        """Initialize Touch and Go dataset class.

        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.transform = get_transform(opt)
        
        if opt.isTrain:
            with open(os.path.join('./datasets/touch_and_go', 'train.txt'),'r') as f:
                data = f.read().split('\n')
        else:
            with open(os.path.join('./datasets/touch_and_go', 'test.txt'),'r') as f:
                data = f.read().split('\n')
        
        self.length = len(data)
        self.data = data
        self.root = opt.dataroot

    def __getitem__(self, index):
        """Return a data point containing (Image A, Gelsight A, Image B and Gelsight B).
        """
        index_A = index

        # Random generate index_B different from index_A
        index_B = 0
        while True:
            index_B = random.randint(0, self.length - 1)
            if index_A != index_B:
                break
        
        assert index_A < self.length,'index_A out of range'
        assert index_B < self.length,'index_B out of range'

        A_raw_path, _ = self.data[index_A].strip().split(',') # mother path for A
        B_raw_path, _ = self.data[index_B].strip().split(',') # mother path for B
        A_dir, A_idx = os.path.join(self.root, A_raw_path[:15]) , A_raw_path[16:]
        B_dir, B_idx = os.path.join(self.root, B_raw_path[:15]) , B_raw_path[16:]

        # Read A images and gelsight
        A_img_path = os.path.join(A_dir, 'video_frame', A_idx)
        A_gelsight_path = os.path.join(A_dir, 'gelsight_frame', A_idx)
        A_img = Image.open(A_img_path).convert('RGB')
        A_gel = Image.open(A_gelsight_path).convert('RGB')

        # Read B images and gelsight
        B_img_path = os.path.join(B_dir, 'video_frame', B_idx)
        B_gelsight_path = os.path.join(B_dir, 'gelsight_frame', B_idx)
        B_img = Image.open(B_img_path).convert('RGB')
        B_gel = Image.open(B_gelsight_path).convert('RGB')

        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)
        
        # transform
        A_img = transform(A_img)
        A_gel = transform(A_gel)
        B_img = transform(B_img)
        B_gel = transform(B_gel)

        return {'A': A_img, 'B': B_img, 'A_touch': A_gel, 'B_touch': B_gel, 'A_paths': A_img_path, 'B_paths': B_img_path}

    def __len__(self):
        """Return the total number of images."""
        return self.length
