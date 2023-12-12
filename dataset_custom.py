import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.video_folders = [f for f in os.listdir(root_dir) if f.startswith('video_')]
        self.data = self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        frames = [f"image_{j}.png" for j in range(22)]
        masks = np.load(os.path.join(self.root_dir, video_folder, 'mask.npy'))

       
        frame_list=[]
        mask_list=[]
        for i in range(22):
            frame_path = os.path.join(self.root_dir, video_folder, frames[i])
            frame =Image.open(frame_path).convert('RGB')

            if self.transform:
                frame = self.transform(frame)
            #print(frame.shape)
            frame_list.append(frame)

            mask = torch.Tensor(masks[i])  # Assuming the mask is a single-channel array
            mask_list.append(mask)
        mask_batch=torch.stack(mask_list)
        frame_batch=torch.stack(frame_list)

            #sample_list.append({'frame': frame, 'mask': mask})

        return frame_batch,mask_batch

    def _load_data(self):
        data = []
        for video_folder in self.video_folders:
            data.append({'video_folder': video_folder})
        return data

# Example usage
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     # Add more transforms if needed
# ])
# batch_size=2
# dataset = CustomDataset(root_dir='./dataset/train', transform=transform)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
# for batch in dataloader:
#     images, masks = batch
#     print(images.shape,masks.shape)
#     images=images.view(batch_size * 22, 3, 160, 240)
#     masks=masks.view(batch_size * 22,160, 240)
#     print(images.shape,masks.shape)
