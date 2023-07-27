import numpy as np
import torchvision
from PIL import Image
from cleanfid import fid
import os
import torch
def store_dataset_as_folder(dataset, out_path, target_resolution):
    os.makedirs(out_path, exist_ok=True)
    batch_size = 64
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size, shuffle=False,
                                             drop_last=False, num_workers=4)
    dataset.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(target_resolution),
            torchvision.transforms.ToTensor()
    ])
    for i, batch in enumerate(dataloader):
        imgs, target = batch
        for j, img in enumerate(imgs):
            img = Image.fromarray(np.uint8(np.transpose(img.numpy(), (1, 2, 0)) * 255))
            img.save(os.path.join(out_path, f'{i * batch_size + j}.png'))

if __name__ == '__main__':
    # 1. set your paths
    path_to_stl = '/absolute/path/to/stl'
    path_for_intermediate_img_storage = '/path/for/intermediate/images'

    # 2. store dataset statistics
    dataset = torchvision.datasets.STL10(root=path_to_stl, split='train+unlabeled')
    store_dataset_as_folder(dataset, path_for_intermediate_img_storage, target_resolution=48)
    fid.make_custom_stats('stl10-train', path_for_intermediate_img_storage, mode='legacy_tensorflow', num_workers=4)