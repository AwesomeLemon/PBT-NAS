import torchvision
import torch.utils.data

class STL10AdvNasDataProvider:
    def __init__(self, dataset_path, train_batch_size, n_workers, **kwargs):
        self.n_workers = n_workers
        self.train_batch_size = train_batch_size
        self.dataset_path = dataset_path

    def create_dataloaders(self):
        Dt = torchvision.datasets.STL10 # create here because otherwise passing the Dataset object via Ray is slow
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(48),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = Dt(root=self.dataset_path, split='train+unlabeled', transform=transform, download=True)

        dataloaders = {'train': torch.utils.data.DataLoader(dataset, batch_size=self.train_batch_size, shuffle=True,
                                                            num_workers=self.n_workers,
                                                            pin_memory=True
                                                            ),
                       'val': None}
        return dataloaders