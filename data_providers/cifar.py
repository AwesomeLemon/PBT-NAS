import torchvision
import torch.utils.data

class CIFAR10AdvNasDataProvider:
    def __init__(self, dataset_path, train_batch_size, n_workers, **kwargs):
        Dt = torchvision.datasets.CIFAR10
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.dataset = Dt(root=dataset_path, train=True, transform=transform, download=True)
        self.n_workers = n_workers
        self.train_batch_size = train_batch_size

    def create_dataloaders(self):
        dataloaders = {'train': torch.utils.data.DataLoader(self.dataset, batch_size=self.train_batch_size, shuffle=True,
                                                            num_workers=self.n_workers, pin_memory=True),
                       'val': None}
        return dataloaders