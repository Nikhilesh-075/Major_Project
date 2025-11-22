from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=8, img_size=380, **kwargs):
    """
    Create train/val/test dataloaders with flexible DataLoader options.
    Extra arguments like num_workers, pin_memory, prefetch_factor
    can be passed via kwargs.
    """
    transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


    train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=transform)
    val_dataset   = datasets.ImageFolder(f"{data_dir}/val", transform=transform)
    test_dataset  = datasets.ImageFolder(f"{data_dir}/test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader
