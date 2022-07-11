from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler


def make_dataloaders_from_dataset(ds, batch_size=64):
    indices = list(range(len(ds)))

    train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=17)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.3, random_state=17)
    # print(len(train_indices), len(val_indices), len(test_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(ds, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(ds, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(ds, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader


def make_dataloaders_from_datasets(train_ds, val_ds, test_ds, batch_size=64):
    train_loader = DataLoader(train_ds, batch_size=batch_size,  shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,  shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size,  shuffle=True)

    return train_loader, val_loader, test_loader
