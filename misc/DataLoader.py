import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

input_size = (28,28)
mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225]




# Define transformations (adjust as needed)
traintransform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    transforms.RandomResizedCrop(input_size, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
])


valtransform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


train_path = "/home/ss/STUDY/PyTorch-CPP/Custom_DataLoader/PetImages/Train"

test_path = "/home/ss/STUDY/PyTorch-CPP/Custom_DataLoader/PetImages/Test"


print("[INFO] loading the training and validation dataset...")
train_dataset = ImageFolder(root=train_path,transform=traintransform)
test_dataset = ImageFolder(root=test_path,transform=valtransform)
print("[INFO] training dataset contains {} samples...".format(len(train_dataset)))
print("[INFO] validation dataset contains {} samples...".format(len(test_dataset)))


def custom_collate(batch):
    """
    Custom collate function to convert PIL Images to tensors
    """
    data = [item[0] for item in batch]  # Extract images from the batch
    target = [item[1] for item in batch]  # Extract labels from the batch
    
    # Convert PIL Images to tensors
    data = torch.stack(data, dim=0)
    target = torch.tensor(target)
    
    return data, target

def prepare_data_loaders():

    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=24,
        sampler=train_sampler,collate_fn=custom_collate)

    data_loader_test = torch.utils.data.DataLoader(
        test_dataset, batch_size=24,
        sampler=test_sampler,collate_fn=custom_collate)

    return data_loader, data_loader_test