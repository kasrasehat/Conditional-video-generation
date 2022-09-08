from Dataloader import CustomDataLoader
from torchvision import transforms
from torch.utils.data import DataLoader

batch_size = 8
device = 'cuda'
height, width = 224, 224
trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# dataset
dataset1 = CustomDataLoader(fmri_file='./data/fmri/sub-01_perceptionNaturalImageTraining_original_VC.h5',
                            imagenet_folder='./data/images/training',
                            transform=trans)
dataloader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True, drop_last=True)