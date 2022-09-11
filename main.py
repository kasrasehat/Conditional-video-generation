from Dataloader import CustomDataLoader
from torchvision import transforms
from torch.utils.data import DataLoader

batch_size = 2
device = 'cuda'
height, width = 224, 224
trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# dataset
dataset1 = CustomDataLoader(csv_file='Book1.csv',
                            transform=trans, n=10, height=height, width=width)
dataloader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True, drop_last=True)
for i, data in enumerate(dataloader1):
    print(i)
    print(data[0].shape)
    print(data[1])

