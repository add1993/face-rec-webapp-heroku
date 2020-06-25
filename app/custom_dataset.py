from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from skimage import io

class MyCustomDataset(Dataset):
    def __init__(self, dataset, transforms=None):
        self.transforms = transforms
        self.dataset = dataset

    def __getitem__(self, index):
        img_path,label = self.dataset[index]# Some data read from a file or image
        #data = Image.open(img_path)
        data = io.imread(img_path)
        if self.transforms is not None:
            data = self.transforms(data)
        return (data, label)

    def __len__(self):
        return len(self.dataset) # of how many data(images?) you have

if __name__ == '__main__':
    # Define transforms (1)
    transformations = transforms.Compose([transforms.CenterCrop(100), transforms.ToTensor()])
    # Call the dataset
    #custom_dataset = MyCustomDataset(..., transformations)
