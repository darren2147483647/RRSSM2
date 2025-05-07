from glob import glob

from torch.utils.data import Dataset
from PIL import Image


class MSCOCO(Dataset):
    def __init__(self, root, transform, img_list=None):
        assert root[-1] == '/', "root to COCO dataset should end with \'/\', not {}.".format(
            root)

        if img_list:
            self.image_paths = []
            with open(img_list, 'r') as r:
                lines = r.read().splitlines()
                for line in lines:
                    self.image_paths.append(root + line)
        else:
            self.image_paths = sorted(glob(root + "*.jpg"))
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            object: image.
        """
        img_path = self.image_paths[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.image_paths)
    

class Kodak(Dataset):
    def __init__(self, root, transform):

        assert root[-1] == '/', "root to Kodak dataset should end with \'/\', not {}.".format(
            root)

        self.image_paths = sorted(glob(root + "*.png"))
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            object: image.
        """
        img_path = self.image_paths[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.image_paths)

#add dataset setting
class RIVERAVSSD(Dataset):
    def __init__(self, root, transform):
        assert root[-1] == '/', "root to the dataset should end with \'/\', not {}.".format(
            root)

        self.image_paths = sorted(glob(root + "*.jpg")+glob(root + "*.png"))
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            object: image.
        """
        img_path = self.image_paths[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.image_paths)
    
class RIVERAVSSD_rd(Dataset):
    def __init__(self, root1, root2 , transform):
        assert root1[-1] == '/', "root to the dataset should end with \'/\', not {}.".format(
            root1)
        assert root2[-1] == '/', "root to the dataset should end with \'/\', not {}.".format(
            root2)

        self.image_paths1 = sorted(glob(root1 + "*.jpg")+glob(root1 + "*.png"))
        self.image_paths2 = sorted(glob(root2 + "*.jpg")+glob(root2 + "*.png"))
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            object: image.
        """
        img_path1 = self.image_paths1[index]

        img1 = Image.open(img_path1).convert('RGB')

        if self.transform is not None:
            img1 = self.transform(img1)
            
        img_path2 = self.image_paths2[index]

        img2 = Image.open(img_path2).convert('RGB')

        if self.transform is not None:
            img2 = self.transform(img2)

        return img1, img2

    def __len__(self):
        return len(self.image_paths1)
class ZipDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.length = min(len(dataset1), len(dataset2))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.dataset1[idx], self.dataset2[idx]