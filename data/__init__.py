"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 
import csv
import pandas, random


images_path = "/content/AttentionGAN/datasets/DeepFashion/real/img/"
# images_path = "../IUV/img/"
test_pairs_path = "/content/AttentionGAN/datasets/DeepFashion/fasion-resize-pairs-test.csv"
train_pairs_path = "/content/AttentionGAN/datasets/DeepFashion/fasion-resize-pairs-train.csv"

color_save_path = "/content/AttentionGAN/datasets/DeepFashion/segments/colors/"
layer_save_path = "/content/AttentionGAN/datasets/DeepFashion/segments/layers/"
csv_save_path = "/content/AttentionGAN/datasets/DeepFashion/segments/colors.csv"

colnames = ['Name', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12']

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def getFilePath(raw_name):

    #array to store the different folders to go in to reach the final image
    folders = []
    
    b = raw_name.split("fashion")[1]
    a = raw_name.split("fashion")[1]

    #checking if there's women or men in the name and storing it in folders
    try:
        a = a.split("WOMEN")[1]
        folders.append("WOMEN")
    except:
        a = a.split("MEN")[1]
        folders.append("MEN")

    #storing the wearable type in folders
    folders.append(a.split("id",1)[0])
    a = a.split("id",1)[1]

    #storing the image name (in the folders there's an _ after id
    folders.append("id_"+a[:8])
    a = a[8:]
    a = a[:4] + "_" + a[4:]
    folders.append(a)
##    print(b)
    
    return folders

def getColorsFromIndex(L, index):
    colors = []

    for i in L:
        colors.append([int(val) for val in i[index].split("_")])

    return colors

def changeColors(colored_img, color_list, avail_ims, name):
        n_len = len(color_list[0])

        colored_image = colored_img.copy()

        n = random.randint(0, n_len)        
        orig_index = avail_ims.index(name)
        original_colors = getColorsFromIndex(color_list, orig_index)

        new_colors = getColorsFromIndex(color_list, n)

        # print("original_colors", original_colors)
        # print("new_colors", new_colors)

        '''ORDER : chest, head, LThigh, LShin, RThigh, RShin,  LArm, LF, RArm, RF, RFeet, LFeet'''

        for i in range(len(color_list)):
            colored_image[:,:,0] = np.where(colored_image[:,:,0]==original_colors[i][0], new_colors[i][0], colored_image[:,:,0])
            colored_image[:,:,1] = np.where(colored_image[:,:,1]==original_colors[i][1], new_colors[i][1], colored_image[:,:,1])
            colored_image[:,:,2] = np.where(colored_image[:,:,2]==original_colors[i][2], new_colors[i][2], colored_image[:,:,2])


        return colored_image

def create_segments(colored_img, color_list, avail_ims, name):
    # n_len = len(color_list[0])

    colored_image = colored_img.copy()

    # n = random.randint(0, n_len)        
    orig_index = avail_ims.index(name)
    original_colors = getColorsFromIndex(color_list, orig_index)

    # new_colors = getColorsFromIndex(color_list, n)

    # print("original_colors", original_colors)
    # print("new_colors", new_colors)

    '''ORDER : chest, head, LThigh, LShin, RThigh, RShin,  LArm, LF, RArm, RF, RFeet, LFeet'''


    for i in range(len(color_list)):
        colored_image[:,:,0] = np.where(colored_image[:,:,0]==original_colors[i][0], i, colored_image[:,:,0])
        # colored_image[:,:,1] = np.where(colored_image[:,:,1]==original_colors[i][1], new_colors[i][1], colored_image[:,:,1])
        # colored_image[:,:,2] = np.where(colored_image[:,:,2]==original_colors[i][2], new_colors[i][2], colored_image[:,:,2])

    segment_image = colored_image[:,:,0]

    return segment_image

def create_dataset_custom(csv_data, avail_ims, L, batch_size=4):
        index = 0
        # for inp,out in csv_data:
        n_pairs = len(csv_data)
        # x = np.zeros((4,256,256, 3+10+3+10+3))
        # y = np.zeros((4,256,256, 3+10+3+10+3))
        x = np.zeros((4,256,256, 3+1+3+1+3))
        y = np.zeros((4,256,256, 3+1+3+1+3))
        A_paths = []
        B_paths = []

        while(index<batch_size):
                
                # Generates a random number between 
                # a given positive range 
                n = random.randint(0, n_pairs-1)        

                inp , out = csv_data[n]

                

                if((inp in avail_ims) and (out in avail_ims)):
                        # print("matched")

                        frm = getFilePath(inp)
                        to = getFilePath(out)

                        A_paths.append(frm)
                        B_paths.append(to)

                        image_path_frm = images_path + frm[0] + "/" + frm[1] + "/" + frm[2] + "/" + frm[3]
                        image_path_to = images_path + to[0] + "/" + to[1] + "/" + to[2] + "/" + to[3]


                        #load real image
                        r_img1 = cv2.imread(image_path_frm)


                        #load layers for 1
                        
                        # image = np.load(layer_save_path + "layers_" + inp + ".npz")

                        # image = image["array1"]
                        # image[:,:,3] = np.where(image[:,:,11], 1.0,image[:,:,3])
                        # image[:,:,5] = np.where(image[:,:,10], 1.0,image[:,:,5])
                        # image = image[:,:,:10]
                        # l_img1 = image

                        #load colors for 1
                        c_img1 = cv2.imread(color_save_path + "color_" + inp + ".png")

                        #load real image for 2
                        r_img2 = cv2.imread(image_path_to)

                        #load layers for 2
                        # image = np.load(layer_save_path + "layers_" + out + ".npz")

                        # image = image["array1"]
                        # image[:,:,3] = np.where(image[:,:,11], 1.0,image[:,:,3])
                        # image[:,:,5] = np.where(image[:,:,10], 1.0,image[:,:,5])
                        # image = image[:,:,:10]
                        # l_img2 = image

                        #load colors for 2
                        c_img2 = cv2.imread(color_save_path + "color_" + out + ".png")

                        #change colors of c_img1
                        c_aug2 = changeColors(c_img2, L, avail_ims, out)

                        seg_1 = create_segments(c_img1, L, avail_ims, inp)

                        seg_2 = create_segments(c_img2, L, avail_ims, out)

    ##          cv2.imshow("augmenty", c_aug2)

                        # try:
                        #   x = np.dstack((x,r_img1, l_img1, c_img1, l_img2, c_img2))
                        #   y = np.dstack((y,r_img2, l_img2, c_img2, l_img1, c_img1))
                        # except:
                        #   x = 

                        im = (r_img1/255.0 - 0.5) * 2

                        # im = np.dstack((im, l_img1, c_img1, l_img2, c_img2))

                        #replacing original colors with the augmented ones
                        # im = np.dstack((im, (l_img1-0.5)*2, (c_img1/255.0 - 0.5) * 2, (l_img2-0.5)*2, (c_aug2/255.0 - 0.5) * 2))
                        im = np.dstack((im, seg_1, (c_img1/255.0 - 0.5) * 2, seg_2, (c_aug2/255.0 - 0.5) * 2))

                        # print(x.shape ,im.shape, r_img1.shape, l_img1.shape, c_img1.shape, l_img2.shape,c_img2.shape)
                        
                        x[index,:,:,:] = im
                        im = (r_img2/255.0 - 0.5) * 2
                        
                        # im = np.dstack((im, (l_img2-0.5)*2, (c_aug2/255.0 - 0.5) * 2 , (l_img1-0.5)*2, (c_img1/255.0 - 0.5) * 2))
                        im = np.dstack((im, seg_2, (c_aug2/255.0 - 0.5) * 2 , seg_2, (c_img2/255.0 - 0.5) * 2))
                        y[index,:,:,:] = im

                        
                        index += 1
        # a = np.moveaxis(x, 3, 1);b = np.moveaxis(y, 3, 1);a = torch.from_numpy(a.astype("float32"));b = torch.from_numpy(b.astype("float32"));
        a = torch.from_numpy(x.astype("float32"));b = torch.from_numpy(y.astype("float32"));a = a.permute(0, 3, 1, 2); b = b.permute(0,3,1,2);
        # c = a.permute(0,2,3,1)
        '''dict_keys(['A', 'B', 'A_paths', 'B_paths'])'''
        #for colab
    ##  a = torch.from_numpy(x)
    ##  b = torch.from_numpy(y)
    ##  a = x
    ##  b = y

        
        return {'A':a, 'B':b, 'A_paths': A_paths, 'B_paths':B_paths}

def create_dataset(opt, csv_data, avail_ims, L):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
##    data_loader = CustomDatasetDataLoader(opt)
##    dataset = data_loader.load_data_custom()
    dataset = create_dataset_custom(csv_data, avail_ims, L)

    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)

        print("from init dataset_class", dataset_class)
        print("from init dataset", dataset)

        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
