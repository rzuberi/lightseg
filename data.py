import numpy as np
import tifffile
#from cellpose import models, core
import random
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def import_images(images_path,normalisation=False,num_imgs=20):
    images = [np.squeeze(tifffile.imread(images_path + str(i) + '.tif')) for i in range(num_imgs)]
    if normalisation == True:
        return [(image-np.min(image))/(np.max(image)-np.min(image)) for image in images]
    return images

def get_random_crops(images, masks, cellprobs):

    imgs = [(image-np.min(image))/(np.max(image)-np.min(image)) for image in images]
    masks = [(mask-np.min(mask))/(np.max(mask)-np.min(mask)) for mask in masks] #they are already 0 and 1 no need to normalise
    cellprobs = [(cellprob-np.min(cellprob))/(np.max(cellprob)-np.min(cellprob)) for cellprob in cellprobs]

    #make random crops with them
    imgs_aug = []
    mks_aug = []
    cellprob_aug = []
    for i in range(len(imgs)):
        img = imgs[i]
        mask = masks[i]
        cellprob = cellprobs[i]
        for j in range(1000):
            #crop_width = random.randint(5,256)
            #crop_height = random.randint(5,256)
            #crop_val = random.randint(5,256)
            crop_val = 256
            assert img.shape[0] >= crop_val
            assert img.shape[1] >= crop_val
            assert img.shape[0] == mask.shape[0]
            assert img.shape[1] == mask.shape[1]
            x = random.randint(0, img.shape[1] - crop_val)
            y = random.randint(0, img.shape[0] - crop_val)
            img_cropped = np.array(img[y:y+crop_val, x:x+crop_val],dtype=np.float16)
            mask_cropped = mask[y:y+crop_val, x:x+crop_val]
            cellprob_cropped = cellprob[y:y+crop_val, x:x+crop_val]

            #Filters out the masks that have only background
            if len(np.unique(mask_cropped)) == 1:
                j -= 1
                continue
            
            for i in range(4):
                img_cropped_exp = np.expand_dims(img_cropped,-1)
                mask_cropped_exp = np.expand_dims(mask_cropped,-1)
                cellprob_cropped_exp = np.expand_dims(cellprob_cropped,-1)

                img_cropped_exp = np.moveaxis(img_cropped_exp, -1, 0)
                mask_cropped_exp = np.moveaxis(mask_cropped_exp,-1,0)
                cellprob_cropped_exp = np.moveaxis(cellprob_cropped_exp,-1,0)

                imgs_aug.append(img_cropped_exp)
                mks_aug.append(mask_cropped_exp)
                cellprob_aug.append(cellprob_cropped_exp)

                img_cropped = np.rot90(img_cropped)
                mask_cropped = np.rot90(mask_cropped)
                cellprob_cropped = np.rot90(cellprob_cropped)
        
    imgs_aug = torch.tensor(imgs_aug)
    mks_aug = torch.tensor(np.array(mks_aug))
    cellprob_aug = torch.tensor(np.array(cellprob_aug))

    return imgs_aug, mks_aug, cellprob_aug

def get_data_loaders(imgs_aug, mks_aug, cellprob_aug):
    random_state = 10

    X_train_img, X_test_img, y_train_cp, y_test_cp = train_test_split(imgs_aug, cellprob_aug, test_size=0.33, random_state=random_state)

    X_train_cp, X_test_cp, y_train_mks, y_test_mks = train_test_split(cellprob_aug, mks_aug, test_size=0.33, random_state=random_state)

    trainDS_img = [(X_train_img[i],y_train_cp[i]) for i in range(len(X_train_img))]
    testDS_img  = [(X_test_img[i],y_test_cp[i]) for i in range(len(X_test_img))]
    trainLoader_img = DataLoader(trainDS_img, shuffle=True,
	                    batch_size=5, pin_memory=True,
	                    num_workers=2)
    testLoader_img = DataLoader(testDS_img, shuffle=True,
                            batch_size=5, pin_memory=True,
                            num_workers=2)

    trainDS_cp = [(X_train_cp[i],y_train_mks[i]) for i in range(len(X_train_cp))]
    testDS_cp  = [(X_test_cp[i],y_test_mks[i]) for i in range(len(X_test_cp))]
    trainLoader_cp = DataLoader(trainDS_cp, shuffle=True,
	                    batch_size=5, pin_memory=True,
	                    num_workers=2)
    testLoader_cp = DataLoader(testDS_cp, shuffle=True,
                            batch_size=5, pin_memory=True,
                            num_workers=2)

    return trainLoader_img, testLoader_img, trainLoader_cp, testLoader_cp

#def get_cellpose_data(images_path, augment=False, learning_rate=0.01):
#    num_imgs = 5
#
#    images = import_images(images_path,num_imgs=num_imgs)
#    images = np.array(images)
#    images = images
#
#    model = models.CellposeModel(model_type='nuclei',gpu=core.use_gpu())
#    masks, flows, styles = model.eval(images,channels=[[0,0]],cellprob_threshold=False)
#
#    masks = np.array(masks)
#    masks = np.where(masks>0,1,0)
#    flows = np.array(flows)
#    
#    cellprobs = np.array([flows[2][i] for i in range(num_imgs)])
#    return images, masks, cellprobs