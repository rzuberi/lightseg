from u_net import UNet
import torch
import numpy as np
#from data import get_cellpose_data
from train_network import train_model
from statistics import mean
import os
import math
from skimage import measure

class CellMaskModel():

    #TODO: make sure we can change the cellpose model if need
    def __init__(self, model_tye='nuclei',device="cpu"):
        self.device = device
        self.unet_cp = UNet()
        self.unet_cp.to(device)
        self.unet_mask = UNet()
        self.unet_mask.to(device)
        self.trainLoader_img = None
        self.testLoader_img = None
        self.trainLoader_cp = None
        self.testLoader_cp = None

    def import_model(self, first_network_path, second_network_path):
        if self.device =='cpu':
            self.unet_cp.load_state_dict(torch.load(first_network_path,map_location=torch.device('cpu')))
            self.unet_mask.load_state_dict(torch.load(second_network_path,map_location=torch.device('cpu')))
        else:
            self.unet_cp.load_state_dict(torch.load(first_network_path))
            self.unet_mask.load_state_dict(torch.load(second_network_path))

    def save_model(self, first_network_path, second_network_path):
        torch.save(self.unet_cp.state_dict(), first_network_path)
        torch.save(self.unet_mask.state_dict(), second_network_path)

    #def get_data(self, images_path):
    #    images, masks, cellprobs = get_cellpose_data(images_path, augment=False, learning_rate=0.01)
    #    imgs_aug, mks_aug, cellprob_aug = get_random_crops(images[:,:,:,0],masks,cellprobs)
    #    self.trainLoader_img, self.testLoader_img, self.trainLoader_cp, self.testLoader_cp = get_data_loaders(imgs_aug, mks_aug, cellprob_aug)

    def train_models(self,num_epochs,learning_rate=0.001):
        self.unet_cp = train_model(self.unet_cp,self.trainLoader_img,self.testLoader_img,learning_rate=learning_rate,num_epochs=num_epochs,device=self.device,loss="mse")
        #self.unet_mask = train_model(self.unet_mask,self.trainLoader_cp,self.testLoader_cp,learning_rate=learning_rate,num_epochs=num_epochs,device=self.device,loss="BCEwithLogits")
        self.unet_mask = train_model(self.unet_mask,self.trainLoader_cp,self.testLoader_cp,learning_rate=learning_rate,num_epochs=num_epochs,device=self.device,loss="dice")

    def get_pred(self,x,channel,encFeats=False,encFeats_list=False):
        if len(x.shape) == 3:
            x = x[:,:,channel]
        x, pad_val = self.expand_div_256(x)
        arrays = self.blockshaped(x,256,256)
        masks_crops = []
        inter_preds = []
        encFeats_cps = []
        encFeats_cps_list = []
        encFeats_masks = []
        self.unet_cp.eval()
        self.unet_mask.eval()

        for i in range(len(arrays)):
            x = torch.tensor(np.expand_dims(arrays[i],0)).type(torch.float32).to(self.device)
            x = torch.unsqueeze(x,0)

            encFeats_cp, cp_pred = self.unet_cp(x)
            encFeats_cp_lowest = encFeats_cp[2][0]
            encFeats_cps_list.append(encFeats_cp[2].detach().numpy())
            encFeats_cp_flattened = torch.mean(encFeats_cp_lowest, axis=0)
            encFeats_cps.append(encFeats_cp_flattened)

            inter_preds.append(cp_pred.cpu().detach().numpy())
            
            encFeats_mask, mask_pred = self.unet_mask(cp_pred.to(self.device))
            encFeats_mask_lowest = encFeats_mask[2][0]
            encFeats_mask_flattened = torch.mean(encFeats_mask_lowest, axis=0)
            encFeats_masks.append(encFeats_mask_flattened)

            #mask_pred = torch.sigmoid(mask_pred)
            mask_tresh = np.where(np.squeeze(mask_pred.cpu().detach().numpy())>0.5,1,0)
            masks_crops.append(mask_tresh)
        
        cp = self.stack_img(inter_preds)
        cp = cp[pad_val:-pad_val, pad_val:-pad_val]
        

        mask = self.stack_img(masks_crops)
        mask = mask[pad_val:-pad_val, pad_val:-pad_val]
        
        instance_mask = self.instance_seg(mask)

        #encFeats = self.stack_img(encFeats_cp_flattened.detach().numpy())

        if encFeats:
            return cp, mask, instance_mask, encFeats_cps, encFeats_masks
        elif encFeats_list:
            return cp, mask, instance_mask, encFeats_cps_list, encFeats_masks
        return cp, mask, instance_mask

    def eval(self,images,channel=0,encFeats=False,encFeats_list=False):
        #TODO check if images have multiple channels, and remove them
        instance_masks = []
        masks = []
        cps = []
        encFeats_cps = []
        encFeats_masks = []
        if images.shape[0] == images.shape[1]:
            images = np.expand_dims(images,0)
        for x in images:
            if encFeats or encFeats_list:
                cp, mask, instance_mask, encFeats_cp, encFeats_mask = self.get_pred(x,channel,encFeats=encFeats,encFeats_list=encFeats_list)
                print('iofhef')
                print('xxx',len(encFeats_cp))
                encFeats_cps.append(encFeats_cp)
                encFeats_masks.append(encFeats_mask)
            else:
                print('elseifhof')
                cp, mask, instance_mask = self.get_pred(x,channel,encFeats=encFeats)
            cps.append(cp)
            masks.append(mask)
            instance_masks.append(instance_mask)
        
        if encFeats or encFeats_list:
            return cps, masks, instance_masks, encFeats_cps, encFeats_masks
        return cps, masks, instance_masks
    
    def dice_evaluate(self):
        self.unet_cp.eval()
        self.unet_mask.eval()
        dice_coeffs = []
        for ((x_img,y_img),(x_cp,y_cp)) in zip(self.testLoader_img,self.testLoader_cp):
            (x_img,y_img) = (x_img.type(torch.float32).to(self.device), y_img.type(torch.float32).to(self.device))
            encFeats, cp_pred = self.unet_cp(x_img)

            (x_cp,y_cp) = (x_cp.type(torch.float32).to(self.device), y_cp.type(torch.float32).to(self.device))
            encFeats, mask_pred = self.unet_mask(cp_pred.to(self.device))
            mask_pred = torch.sigmoid(mask_pred)
            mask_tresh = np.where(np.squeeze(mask_pred.cpu().detach().numpy())>0.5,1,0)
            
            dice = self.dice_coeff(mask_tresh,np.squeeze(y_cp.cpu().detach().numpy()))
            dice_coeffs.append(dice)

        return mean(dice_coeffs)
    
    def expand_div_256(self, img):
        expand_to = img.shape[0]
        for i in range(img.shape[0],img.shape[0]+10000):
            if i % 256 == 0:
                expand_to = i
                break

        return np.pad(img, pad_width=int((expand_to-img.shape[0])/2),mode='constant', constant_values=0), int((expand_to-img.shape[0])/2), 

    def blockshaped(self, arr, nrows, ncols):
        """
        Return an array of shape (n, nrows, ncols) where
        n * nrows * ncols = arr.size
        If arr is a 2D array, the returned array should look like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
        h, w = arr.shape
        assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
        assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
        return (arr.reshape(h//nrows, nrows, -1, ncols)
                .swapaxes(1,2)
                .reshape(-1, nrows, ncols))
    
    def stack_img(self, arrays, colrow=256):
        it = int(math.sqrt(len(arrays)))
        preds = np.array(arrays)
        preds_r = np.reshape(preds,(it,it,colrow,colrow))
        stacks = []
        for i in range(len(preds_r)):
            stacked = preds_r[i,0]
            for j in range(1,len(preds_r)):
                stacked = np.hstack((stacked,preds_r[i,j]))
            stacks.append(stacked)
        stacks = np.array(stacks)
        stacked = stacks[0]
        for i in range(1,len(preds_r)):
            stacked = np.vstack((stacked,stacks[i]))

        return stacked

    def dice_coeff(self,array1,array2):
        im1 = np.asarray(array1).astype(np.bool)
        im2 = np.asarray(array2).astype(np.bool)

        print('im1 shape:',im1.shape)
        print('im2 shape:',im2.shape)

        if im1.shape != im2.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

        intersection = np.logical_and(im1, im2)

        return 2. * intersection.sum() / (im1.sum() + im2.sum())

    def instance_seg(self,binary_mask):
        return measure.label(binary_mask, background=0,connectivity=1)