import numpy as np
from data import import_images
from cellmask_model import CellMaskModel
import os
import matplotlib.pyplot as plt
import torch
import cv2
import time

if __name__ == '__main__':
    #import the cell_centers2.txt file without numpy
    #with open('cellmask/cell_centers2.txt') as f:
    #    cell_centers = [np.array(eval(line)).tolist() for line in f]
    #cell_centers = [[(item[0], item[1]) for item in arr] for arr in cell_centers]

    #import the images#
    images = np.array(import_images('images/',num_imgs=3,normalisation=True))

    model = CellMaskModel()
    model.import_model(os.getcwd() + '/saved_weights/cp_model', os.getcwd() + '/saved_weights/mask_model')
    instance_masks, masks, cps = model.eval(images) #Making predictions

    def get_encodings(images, model):
        #get the encoder of the model
        encoder = model.unet_cp.encoder
        #torch.from_numpy(encFeats[0]).expand(0).expand(0
        encFeats = encoder(torch.from_numpy(images).unsqueeze(1).type(torch.float32))[2].detach().numpy()

       
        encFeats_interpolated = []
        for i in range(len(encFeats)):
            encFeats_per_channel = []
            for j in range(len(encFeats[i])):
                encFeats_per_channel.append(cv2.resize(encFeats[i][j], (1080,1080), interpolation=cv2.INTER_CUBIC))
            encFeats_interpolated.append(encFeats_per_channel)
        encFeats = np.array(encFeats_interpolated)
        return encFeats  
   
    def get_encFeats_per_cell_per_mask(instance_masks, encFeats):
        encFeats_per_cell_per_mask = []
        for i in range(len(instance_masks)):
            instance_mask = instance_masks[i]
            encoding_features = encFeats[i]
            encoding_features_per_cell_2 = []
            for mask_val in range(1, np.max(instance_mask)+1):
                mask = instance_mask == mask_val
                masked_encFeats = encoding_features[:, mask]
                encoding_features_per_cell_2.append(masked_encFeats)
            encFeats_per_cell_per_mask.append(encoding_features_per_cell_2)#
    
        encFeats_per_cell_per_mask = np.array(encFeats_per_cell_per_mask)
        return encFeats_per_cell_per_mask
   
    def get_pairs(encFeats_per_cell_per_mask):
        cos_sims_for_each_cell = []
        for cell_encFeats in encFeats_per_cell_per_mask[0]:
            cos_sims_for_cell = []
            arr1 = cell_encFeats.flatten()
            for cell_encFeats2 in encFeats_per_cell_per_mask[1]:
                arr2 = cell_encFeats2.flatten()
                if arr1.shape[0] > arr2.shape[0]:
                    arr2 = np.pad(arr2, (0, arr1.shape[0] - arr2.shape[0]), 'constant')
                elif arr1.shape[0] < arr2.shape[0]:
                    arr1 = np.pad(arr1, (0, arr2.shape[0] - arr1.shape[0]), 'constant')
                cos_sim = np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))
                cos_sims_for_cell.append(cos_sim)
            cos_sims_for_each_cell.append(cos_sims_for_cell)

        cos_sims_for_each_cell = np.array(cos_sims_for_each_cell)

        lst = cos_sims_for_each_cell.tolist()

        pairs = []
        length = min(len(lst),len(lst[0]))
        for i in range(length):
            lst = np.array(lst)
            max_index = np.argmax(lst)
            row, col = np.unravel_index(max_index, lst.shape)
            pairs.append((row,col))
            lst[:,col] = 0
            lst[row,:] = 0
        return pairs

    def merge(arr,arr1):
        result = []
        for i in arr:
            for j in arr1:
                if i[-1] == j[0]:
                    res = [num for num in i]
                    res.append(j[1])
                    result.append(res)
        result = np.array(result)
        return result

    def merge_list(list_of_pairs):
        matches = list_of_pairs[0]
        for i in range(len(list_of_pairs)-1):
            matches = merge(matches,list_of_pairs[i+1])
        matches = np.array(matches)
        return np.squeeze(matches)

    start = time.time()
    encFeats_per_cell_per_mask = get_encFeats_per_cell_per_mask(instance_masks, get_encodings(images, model))
    print('time taken for encFeats_per_cell_per_mask: ', time.time()-start)
    print(len(encFeats_per_cell_per_mask))
    #start = time.time()
    #pairs_1 = get_pairs([encFeats_per_cell_per_mask[0],encFeats_per_cell_per_mask[1]])
    #pairs_2 = get_pairs([encFeats_per_cell_per_mask[1],encFeats_per_cell_per_mask[2]])
    #print('time taken to get_pairs: ', time.time()-start)
    all_pairs = []
    for i in range(len(encFeats_per_cell_per_mask)-1):
        start = time.time()
        pairs = get_pairs([encFeats_per_cell_per_mask[i],encFeats_per_cell_per_mask[i+1]])
        print('time taken to get_pairs: ', time.time()-start)

    #print(merge_list([pairs_1,pairs_2]))
    #print('time taken: ', time.time()-start)