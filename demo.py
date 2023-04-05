import os
from cellmask_model import CellMaskModel
import matplotlib.pyplot as plt
from data import import_images
import numpy as np

if __name__ == '__main__':
    
    # Importing images
    images_path = os.getcwd() + '/images/'
    images = import_images(images_path,normalisation=True,num_imgs=3)
    
    # Getting model
    model = CellMaskModel()
    model.import_model(os.getcwd() + '/saved_weights/cp_model', os.getcwd() + '/saved_weights/mask_model')

    instance_masks, masks, cps = model.eval(images) #Making predictions

    #Plotting predictions

    for i in range(len(instance_masks)):
        plt.subplot(3,len(instance_masks),i+1)
        plt.axis('off')
        #if images[i][0] == 3:
            #change the color channel to the end
        #    images[i] = np.moveaxis(images[i], 0, -1)
        #if images[i].shape[2] != 1:
        #    images[i] = images[i][:,:,0]
        #else:
        plt.imshow(images[i])

    for i in range(len(instance_masks)):
        plt.subplot(3,len(instance_masks),i+1+len(instance_masks))
        plt.axis('off')
        plt.imshow(cps[i])

    for i in range(len(instance_masks)):
        plt.subplot(3,len(instance_masks),i+1+len(instance_masks)+len(instance_masks))
        plt.axis('off')
        plt.imshow(instance_masks[i])

    plt.subplots_adjust(wspace=0, hspace=0.033)
    plt.show()