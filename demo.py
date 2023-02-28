import os
from cellmask_model import CellMaskModel
import matplotlib.pyplot as plt
from data import import_images

if __name__ == '__main__':
    
    # Importing images
    images_path = os.getcwd() + '/images/'
    images = import_images(images_path,normalisation=True,num_imgs=5)
    #imgs = [(image-np.min(image))/(np.max(image)-np.min(image)) for image in images]

    # Getting model
    model = CellMaskModel()
    model.import_model(os.getcwd() + '/saved_weights/cp_model', os.getcwd() + '/saved_weights/mask_model')

    instance_masks, masks, cps = model.eval(images) #Making predictions

    #Plotting predictions
    for i in range(0,len(instance_masks)*3,3):
        plt.subplot(len(instance_masks),3,i+1)
        plt.axis('off')
        plt.imshow(images[int(i/3)][:,:,0])

        plt.subplot(len(instance_masks),3,i+2)
        plt.axis('off')
        plt.imshow(cps[int(i/3)])

        plt.subplot(len(instance_masks),3,i+3)
        plt.axis('off')
        plt.imshow(instance_masks[int(i/3)])
    plt.show()