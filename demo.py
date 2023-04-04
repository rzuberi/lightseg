import os
from cellmask_model import CellMaskModel
import matplotlib.pyplot as plt
from data import import_images
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QSlider, QCheckBox, QGridLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image, ImageQt
import sys
import numpy as np
import cv2

class ImageViewer(QWidget):
    def __init__(self, images, instance_masks):
        super().__init__()

        self.images = images
        self.instance_masks = instance_masks
        
        # Create the slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(images)-1)
        self.slider.valueChanged.connect(self.on_slider_changed)

        # Create the checkbox
        self.checkbox = QCheckBox('display instance masks', self)
        self.checkbox.stateChanged.connect(self.on_checkbox_changed)
        
        # Create the image label
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(1080, 1080)


        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.slider)
        layout.addWidget(self.checkbox)
        layout.addWidget(self.image_label)

         # Add image boxes
        self.num_images = len(images)
        self.image_boxes = []
        if self.num_images <= 5:
            for i in range(self.num_images):
                image_box = QLabel(self)
                image_box.setFixedSize(216, 216)
                image_box.setAlignment(Qt.AlignCenter)
                self.image_boxes.append(image_box)
                layout.addWidget(image_box)
        else:
            self.slider2 = QSlider(Qt.Horizontal)
            self.slider2.setMinimum(0)
            self.slider2.setMaximum(self.num_images - 5)
            self.slider2.valueChanged.connect(self.on_slider2_changed)
            layout.addWidget(self.slider2)
            for i in range(5):
                image_box = QLabel(self)
                image_box.setFixedSize(216, 216)
                image_box.setAlignment(Qt.AlignCenter)
                self.image_boxes.append(image_box)
                layout.addWidget(image_box)
        
        # Mouse tracking
        self.setMouseTracking(True)

        self.image_x = -1
        self.image_y = -1
        
        # Add the image boxes to the layout
        hbox = QHBoxLayout()
        for box in self.image_boxes:
            hbox.addWidget(box)
        layout.addLayout(hbox)
        
        self.setLayout(layout)
        
        
        # Initialize the image
        self.use_alternate_images = False
        self.update_image(0)

        num_boxes = len(self.images) if len(self.images) <= 5 else 5
        for i in range(num_boxes):
            box = QLabel(self)
            box.setFixedSize(200, 200)
            self.image_boxes.append(box)
        layout.addWidget(QLabel("Image boxes:"))
        box_layout = QHBoxLayout()
        for i in range(num_boxes):
            box_layout.addWidget(self.image_boxes[i])
        layout.addLayout(box_layout)
    
    def on_slider_changed(self, value):
        self.update_image(value)
    
    def on_slider2_changed(self, value):
        print('wwwwww', self.image_x)
        if self.image_x != -1:
            print('ahrahaofahfoafoad')
            self.update_image_box()
    
    def on_checkbox_changed(self, state):
        self.use_alternate_images = (state == Qt.Checked)
        self.update_image(self.slider.value())

    def update_image_box(self):
        # Get the value of the pixel at the cursor position

        image_x = self.image_x
        image_y = self.image_y

        pixmap = self.image_label.pixmap()
        if pixmap:
            qimage = pixmap.toImage()

            # Get the cropped image around the pixel
            images = self.images
            instance_masks = self.instance_masks

            # Set the tooltip text to display the pixel value
            pixel_value  = instance_masks[self.slider.value()][image_y][image_x]

            #take the pixel_value and the slider value
            #look through the list of matches at the index of the slider value if we find the pixel value
            #if we do, then we take that list as the list of matches that will be displayed
            #we then slice the list of matches between the slider value and the slider value + 5

            self.image_label.setToolTip(f"Pixel value: {pixel_value}")

            if pixel_value > 0:

                
                images_to_use = []
                for i in range(self.slider2.value(),self.slider2.value()+5):
                    if i < len(images):
                        images_to_use.append(images[i])
                    else:
                        #random array of zeros
                        rand_arr = np.zeros((images[0].shape[0], images[0].shape[1]))
                        images_to_use.append(rand_arr)

                instance_masks_to_use = []
                for i in range(self.slider2.value(),self.slider2.value()+5):
                    if i < len(instance_masks):
                        instance_masks_to_use.append(instance_masks[i])
                    else:
                        instance_masks_to_use.append(instance_masks[i - len(instance_masks)])

                for i, image in enumerate(images_to_use):
                    #pixel value her will have to change to the pixel value in teh list of matches
                    #so probably mrore something like center_x, center_y = np.where(instance_masks_to_use[i] == list_of_matches_sliced[i])
                    center_x, center_y = np.where(instance_masks_to_use[i] == pixel_value)
                    #center_x, center_y = image_x, image_y

                    #print('center_x',center_x,'center_y',center_y)
                    if len(center_x) > 0:
                        center_x = int(center_x.mean())
                        center_y = int(center_y.mean())
                        crop_size = 30
                        x1 = max(0, center_x - crop_size // 2)
                        y1 = max(0, center_y - crop_size // 2)
                        x2 = min(image.shape[0], center_x + crop_size // 2)
                        y2 = min(image.shape[1], center_y + crop_size // 2)
                        cropped_image = image[x1:x2, y1:y2]
                        #cropped_image = np.resize(cropped_image,(200,200))
                        cropped_image = cv2.resize(cropped_image, dsize=(216, 216), interpolation=cv2.INTER_CUBIC)
                        q_image = QImage(cropped_image.data.tobytes(), cropped_image.shape[0], cropped_image.shape[1], QImage.Format_Grayscale16)
                        # Convert the QImage to a QPixmap
                        pixmap = QPixmap.fromImage(q_image)

                        self.image_boxes[i].setPixmap(pixmap)

    def mousePressEvent(self, event):
        # Get the position of the cursor in the label widget
        pos = event.pos()
        x = pos.x()
        y = pos.y()

        print('slider value',self.slider.value())

        label_pos = self.image_label.pos()
        label_x = label_pos.x()
        label_y = label_pos.y()

        # Calculate the x and y position on the image
        self.image_x = x - label_x
        self.image_y = y - label_y

        
        if self.image_x > 0 and self.image_y > 0 and self.image_x < 1080 and self.image_y < 1080:
            self.update_image_box()
    
    def update_image(self, index):

        if self.use_alternate_images:
            # Get the image from the alternate images
            instance_masks = self.instance_masks

            scaled_image = (255 * instance_masks[index] / 150).astype(np.uint8)
            q_image = QImage(scaled_image.data, 1080, 1080, QImage.Format_Grayscale8)
            # Convert the QImage to a QPixmap
            pixmap = QPixmap.fromImage(q_image)

        else:
            # Get the image from the original images
            images = self.images

            q_image = QImage(images[index].data, 1080, 1080, QImage.Format_Grayscale16)
            # Convert the QImage to a QPixmap
            pixmap = QPixmap.fromImage(q_image)
        
        # Set the pixmap on the image label
        self.image_label.setPixmap(pixmap)

if __name__ == '__main__':
    print('hey')
    # Importing images
    images_path = os.getcwd() + '\\data\\'
    images_for_display = import_images(images_path,normalisation=False,num_imgs=10)
    print('hey2')
    # Getting model
    model = CellMaskModel()
    images_for_model = import_images(images_path,normalisation=True,num_imgs=10)
    model.import_model(os.getcwd() + '/saved_weights/cp_model', os.getcwd() + '/saved_weights/mask_model')
    instance_masks, masks, cps = model.eval(images_for_model) #Making predictions
    print('hey2')
    print(np.unique(instance_masks[0]))

    app = QApplication(sys.argv)
    w = ImageViewer(images_for_display, instance_masks)
    w.show()
    sys.exit(app.exec_())
    