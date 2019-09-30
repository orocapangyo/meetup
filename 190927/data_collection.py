# IPython Libraries for display and widgets
import traitlets
import ipywidgets.widgets as widgets
from IPython.display import display

# Camera and Motor Interface for JetBot
from jetbot import Robot, Camera, bgr8_to_jpeg

# Python basic pakcages for image annotation
from uuid import uuid1
import os
import json
import glob
import datetime
import numpy as np
import cv2
import time

###########################################################################
camera = Camera()

image_widget = widgets.Image(format='jpeg', width=224, height=224)
target_widget = widgets.Image(format='jpeg', width=224, height=224)

x_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.001, description='x')
y_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.001, description='y')

def display_xy(camera_image):
    image = np.copy(camera_image)
    x = x_slider.value
    y = y_slider.value
    x = int(x * 224 / 2 + 112)
    y = int(y * 224 / 2 + 112)
    image = cv2.circle(image, (x, y), 8, (0, 255, 0), 3)
    image = cv2.circle(image, (112, 224), 8, (0, 0,255), 3)
    image = cv2.line(image, (x,y), (112,224), (255,0,0), 3)
    jpeg_image = bgr8_to_jpeg(image)
    return jpeg_image

time.sleep(1)
traitlets.dlink((camera, 'value'), (image_widget, 'value'), transform=bgr8_to_jpeg)
traitlets.dlink((camera, 'value'), (target_widget, 'value'), transform=display_xy)

display(widgets.HBox([image_widget, target_widget]), x_slider, y_slider)
###############################################################################

###############################################################################
controller = widgets.Controller(index=0)
display(controller)
###############################################################################

### Connect Gamepad Controller to Label Images
widgets.jsdlink((controller.axes[0], 'value'), (x_slider, 'value'))
widgets.jsdlink((controller.axes[1], 'value'), (y_slider, 'value'))

###############################################################################
### Collect Data
DATASET_DIR = 'dataset_xy'
# we have this "try/except" statement because these next functions can throw an error if the directories exist already
try:
    os.makedirs(DATASET_DIR)
except FileExistsError:
    print('Directories not created becasue they already exist')

for b in controller.buttons:
    b.unobserve_all()

count_widget = widgets.IntText(description='count', value=len(glob.glob(os.path.join(DATASET_DIR, '*.jpg'))))

def xy_uuid(x, y):
    return 'xy_%03d_%03d_%s' % (x * 50 + 50, y * 50 + 50, uuid1())

def save_snapshot(change):
    if change['new']:
        uuid = xy_uuid(x_slider.value, y_slider.value)
        image_path = os.path.join(DATASET_DIR, uuid + '.jpg')
        with open(image_path, 'wb') as f:
            f.write(image_widget.value)
        count_widget.value = len(glob.glob(os.path.join(DATASET_DIR, '*.jpg')))

controller.buttons[13].observe(save_snapshot, names='value')

display(widgets.VBox([
    target_widget,
    count_widget
]))

def timestr():
    return str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
###############################################################################

!zip -r -q road_following_{DATASET_DIR}_{timestr()}.zip {DATASET_DIR}