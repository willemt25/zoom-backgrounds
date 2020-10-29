import numpy as np
import imageio
import matplotlib.pyplot as plt
import cv2
from matplotlib.widgets import Cursor
import sys

coords = np.empty((0, 2), int)

def get_obj_coords(img):

    global coords

    fig, axes = plt.subplots(1, 1, sharey=True)
    axes.axis("off")
    axes.imshow(img)
    axes.set_title("Select the object you would like to remove.")

    cid = fig.canvas.mpl_connect('button_press_event', handle_click)

    cursor = Cursor(axes, useblit=True, color='red', linewidth=2)

    plt.show()

    return coords


def handle_click(event):
    global coords

    coords = np.append(coords, np.array([[int(event.xdata), int(event.ydata)]]), axis=0)
    plt.scatter([int(event.xdata)], [int(event.ydata)], c='white', s=5)

def preprocess_image(original_img, img):
    selection_img = np.zeros_like(original_img)
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j] == 0:
                selection_img[i][j] = [255, 255, 255]
            else:
                selection_img[i][j] = original_img[i][j]

    return selection_img

def connected_comp(img):
    num_labels, labels = cv2.connectedComponents(img)

    # Map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    return labeled_img

def select_object(labeled_img):
    ret = np.zeros((labeled_img.shape[0:2]))

    for pixel in input_pixels:
        y = pixel[0]
        x = pixel[1]
        group_to_remove = labeled_img[x][y]
        print(group_to_remove)
        for a in range(len(labeled_img)):
            for b in range(len(labeled_img[0])):
                if np.array_equal(labeled_img[a][b],group_to_remove):
                    ret[a][b] = 1

    return ret


if __name__ == '__main__':

    img_name = sys.argv[1]

    original_img = imageio.imread(img_name)

    img = cv2.threshold(cv2.imread(img_name, 0), 160, 255, cv2.THRESH_BINARY_INV)[1]

    selection_img = preprocess_image(original_img, img)

    labeled_img = connected_comp(img)

    input_pixels = get_obj_coords(selection_img)

    selected_object = select_object(labeled_img)

    np.save(img_name, selected_object)
    
    #this portion is just for displaying the array of 1's and 0's
    # display_img = np.zeros_like(labeled_img)
    # for x in range(len(selected_object)):
    #     for y in range(len(selected_object[0])):
    #         if selected_object[x][y] == 1:
    #             display_img[x][y] = [255,0,0]
    #         else:
    #             display_img[x][y] = [0,0,0]
    # plt.imshow(display_img, cmap='gray')
    # plt.axis('off')
    # plt.title("Returned Binary Image")
    # plt.show()
