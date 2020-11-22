import numpy as np
import imageio
import matplotlib.pyplot as plt
import cv2
from matplotlib.widgets import Cursor
import sys
import pathlib

coords = np.empty((0, 2), int)
background_coords = np.empty((0, 2), int)

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

def get_background_coords(img):

    global background_coords

    fig, axes = plt.subplots(1, 1, sharey=True)
    axes.axis("off")
    axes.imshow(img)
    axes.set_title("Select the background closest to the object.")

    cid = fig.canvas.mpl_connect('button_press_event', handle_click_background)

    cursor = Cursor(axes, useblit=True, color='red', linewidth=2)

    plt.show()

    return background_coords


def handle_click_background(event):
    global background_coords

    background_coords = np.append(background_coords, np.array([[int(event.xdata), int(event.ydata)]]), axis=0)
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

def select_object(labeled_img, neighborhood_size):
    ret = np.zeros((labeled_img.shape[0:2]))

    background = background_pixels[0]
    background_pixel = labeled_img[background[1]][background[0]]

    for a in range(len(labeled_img)):
        for b in range(len(labeled_img[0])):
            curr = labeled_img[a][b]

            matched = False

            for pixel in input_pixels:
                y = pixel[0]
                x = pixel[1]
                group_to_remove = labeled_img[x][y]

                if np.array_equal(curr, group_to_remove):
                    ret[a][b] = -100000
                    matched = True
                    break

    modified = []
    buffer_size = int((neighborhood_size - 1) / 2)
    for a in range(len(ret)):
        for b in range(len(ret[0])):
            if ret[a][b] == -100000 and (a,b) not in modified:
                if a+1 < len(ret) and ret[a+1][b] == 0:
                    for i in range(1,buffer_size):
                        if a+i < len(ret):
                            ret[a+i][b] = -100000
                            modified.append((a+i,b))
                if b+1 < len(ret[0]) and ret[a][b+1] == 0:
                    for i in range(1,buffer_size):
                        if b+i < len(ret[0]):
                            ret[a][b+i] = -100000
                            modified.append((a,b+i))
                if a-1 > 0 and ret[a-1][b] == 0:
                    for i in range(1,buffer_size):
                        if a-i > 0:
                            ret[a-i][b] = -100000
                            modified.append((a-i,b))
                if b-1 > 0 and ret[a][b-1] == 0:
                    for i in range(1,buffer_size):
                        if b-i > 0:
                            ret[a][b-i] = -100000
                            modified.append((a,b-i))

    return ret


if __name__ == '__main__':
    img_name = sys.argv[1]
    neighborhood_size = int(sys.argv[2])
    constant = float(sys.argv[3])
    print(pathlib.Path().absolute())

    original_img = imageio.imread('data/' + img_name)

    img = cv2.adaptiveThreshold(cv2.imread('data/' + img_name, 0) ,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,neighborhood_size,constant)

    selection_img = preprocess_image(original_img, img)

    # imageio.imwrite('results/' + img_name + '.selection.jpeg', selection_img)

    labeled_img = connected_comp(img)

    plt.imshow(labeled_img)
    plt.show()

    # imageio.imwrite('results/' + img_name + '.labeled.jpeg', labeled_img)

    input_pixels = get_obj_coords(selection_img)

    background_pixels = get_background_coords(original_img)

    selected_object = select_object(labeled_img, neighborhood_size)

    plt.imshow(selected_object)
    plt.show()

    np.save('results/' + img_name, selected_object)

    # #this portion is just for displaying the array of 1's and 0's
    # display_img = np.zeros_like(labeled_img)
    # for x in range(len(selected_object)):
    #     for y in range(len(selected_object[0])):
    #         if selected_object[x][y] == -100000:
    #             display_img[x][y] = [255,0,0]
    #         elif selected_object[x][y] == 100000:
    #             display_img[x][y] = [0,0,255]
    #         else:
    #             display_img[x][y] = [0,255,0]
    # plt.imshow(display_img, cmap='gray')
    # plt.axis('off')
    # plt.title("Returned Binary Image")
    # plt.show()

    # imageio.imwrite('results/' + img_name + '.naive.jpeg', display_img)

    imageio.imwrite('finalResults/selectionImages/' + img_name + '.selection.jpeg', selection_img)
    imageio.imwrite('finalResults/labeledImages/' + img_name + '.labeled.jpeg', labeled_img)

