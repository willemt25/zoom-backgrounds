import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
from scipy import ndimage, signal
from imageio import imread, imsave, imwrite
from skimage.color import rgb2gray

def compute_gradients(img):
    gx = cv2.Scharr(img, -1, 1, 0)
    gy = cv2.Scharr(img,-1, 0, 1)

    return gx, gy

def energy_image(img, mask, useMask):
    energyImage = np.zeros_like(img)

    img = rgb2gray(img)
    gx, gy = compute_gradients(img)
    energyImage = abs(gx) + abs(gy)

    energyImage = energyImage / energyImage.max()

    if useMask :
      energyImage += mask

    return energyImage

def cumulative_minimum_energy_map(energyImage):
    cumulativeEnergyMap = np.zeros_like(energyImage)

    for row in range(len(energyImage)):
        cumulativeEnergyMap[row] = energyImage[row] + ndimage.minimum_filter(cumulativeEnergyMap, footprint=[[1, 1, 1], [0, 0, 0], [0, 0, 0]])[row]

    return cumulativeEnergyMap

def find_optimal_vertical_seam(cumulativeEnergyMap):

    verticalSeam = [0]*cumulativeEnergyMap.shape[0]

    current_column = -1
    curr_min = float('inf')

    for i in reversed(range(cumulativeEnergyMap.shape[0])):
      if i == (cumulativeEnergyMap.shape[0] - 1):
          for j in range(cumulativeEnergyMap.shape[1]):
              if cumulativeEnergyMap[i][j] < curr_min:
                  curr_min = cumulativeEnergyMap[i][j]
                  current_column = j
          verticalSeam[i] = current_column
      else:
          j = current_column
          curr_min = cumulativeEnergyMap[i][j]

          if j > 0 and cumulativeEnergyMap[i][j - 1] <= curr_min:
              curr_min = cumulativeEnergyMap[i][j - 1]
              current_column = j - 1

          if j < (cumulativeEnergyMap.shape[1] - 1) and cumulativeEnergyMap[i][j + 1] < curr_min:
              curr_min = cumulativeEnergyMap[i][j + 1]
              current_column = j + 1

          verticalSeam[i] = current_column

    return verticalSeam


def reduce_width(img, mask, energyImage):
    reducedEnergyImageSize = (energyImage.shape[0], energyImage.shape[1] - 1)
    reducedColorImageSize = (img.shape[0], img.shape[1] - 1, 3)
    reducedMaskSize = (mask.shape[0], mask.shape[1] - 1)

    reducedColorImage = np.zeros(reducedColorImageSize)
    reducedEnergyImage = np.zeros(reducedEnergyImageSize)
    reducedMask = np.zeros(reducedMaskSize)

    cumulativeMinimalEnergyMap = cumulative_minimum_energy_map(energyImage)

    seam = find_optimal_vertical_seam(cumulativeMinimalEnergyMap)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (j < seam[i]):
                reducedColorImage[i][j] = img[i][j]
                reducedMask[i][j] = mask[i][j]
                reducedEnergyImage[i][j] = energyImage[i][j]
            if (j > seam[i]):
                reducedColorImage[i][j - 1] = img[i][j]
                reducedMask[i][j - 1] = mask[i][j]
                reducedEnergyImage[i][j - 1] = energyImage[i][j]

    return reducedColorImage.astype(np.uint8), reducedEnergyImage, reducedMask

def increase_width(img, energyImage):
    increasedEnergyImageSize = (energyImage.shape[0], energyImage.shape[1] + 1)
    increasedColorImageSize = (img.shape[0], img.shape[1] + 1, 3)

    increasedColorImage = np.zeros(increasedColorImageSize)
    increasedEnergyImage = np.zeros(increasedEnergyImageSize)

    cumulativeMinimalEnergyMap = cumulative_minimum_energy_map(energyImage)

    seam = find_optimal_vertical_seam(cumulativeMinimalEnergyMap)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (j < seam[i]):
                increasedColorImage[i][j] = img[i][j]
                increasedEnergyImage[i][j] = energyImage[i][j]
            if (j == seam[i]):
                increasedColorImage[i][j] = img[i][j]
                increasedEnergyImage[i][j] = energyImage[i][j]
            if (j >= seam[i]):
                increasedColorImage[i][j + 1] = img[i][j]
                increasedEnergyImage[i][j + 1] = energyImage[i][j]

    return increasedColorImage.astype(np.uint8), increasedEnergyImage

def get_object_dimensions(image_mask):
    rows,cols = np.where(image_mask < 0)
    object_height = np.amax(rows) - np.amin(rows) + 1
    object_width = np.amax(cols) - np.amin(cols) + 1
    return object_height, object_width

def swapaxes(image, mask):
    image = np.swapaxes(image, 0, 1)
    mask = np.swapaxes(mask, 0, 1)

    return image, mask

if __name__ == '__main__':
  img_name = sys.argv[1]

  # 1 = background, -10000 = object, 10000 = foreground
  # read in mask
  mask = np.load('results/' + img_name + '.npy')

  imwrite('finalResults/masks/' + img_name + '.mask.jpeg', mask)
  # plt.imshow(mask)
  # plt.show()

  # read in image
  image = imread('data/' + img_name)

  # get object dimensions
  obj_height, obj_width = get_object_dimensions(mask)

  # transpose mask and image if the object height < width; vertical seams
  is_transposed = False
  if obj_height < obj_width:
    image, mask = swapaxes(image, mask)
    is_transposed = True

  # get original image shape
  orig_shape = image.shape

  # compute energy image accounting for mask
  nrg_img = energy_image(image, mask, True)

  # until object is gone
  while (len(np.where(mask < 0)[0]) > 0):
    # delete minimal seam from image, energy image, and mask
    image, nrg_img, mask = reduce_width(image, mask, nrg_img)

  imwrite('finalResults/reducedImages/' + img_name + '.reduced.jpeg', image)
  plt.imshow(image)
  plt.show()

  # compute energy image without mask
  nrg_img = energy_image(image, mask, False)

  # until image reaches original dimensions
  while (image.shape[0] != orig_shape[0] or image.shape[1] != orig_shape[1]):
    # add minimal seam back to image and energy image
    image, nrg_img = increase_width(image, nrg_img)

  # transpose image if it was transposed in the beginning
  if is_transposed:
    image, mask = swapaxes(image, mask)

  plt.imshow(image)
  plt.show()

  imwrite('finalResults/finalImages/' + img_name + '.result.jpeg', image)
  # imwrite('results/' + img_name + '.result.jpeg', image)
