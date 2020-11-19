import numpy as np
from imageio import imread, imwrite
from matplotlib import pyplot as plt
from PIL import Image as im
import sys

class SeamCarve:
    max_energy = 1000000.0

    def __init__(self, img):
        self.array = img.astype(int)
        self.height, self.width = img.shape[:2]
        self.energy_array = np.empty((self.height, self.width))
        self.compute_energy_array()

    def compute_energy(self, i, j):
        if (i == 0 or i == self.height - 1) or (j == 0 or j == self.width - 1):
            return self.max_energy
        b = abs(self.array[i - 1, j, 0] - self.array[i + 1, j, 0])
        g = abs(self.array[i - 1, j, 1] - self.array[i + 1, j, 1])
        r = abs(self.array[i - 1, j, 2] - self.array[i + 1, j, 2])

        b += abs(self.array[i, j - 1, 0] - self.array[i, j + 1, 0])
        g += abs(self.array[i, j - 1, 1] - self.array[i, j + 1, 1])
        r += abs(self.array[i, j - 1, 2] - self.array[i, j + 1, 2])
        energy = b + g + r
        return energy

    def swapaxes(self):
        self.energy_array = np.swapaxes(self.energy_array, 0, 1)
        self.array = np.swapaxes(self.array, 0, 1)
        self.height, self.width = self.width, self.height

    def compute_energy_array(self):
        self.energy_array[[0, -1], :] = self.max_energy
        self.energy_array[:, [0, -1]] = self.max_energy
        self.energy_array[1:-1, 1:-1] = np.add.reduce(np.abs(self.array[:-2, 1:-1] - self.array[2:, 1:-1]), -1)
        self.energy_array[1:-1, 1:-1] += np.add.reduce(np.abs(self.array[1:-1, :-2] - self.array[1:-1, 2:]), -1)

    def compute_seam(self, horizontal=False):
        if horizontal:
            self.swapaxes()
        energy_sum_array = np.empty_like(self.energy_array)
        energy_sum_array[0] = self.energy_array[0]
        for i in range(1, self.height):
            energy_sum_array[i, :-1] = np.minimum(energy_sum_array[i - 1, :-1], energy_sum_array[i - 1, 1:])
            energy_sum_array[i, 1:] = np.minimum(energy_sum_array[i, :-1], energy_sum_array[i - 1, 1:])
            energy_sum_array[i] += self.energy_array[i]
        seam = np.empty(self.height, dtype=int)
        seam[-1] = np.argmin(energy_sum_array[-1, :])
        seam_energy = energy_sum_array[-1, seam[-1]]
        for i in range(self.height - 2, -1, -1):
            l, r = max(0, seam[i + 1] - 1), min(seam[i + 1] + 2, self.width)
            seam[i] = l + np.argmin(energy_sum_array[i, l: r])
        if horizontal:
            self.swapaxes()
        return seam_energy, seam

    def carve(self, horizontal=False, seam=None, remove=True):
        if horizontal:
            self.swapaxes()
        if seam is None:
            seam = self.compute_seam()[1]
        if remove:
            self.width -= 1
        else:
            self.width += 1
        new_array = np.empty((self.height, self.width, 3))
        new_energy_array = np.empty((self.height, self.width))
        mp_deleted_count = 0
        for i, j in enumerate(seam):
            if remove:
                if self.energy_array[i, j] < 0:
                    mp_deleted_count += 1
                new_energy_array[i] = np.delete(self.energy_array[i], j)
                new_array[i] = np.delete(self.array[i], j, 0)
            else:
                new_energy_array[i] = np.insert(self.energy_array[i], j, 0, 0)
                new_pixel = self.array[i, j]
                if not (i == 0 or i == self.height - 1) or (j == 0 or j == self.width - 1):
                    new_pixel = (self.array[i, j - 1] + self.array[i, j + 1]) // 2
                new_array[i] = np.insert(self.array[i], j, new_pixel, 0)
        self.array = new_array
        self.energy_array = new_energy_array
        for i, j in enumerate(seam):
            for k in range(j - 1, j + 1):
                if 0 <= k < self.width and self.energy_array[i, k] >= 0:
                    self.energy_array[i, k] = self.compute_energy(i, k)
        if horizontal:
            self.swapaxes()
        return mp_deleted_count

    def resize(self, new_height=None, new_width=None):
        if new_height is None:
            new_height = self.height
        if new_width is None:
            new_width = self.width
        while self.width != new_width:
            self.carve(horizontal=False, remove=self.width > new_width)
        while self.height != new_height:
            self.carve(horizontal=True, remove=self.height > new_height)

    def remove_mask(self, mask):
        mp_count = np.count_nonzero(mask == 1)
        self.energy_array[mask == 1] *= -(self.max_energy ** 2)
        self.energy_array[mask == 1] -= (self.max_energy ** 2)
        
        self.energy_array[mask == 2] *= (self.max_energy ** 4)
        self.energy_array[mask == 2] += (self.max_energy ** 2)
        
        while mp_count:
            v_seam_energy, v_seam = self.compute_seam(False)
            h_seam_energy, h_seam = self.compute_seam(True)
            horizontal, seam = False, v_seam
            if v_seam_energy > h_seam_energy:
                horizontal, seam = True, h_seam
                
            mp_count -= self.carve(horizontal, seam)
        
    def image(self):
        return self.array.astype(np.uint8)


if __name__ == '__main__':

    img_name = sys.argv[1]
    
    mask = np.load('results/' + img_name + '.npy')
    
#    plt.imshow(mask)
#    plt.show()
    
    image = imread('data/' + img_name)
    
#    plt.imshow(image)
#    plt.show()
    newImg = SeamCarve(image)
    newImg.remove_mask(mask)
    
#    plt.imshow(newImg.image())
#    plt.show()
    
    newImg.resize(new_height=len(image), new_width=len(image[0]))

    plt.imshow(newImg.image())
    plt.show()
    
    imwrite('results/' + img_name + '.removed.jpeg', newImg.image())
