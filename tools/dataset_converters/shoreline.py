#-----------------------------------
# adapted from
# https://stackoverflow.com/questions/74339154/how-to-convert-rle-format-of-label-studio-to-black-and-white-image-masks
#-----------------------------------

from typing import List
import numpy as np
import json
from PIL import Image

class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i:self.i + size]
        self.i += size
        return int(out, 2)


def access_bit(data, num):
    """ from bytes array to bits by num position"""
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def bytes2bit(data):
    """ get bit string from bytes data"""
    return ''.join([str(access_bit(data, i)) for i in range(len(data) * 8)])


def rle_to_mask(rle: List[int], height: int, width: int) -> np.array:
    """
    Converts rle to image mask
    Args:
        rle: your long rle
        height: original_height
        width: original_width

    Returns: np.array
    """

    rle_input = InputStream(bytes2bit(rle))

    num = rle_input.read(32)
    word_size = rle_input.read(5) + 1
    rle_sizes = [rle_input.read(4) + 1 for _ in range(4)]
    # print('RLE params:', num, 'values,', word_size, 'word_size,', rle_sizes, 'rle_sizes')

    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = rle_input.read(1)
        j = i + 1 + rle_input.read(rle_sizes[rle_input.read(2)])
        if x:
            val = rle_input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = rle_input.read(word_size)
                out[i] = val
                i += 1

    image = np.reshape(out, [height, width, 4])[:, :, 3]
    return image

with open('090224_delivery.json', 'r') as f:
    data = json.load(f)

palette = {
    'Man-made structure': [255, 0, 0],       # Red color for man-made structures
    'Sandy beach': [255, 255, 102],          # Light yellow color for sandy beaches
    'Rocky outcrop': [153, 102, 51],         # Brown color for rocky outcrops
    'Gravel beach': [204, 204, 204],         # Light gray color for gravel beaches
    'Block beach': [102, 102, 102],          # Dark gray color for block beaches
    'Ice and snow': [255, 255, 255],         # White color for ice and snow
    'Clay beach': [204, 153, 102],           # Light brown color for clay beaches
    'Wetland area': [51, 153, 102]           # Green color for wetland areas
}



for entry in data:
    image_name = entry['file_upload']
    image_path = "./images/" + image_name
    image = np.array(Image.open(image_path))

    result = entry['annotations'][0]['result']
    for data in result:
        label = data['value']['brushlabels'][0]
        color = palette.get(label)
        mask = rle_to_mask(
            data['value']['rle'], 
            data['original_height'], 
            data['original_width']
            
        )
        image[mask > 0] = color 

    # save mask
    tmp = Image.fromarray(image)
    tmp.save(f"./labels/{image_name}")
    
    

