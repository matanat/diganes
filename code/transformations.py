from skimage.filters import threshold_otsu
from skimage import color
import numpy as np
import itertools
from PIL import Image
import os
import pandas as pd

class RandomPermutation(object):
    """Create a random permutation of a PIL image.

    """
    def __init__(self, cache_dir=None, cache_prefix='per_'):
        '''
        Should we use offline cache instead of creating image permutations on the spot

        '''
        self.cache_dir = cache_dir
        self.cache_prefix = cache_prefix

        if self.cache_dir != None:
            self.cached_files = pd.DataFrame(
                                [x for x in os.listdir(cache_dir)if (x.startswith(cache_prefix) and x.endswith(".jpg"))],
                                columns=['Filename'])

    def __call__(self, image):

        if self.cache_dir != None:
            cache = self.cached_files
            _, filename = os.path.split(image.filename)
            cache_match = cache[cache['Filename'].str.contains(filename)]

            chosen_index = np.random.randint(-1, len(cache_match))

            #chosen_index == -1 is identity
            if chosen_index != -1:
                chosen_per = cache_match.iloc[chosen_index, 0]
                image = Image.open(os.path.join(self.cache_dir, chosen_per))

        else:
            cuts = self.get_possible_cuts(image)
            permutations = self.get_possible_permutations(image.width, cuts)

            chosen_index = np.random.randint(0, len(permutations))

            #permutations[0] is identity
            if chosen_index != 0:
                image = self.get_permutated_image(image, permutations[chosen_index])

        return image.convert('RGB')

    def get_possible_cuts(self, image):
        '''
        Get a PIL image and returns a list of x values in which the image could be cut
        without going through any pictorial elements.
        '''
        #convert to gray scale and seperate between black and white pixels
        image = color.rgb2gray(np.array(image))
        thresh = threshold_otsu(image)
        binary = image > thresh

        #mean of black pixes in every column
        H = binary.shape[0]
        W = binary.shape[1]
        take = int(H * 0.05) #ignore top and bottom frame
        pix = np.invert(binary)[take:H-take,:].mean(axis=0)

        #look for local minima
        argmin = np.argsort(pix)
        min_pts = {idx : val for val, idx in zip(pix[argmin], argmin)}

        cuts = set()
        for k in min_pts:
            #Only intrestd with columns with less then 4% of black pixels
            if min_pts[k] > 0.04:
                break

            #Don't look close to image edges
            if k < int(W * 0.15) or k > int(W - (W * 0.15)):
                continue

            #Find local minimum
            skip = False
            cuts_copy = cuts
            for cut in cuts:
                if (k - cut) < int(W * 0.15):
                    if min_pts[k] < min_pts[cut]:
                        cuts_copy.remove(cut)
                    else:
                        skip = True
                    break

            cuts = cuts_copy
            if skip == True:
                continue

            cuts.add(k)

        return cuts

    def get_possible_permutations(self, image_width, cuts):
        '''
        Get a list of cuts returns a list of all posibble permutations of image tiles.
        A permutation is a dictionary of tile_number: (tile_start_x, tile_end_x)
        '''
        if len(cuts) == 0:
            return [{0 : (0, image_width - 1)}]

        #cuts to image tiles
        tiles = {}
        begin = 0
        for i, cut in enumerate(sorted(cuts)):
            tiles[i] = (begin, cut)
            begin = cut + 1
        tiles[i + 1] = (begin, image_width - 1)

        #get all tiles permutations
        permutations = []
        for per in itertools.permutations(tiles, len(tiles)):
            ordering = {}
            for tile in per:
                ordering[tile] = tiles[tile]
            permutations.append(ordering)

        return permutations

    def get_permutated_image(self, image, permutation):
        '''
        Get a PIL image and a chosen permutation, returns the permutated image.
        If cache is used, look for permutated image in same directory.
        '''

        #crop tiles
        new_image_order = []
        for tile in permutation:
            (x, width) = permutation[tile]
            region = image.crop((x, 0, width, image.height - 1))
            new_image_order.append(region)

        per_image = Image.new(mode="RGB", size=image.size)

        #paste tiles in target
        x = 0
        for region in new_image_order:
            per_image.paste(region, (x, 0))
            x += region.width

        return per_image

#from https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
class ReverseTransform(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)

        #return to RGB
        tensor = tensor.permute(1, 2, 0)
        return tensor
