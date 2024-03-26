import numbers
import random
import numpy as np
import PIL
import skimage.transform
import torchvision
import math
import torch

from src.datasets.data_utils.volume_transforms import ClipToTensor


from . import video_functional as F
from IPython import embed

def video_bbox_prcoess(is_train, img_res, gaze_pred_bbox=False):

    if gaze_pred_bbox:
        mean = ([0.031, 0.031, 0.031], [0.485, 0.456, 0.406])
        std  = ([0.100, 0.100, 0.100], [0.229, 0.224, 0.225])
        normalize = RGB_Gaze_Normalize(mean, std)
    else:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        normalize = Normalize(mean, std)

    if is_train:
        return Compose([
            RandomHorizontalFlip(),
            RandomSelect(
                Resize((img_res, img_res)),
                Compose([
                    RandomResize([400, 500, 600]),
                    RandomSizeCrop(384, 600),
                    Resize((img_res, img_res)),
                ])
            ),
            ToTensor(),
            normalize
        ])
    else:
        return Compose([
            Resize((img_res, img_res)),
            ToTensor(),
            normalize
        ])


class ToTensor(object):
    def __call__(self, clip, bbox=None, is_gaze=False):
        return ClipToTensor(channel_nb=3)(clip), bbox

class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, bbox=None, is_gaze=False):
        if random.random() < self.p:
            return self.transforms1(img, bbox, is_gaze)
        return self.transforms2(img, bbox, is_gaze)


class Compose(object):
    """Composes several transforms

    Args:
    transforms (list of ``Transform`` objects): list of transforms
    to compose
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip, bbox=None, is_gaze=False):
        if bbox is not None:
            for t in self.transforms:
                clip, bbox = t(clip, bbox, is_gaze)
            return clip, bbox
        else:
            for t in self.transforms:
                clip, bbox = t(clip, bbox, is_gaze)
            return clip


class RandomHorizontalFlip(object):
    """Horizontally flip the list of given images randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip, bbox=None, is_gaze=False):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        PIL.Image or numpy.ndarray: Randomly flipped clip
        """
        if random.random() < self.p:
            if bbox is not None:
                bbox = torch.as_tensor([1-bbox[2], bbox[1], 1-bbox[0], bbox[3]])
            
            if isinstance(clip[0], np.ndarray):
                return [np.fliplr(img) for img in clip], bbox
            elif isinstance(clip[0], PIL.Image.Image):
                return [
                    img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in clip
                ], bbox
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                ' but got list of {0}'.format(type(clip[0])))
        return clip, bbox

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    """Vertically flip the list of given images randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip, bbox=None, is_gaze=False):
        """

        Args:
            img (PIL.Image or numpy.ndarray): List of images to be flipped
            in format (h, w, c) in numpy.ndarray

        Returns:
            PIL.Image or numpy.ndarray: Randomly flipped clip
        """
        if random.random() < self.p:

            if bbox is not None:
                bbox = torch.as_tensor([bbox[0], 1-bbox[3], bbox[2], 1-bbox[1]])
            
            if isinstance(clip[0], np.ndarray):
                return [np.flipud(img) for img in clip], bbox
            elif isinstance(clip[0], PIL.Image.Image):
                return [
                    img.transpose(PIL.Image.FLIP_TOP_BOTTOM) for img in clip
                ], bbox
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                ' but got list of {0}'.format(type(clip[0])))
        return clip, bbox

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomGrayscale(object):
    """Randomly convert image to grayscale with a probability of p (default 0.1).
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading
    dimensions
    Args:
        p (float): probability that image should be converted to grayscale.
    Returns:
        PIL Image or Tensor: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b
    """
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
    def __call__(self, clip, bbox=None, is_gaze=False):
        """
        Args:
            list of imgs (PIL Image or Tensor): Image to be converted to grayscale.
        Returns:
            PIL Image or Tensor: Randomly grayscaled image.
        """
        num_output_channels = 1 if clip[0].mode == 'L' else 3
        if torch.rand(1)<self.p:
            for i in range(len(clip)):
                clip[i]=F.to_grayscale(clip[i],num_output_channels)
        return clip, bbox

class RandomResize(object):
    def __init__(self, sizes, max_size=None, interpolation='nearest'):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size
        self.interpolation = interpolation


    def __call__(self, clip, bbox=None, is_gaze=False):
        size = random.choice(self.sizes)
        resized = F.resize_clip(
            clip, size, interpolation=self.interpolation)
        return resized, bbox


class Resize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size

    The larger the original image is, the more times it takes to
    interpolate

    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, size, interpolation='nearest'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip, bbox=None, is_gaze=False):
        resized = F.resize_clip(
            clip, self.size, interpolation=self.interpolation)
        return resized, bbox


class RandomCrop(object):
    """Extract random crop at the same location for a list of images

    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip, bbox=None, is_gaze=False):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        y1 = random.randint(0, im_h - h)
        x1 = random.randint(0, im_w - w)
        cropped = F.crop_clip(clip, y1, x1, h, w)

        if bbox is not None:
            max_size = torch.as_tensor([w, h], dtype=torch.float32)
            raw_bbox = torch.as_tensor([bbox[0]*im_w, bbox[1]*im_h, bbox[2]*im_w, bbox[3]*im_h])
            cropped_boxes = raw_bbox - torch.as_tensor([x1, y1, x1, y1])
            cropped_boxes = torch.min(cropped_boxes.reshape(2, 2), max_size)
            cropped_boxes = cropped_boxes.clamp(min=0)
            area = (cropped_boxes[1, :] - cropped_boxes[0, :]).prod(dim=0)
            bbox = cropped_boxes.reshape(4)
            bbox = torch.as_tensor([bbox[0]/im_w, bbox[1]/im_h, bbox[2]/im_w, bbox[3]/im_h])
                
        return cropped, bbox

# class RandomResizedCrop(object):
#     """Crop the given list of PIL Images to random size and aspect ratio.

#     A crop of random size (default: of 0.08 to 1.0) of the original size and a random
#     aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
#     is finally resized to given size.
#     This is popularly used to train the Inception networks.

#     Args:
#         size: expected output size of each edge
#         scale: range of size of the origin size cropped
#         ratio: range of aspect ratio of the origin aspect ratio cropped
#         interpolation: Default: PIL.Image.BILINEAR
#     """

#     def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='bilinear'):
#         if isinstance(size, (tuple, list)):
#             self.size = size
#         else:
#             self.size = (size, size)
#         if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
#             warnings.warn("range should be of kind (min, max)")

#         self.interpolation = interpolation
#         self.scale = scale
#         self.ratio = ratio

#     @staticmethod
#     def get_params(clip, scale, ratio):
#         """Get parameters for ``crop`` for a random sized crop.

#         Args:
#             img (list of PIL Image): Image to be cropped.
#             scale (tuple): range of size of the origin size cropped
#             ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

#         Returns:
#             tuple: params (i, j, h, w) to be passed to ``crop`` for a random
#                 sized crop.
#         """
#         if isinstance(clip[0], np.ndarray):
#             height, width, im_c = clip[0].shape
#         elif isinstance(clip[0], PIL.Image.Image):
#             width, height = clip[0].size
#         area = height * width

#         for _ in range(10):
#             target_area = random.uniform(*scale) * area
#             log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
#             aspect_ratio = math.exp(random.uniform(*log_ratio))

#             w = int(round(math.sqrt(target_area * aspect_ratio)))
#             h = int(round(math.sqrt(target_area / aspect_ratio)))

#             if 0 < w <= width and 0 < h <= height:
#                 i = random.randint(0, height - h)
#                 j = random.randint(0, width - w)
#                 return i, j, h, w

#         # Fallback to central crop
#         in_ratio = float(width) / float(height)
#         if (in_ratio < min(ratio)):
#             w = width
#             h = int(round(w / min(ratio)))
#         elif (in_ratio > max(ratio)):
#             h = height
#             w = int(round(h * max(ratio)))
#         else:  # whole image
#             w = width
#             h = height
#         i = (height - h) // 2
#         j = (width - w) // 2
#         return i, j, h, w

#     def __call__(self, clip):
#         """
#         Args:
#             clip: list of img (PIL Image): Image to be cropped and resized.

#         Returns:
#             list of PIL Image: Randomly cropped and resized image.
#         """
#         i, j, h, w = self.get_params(clip, self.scale, self.ratio)
#         imgs=F.crop_clip(clip,i,j,h,w)
#         return F.resize_clip(clip,self.size,self.interpolation)
#         # return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

#     def __repr__(self):
#         interpolate_str = _pil_interpolation_to_str[self.interpolation]
#         format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
#         format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
#         format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
#         format_string += ', interpolation={0})'.format(interpolate_str)
#         return format_string


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size


    def __call__(self, clip, bbox=None, is_gaze=False):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        w = random.randint(self.min_size, min(im_w, self.max_size))
        h = random.randint(self.min_size, min(im_h, self.max_size))


        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        y1 = random.randint(0, im_h - h)
        x1 = random.randint(0, im_w - w)
        cropped = F.crop_clip(clip, y1, x1, h, w)

        if bbox is not None:
            max_size = torch.as_tensor([w, h], dtype=torch.float32)
            raw_bbox = torch.as_tensor([bbox[0]*im_w, bbox[1]*im_h, bbox[2]*im_w, bbox[3]*im_h])
            cropped_boxes = raw_bbox - torch.as_tensor([x1, y1, x1, y1])
            cropped_boxes = torch.min(cropped_boxes.reshape(2, 2), max_size)
            cropped_boxes = cropped_boxes.clamp(min=0)
            area = (cropped_boxes[1, :] - cropped_boxes[0, :]).prod(dim=0)
            bbox = cropped_boxes.reshape(4)
            bbox = torch.as_tensor([bbox[0]/im_w, bbox[1]/im_h, bbox[2]/im_w, bbox[3]/im_h])
                
        return cropped, bbox

# class RandomRotation(object):
#     """Rotate entire clip randomly by a random angle within
#     given bounds

#     Args:
#     degrees (sequence or int): Range of degrees to select from
#     If degrees is a number instead of sequence like (min, max),
#     the range of degrees, will be (-degrees, +degrees).

#     """

#     def __init__(self, degrees):
#         if isinstance(degrees, numbers.Number):
#             if degrees < 0:
#                 raise ValueError('If degrees is a single number,'
#                                  'must be positive')
#             degrees = (-degrees, degrees)
#         else:
#             if len(degrees) != 2:
#                 raise ValueError('If degrees is a sequence,'
#                                  'it must be of len 2.')

#         self.degrees = degrees

#     def __call__(self, clip):
#         """
#         Args:
#         img (PIL.Image or numpy.ndarray): List of images to be cropped
#         in format (h, w, c) in numpy.ndarray

#         Returns:
#         PIL.Image or numpy.ndarray: Cropped list of images
#         """
#         angle = random.uniform(self.degrees[0], self.degrees[1])
#         if isinstance(clip[0], np.ndarray):
#             rotated = [skimage.transform.rotate(img, angle) for img in clip]
#         elif isinstance(clip[0], PIL.Image.Image):
#             rotated = [img.rotate(angle) for img in clip]
#         else:
#             raise TypeError('Expected numpy.ndarray or PIL.Image' +
#                             'but got list of {0}'.format(type(clip[0])))

#         return rotated


class CenterCrop(object):
    """Extract center crop at the same location for a list of images

    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = int(round((im_w - w) / 2.))
        y1 = int(round((im_h - h) / 2.))
        cropped = F.crop_clip(clip, y1, x1, h, w)

        if bbox is not None:
            max_size = torch.as_tensor([w, h], dtype=torch.float32)
            raw_bbox = torch.as_tensor([bbox[0]*im_w, bbox[1]*im_h, bbox[2]*im_w, bbox[3]*im_h])
            cropped_boxes = raw_bbox - torch.as_tensor([x1, y1, x1, y1])
            cropped_boxes = torch.min(cropped_boxes.reshape(2, 2), max_size)
            cropped_boxes = cropped_boxes.clamp(min=0)
            area = (cropped_boxes[1, :] - cropped_boxes[0, :]).prod(dim=0)
            bbox = cropped_boxes.reshape(4)
            bbox = torch.as_tensor([bbox[0]/im_w, bbox[1]/im_h, bbox[2]/im_w, bbox[3]/im_h])
           

        return cropped


# class ColorJitter(object):
#     """Randomly change the brightness, contrast and saturation and hue of the clip

#     Args:
#     brightness (float): How much to jitter brightness. brightness_factor
#     is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
#     contrast (float): How much to jitter contrast. contrast_factor
#     is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
#     saturation (float): How much to jitter saturation. saturation_factor
#     is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
#     hue(float): How much to jitter hue. hue_factor is chosen uniformly from
#     [-hue, hue]. Should be >=0 and <= 0.5.
#     """

#     def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
#         self.brightness = brightness
#         self.contrast = contrast
#         self.saturation = saturation
#         self.hue = hue

#     def get_params(self, brightness, contrast, saturation, hue):
#         if brightness > 0:
#             brightness_factor = random.uniform(
#                 max(0, 1 - brightness), 1 + brightness)
#         else:
#             brightness_factor = None

#         if contrast > 0:
#             contrast_factor = random.uniform(
#                 max(0, 1 - contrast), 1 + contrast)
#         else:
#             contrast_factor = None

#         if saturation > 0:
#             saturation_factor = random.uniform(
#                 max(0, 1 - saturation), 1 + saturation)
#         else:
#             saturation_factor = None

#         if hue > 0:
#             hue_factor = random.uniform(-hue, hue)
#         else:
#             hue_factor = None
#         return brightness_factor, contrast_factor, saturation_factor, hue_factor

#     def __call__(self, clip):
#         """
#         Args:
#         clip (list): list of PIL.Image

#         Returns:
#         list PIL.Image : list of transformed PIL.Image
#         """
#         if isinstance(clip[0], np.ndarray):
#             raise TypeError(
#                 'Color jitter not yet implemented for numpy arrays')
#         elif isinstance(clip[0], PIL.Image.Image):
#             brightness, contrast, saturation, hue = self.get_params(
#                 self.brightness, self.contrast, self.saturation, self.hue)

#             # Create img transform function sequence
#             img_transforms = []
#             if brightness is not None:
#                 img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
#             if saturation is not None:
#                 img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
#             if hue is not None:
#                 img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
#             if contrast is not None:
#                 img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
#             random.shuffle(img_transforms)

#             # Apply to all images
#             jittered_clip = []
#             for img in clip:
#                 for func in img_transforms:
#                     jittered_img = func(img)
#                 jittered_clip.append(jittered_img)

#         else:
#             raise TypeError('Expected numpy.ndarray or PIL.Image' +
#                             'but got list of {0}'.format(type(clip[0])))
#         return jittered_clip

class Normalize(object):
    """Normalize a clip with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, clip, bbox=None, is_gaze=False):
        """
        Args:
            clip (Tensor): Tensor clip of size (T, C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor clip.
        """
        return F.normalize(clip, self.mean, self.std), bbox

class RGB_Gaze_Normalize(object):
    """Normalize a clip with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, clip, bbox=None, is_gaze=False):
        """
        Args:
            clip (Tensor): Tensor clip of size (T, C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor clip.
        """
        if is_gaze:
            return F.normalize(clip, self.mean[0], self.std[0]), bbox
        else:
            return F.normalize(clip, self.mean[1], self.std[1]), bbox

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
