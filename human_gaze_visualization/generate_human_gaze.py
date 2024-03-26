import argparse
import csv
import os
import os.path as op
import pickle
import shutil

import matplotlib
import numpy as np
from matplotlib import image, pyplot

import cv2
import torch
from tqdm import tqdm       
import json



def draw_display(dispsize, imagefile=None):
    """Returns a matplotlib.pyplot Figure and its axes, with a size of
    dispsize, a black background colour, and optionally with an image drawn
    onto it

    arguments

    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)

    returns
    fig, ax		-	matplotlib.pyplot Figure and its axes: field of zeros
                    with a size of dispsize, and an image drawn onto it
                    if an imagefile was passed
    """

    # construct screen (black background)
    screen = np.zeros((dispsize[1], dispsize[0], 3), dtype='float32')
    # if an image location has been passed, draw the image
    if imagefile != None:
        # check if the path to the image exists
        if not os.path.isfile(imagefile):
            raise Exception("ERROR in draw_display: imagefile not found at '%s'" % imagefile)
        # load image
        img = image.imread(imagefile)

        # width and height of the image
        w, h = len(img[0]), len(img)
        # x and y position of the image on the display
        x = int(dispsize[0] / 2 - w / 2)
        y = int(dispsize[1] / 2 - h / 2)
        # draw the image on the screen
        screen[y:y + h, x:x + w, :] += img[::-1, :, :]
    # dots per inch
    dpi = 100.0
    # determine the figure size in inches
    figsize = (dispsize[0] / dpi, dispsize[1] / dpi)
    # create a figure
    fig = pyplot.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = pyplot.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plot display
    ax.axis([0, dispsize[0], 0, dispsize[1]])
    ax.imshow(screen)  # , origin='upper')

    return fig, ax

def gaussian(x, sx, y=None, sy=None):
    """Returns an array of np arrays (a matrix) containing values between
    1 and 0 in a 2D Gaussian distribution

    arguments
    x		-- width in pixels
    sx		-- width standard deviation

    keyword argments
    y		-- height in pixels (default = x)
    sy		-- height standard deviation (default = sx)
    """

    # square Gaussian if only x values are passed
    if y == None:
        y = x
    if sy == None:
        sy = sx
    # centers
    xo = x / 2
    yo = y / 2
    # matrix of zeros
    M = np.zeros([y, x], dtype=float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            M[j, i] = np.exp(
                -1.0 * (((float(i) - xo) ** 2 / (2 * sx * sx)) + ((float(j) - yo) ** 2 / (2 * sy * sy))))

    return M

def draw_heatmap(gazepoints, dispsize, imagefile=None, alpha=0.5, savefilename=None, gaussianwh=200, gaussiansd=None):
    """Draws a heatmap of the provided fixations, optionally drawn over an
    image, and optionally allocating more weight to fixations with a higher
    duration.

    arguments

    gazepoints		-	a list of gazepoint tuples (x, y)
    
    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)
    alpha		-	float between 0 and 1, indicating the transparancy of
                    the heatmap, where 0 is completely transparant and 1
                    is completely untransparant (default = 0.5)
    savefilename	-	full path to the file in which the heatmap should be
                    saved, or None to not save the file (default = None)

    returns

    fig			-	a matplotlib.pyplot Figure instance, containing the
                    heatmap
    """

    # IMAGE
    fig, ax = draw_display(dispsize, imagefile=imagefile)

    # HEATMAP
    # Gaussian
    gwh = gaussianwh
    gsdwh = gwh / 6 if (gaussiansd is None) else gaussiansd
    gaus = gaussian(gwh, gsdwh)
    # matrix of zeroes
    strt = int(gwh / 2)
    heatmapsize = dispsize[1] + 2 * strt, dispsize[0] + 2 * strt
    heatmap = np.zeros(heatmapsize, dtype=float)
    # create heatmap
    for i in range(0, len(gazepoints)):
        # get x and y coordinates
        x = strt + gazepoints[i][0] - int(gwh / 2)
        y = strt + gazepoints[i][1] - int(gwh / 2)
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj = [0, gwh]
            vadj = [0, gwh]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x - dispsize[0])
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y - dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:
                heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * gazepoints[i][2]
            except:
                # fixation was probably outside of display
                pass
        else:
            # add Gaussian to the current heatmap
            heatmap[y:y + gwh, x:x + gwh] += gaus * gazepoints[i][2]
    # resize heatmap
    heatmap = heatmap[strt:dispsize[1] + strt, strt:dispsize[0] + strt]
    # remove zeros
    lowbound = np.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lowbound] = np.NaN
    # draw heatmap on top of image
    heatmap = heatmap[::-1, :]
    ax.imshow(heatmap, cmap='gray', alpha=alpha)

    # FINISH PLOT
    # invert the y axis, as (0,0) is top left on a display
    # ax.invert_yaxis()
    # save the figure if a file name was provided
    if savefilename != None:
        fig.savefig(savefilename)

    pyplot.cla()
    pyplot.close("all")
    return None

def bbox_xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] where xy=top-left
    # to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    if len(x.shape) == 2:
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0]  # top left x
        y[:, 1] = x[:, 1]  # top left y
        y[:, 2] = x[:, 0] + x[:, 2]  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3]  # bottom right y
    else:
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[0] = x[0]  # top left x
        y[1] = x[1]  # top left y
        y[2] = x[0] + x[2]  # bottom right x
        y[3] = x[1] + x[3]  # bottom right y
    return y

##################
#     Parsing    #
##################

parser = argparse.ArgumentParser(description='Parameters required for processing.')

#optional args
parser.add_argument('-a', '--alpha', type=float, default='0.5', required=False, help='alpha for the gaze overlay')
parser.add_argument('-o',  '--output-name', type=str, required=False, help='name for the output file')
parser.add_argument('-b',  '--background-image', type=str, default=None, required=False, help='path to the background image')

#advanced optional args
parser.add_argument('-n', '--n-gaussian-matrix', type=int, default='200', required=False, help='width and height of gaussian matrix')
parser.add_argument('-sd',  '--standard-deviation', type=float, default=None ,required=False, help='standard deviation of gaussian distribution')


args = vars(parser.parse_args())

alpha = args['alpha']
output_name = args['output_name'] if args['output_name'] is not None else 'output'
background_image = args['background_image']
ngaussian = args['n_gaussian_matrix']
sd = args['standard_deviation']


raw_video_path = "datasets"
root_path =  "anomaly_dataset_gt/anomaly_dataset_gt_test.json"

inpainted_output = dict()

for mode in ['test']:
    with open(root_path, "r") as f:
        json_data = json.load(f)
    filenames = [i for i in json_data.keys()]
    filenames.sort()
    for idx in tqdm(range(0, len(filenames))):
        filename = filenames[idx]
        
        # gt_bbox     = torch.tensor([0.,0.,0.,0.])
        xyxy     = torch.tensor(json_data[filename]['x1y1x2y2_bbox'])
        # Vehicle, Pedestrian, Infrastructure, Cyclist
        # class_label = torch.tensor(0)
        class_label = torch.tensor(json_data[filename]['have_anomaly'])

        image_key = json_data[filename]['image_path']
        assert (not image_key.startswith('/')) and (image_key.endswith('.png') or image_key.endswith('.jpg') or image_key.endswith('.jpeg'))
        img_path = op.join(raw_video_path, image_key)

        img_name = op.basename(img_path)
        img_data = cv2.imread(img_path)
        
        img_h, img_w = img_data.shape[:2]
        # img_h, img_w = 10, 10
        print(img_path, img_w, img_h)
        gaze_data = [(int((xyxy[0]+xyxy[2])/2*img_w), int((xyxy[1]+xyxy[3])/2*img_h), 1)]
        
        # # visualize
        # cv2.rectangle(img_data, (int(xyxy[0]*img_w), int(xyxy[1]*img_h)), (int(xyxy[2]*img_w), int(xyxy[3]*img_h)), (0, 0, 255), 4)
        # cv2.imwrite(f"{idx}_2.jpg", img_data)

        if os.path.exists(op.join(output_name, 'fake_saliency_my')) and os.path.exists(op.join(output_name, 'fake_saliency_my', img_name)):
            try:
                if cv2.imread(op.join(output_name, 'fake_saliency_my', image_key)).shape == img_data.shape:
                    continue
            except:
                pass
            
        os.makedirs(op.dirname(op.join(output_name, 'fake_saliency_my', image_key)), exist_ok=True)
        
        draw_heatmap(gaze_data, (img_w, img_h), alpha=alpha, savefilename=op.join(output_name, 'fake_saliency_my', image_key), imagefile=background_image, gaussianwh=ngaussian, gaussiansd=sd)
        # draw_heatmap(gaze_data, (img_w, img_h), alpha=alpha, savefilename=op.join(f"{idx}_1.jpg"), imagefile=background_image, gaussianwh=ngaussian, gaussiansd=sd)

        # if idx == 2:
        #     break
