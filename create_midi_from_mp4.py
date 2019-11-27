
import imageio.core.util
from IPython import get_ipython
import glob
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import imageio
import shutil
import time
import tqdm
import cv2
import binascii
import struct
from PIL import Image
from midiutil import MIDIFile
import PIL
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
from skimage import io
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage.filters import threshold_yen, threshold_isodata
from skimage.color import rgb2hsv
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from skimage.morphology import disk, binary_dilation, binary_erosion
from skimage import data, img_as_float, img_as_ubyte
from IPython.core.debugger import set_trace
import matplotlib.colors as colors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg
from skimage.exposure import rescale_intensity
from skimage import filters
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
import matplotlib.patches as mpatches
from skimage.morphology import closing, square
from skimage.filters.rank import median
from skimage.measure import compare_psnr
from skimage.util import random_noise
from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage.morphology import disk
from skimage.filters.rank import enhance_contrast_percentile
from sklearn import mixture
import matplotlib as mpl
from scipy import linalg
import itertools
import cv2
from skimage.measure import label, regionprops
from skimage.color import rgb2hsv
from skimage import io
from pathlib import Path
from skimage.color import label2rgb
from skimage.morphology import watershed
from skimage.segmentation import slic, join_segmentations
from skimage.measure import label
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.filters import sobel
from skimage.color import rgb2gray
from skimage.data import astronaut
import warnings
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    io.imsave


def ignore_warnings(*args, **kwargs):
    pass


imageio.core.util._precision_warn = ignore_warnings


def get_video_info(filename):
    vid = imageio.get_reader(filename,  'ffmpeg')
    info = vid.get_meta_data()
    frameRate = info['fps']
    Duration = info['duration']
    nFrames = vid.count_frames()
    sr = int(np.floor(frameRate*.5))
    return vid, info, frameRate, Duration, nFrames, sr


def get_frames(filename, skip=False):

    home = str(Path.home())
    framePath = os.path.join(home, 'tempFrames')

    if skip:
        return glob.glob(os.path.join(framePath, '*.png'))

    if os.path.isdir(framePath):
        shutil.rmtree(framePath, ignore_errors=True)
    os.makedirs(framePath, exist_ok=True)

    vid, info, frameRate, Duration, nFrames, sr = get_video_info(filename)

    frame = vid.get_data(0)

    t = time.time()
    for fIDX in tqdm.tqdm(range(0, nFrames, sr), desc='Extract frames', unit='frames'):
        frame = vid.get_data(fIDX)
        # crop frame
        frame = frame[int(np.round(frame.shape[0] * .2)):int(np.round(frame.shape[0] * .6)), :, :]
        # write frame
        io.imsave(os.path.join(framePath, f'f{fIDX:08d}.png'), frame)

    return glob.glob(os.path.join(framePath, 'f*.png'))


def get_median(frame_list):
    imgs = np.asarray(io.ImageCollection(frame_list))
    return np.median(imgs, axis=0).astype(float) / 255


def show_img(img, title=''):
    plt.figure(figsize=(20, 4))
    plt.imshow(img_as_ubyte(img))
    plt.colorbar()
    plt.title(title)
    plt.show()


def show_hist(img):
    plt.figure(figsize=(20, 4))
    f00 = img.flatten()
    f00 = f00[f00 > 0]
    plt.hist(f00, bins=100)
    plt.show()


@adapt_rgb(each_channel)
def median_each(image):
    k = int(image.shape[1] / 500)
    return filters.median(image, disk(k))


def filt_img(rgb_img, rgb_med, plot_on=False):

    thresh_hue = .2
    thresh_sat = .2

    hsv_img = rgb2hsv(rgb_img)

    # mask background pixels similar to median image
    s = np.std(rgb_med, axis=(0, 1))
    m = np.median(rgb_med, axis=(0, 1))
    lower = m - s
    upper = m + s
    med_mask = np.invert(cv2.inRange(rgb_img, lower, upper)) / 255
    # new version
    med_mask = ~np.concatenate((rgb_img >= rgb_med - 0.1, rgb_img <= rgb_med + 0.1), axis=2).all(axis=2)
    img01 = rgb_img.copy() * med_mask[..., None]

    # mask background pixels similar within a row
    img02 = img01.copy()
    for col in range(0, img02.shape[2]):
        for h in range(0, img02.shape[0]):
            f00 = img02[h, :, col]
            s = np.std(f00)
            m = np.median(f00)
            mask = np.logical_and(f00 < m + s / 2, f00 > m - s / 2)
            f00[mask] = 0
            img02[h, :, col] = f00

    # mask gray background pixels that are below a certain hue and saturation
    hsv_thresh_mask = img_as_float(np.logical_and(
        hsv_img[:, :, 0] > thresh_hue,
        hsv_img[:, :, 1] > thresh_sat))
    img03 = img02.copy() * hsv_thresh_mask[..., None].astype(float)

    # mask edges to separate blobs
    edge_mask = binary_erosion(
        (sobel(img03[:, :, 0]) + sobel(img03[:, :, 1]) + sobel(img03[:, :, 2])) < .1)
    img04 = img03.copy() * edge_mask[..., None].astype(float)

    # remove noise: wavelet filter
    sigma_est = estimate_sigma(img04, multichannel=True, average_sigmas=True)
    img05 = denoise_wavelet(img04.copy(), multichannel=True, convert2ycbcr=True,
                            method='VisuShrink', mode='soft',
                            sigma=sigma_est / 4)

    # remove noise: median filter
    img06 = img_as_float(median_each(img05.copy()))

    if plot_on:
        show_img(rgb_img, title='Original')
        show_img(img01, title='minus median')
        show_img(img02, title='minus row background')
        show_img(img03, title='thresholded')
        show_img(img04, title='removing edges')
        show_img(img05, title='wavelet filter')
        show_img(img06, title='median')

    return img06


def find_dominant_color(img):

    NUM_CLUSTERS = 5

    ar = np.asarray(img)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences
    index_max = scipy.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    # colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')

    return peak


def get_key(img, plot_on=False):

    k = int(img.shape[1] / 500)
    thr = 0.1
    img_bin = closing(rgb2gray(img.copy()) > thr, square(k))

    # label image regions
    label_image = label(img_bin > 0)

    key_col = []
    key_start = []
    key_end = []
    key_loc = []
    key_width = []
    rects = []
    for region in tqdm.tqdm(regionprops(label_image), desc='Get key properties', unit='keys'):
        # take regions with large enough areas
        if region.area >= k * 20:
            minr, minc, maxr, maxc = region.bbox
            c0 = int((minc + maxc) / 2 - k)
            c1 = int((minc + maxc) / 2 + k)
            r0 = int((minr + maxr) / 2 - k)
            r1 = int((minr + maxr) / 2 + k)
            # c = np.median(img[r0:r1, c0:c1, :], axis=(0, 1))
            # c = np.median(img[minr:maxr, minc:maxc, :], axis=(0, 1))
            c = np.median(img[minr+1:maxr-1, minc+k:maxc-k, :], axis=(0, 1))
            if np.isnan(c).any():
                c = np.median(img[minr+1:maxr-1, minc:maxc, :], axis=(0, 1))
            # c = find_dominant_color(img[minr:maxr, minc:maxc, :])
            key_col.append(c)
            key_start.append(minr - k)
            key_end.append(maxr + k)
            key_loc.append(int((minc + maxc) / 2))
            key_width.append(maxc - minc)
            # for plotting only
            rects.append(mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                            fill=False, edgecolor='red', linewidth=2))

    key_col = np.vstack(key_col)
    key_start = np.asanyarray(key_start)
    key_end = np.asanyarray(key_end)
    key_loc = np.asanyarray(key_loc)
    key_width = np.asanyarray(key_width)

    if plot_on:
        show_img(img_bin, title='binarized')
        show_img(label_image, title='labeled')

        _, ax = plt.subplots(figsize=(10, 6))
        plt.imshow(img)
        for rect in rects:
            ax.add_patch(rect)
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()

    return key_col, key_loc, key_width, key_start, key_end


def find_offset(img0, img1):
    m = []
    for i in range(1, img0.shape[0]):
        diff_img = np.abs(img0[:-i, :, :] - img1[i:, :, :])
        m.append(np.mean(diff_img))
    return np.argmin(m)


def get_offset(frame_list):
    offset = []
    i0 = int(len(frame_list) / 2 - 10)
    i1 = int(len(frame_list) / 2 + 10)
    img1 = img_as_float(io.imread(frame_list[i0]))
    for id in range(i0 + 1, i1):
        img0 = img1
        img1 = img_as_float(io.imread(frame_list[id]))

        offset.append(find_offset(img0, img1))

    return np.median(offset).astype(int)


def create_filt_pianoroll(frame_list):
    i0 = int(len(frame_list) / 2 - 30)
    i1 = int(len(frame_list) / 2 + 30)
    off = get_offset(frame_list)
    rgb_med = get_median(frame_list[i0:i1])

    frame_shape = io.imread(frame_list[0]).shape
    pianoroll = np.zeros((1, frame_shape[1], frame_shape[2]))
    for fr in tqdm.tqdm(frame_list, desc='Concatenate frames', unit='frames'):
        rgb_img = img_as_float(io.imread(fr))
        img = filt_img(rgb_img, rgb_med, plot_on=False)
        pianoroll = np.concatenate((img[:off, :, :], pianoroll), axis=0)

    return np.flip(pianoroll, axis=0)


def calc_keys(img_shape):
    # location of black and white keys to get identity, in x direction
    wkey_border = np.linspace(0, img_shape[1], 53)  # 52 white keys -> 53 borders
    wkey_border[0] = 1
    wkey_border = wkey_border[:-1]
    wkey_center = np.round(wkey_border + np.mean(np.diff(wkey_border)) / 2)
    bkey_center = np.round(wkey_border)
    no_black = [0, 2, 5, 9, 12, 16, 19, 23, 26, 30, 33, 37, 40, 44, 47, 51]
    bkey_center = np.delete(bkey_center, no_black)
    key_pos = np.sort(np.concatenate((wkey_center, bkey_center), axis=0))
    is_white = [k in wkey_center for k in key_pos]
    wkey_width = img_shape[1] / len(wkey_center) - 2
    bkey_width = (wkey_width + 2) * (12.7 / 23.6) - 2
    return key_pos, is_white, wkey_width, bkey_width, np.arange(0, 88)


def cluster_key_col(key_col, method='dpgmm3d', n_components=5, plot_on=False):

    if '2d' in method:
        data = key_col[:, 0:2]
    elif '3d' in method:
        data = key_col

    if method == 'gmm2d':
        # Fit a Dirichlet process Gaussian mixture using five components
        model = mixture.GaussianMixture(n_components=n_components, covariance_type='full')
    elif method == 'dpgmm2d':
        model = BayesianGaussianMixture(n_components=n_components, covariance_type='full')
    elif method == 'gmm3d':
        # Fit a Dirichlet process Gaussian mixture using five components
        model = mixture.GaussianMixture(n_components=n_components, covariance_type='full')
    elif method == 'dpgmm3d':
        model = BayesianGaussianMixture(n_components=n_components, covariance_type='full')

    pred = model.fit_predict(data)

    if plot_on:
        if '2d' in method:
            plot_results(data, pred, model.means_, model.covariances_, 0,
                         f'Gaussian Mixture 2d ({n_components} components)')
        elif '3d' in method:
            plot_results3d(data, pred, model.means_, model.covariances_, 1,
                           f'Gaussian Mixture 3d ({n_components} components)')

    return pred


def get_rank(lst):
    array = np.array(lst)
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks


def plot_results(X, Y_, means, covariances, index, title):
    colors = ['navy', 'c', 'cornflowerblue', 'gold', 'darkorange']
    plt.figure(figsize=(20, 10))
    splot = plt.subplot(2, 1, 1 + index)
    for i in np.unique(Y_):
        v, w = linalg.eigh(covariances[i])
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=colors[i])

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(
            means[i], v[0], v[1], 180. + angle, color=colors[i])
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(np.min(X[:, 0]) - .1, np.max(X[:, 0]) + .1)
    plt.ylim(np.min(X[:, 1]) - .1, np.max(X[:, 1]) + .1)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    plt.show()


def plot_results3d(X, Y_, means, covariances, index, title):
    colors = ['navy', 'c', 'cornflowerblue', 'gold', 'darkorange']
    fig = plt.figure(figsize=(20, 10))
    splot = fig.add_subplot(2, 1, 1 + index, projection='3d')
    for i in np.unique(Y_):
        splot.scatter(X[Y_ == i, 0], X[Y_ == i, 1], X[Y_ == i, 2], c=colors[i])
    plt.show()


def write_midi(keys, midi_filename):
    channel = 0
    time = 0    # In beats
    duration = 1    # In beats
    tempo = 60   # In BPM
    volume = 100  # 0-127, as per the MIDI standard

    MyMIDI = MIDIFile(2)
    for track in [0, 1]:
        MyMIDI.addTempo(track, time, tempo)

    for index, k in tqdm.tqdm(keys.iterrows(), desc='Add notes', unit='notes'):
        MyMIDI.addNote(int(k.left_hand), channel, int(k.key_piano_pos), k.key_start, k.key_duration, volume)

    with open(midi_filename, "wb") as output_file:
        MyMIDI.writeFile(output_file)


def remove_color_cluster(keys, plot_on=False):
    classes, counts = np.unique(keys['color_cluster'], return_counts=True)
    if len(classes) > 4:
        class_min = classes[np.argmin(counts)]
        loc_class_min = np.argwhere(keys['color_cluster'] == class_min).flatten()
        if plot_on:
            for i in loc_class_min:
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(pianoroll[keys['key_start'][i]-10:keys['key_end'][i]+10,
                                       keys['key_loc'][i]-keys['key_width'][i]:keys['key_loc'][i]+keys['key_width'][i], :])
                plt.suptitle(f'Key: {i}, outlier color, color cluster: {keys["color_cluster"][i]}')
                ax[1].imshow(pianoroll[keys['key_start'][i]-100:keys['key_end'][i]+100, :, :])
                plt.show()

        keys.drop(keys[keys['color_cluster'] == class_min].index, inplace=True)

    return keys


def check_inconsistency(keys, plot_on=False):
    keys['inconsistent'] = keys['is_white_pos'].ne(keys['is_white_width'])
    if plot_on:
        for i in keys.index[keys['inconsistent']].tolist():
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(pianoroll[keys['key_start'][i]-10:keys['key_end'][i]+10,
                                   keys['key_loc'][i]-keys['key_width'][i]:keys['key_loc'][i]+keys['key_width'][i], :])
            plt.suptitle(f'Key: {i}, outlier width, width: {key_width[i]}')
            ax[1].imshow(pianoroll[keys['key_start'][i]-100:keys['key_end'][i]+100, :, :])
            plt.show()
    return keys


def remove_intro_outro(pianoroll):
    mc = np.empty([pianoroll.shape[0], pianoroll.shape[2]])
    for c in range(0, pianoroll.shape[2]):
        for r in range(0, pianoroll.shape[0]):
            m = np.mean(pianoroll[r, pianoroll[r, :, c] > 0.0001, :])
            if np.isnan(m):
                m = 0
            mc[r, c] = m

    i0 = int(mc.shape[0]/2 - mc.shape[0]/10)
    i1 = int(mc.shape[0]/2 + mc.shape[0]/10)
    mm = np.median(mc[i0:i1, :], axis=0)
    ms = np.std(mc[i0:i1, :], axis=0)

    mcb = np.empty([pianoroll.shape[0], pianoroll.shape[2]])
    from scipy import signal
    win = signal.hann(200)
    for c in range(0, mc.shape[1]):
        filtered = signal.convolve(mc[:, c], win, mode='same') / sum(win)
        mcb[:, c] = np.logical_and(filtered < mm[c] + ms[c], filtered > mm[c] - ms[c])

    mcb[int(mc.shape[0]/10):int(mc.shape[0] - mc.shape[0]/10), :] = 1.0
    mcb = np.mean(mcb, axis=1) == 1.0

    # plt.plot(mcb)
    # plt.show()

    # plt.plot(mc)
    # plt.show()

    return pianoroll[mcb, :, :]


if __name__ == '__main__':

    ext = 'mp4'
    video_path = f'videos/*.{ext}'
    video_list = glob.glob(video_path)

    for song in video_list:
        print(os.path.basename(song))

        pianoroll_filename = os.path.join('pianoroll', os.path.basename(song).replace(f'.{ext}', '.png'))
        midi_filename = os.path.join('midi', os.path.basename(song).replace(f'.{ext}', '.midi'))

        if os.path.isfile(midi_filename):
            print('Midi exists. Skipping.')
            continue

        if not os.path.isfile(pianoroll_filename):
            frame_list = get_frames(song, skip=False)
            pianoroll = create_filt_pianoroll(frame_list)
            pianoroll_uint8 = 255 * pianoroll.copy()  # for saving, scale by 255
            pianoroll_uint8 = pianoroll_uint8.astype(np.uint8)
            io.imsave(pianoroll_filename, pianoroll_uint8)  # write frame
        else:
            print('Pianoroll exists. Loading...')
            PIL.Image.MAX_IMAGE_PIXELS = 933120000
            pianoroll = img_as_float(io.imread(pianoroll_filename))

        vid, info, frameRate, Duration, nFrames, sr = get_video_info(song)
        # show_img(pianoroll)

        pianoroll = remove_intro_outro(pianoroll)

        key_pos, is_white, wkey_width, bkey_width, key_idx = calc_keys(pianoroll.shape)
        key_col, key_loc, key_width, key_start, key_end = get_key(pianoroll)
        color_cluster = cluster_key_col(key_col, method='dpgmm3d', n_components=5, plot_on=False)

        # plt.hist(key_width)
        # plt.show()

        keys = pd.DataFrame({'key_loc': key_loc,
                             'key_width': key_width,
                             'key_start': key_start - np.min(key_start),
                             'key_end': key_end - np.min(key_start),
                             'key_duration': key_end-key_start,
                             'key_col': key_col.tolist(),
                             'color_cluster': color_cluster})

        # correct to true duration of video
        dur_fact = keys['key_end'].max() / Duration
        keys['key_end'] /= dur_fact
        keys['key_start'] /= dur_fact
        keys['key_duration'] /= dur_fact

        # position from pixel space to piano space
        keys['key_pos'] = keys['key_loc'].apply(lambda x: np.argmin(np.abs(key_pos - x)))
        keys['key_piano_pos'] = keys['key_pos'] + 20
        keys['is_white_pos'] = keys['key_pos'].apply(lambda x: is_white[x])
        keys['is_white_width'] = keys['key_width'].apply(lambda x: np.argmin(
            np.abs(np.array([bkey_width, wkey_width]) - x)).astype(bool))

        # remove outlier color cluster
        keys = remove_color_cluster(keys, plot_on=False)

        m = keys.groupby('color_cluster').mean()['key_loc']
        keys['color_cluster_avgpos'] = keys["color_cluster"].astype(str).replace(dict(zip(m.index.astype(str), m)))
        keys['color_cluster_rank'] = keys["color_cluster"].astype(
            str).replace(dict(zip(m.index.astype(str), get_rank(m))))
        keys['left_hand'] = keys['color_cluster_rank'].apply(lambda x: x in [0, 1])

        # number of keys per color cluster
        k_unique = keys.groupby('is_white_pos')['color_cluster_rank'].value_counts()
        # assert(len(k_unique) == 4)

        keys = check_inconsistency(keys)

        write_midi(keys, midi_filename)
