# This script will load a stack of .npz files, each containing both
# volume-rendering and label data. Each .npz file will be turned into
# one frame (in .png format), by compositing the volume renderings and labels.

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm, LinearSegmentedColormap

from tqdm.auto import tqdm

from itertools import zip_longest

from render_labels import alpha_composite_many


def volrender_data_to_img(arr, cmap='bwr_r',
                          lum_norm=Normalize(vmin=0.0, vmax=1.5),
                          hue_norm=Normalize(vmin=-0.1, vmax=0.1)):
    # The first channel defines the luminosity of the image
    lum = lum_norm(arr[0])

    if len(arr) == 1:
        hue = np.full_like(lum, hue_norm(0))
    else:
        # If there is a 2nd channel, use it to set the hue
        hue = hue_norm(arr[1] / (arr[0]+1e-5))

    # Clip luminosities and hues to range (0,1)
    lum = np.clip(lum, 0, 1).T
    hue = np.clip(hue, 0, 1).T

    # Transform "hue" into color using cmap
    c = plt.get_cmap(cmap)(hue)[:,:,:]

    # Multiply alpha by "lum" to get final color
    c[:,:,3] = lum

    # Convert to RGB
    c = (c * 255).astype(np.uint8)

    # Convert to PIL and return
    return Image.fromarray(c, mode='RGBA')


def main():
    n_frames = 125
    gen_every_n_frames = 30 # If >1, only every nth frame will be generated

    # Backround color
    background_rgba = (0,0,0,255)

    # Normalizations to apply to luminosity and hue channels
    luminosity_norm = PowerNorm(1.0, vmin=0.0, vmax=1.5)
    hue_norm = Normalize(vmin=-0.1, vmax=0.1)

    # The colormap to be used to set the hue
    cmap = LinearSegmentedColormap.from_list(
        'magenta_white_cyan',
        [(1, 0, 1),  # Magenta
         (1, 1, 1),  # White
         (0, 1, 1)], # Cyan
        N=256
    )

    # Input/output filename patterns
    in_fname_pattern = 'frames/frame_data_{frame:04d}.npz'
    out_fname_pattern = 'frames/frame_{frame:04d}.png'

    canvas = None

    # Process each frame independently
    for frame in tqdm(range(0,n_frames,gen_every_n_frames)):
        # Load the label and volume-rendered data
        with np.load(in_fname_pattern.format(frame=frame)) as f:
            label_data = f['labels']
            volrender_data = f['volrender']
        
        # Interleave the labels and volume renderings
        img_stack = []
        for i,(label,volrender) in enumerate(zip_longest(label_data,
                                                         volrender_data)):
            if volrender is not None:
                img = volrender_data_to_img(
                    volrender.astype('f4'),
                    cmap=cmap,
                    lum_norm=luminosity_norm,
                    hue_norm=hue_norm
                )
                img_stack.append(img)
            if label is not None:
                img = Image.fromarray(label, mode='RGBA')
                img_stack.append(img)
        
        # Create a background canvas, if none exists yet
        if canvas is None:
            canvas = Image.new('RGBA', img_stack[0].size, background_rgba)

        # Stack images
        img_stack = [canvas] + img_stack[::-1]
        img = alpha_composite_many(img_stack)

        # Save frame
        img.save(out_fname_pattern.format(frame=frame))

    return 0

if __name__ == "__main__":
    main()
