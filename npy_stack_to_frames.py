# This script will load a stack of .npy files, and then convert them to PNGs.
# For each frame, there will be two .npy files: one defines the luminosity, and
# the other will be turned into a hue.

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm, LinearSegmentedColormap

from tqdm.auto import tqdm

def load_frame(fname_pattern, frame, n_channels):
    arr = []
    for channel in range(n_channels):
        fname = fname_pattern.format(frame=frame, channel=channel)
        arr.append(np.load(fname).T)
    arr = np.stack(arr, axis=0)
    return arr

def frame_stack_to_img(arr, cmap='bwr_r',
                       lum_norm=PowerNorm(0.5, vmin=0.0, vmax=1.5),
                       hue_norm=Normalize(vmin=-0.1, vmax=0.1)):
    lum = lum_norm(arr[0])
    hue = hue_norm(arr[1] / (arr[0]+1e-5))

    # Clip luminosities and hues to range (0,1)
    lum = np.clip(lum, 0, 1)
    hue = np.clip(hue, 0, 1)

    # Transform "hue" into color using cmap
    c = plt.cm.get_cmap(cmap)(hue)[:,:,:3]

    # Multiply by "lum" to get final color
    c *= lum[:,:,None]

    print(np.max(c))

    # Convert to RGB
    c = (c * 255).astype(np.uint8)

    # Convert to PIL and return
    return Image.fromarray(c, mode='RGB')


def main():
    n_frames = 125
    n_channels = 2
    gen_every_n_frames = 10

    in_fname_pattern = 'frames/dust_video_v2_{frame:04d}_{channel:d}.npy'
    out_fname_pattern = 'frames/dust_video_v2_{frame:04d}.png'

    cmap = LinearSegmentedColormap.from_list(
        'magenta_white_cyan',
        [(1, 0, 1),  # Magenta
         (1, 1, 1),  # White
         (0, 1, 1)], # Cyan
        N=256
    )

    # Process each frame independently
    for frame in tqdm(range(0,n_frames,gen_every_n_frames)):
        arr = load_frame(in_fname_pattern, frame, n_channels)
        img = frame_stack_to_img(arr, cmap=cmap)
        img.save(out_fname_pattern.format(frame=frame))

    return 0

if __name__ == "__main__":
    main()
