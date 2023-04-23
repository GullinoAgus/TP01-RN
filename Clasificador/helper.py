import numpy as np
import matplotlib as mpl
from io import BytesIO
from ipywidgets.widgets import Image
import matplotlib.pyplot as plt

def showpics(img_array, num_of_img, title):

    fig, axes = plt.subplots(1, num_of_img+1)
    axes[0].text(0, 0, title)
    axes[0].axis("off")
    for ax, i in zip(axes[1:], range(num_of_img)):
        ax.imshow(img_array[i])
        ax.axis("off")
        
    plt.show()
    
def arr2img(arr):
    """Display a 2- or 3-d numpy array as an image."""
    if arr.ndim == 2:
        format, cmap = 'png', mpl.cm.gray
    elif arr.ndim == 3:
        format, cmap = 'jpg', None
    else:
        raise ValueError("Only 2- or 3-d arrays can be displayed as images.")
    # Don't let matplotlib autoscale the color range so we can control overall luminosity
    vmax = 255 if arr.dtype == 'uint8' else 1.0
    with BytesIO() as buffer:
        mpl.image.imsave(buffer, arr, format=format, cmap=cmap, vmin=0, vmax=vmax, pil_kwargs={'interpolator':'nearest'})
        out = buffer.getvalue()
    return Image(value=out, width=100, embed = True)