"""
The image_gen module contains functions for generating image_gen rgb numpy arrays and images.
The module can be used in three ways.
The image_gen.rgb function takes arguments that specify the type of pattern and return a numpy array.
>>> rgb = image_gen.rgb(lx=128, ly=128, ...)
>>> print(rgb.shape)
(128, 128, 3)
The image_gen.plot function takes the same arguments as image_gen.rgb and plots the rgb array.
>>> image_gen.plot(lx=128, ly=128, ...)
Finally the module can be used from the command line.
The command line arguments are the same as the rgb function arguments
[user]$ python image_gen.py --lx 128 --ly 128
"""
from matplotlib import pyplot as plt
import numpy as np

def rgb(lx, ly):
    x = np.linspace(0, 1, lx)
    y = np.linspace(0, 1, ly)
    xs, ys = np.meshgrid(x, y)
    def background_pattern(xs, ys):
        r = 0.5+0.5*np.cos(xs*4*np.pi)
        g = 0.5+0.5*np.cos(ys*4*np.pi)
        b = 0.*xs
        return np.stack([r, g, b], axis=-1)
    def white(xs, ys):
        return np.full((lx, ly, 3), 1.)
    def inside_shape(xs, ys):
        def inside_circle(xs, ys, x0, y0, r0):
            return ((xs-x0)**2+(ys-y0)**2)<r0**2
        r0 = np.random.normal(0, 0.1)
        x0, y0 = np.random.random(2)*0.6-0.2
        return np.tile(np.expand_dims(inside_circle(xs, ys, x0, y0, r0), 2), (1, 1, 3))
    return np.where(
        inside_shape(xs, ys),
        white(xs, ys),
        background_pattern(xs, ys)
    )

def plot(*args, **kwargs):
    plt.imshow(rgb(*args, **kwargs))
    plt.show()

if __name__=='__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Display a image_gen.rgb rgb array as an image.')
    parser.add_argument('--lx', type=int, help='image width')
    parser.add_argument('--ly', type=int, help='image height')
    args = parser.parse_args()

    plot(**vars(args))
