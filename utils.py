import numpy as np
import cv2

import OpenEXR
import Imath

def WriteEXR(color, exr_name, layer="input"):
    if layer == 'depth':
        color = np.squeeze(color[:,:,0])
        type_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        HEADER = OpenEXR.Header(color.shape[1], color.shape[0])
        HEADER['channels'] = dict([(c, type_chan) for c in "R"])
        exr = OpenEXR.OutputFile(exr_name, HEADER)
        exr.writePixels({"R": np.squeeze(color)/1.0})
        exr.close()
    elif layer in ["input", "output", "velocity"]:
        type_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        HEADER = OpenEXR.Header(color.shape[1], color.shape[0])
        HEADER['channels'] = dict([(c, type_chan) for c in "RGBA"])
        exr = OpenEXR.OutputFile(exr_name, HEADER)
        exr.writePixels({"R": np.squeeze(color[:,:,0]/1.0), "G":np.squeeze(color[:,:,1]/1.0), "B":np.squeeze(color[:,:,2]/1.0), "A":np.squeeze(color[:,:,3]/1.0)})
        exr.close()

def ReadExr(path, file_type="input"):
    exrfile = OpenEXR.InputFile(path)
    header = exrfile.header()
    (r, g, b, a) = exrfile.channels("RGBA")
    r = np.frombuffer(r)
    # print(r.shape)
    # print("file:{} {}".format(path, header))

    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()
    
    # convert all channels in the image to numpy arrays
    for c in header['channels']:
        if file_type == "depth":
            C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
            C = np.frombuffer(C, dtype=np.float32)
        else:
            C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.HALF))
            C = np.frombuffer(C, dtype=np.float16)
        C = np.reshape(C, isize)
        channelData[c] = C
    
    if file_type == "input":
        colorChannels = ['R', 'G', 'B', 'A']
    elif file_type == "velocity":
        colorChannels = ['R', 'G', 'B', 'A']
    elif file_type == "depth":
        colorChannels = ["R"]
    img = np.concatenate([channelData[c][...,np.newaxis] for c in colorChannels], axis=2)
    return img


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
