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

# class float4(np.ndarray):
#     def __init__(self, x, y, z, w):
#         self.val = np.array((x, y, z, w), dtype=np.float32)
#         self.xy = self.val[:2]
#         self.zw = self.val[-2:]

# class float2(np.ndarray):
#     def __init__(self, x, y):
#         self.val = np.array((x, y), dtype=np.float32)
#         self.xy = self.val[:2]
#     def __call__(self):
#         pass

# class float4(np.ndarray):
#     def __init__(self, x, y, z, w):
#         # super(np.ndarray, self).__init__(data=(x,y,z,w), dtype=np.float32)
#         # super(np.ndarray, self).__new__(np.array((x, y, z, w), dtype=np.float32))
#         pass

# class float2(np.ndarray):
#     def __init__(self, x, y):
#         self.val = np.array((x, y), dtype=np.float32)
#         self.xy = self.val[:2]
#     def __call__(self):
#         pass
    
# def float4(x, y, z, w):
#     return np.array((x, y, z, w), dtype=np.float32)

# def float2(x, y):
#     return np.array((x, y), dtype=np.float32)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def ViewportUVToScreenPos(ViewportUV):
    return float2([2 * ViewportUV.x - 1, 1 - 2 * ViewportUV.y])


class float2(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls).astype(np.float32)
        return obj
    @property
    def x(self, **kwargs):
        return self[0]
    @property
    def y(self, **kwargs):
        return self[1]
    @property
    def xy(self, **kwargs):
        return self

class float4(float2):
    @property
    def zw(self, **kwargs):
        return self[2:4]


def TemporalAASample(x, y, cur, prev, depth, velocity):
    inputView = [320.0, 180.0]
    outputView = [640.0, 360.0]

    OutputViewportSize = float4([outputView[0], outputView[1],  1/outputView[0], 1/outputView[1]])
    FrameExposureScale = 1.0

    InputParams = dotdict({})

    DispatchThreadId =float2([x, y])
    print("xy:{} zw:{}".format(OutputViewportSize.xy, OutputViewportSize.zw))
    ViewportUV = (DispatchThreadId + 0.5) * OutputViewportSize.zw
    InputParams.ViewportUV = ViewportUV
    InputParams.ScreenPos = ViewportUVToScreenPos(ViewportUV)




if __name__ == "__main__":
    input = ReadExr("input_rgb.exr", "input")
    taau_output = ReadExr("taau_output.exr", "input")
    history = ReadExr("history.exr", "input")
    velocity = ReadExr("velocity.exr", "velocity")
    depth = ReadExr("depthz.exr", "depth")

    # input = ReadExr("output/input_rgb.exr", "input")
    # taau_output = ReadExr("output/taau_output.exr", "input")
    # history = ReadExr("output/history.exr", "input")
    # velocity = ReadExr("output/velocity.exr", "velocity")
    # depth = ReadExr("output/depth.exr", "depth")
    x, y = 54, 48
    TemporalAASample(x, y, input, history, depth, velocity)

    # print("input.shape:{}, input.dtype:{}".format(input.shape, input.dtype))
    # print("taau_output.shape:{}, taau_output.dtype:{}".format(taau_output.shape, taau_output.dtype))
    # print("history.shape:{}, history.dtype:{}".format(history.shape, history.dtype))
    # print("velocity.shape:{}, velocity.dtype:{}".format(velocity.shape, velocity.dtype))
    # print("depth.shape:{}, depth.dtype:{}".format(depth.shape, depth.dtype))

    # WriteEXR(input, "output/input_rgb.exr", "input")
    # WriteEXR(taau_output, "output/taau_output.exr", "input")
    # WriteEXR(history, "output/history.exr", "input")
    # WriteEXR(velocity, "output/velocity.exr", "input")
    # WriteEXR(depth, "output/depth.exr", "depth")



    