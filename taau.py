import numpy as np
import cv2

import OpenEXR
import Imath

from utils import *


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

class float3(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls).astype(np.float32)
        return obj
    @property
    def xy(self, **kwargs):
        return self[:2]
    @property
    def z(self, **kwargs):
        return self[2]

class float4(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls).astype(np.float32)
        return obj
    @property
    def xy(self, **kwargs):
        return self[:2]
    @property
    def zw(self, **kwargs):
        return self[2:4]

class int2(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls).astype(np.int32)
        return obj
    @property
    def x(self, **kwargs):
        return self[0]
    @property
    def y(self, **kwargs):
        return self[1]


class SamplerState():
    def __init__(self, filter) -> None:
        self.filter = filter

class Texture2D():
    def __init__(self, texture:np.ndarray) -> None:
        self.Texture = texture
        self.T_w = texture.shape[1]
        self.T_h = texture.shape[0]
    
    def SampleLevel(self, sampler:SamplerState, UV:float2, mip_level=0, pixelOffset=int2([0,0])):
        if sampler.filter == "SF_point":
            x = int(round(UV.x * self.T_w - 0.5) + pixelOffset.x) # -0.5 从像素面积中心坐标，换算成矩阵坐标
            y = int(round(UV.y * self.T_h - 0.5) + pixelOffset.y)
            print("x:{} y:{} float_x:{} float_y:{}".format(x, y, UV.x * self.T_w- 0.5, UV.y * self.T_h- 0.5))
            sample_value =  self.Texture[y, x]
            if sample_value.shape[0] == 4:
                return float4(sample_value)
            else:
                return sample_value

input = ReadExr("input_rgb.exr", "input")
taau_output = ReadExr("taau_output.exr", "input")
history = ReadExr("history.exr", "input")
velocity = ReadExr("velocity.exr", "velocity")
depth = ReadExr("depthz.exr", "depth")

SceneDepthTexture = Texture2D(depth)
SceneDepthTextureSampler = SamplerState("SF_point")


def ViewportUVToScreenPos(ViewportUV):
    return float2([2 * ViewportUV.x - 1, 1 - 2 * ViewportUV.y])

def trunc(x):
    """
    只保留整数部分
    """
    return [int(x[0]), int(x[1])]


def CreateIntermediaryResult():
    IntermediaryResult = dotdict({})
    IntermediaryResult.FilteredTemporalWeight = 1
    IntermediaryResult.InvFilterScaleFactor = 1
    return IntermediaryResult

def SampleCachedSceneDepthTexture(InputParams, PixelOffset:int2) -> float:
	return SceneDepthTexture.SampleLevel(SceneDepthTextureSampler, InputParams.NearestBufferUV, 0, PixelOffset)[0]


def TemporalAASample(x, y):
    inputView = [320.0, 180.0]
    outputView = [640.0, 360.0]
    FrameExposureScale = 1.0
    ViewportUVToInputBufferUV = float4([1.0, 1.0, 0.0, 0.0])
    InputSceneColorSize = float4([inputView[0], inputView[1],  1/inputView[0], 1/inputView[1]])
    InputViewSize = InputSceneColorSize
    OutputViewportSize = float4([outputView[0], outputView[1],  1/outputView[0], 1/outputView[1]])
    TemporalJitterPixels = float2([-0.0625, 0.31481]) # TODO:rdc中看到的值精度会不会差些？
    InputViewMin = float2([0.0, 0.0])



    InputParams = dotdict({})
    DispatchThreadId =float2([x, y])
    # print("xy:{} zw:{}".format(OutputViewportSize.xy, OutputViewportSize.zw))
    ViewportUV = (DispatchThreadId + 0.5) * OutputViewportSize.zw
    InputParams.ViewportUV = ViewportUV
    InputParams.ScreenPos = ViewportUVToScreenPos(ViewportUV)
    InputParams.NearestBufferUV = ViewportUV * ViewportUVToInputBufferUV.xy + ViewportUVToInputBufferUV.zw

    kResponsiveStencilMask = np.uint32(1 << 3)

    SceneStencilUV = int2(trunc(InputParams.NearestBufferUV * InputSceneColorSize.xy))
    # SceneStencilRef = StencilTexture.Load(int3(SceneStencilUV, 0)).g
    SceneStencilRef = np.uint32(0.0) # TODO: StencilTexture.Load
    InputParams.bIsResponsiveAAPixel = 1.0 if (SceneStencilRef & kResponsiveStencilMask) else 0.0

    PPCo = float2(ViewportUV * InputViewSize.xy + TemporalJitterPixels)
    PPCk = float2(np.floor(PPCo) + 0.5)
    PPCt = float2(np.floor(PPCo - 0.5) + 0.5)
    InputParams.NearestBufferUV = InputSceneColorSize.zw * (InputViewMin + PPCk)
    InputParams.NearestTopLeftBufferUV = InputSceneColorSize.zw * (InputViewMin + PPCt)

    IntermediaryResult = CreateIntermediaryResult()

    
    

    # PrecacheInputSceneDepth(InputParams) 
    z = SampleCachedSceneDepthTexture(InputParams, int2([0, 0]))
    PosN = float3([InputParams.ScreenPos.x, InputParams.ScreenPos.y, z])

    VelocityOffset = float2([0.0, 0.0])

    x = SampleCachedSceneDepthTexture(InputParams, int2([- 1 , - 1] ))
    y = SampleCachedSceneDepthTexture(InputParams, int2( [ 1 , - 1] ))
    z = SampleCachedSceneDepthTexture(InputParams, int2([-1 ,  1] ))
    w = SampleCachedSceneDepthTexture(InputParams, int2( [ 1 ,  1 ]))
    Depths = float4([x, y, z, w])
    print(Depths)

    DepthOffset = float2( 1 ,  1 )


    #     if(Depths.x > Depths.y)
    #     {
    #         DepthOffsetXx = - 1 ;
    #     }
    #     if(Depths.z > Depths.w)
    #     {
    #         DepthOffset.x = - 1 ;
    #     }
    #     float DepthsXY = max(Depths.x, Depths.y);
    #     float DepthsZW = max(Depths.z, Depths.w);
    #     if(DepthsXY > DepthsZW)
    #     {
    #         DepthOffset.y = - 1 ;
    #         DepthOffset.x = DepthOffsetXx;
    #     }
    #     float DepthsXYZW = max(DepthsXY, DepthsZW);
    #     if(DepthsXYZW > PosN.z)
    #     {



    #         VelocityOffset = DepthOffset * InputSceneColorSize.zw;


    #         PosN.z = DepthsXYZW;
    #     }








if __name__ == "__main__":
    # input = ReadExr("output/input_rgb.exr", "input")
    # taau_output = ReadExr("output/taau_output.exr", "input")
    # history = ReadExr("output/history.exr", "input")
    # velocity = ReadExr("output/velocity.exr", "velocity")
    # depth = ReadExr("output/depth.exr", "depth")
    x, y = 54, 48
    # x, y = 87, 99
    TemporalAASample(x, y)

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



    