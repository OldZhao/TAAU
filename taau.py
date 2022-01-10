import numpy as np
import cv2
np.set_printoptions(precision=10)
np.set_printoptions(suppress=True)

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
    def x(self, **kwargs):
        return self[0]
    @property
    def y(self, **kwargs):
        return self[1]
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
    @property
    def x(self, **kwargs):
        return self[0]
    @property
    def y(self, **kwargs):
        return self[1]
    @property
    def z(self, **kwargs):
        return self[2]
    @property
    def w(self, **kwargs):
        return self[3]

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
            
           
            print("sample_value:{}".format(sample_value))
            print("sample_value:{}".format(sample_value[0]))
            if sample_value.shape[0] == 4:
                return float4(sample_value)
            else:
                return sample_value


inputView = [320.0, 180.0]
outputView = [640.0, 360.0]
FrameExposureScale = 1.0
ViewportUVToInputBufferUV = float4([1.0, 1.0, 0.0, 0.0])
InputSceneColorSize = float4([inputView[0], inputView[1],  1/inputView[0], 1/inputView[1]])
InputViewSize = InputSceneColorSize
OutputViewportSize = float4([outputView[0], outputView[1],  1/outputView[0], 1/outputView[1]])
TemporalJitterPixels = float2([-0.0625, 0.31481]) # TODO:rdc中看到的值精度会不会差些？
InputViewMin = float2([0.0, 0.0])
View_ClipToPrevClip = np.array([[1.00, 1.45057E-09, 0.00, -1.37615E-08],
                                [2.51079E-09, 1.00, 0.00, -2.23517E-08],
                                [0.00, 0.00, 1.00, 0.00],
                                [1.13967E-09, -8.94070E-08, 0.00, 1.00]], dtype=np.float32) # 当前clip到上一帧clip的变换矩阵


input = ReadExr("input_rgb.exr", "input")
taau_output = ReadExr("taau_output.exr", "input")
history = ReadExr("history.exr", "input")
velocity = ReadExr("velocity.exr", "velocity")
depth = ReadExr("depthz.exr", "depth")

SceneDepthTexture = Texture2D(depth)
SceneDepthTextureSampler = SamplerState("SF_point")

GBufferVelocityTexture = Texture2D(velocity)
GBufferVelocityTextureSampler = SamplerState("SF_point")

# struct FTAAHistoryPayload
# {
# 	// Transformed scene color and alpha channel.
# 	float4 Color;

# 	// Radius of the circle of confusion for DOF.
# 	float CocRadius;
# };

def  DecodeVelocityFromTexture(EncodedV:float4)->float3:
    InvDiv = 1.0 / (0.499 * 0.5)
    print("InvDiv:{}".format(InvDiv))

    
    xy_ = EncodedV.xy * InvDiv - 32767.0 / 65535.0 * InvDiv
    print("xy_:{}".format(xy_))


    z_ = np.float32((np.uint32(np.round(EncodedV.z * 65535.0)) << 16) | np.uint32(np.round(EncodedV.w * 65535.0)))

    V = float3([xy_[0], xy_[1], z_])

    return V



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

def SampleCachedSceneColorTexture(InputParams, PixelOffset:int2) -> float:
	return SceneDepthTexture.SampleLevel(SceneDepthTextureSampler, InputParams.NearestBufferUV, 0, PixelOffset)

kOffsets3x3 =[int2(-1, -1),
            int2( 0, -1),
            int2( 1, -1),
            int2(-1, 0),
            int2( 0, 0),
            int2( 1, 0),
            int2(-1, 1),
            int2( 0, 1),
            int2( 1, 1)]





def ComputeNeighborhoodBoundingbox(InputParams, IntermediaryResult, OutNeighborMin, OutNeighborMax):
    kNeighborsCount = 9
    Neighbors = [dotdict({}) for i in range(kNeighborsCount)]
    
    for i in range(kNeighborsCount):
        Neighbors[i].Color = SampleCachedSceneColorTexture(InputParams, kOffsets3x3[i]).Color
        Neighbors[i].CocRadius = SampleCachedSceneColorTexture(InputParams, kOffsets3x3[i]).CocRadius

    FTAAHistoryPayload NeighborMin;
    FTAAHistoryPayload NeighborMax;


    {
    float2 PPCo = InputParams.ViewportUV * InputViewSize.xy + TemporalJitterPixels;
    float2 PPCk = floor(PPCo) + 0.5;
    float2 dKO = PPCo - PPCk;


    NeighborMin = Neighbors[4];
    NeighborMax = Neighbors[4];


    float DistthresholdLerp = UpscaleFactor - 1;
    float DistThreshold = lerp(1.51, 1.3, DistthresholdLerp);


    const uint Indexes[9] = kSquareIndexes3x3;




    [unroll]
    for( uint i = 0; i <  9 ; i++ )
    {
    uint NeightborId = Indexes[i];
    if (NeightborId != 4)
    {
    float2 dPP = float2(kOffsets3x3[NeightborId]) - dKO;

    [flatten]
    if (dot(dPP, dPP) < (DistThreshold * DistThreshold))
    {
        NeighborMin = MinPayload(NeighborMin, Neighbors[NeightborId]);
        NeighborMax = MaxPayload(NeighborMax, Neighbors[NeightborId]);
    }
    }
    }
    }

    OutNeighborMin = NeighborMin;
    OutNeighborMax = NeighborMax;


def TemporalAASample(x, y):


    InputParams = dotdict({})
    DispatchThreadId =float2([x, y])
    # print("xy:{} zw:{}".format(OutputViewportSize.xy, OutputViewportSize.zw))
    ViewportUV = (DispatchThreadId + 0.5) * OutputViewportSize.zw
    InputParams.ViewportUV = ViewportUV
    InputParams.ScreenPos = ViewportUVToScreenPos(ViewportUV)
    print("InputParams.ScreenPos:{}".format(InputParams.ScreenPos))
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

    
    

    #### 选取中心点及附近4点最大的深度值作为该点深度值
    # PrecacheInputSceneDepth(InputParams) 
    z = SampleCachedSceneDepthTexture(InputParams, int2([0, 0]))
    PosN = float3([InputParams.ScreenPos.x, InputParams.ScreenPos.y, z])

    VelocityOffset = float2([0.0, 0.0])

    x = SampleCachedSceneDepthTexture(InputParams, int2([- 1 , - 1] ))
    y = SampleCachedSceneDepthTexture(InputParams, int2( [ 1 , - 1] ))
    z = SampleCachedSceneDepthTexture(InputParams, int2([-1 ,  1] ))
    w = SampleCachedSceneDepthTexture(InputParams, int2( [ 1 ,  1 ]))
    Depths = float4([x, y, z, w])
    print("Depths:{}".format(Depths))

    # DepthOffset = float2( 1 ,  1)
    DepthOffset_x = 1
    DepthOffset_y = 1
    DepthOffsetXx = 1
    if(Depths.x > Depths.y):
        DepthOffsetXx = - 1
    if(Depths.z > Depths.w):
        DepthOffset_x = - 1
    DepthsXY = max(Depths.x, Depths.y)
    DepthsZW = max(Depths.z, Depths.w)
    if(DepthsXY > DepthsZW):
        DepthOffset_y = - 1
        DepthOffset_x = DepthOffsetXx
    DepthsXYZW = max(DepthsXY, DepthsZW)
    DepthOffset = float2([DepthOffset_x, DepthOffset_y])
    print("DepthsXYZW:{}".format(DepthsXYZW))
    if(DepthsXYZW > PosN.z): # 周围4个点的深度大于中间
        VelocityOffset = float2(DepthOffset * InputSceneColorSize.zw)
        # PosN.z = DepthsXYZW
        PosN = float3([PosN.x, PosN.y, DepthsXYZW])

    print("VelocityOffset:{} PosN:{}".format(VelocityOffset, PosN))
    

    OffScreen = False
    Velocity = 0
    HistoryBlur = 0
    HistoryScreenPosition = InputParams.ScreenPos


    ThisClip = float4( [PosN.x, PosN.y, PosN.z, 1 ])
    # PrevClip = View_ClipToPrevClip @ ThisClip
    # print("PrevClip:{}".format(PrevClip))
    PrevClip = ThisClip.T @ View_ClipToPrevClip
    print("PrevClip:{} {}".format(PrevClip, type(PrevClip)))

    PrevScreen = float2(PrevClip.xy / PrevClip.w)
    BackN = float2(PosN.xy - PrevScreen) 
    print("PosN.xy:{} PrevScreen:{} BackN:{}".format(PosN.xy, PrevScreen, BackN))



    print("InputParams.NearestBufferUV:{}".format(InputParams.NearestBufferUV))
    EncodedVelocity = GBufferVelocityTexture.SampleLevel(GBufferVelocityTextureSampler, InputParams.NearestBufferUV + VelocityOffset, 0)
    # EncodedVelocity = float4([0.49993, EncodedVelocity.y, 0, 0])
    print("EncodedVelocity:{}".format(EncodedVelocity))
    DynamicN = EncodedVelocity.x > 0.0
    if(DynamicN):
        BackN = DecodeVelocityFromTexture(EncodedVelocity).xy  # 此函数存在严重的精度问题
    print("BackN:{}".format(BackN))
    BackTemp = BackN * OutputViewportSize.xy
    print("BackTemp:{}".format(BackTemp))
    
    Velocity = np.sqrt(np.dot(BackTemp, BackTemp))
    print("Velocity:{}".format(Velocity))

    HistoryScreenPosition = InputParams.ScreenPos - BackN

    HistoryScreenPosition = float2([0.00181, 0.19557])

    OffScreen = max(abs(HistoryScreenPosition.x), abs(HistoryScreenPosition.y)) >= 1.0
    print("HistoryScreenPosition:{} OffScreen:{}".format(HistoryScreenPosition, OffScreen))

        

    ComputeNeighborhoodBoundingbox(InputParams, IntermediaryResult, NeighborMin, NeighborMax)





if __name__ == "__main__":
    # input = ReadExr("output/input_rgb.exr", "input")
    # taau_output = ReadExr("output/taau_output.exr", "input")
    # history = ReadExr("output/history.exr", "input")
    # velocity = ReadExr("output/velocity.exr", "velocity")
    # depth = ReadExr("output/depth.exr", "depth")
    x, y = 54, 48
    # x, y = 87, 99
    x, y = 320, 144
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



    