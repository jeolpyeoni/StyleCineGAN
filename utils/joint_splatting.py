import torch
import numpy as np

from utils.softmax_splatting import FunctionSoftsplat


backwarp_tenGrid = {}
def backwarp(tenIn, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).cuda()
    # end

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenIn.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenIn.shape[2] - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=tenIn, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)




def joint_splatting(feature_map1, weights1, flow1,
                    feature_map2, weights2, flow2, mode="full", output_size=None):
        
    assert (feature_map1.shape == feature_map2.shape)
    assert (flow1.shape == flow2.shape)
    assert (feature_map1.shape[-2:] == flow1.shape[-2:])
    
    flow2_offset = flow2.clone().cuda()
    flow2_offset[:, 0, :, :] -= feature_map1.shape[-1]

    flow = torch.cat([flow1, flow2_offset], dim=-1)
    feature_map = torch.cat([feature_map1, feature_map2], dim=-1)
    blending_weights = torch.cat([weights1, weights2], dim=-1)
    
    if mode == "full":
        result_softsplat = FunctionSoftsplat(tensorInput=feature_map,
                                             tensorFlow=flow,
                                             tensorMetric=blending_weights,
                                             strType='linear',
                                             output_size=output_size)
        
    elif mode == "no_forward_warping":
        
        result_softsplat = backwarp(feature_map, flow)
        if output_size is not None:
            result_softsplat = result_softsplat[:, :, :output_size[0], :output_size[1]]

    
    return result_softsplat

