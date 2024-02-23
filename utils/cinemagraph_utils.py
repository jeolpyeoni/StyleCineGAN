import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.joint_splatting import joint_splatting, backwarp


def euler_integration(motion, destination_frame, return_all_frames=False):
    """
    Repeatedly integrates the Eulerian motion field to get the displacement map to a future frame.

    :param motion: The Eulerian motion field to be integrated.
    :param destination_frame: The number of times the motion field should be integrated.
    :param return_all_frames: Return the displacement maps to all intermediate frames, not only the last frame.
    :return: The displacement map resulting from repeated integration of the motion field.
    """

    assert (motion.dim() == 4)
    b, c, height, width = motion.shape
    assert (b == 1), 'Function only implemented for batch = 1'
    assert (c == 2), f'Input motion field should be Bx2xHxW. Given tensor is: {motion.shape}'

    y, x = torch.meshgrid(
        [torch.linspace(0, height - 1, height, device='cuda'),
         torch.linspace(0, width - 1, width, device='cuda')])
    coord = torch.stack([x, y], dim=0).long()

    destination_coords = coord.clone().float().cuda()
    motion = motion.cuda()

    if return_all_frames:
        displacements = torch.zeros(destination_frame + 1, 2, height, width, device='cuda')
        visible_pixels = torch.ones(b + 1, 1, height, width, device='cuda')

    else:
        displacements = torch.zeros(1, 2, height, width, device='cuda')
        visible_pixels = torch.ones(1, 1, height, width, device='cuda')

    invalid_mask = torch.zeros(1, height, width, device='cuda').bool()

    for frame_id in range(1, destination_frame + 1):
        destination_coords = destination_coords + motion[0][:, torch.round(destination_coords[1]).long(),
                                                  torch.round(destination_coords[0]).long()]
        out_of_bounds_x = torch.logical_or(destination_coords[0] > (width - 1), destination_coords[0] < 0)
        out_of_bounds_y = torch.logical_or(destination_coords[1] > (height - 1), destination_coords[1] < 0)
        invalid_mask = torch.logical_or(out_of_bounds_x.unsqueeze(0), invalid_mask)
        invalid_mask = torch.logical_or(out_of_bounds_y.unsqueeze(0), invalid_mask)

        # Set the displacement of invalid pixels to zero, to avoid out-of-bounds access errors
        destination_coords[invalid_mask.expand_as(destination_coords)] = coord[
            invalid_mask.expand_as(destination_coords)].float()

        if return_all_frames:
            displacements[frame_id] = (destination_coords - coord.float()).unsqueeze(0)
        else:
            displacements = (destination_coords - coord.float()).unsqueeze(0)

    return displacements, visible_pixels


def pad_tensor(tensor, mode="reflect", number=None):
    cut_size = 0
    size = tensor.size(2)
    pad_size = int(size / 4) + int(size / 8) + cut_size
    pad = (pad_size, pad_size, pad_size, pad_size)

    if mode=="reflect":
        pad = torch.nn.ReflectionPad2d(pad_size)
        padded_tensor = pad(tensor)    
    elif mode=="constant":
        padded_tensor = torch.nn.functional.pad(tensor, pad, "constant", number)

    return padded_tensor


def crop_padded_tensor(padded_tensor, size):
    padded_size = padded_tensor.size(2) - size
    start_idx = int(padded_size/2)
    end_idx = start_idx + size
    
    cropped_tensor = padded_tensor[:,:,start_idx:end_idx, start_idx:end_idx]
    return cropped_tensor


def resize_feature(feature, size):
    if feature.size(2) < size:
        while feature.size(2) < size:

            up_height = feature.shape[2] * 2
            up_width = feature.shape[3] * 2

            feature = nn.functional.interpolate(feature, size=(up_height, up_width), mode='bilinear', align_corners=False)
    
    elif feature.size(2) > size:
        
        down_height = int(feature.shape[2] / 2)
        down_width = int(feature.shape[3] / 2)
        
        feature = nn.functional.interpolate(feature, size=(down_height, down_width), mode='bilinear', align_corners=False)
        
    return feature


def resize_flow(flow, size):
    flow_size = flow.size(2)
    
    while flow.size(2) != size:
        
        mode = 'downsample'
    
        if flow_size > size:
            height = int(flow.size(2) / 2)
            width = int(flow.size(3) / 2)
            
        elif flow_size < size:
            height = int(flow.size(2) * 2)
            width = int(flow.size(3) * 2)
            mode = 'upsample'
            
        flow = nn.functional.interpolate(flow, size=(height, width), mode='bilinear', align_corners=False)
        
        if mode == 'downsample':
            flow /= 2
        elif mode == 'upsample':
            flow *= 2

    return flow


def blend_feature(feature, flow, idx, n_frames):
    
    size = feature.size(2)
    pad_size = int(size / 2)
    alpha = idx / (n_frames - 1)
    
    cut_size = 0
    
    if feature.size(2) == 1024:
        cut_size = 3
    elif feature.size(2) == 512:
        cut_size = 2
    elif feature.size(2) == 256:
        cut_size = 1
        
    if not cut_size == 0:
        feature = feature[:,:,cut_size:-cut_size,cut_size:-cut_size]
        flow = flow[:,:,cut_size:-cut_size,cut_size:-cut_size]
        
    ### Reflection padding for flow
    future_flow = pad_tensor(flow, mode="reflect").float().cuda()
    past_flow = pad_tensor(-flow, mode="reflect").float().cuda()
    
    
    ## Euler integration to get optical flow fro motion field
    future_flow, _ = euler_integration(future_flow, idx)
    past_flow, _ = euler_integration(past_flow, n_frames-idx-1)
    
    
    ### Define importance metric Z
    Z = torch.ones(1, 1, size-2*cut_size, size-2*cut_size)
    future_Z = Z.float().cuda()
    future_Z = pad_tensor(future_Z, mode="reflect").float().cuda() * (1 - alpha)
    
    past_Z = Z.float().cuda()
    past_Z = pad_tensor(past_Z, mode="reflect").float().cuda() * alpha
    
    
    ### Pad feature, and get segmentation mask for feature and flow regions
    feature = pad_tensor(feature, mode="reflect").float().cuda()
    blended = joint_splatting(feature, future_Z, future_flow,
                                  feature, past_Z, past_flow,
                                  mode="full",
                                  output_size=feature.shape[-2:]
                                 )
    
    out = blended.cuda()
    return out.type(torch.float)


def warp_one_level(out, flow, idx, n_frames):
    
    orig_size = out.size(2)
    flow = resize_flow(flow, out.size(2))
    
    out = blend_feature(out, flow, idx, n_frames)
    out = feature_inpaint_conv(out, flow, idx, n_frames)
    out = crop_padded_tensor(out, orig_size)
    
    return out
    

from math import sqrt as sqrt
import heapq
import numpy as np

# flags
KNOWN = 0
BAND = 1
UNKNOWN = 2
# extremity values
INF = 1e6 # dont use np.inf to avoid inf * 0
EPS = 1e-6

# solves a step of the eikonal equation in order to find closest quadrant
def _solve_eikonal(y1, x1, y2, x2, height, width, dists, flags):
    # check image frame
    if y1 < 0 or y1 >= height or x1 < 0 or x1 >= width:
        return INF

    if y2 < 0 or y2 >= height or x2 < 0 or x2 >= width:
        return INF

    flag1 = flags[y1, x1]
    flag2 = flags[y2, x2]

    # both pixels are known
    if flag1 == KNOWN and flag2 == KNOWN:
        dist1 = dists[y1, x1]
        dist2 = dists[y2, x2]
        d = 2.0 - (dist1 - dist2) ** 2
        if d > 0.0:
            r = sqrt(d)
            s = (dist1 + dist2 - r) / 2.0
            if s >= dist1 and s >= dist2:
                return s
            s += r
            if s >= dist1 and s >= dist2:
                return s
            # unsolvable
            return INF

    # only 1st pixel is known
    if flag1 == KNOWN:
        dist1 = dists[y1, x1]
        return 1.0 + dist1

    # only 2d pixel is known
    if flag2 == KNOWN:
        dist2 = dists[y2, x2]
        return 1.0 + dist2

    # no pixel is known
    return INF

# returns gradient for one pixel, computed on 2 pixel range if possible
def _pixel_gradient(y, x, height, width, vals, flags):
    val = vals[y, x]

    # compute grad_y
    prev_y = y - 1
    next_y = y + 1
    if prev_y < 0 or next_y >= height:
        grad_y = INF
    else:
        flag_prev_y = flags[prev_y, x]
        flag_next_y = flags[next_y, x]

        if flag_prev_y != UNKNOWN and flag_next_y != UNKNOWN:
            grad_y = (vals[next_y, x] - vals[prev_y, x]) / 2.0
        elif flag_prev_y != UNKNOWN:
            grad_y = val - vals[prev_y, x]
        elif flag_next_y != UNKNOWN:
            grad_y = vals[next_y, x] - val
        else:
            grad_y = 0.0

    # compute grad_x
    prev_x = x - 1
    next_x = x + 1
    if prev_x < 0 or next_x >= width:
        grad_x = INF
    else:
        flag_prev_x = flags[y, prev_x]
        flag_next_x = flags[y, next_x]

        if flag_prev_x != UNKNOWN and flag_next_x != UNKNOWN:
            grad_x = (vals[y, next_x] - vals[y, prev_x]) / 2.0
        elif flag_prev_x != UNKNOWN:
            grad_x = val - vals[y, prev_x]
        elif flag_next_x != UNKNOWN:
            grad_x = vals[y, next_x] - val
        else:
            grad_x = 0.0

    return grad_y, grad_x

# compute distances between initial mask contour and pixels outside mask, using FMM (Fast Marching Method)
def _compute_outside_dists(height, width, dists, flags, band, radius):
    band = band.copy()
    orig_flags = flags
    flags = orig_flags.copy()
    # swap INSIDE / OUTSIDE
    flags[orig_flags == KNOWN] = UNKNOWN
    flags[orig_flags == UNKNOWN] = KNOWN

    last_dist = 0.0
    double_radius = radius * 2
    while band:
        # reached radius limit, stop FFM
        if last_dist >= double_radius:
            break

        # pop BAND pixel closest to initial mask contour and flag it as KNOWN
        _, y, x = heapq.heappop(band)
        flags[y, x] = KNOWN

        # process immediate neighbors (top/bottom/left/right)
        neighbors = [(y - 1, x), (y, x - 1), (y + 1, x), (y, x + 1)]
        for nb_y, nb_x in neighbors:
            # skip out of frame
            if nb_y < 0 or nb_y >= height or nb_x < 0 or nb_x >= width:
                continue

            # neighbor already processed, nothing to do
            if flags[nb_y, nb_x] != UNKNOWN:
                continue

            # compute neighbor distance to inital mask contour
            last_dist = min([
                _solve_eikonal(nb_y - 1, nb_x, nb_y, nb_x - 1, height, width, dists, flags),
                _solve_eikonal(nb_y + 1, nb_x, nb_y, nb_x + 1, height, width, dists, flags),
                _solve_eikonal(nb_y - 1, nb_x, nb_y, nb_x + 1, height, width, dists, flags),
                _solve_eikonal(nb_y + 1, nb_x, nb_y, nb_x - 1, height, width, dists, flags)
            ])
            dists[nb_y, nb_x] = last_dist

            # add neighbor to narrow band
            flags[nb_y, nb_x] = BAND
            heapq.heappush(band, (last_dist, nb_y, nb_x))

    # distances are opposite to actual FFM propagation direction, fix it
    dists *= -1.0

# computes pixels distances to initial mask contour, flags, and narrow band queue
def _init(height, width, mask, radius):
    # init all distances to infinity
    dists = np.full((height, width), INF, dtype=float)
    # status of each pixel, ie KNOWN, BAND or UNKNOWN
    flags = mask.astype(int) * UNKNOWN
    # narrow band, queue of contour pixels
    band = []

    mask_y, mask_x = mask.nonzero()
    for y, x in zip(mask_y, mask_x):
        # look for BAND pixels in neighbors (top/bottom/left/right)
        neighbors = [(y - 1, x), (y, x - 1), (y + 1, x), (y, x + 1)]
        for nb_y, nb_x in neighbors:
            # neighbor out of frame
            if nb_y < 0 or nb_y >= height or nb_x < 0 or nb_x >= width:
                continue

            # neighbor already flagged as BAND
            if flags[nb_y, nb_x] == BAND:
                continue

            # neighbor out of mask => mask contour
            if mask[nb_y, nb_x] == 0:
                flags[nb_y, nb_x] = BAND
                dists[nb_y, nb_x] = 0.0
                heapq.heappush(band, (0.0, nb_y, nb_x))


    # compute distance to inital mask contour for KNOWN pixels
    # (by inverting mask/flags and running FFM)
    _compute_outside_dists(height, width, dists, flags, band, radius)

    return dists, flags, band

# returns RGB values for pixel to by inpainted, computed for its neighborhood
def _inpaint_pixel(y, x, img, height, width, dists, flags, radius, dim=3):
    dist = dists[y, x]
    # normal to pixel, ie direction of propagation of the FFM
    dist_grad_y, dist_grad_x = _pixel_gradient(y, x, height, width, dists, flags)
    pixel_sum = np.zeros((dim), dtype=float)
    weight_sum = 0.0

    # iterate on each pixel in neighborhood (nb stands for neighbor)
    for nb_y in range(y - radius, y + radius + 1):
        #  pixel out of frame
        if nb_y < 0 or nb_y >= height:
            continue

        for nb_x in range(x - radius, x + radius + 1):
            # pixel out of frame
            if nb_x < 0 or nb_x >= width:
                continue

            # skip unknown pixels (including pixel being inpainted)
            if flags[nb_y, nb_x] == UNKNOWN:
                continue

            # vector from point to neighbor
            dir_y = y - nb_y
            dir_x = x - nb_x
            dir_length_square = dir_y ** 2 + dir_x ** 2
            dir_length = sqrt(dir_length_square)
            # pixel out of neighborhood
            if dir_length > radius:
                continue

            # compute weight
            # neighbor has same direction gradient => contributes more
            dir_factor = abs(dir_y * dist_grad_y + dir_x * dist_grad_x)
            if dir_factor == 0.0:
                dir_factor = EPS

            # neighbor has same contour distance => contributes more
            nb_dist = dists[nb_y, nb_x]
            level_factor = 1.0 / (1.0 + abs(nb_dist - dist))

            # neighbor is distant => contributes less
            dist_factor = 1.0 / (dir_length * dir_length_square)

            weight = abs(dir_factor * dist_factor * level_factor)

            for i in range(dim):
                pixel_sum[i] += weight * img[nb_y, nb_x, i]

            weight_sum += weight

    return pixel_sum / weight_sum


# main inpainting function
def inpaint(img, mask, radius=5, dim=3):
    if img.shape[0:2] != mask.shape[0:2]:
        raise ValueError("Image and mask dimensions do not match")

    height, width = img.shape[0:2]
    dists, flags, band = _init(height, width, mask, radius)

    # find next pixel to inpaint with FFM (Fast Marching Method)
    # FFM advances the band of the mask towards its center,
    # by sorting the area pixels by their distance to the initial contour
    while band:
        # pop band pixel closest to initial mask contour
        _, y, x = heapq.heappop(band)
        # flag it as KNOWN
        flags[y, x] = KNOWN

        # process his immediate neighbors (top/bottom/left/right)
        neighbors = [(y - 1, x), (y, x - 1), (y + 1, x), (y, x + 1)]
        for nb_y, nb_x in neighbors:
            # pixel out of frame
            if nb_y < 0 or nb_y >= height or nb_x < 0 or nb_x >= width:
                continue

            # neighbor outside of initial mask or already processed, nothing to do
            if flags[nb_y, nb_x] != UNKNOWN:
                continue

            # compute neighbor distance to inital mask contour
            nb_dist = min([
                _solve_eikonal(nb_y - 1, nb_x, nb_y, nb_x - 1, height, width, dists, flags),
                _solve_eikonal(nb_y + 1, nb_x, nb_y, nb_x + 1, height, width, dists, flags),
                _solve_eikonal(nb_y - 1, nb_x, nb_y, nb_x + 1, height, width, dists, flags),
                _solve_eikonal(nb_y + 1, nb_x, nb_y, nb_x - 1, height, width, dists, flags)
            ])
            dists[nb_y, nb_x] = nb_dist

            # inpaint neighbor
            pixel_vals = _inpaint_pixel(nb_y, nb_x, img, height, width, dists, flags, radius, dim)
            
            
            for i in range(dim):
                img[nb_y, nb_x, i] = pixel_vals[i]

            # add neighbor to narrow band
            flags[nb_y, nb_x] = BAND
            # push neighbor on band
            heapq.heappush(band, (nb_dist, nb_y, nb_x))
            
            
def feature_inpaint(feature, flow, idx, n_frames, mode="full"):
    size = feature.size(2)
    bn_feature = torch.ones(1, 1, size, size)
    flow = resize_flow(flow, size)
    
    warped_bn_feature = blend_feature(bn_feature, flow, idx, n_frames)
    warped_bn_feature = crop_padded_tensor(warped_bn_feature, size)

    blank_mask = torch.where(warped_bn_feature==0., 1., 0.).squeeze(0)
    
    if blank_mask.max() == 1.:
        feature = feature * (1 - blank_mask)
        blank_mask = blank_mask.squeeze(0).detach().cpu().numpy()
        fill_feature = feature.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().copy()
        inpaint(fill_feature, blank_mask, 5)
        output = torch.from_numpy(fill_feature).permute(2, 0, 1).unsqueeze(0).cuda()
    
    else:
        output = feature

    return output
    

def feature_inpaint_conv(feature, flow, idx, n_frames):
    size = feature.size(2)
    pad_size = int(size / 2)
    
    bn_feature = torch.ones(1, 1, flow.size(2), flow.size(3))
    warped_bn_feature = blend_feature(bn_feature, flow, idx, n_frames)
    
    if False:
        np_bn_feature = warped_bn_feature.detach().cpu().numpy()
        blank_mask = np.where(np_bn_feature==0, 1, 0)
        blank_mask = torch.from_numpy(blank_mask).cuda()
    else:
        blank_mask = torch.where(warped_bn_feature==0, 1, 0)
    
    full_mask = 1 - blank_mask

    if blank_mask.max() == 1.:

        ### make kernel
        weights = torch.ones(1, 1, 7, 7).cuda() / 49
        filtered = torch.zeros_like(feature)

        for i in range(feature.size(1)):
            feat_channel = feature[:,i,:,:].unsqueeze(0)
            feat_filtered = F.conv2d(feat_channel, weights, padding=3)
            filtered[:,i,:,:] = feat_filtered
    
        ### Apply mask
        output = blank_mask*filtered + full_mask*feature
    
    else:
        output = feature
    
    return output