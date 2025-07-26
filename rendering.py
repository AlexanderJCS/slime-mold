import numpy as np
import taichi as ti
import taichi.math as tm

import config


def normalize(v):
    return v / np.linalg.norm(v)


look_at = tm.vec3(0.0, 0.0, 0.0)
camera_pos = tm.vec3(0.0, 0.0, 2.0)
fov = np.radians(45.0)

world_up = tm.vec3(0.0, 1.0, 0.0)
camera_w = tm.vec3(normalize(camera_pos - look_at))

if abs(camera_w.dot(world_up)) > 0.999:
    world_up = tm.vec3(0.0, 0.0, 1.0)

# noinspection PyUnreachableCode
camera_u = tm.vec3(normalize(np.cross(world_up, camera_w)))
# noinspection PyUnreachableCode
camera_v = tm.vec3(normalize(np.cross(camera_w, camera_u)))

camera_theta = fov / 2.0
camera_half_height = np.tan(camera_theta)
camera_half_width = camera_half_height * config.RESOLUTION[0] / config.RESOLUTION[1]


@ti.func
def start_ray(x: int, y: int) -> tm.vec3:
    horizontal = 2 * camera_half_width * camera_u
    vertical = 2 * camera_half_height * camera_v
    
    u_s = (x + 0.5) / config.RESOLUTION[0]
    v_s = (y + 0.5) / config.RESOLUTION[1]
    
    lower_left = camera_pos - camera_half_width * camera_u - camera_half_height * camera_v - camera_w
    
    return tm.normalize(lower_left + u_s * horizontal + v_s * vertical - camera_pos)


@ti.func
def _intersect_slab(
        orig_comp: float, dir_comp: float,
        min_b: float, max_b: float,
        t_min: float, t_max: float
) -> tm.vec2:
    """Intersect one axis-aligned slab and return updated (t_min, t_max) as a vec2"""
    inv_d = 1.0 / dir_comp
    t0 = (min_b - orig_comp) * inv_d
    t1 = (max_b - orig_comp) * inv_d
    
    if t0 > t1:
        t0, t1 = t1, t0
    
    t_min = max(t_min, t0)
    t_max = min(t_max, t1)
    
    return tm.vec2(t_min, t_max)


@ti.func
def unit_cube_intersection(
        ray_origin: tm.vec3,
        ray_dir: tm.vec3
) -> tm.vec2:
    """Calculate (t_min, t_max) entry/exit distances for unit cube [-0.5, 0.5]^3"""
    t_min = 0.001
    t_max = 10000.0

    # X slab
    slab = _intersect_slab(ray_origin.x, ray_dir.x, -0.5, 0.5, t_min, t_max)
    t_min, t_max = slab.x, slab.y
    
    # Y slab
    slab = _intersect_slab(ray_origin.y, ray_dir.y, -0.5, 0.5, t_min, t_max)
    t_min, t_max = slab.x, slab.y
    
    # Z slab
    slab = _intersect_slab(ray_origin.z, ray_dir.z, -0.5, 0.5, t_min, t_max)
    t_min, t_max = slab.x, slab.y

    return tm.vec2(t_min, t_max)


@ti.func
def sample_volume(volume, pos):
    # pos in [0,1]^3; trilinear interpolation
    p = pos * (tm.ivec3(volume.shape) - 1)
    i0 = tm.floor(p).cast(int)
    f = p - i0.cast(float)
    c000 = volume[i0]
    c100 = volume[i0 + tm.ivec3(1, 0, 0)]
    c010 = volume[i0 + tm.ivec3(0, 1, 0)]
    c001 = volume[i0 + tm.ivec3(0, 0, 1)]
    c110 = volume[i0 + tm.ivec3(1, 1, 0)]
    c101 = volume[i0 + tm.ivec3(1, 0, 1)]
    c011 = volume[i0 + tm.ivec3(0, 1, 1)]
    c111 = volume[i0 + tm.ivec3(1, 1, 1)]
    c00 = c000*(1-f.x) + c100*f.x
    c01 = c001*(1-f.x) + c101*f.x
    c10 = c010*(1-f.x) + c110*f.x
    c11 = c011*(1-f.x) + c111*f.x
    c0 = c00*(1-f.y) + c10*f.y
    c1 = c01*(1-f.y) + c11*f.y
    return c0*(1-f.z) + c1*f.z


@ti.kernel
def render_3d(output: ti.template(), volume: ti.template()):
    for i in ti.grouped(output):
        # initialize ray
        ray_o = camera_pos
        ray_d = start_ray(i.x, i.y)
        t_min, t_max = unit_cube_intersection(ray_o, ray_d)
        if t_min > t_max:
            output[i] = tm.vec3(0.1)
            continue

        # march through the cube
        dist = t_max - t_min
        num_samples = ti.static(128)
        dt = dist / num_samples

        col = tm.vec3(0.0)
        alpha_acc = 0.0

        for s in range(num_samples):
            if alpha_acc > 0.99:
                break  # early exit once almost opaque

            t = t_min + (s + 0.5) * dt  # midpoint sampling

            uvw = ray_o + t * ray_d + 0.5  # location of the sample in [0,1]^3

            if 0.0 <= uvw.x < 1 and 0.0 <= uvw.y < 1 and 0.0 <= uvw.z < 1:
                density = sample_volume(volume, uvw)
                # Beer–Lambert for opacity
                sigma = 1.0  # absorption coefficient
                sample_a = 1.0 - tm.exp(-sigma * density * dt)
                sample_col = tm.vec3(pow(density, 0.6))  # gamma‐corrected white ramp
                weight = (1 - alpha_acc) * sample_a
                col += weight * sample_col
                alpha_acc += weight

        output[i] = tm.clamp(col, 0.0, 1.0)
