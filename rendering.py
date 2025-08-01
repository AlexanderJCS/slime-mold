import numpy as np
import taichi as ti
import taichi.math as tm

import config
import cmap


def normalize(v):
    return v / np.linalg.norm(v)


camera_pos = ti.Vector.field(3, dtype=ti.f32, shape=())  # mutable scalar vec3
look_at = ti.Vector.field(3, dtype=ti.f32, shape=())
fov = ti.field(dtype=ti.f32, shape=())


@ti.func
def get_camera_basis():
    w = (camera_pos[None] - look_at[None]).normalized()
    up = tm.vec3(0.0, 1.0, 0.0)
    if abs(w.dot(up)) > 0.999:
        up = tm.vec3(0.0, 0.0, 1.0)
    u = up.cross(w).normalized()
    v = w.cross(u)
    return u, v, w


@ti.func
def start_ray(x: int, y: int) -> tm.vec3:
    u, v, w = get_camera_basis()
    aspect = config.RESOLUTION[0] / config.RESOLUTION[1]
    half_height = ti.tan(fov[None] / 2.0)
    half_width = half_height * aspect

    pixel_u = (x + 0.5) / config.RESOLUTION[0]
    pixel_v = (y + 0.5) / config.RESOLUTION[1]

    horizontal = 2 * half_width * u
    vertical = 2 * half_height * v
    lower_left = camera_pos[None] - half_width * u - half_height * v - w

    return (lower_left + pixel_u * horizontal + pixel_v * vertical - camera_pos[None]).normalized()


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


cube_size = ti.Vector.field(3, dtype=ti.f32, shape=(), needs_grad=False)
half_size = ti.Vector.field(3, dtype=ti.f32, shape=(), needs_grad=False)

cube_size[None] = tm.vec3(1.0, 2.3, 1.0)
half_size[None] = cube_size[None] * 0.5


@ti.func
def cube_intersection(orig, dir):
    t_min = 0.001
    t_max = 1e5
    H = half_size[None]
    
    # X slab
    slab = _intersect_slab(orig.x, dir.x, -H.x, H.x, t_min, t_max)
    t_min, t_max = slab.x, slab.y
    
    # Y slab
    slab = _intersect_slab(orig.y, dir.y, -H.y, H.y, t_min, t_max)
    t_min, t_max = slab.x, slab.y
    
    # Z slab
    slab = _intersect_slab(orig.z, dir.z, -H.z, H.z, t_min, t_max)
    t_min, t_max = slab.x, slab.y
    return tm.vec2(t_min, t_max)


@ti.func
def wrap(idx, dim):
    return (idx + dim) % dim


@ti.func
def sample_volume(volume, pos):
    grid_size = tm.ivec3(volume.shape)
    p = pos * (grid_size - 1)
    i0 = tm.floor(p).cast(int)
    f = p - i0.cast(float)
    i0x, i0y, i0z = wrap(i0.x, grid_size.x), wrap(i0.y, grid_size.y), wrap(i0.z, grid_size.z)
    i1x, i1y, i1z = wrap(i0.x + 1, grid_size.x), wrap(i0.y + 1, grid_size.y), wrap(i0.z + 1, grid_size.z)
    c000 = volume[i0x, i0y, i0z]
    c100 = volume[i1x, i0y, i0z]
    c010 = volume[i0x, i1y, i0z]
    c001 = volume[i0x, i0y, i1z]
    c110 = volume[i1x, i1y, i0z]
    c101 = volume[i1x, i0y, i1z]
    c011 = volume[i0x, i1y, i1z]
    c111 = volume[i1x, i1y, i1z]
    c00 = c000*(1-f.x) + c100*f.x
    c01 = c001*(1-f.x) + c101*f.x
    c10 = c010*(1-f.x) + c110*f.x
    c11 = c011*(1-f.x) + c111*f.x
    c0 = c00*(1-f.y) + c10*f.y
    c1 = c01*(1-f.y) + c11*f.y
    return c0*(1-f.z) + c1*f.z


@ti.kernel
def render_3d(
        output: ti.template(), volume: ti.template(), gradient_image: ti.template(),
        old_cmap: ti.template(), new_cmap: ti.template(), time: float
):
    for i in ti.grouped(output):
        # initialize ray
        ray_o = camera_pos[None]
        ray_d = start_ray(i.x, i.y)
        t_min, t_max = cube_intersection(ray_o, ray_d)
        if t_min > t_max:
            output[i] = tm.vec3(0)
            continue

        # march through the cube
        dist = t_max - t_min
        num_samples = ti.static(256)
        dt = dist / num_samples

        col = tm.vec3(0.0)
        alpha_acc = 0.0

        for s in range(num_samples):
            if alpha_acc > 0.99:
                break  # early exit once almost opaque

            t = t_min + (s + 0.5) * dt  # midpoint sampling

            p = ray_o + t * ray_d
            local = p + half_size[None]
            uvw = tm.fract(local)

            if any(uvw < 0.0) or any(uvw > 1.0):
                continue
                
            density = tm.max(sample_volume(volume, uvw), 0.0)
            gradient = tm.normalize(sample_volume(gradient_image, uvw))
            gradient_max = max(abs(gradient).x, abs(gradient).y, abs(gradient).z)
            
            # Beer–Lambert for opacity
            sigma = 2.0  # absorption coefficient
            sample_a = 1.0 - tm.exp(-sigma * density * dt)
            sample_col = tm.vec3(pow(density, 0.6))  # gamma‐corrected white ramp
            weight = (1 - alpha_acc) * sample_a
            
            old_color = cmap.interp_cmap(old_cmap, tm.clamp(gradient_max, 0.0001, 0.9999))
            new_color = cmap.interp_cmap(new_cmap, tm.clamp(gradient_max, 0.0001, 0.9999))
            
            color = tm.mix(old_color, new_color, time)
            
            col += weight * sample_col * color  # apply gradient and color mapping
            
            alpha_acc += weight

        # Reinhard tonemapping
        col = col / (1.0 + col)
        output[i] = col
