import numpy as np
import taichi as ti
import taichi.math as tm

import config


def normalize(v):
    return v / np.linalg.norm(v)


look_at = tm.vec3(0.0, 0.0, 0.0)
camera_pos = tm.vec3(0.0, 0.0, 5.0)
fov = np.radians(45.0)

camera_w = tm.vec3(normalize(camera_pos - look_at))
# noinspection PyUnreachableCode
camera_u = tm.vec3(normalize(np.cross(tm.vec3(0.0, 1.0, 0.0), camera_w)))
# noinspection PyUnreachableCode
camera_v = tm.vec3(normalize(np.cross(camera_w, camera_u)))

camera_theta = fov / 2.0
camera_half_height = np.tan(camera_theta)
camera_half_width = camera_half_height * config.SIZE[0] / config.SIZE[1]


@ti.func
def start_ray(x: int, y: int) -> tm.vec3:
    horizontal = 2 * camera_half_width * camera_u
    vertical = 2 * camera_half_height * camera_v
    
    u_s = (x + 0.5) / config.SIZE[0]
    v_s = (y + 0.5) / config.SIZE[1]
    
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


@ti.kernel
def render_3d(output: ti.template(), volume: ti.template()):
    for i, j in output:
        ray_origin = camera_pos
        ray_dir = start_ray(i, j)
        slab = unit_cube_intersection(ray_origin, ray_dir)
        cube_t_min, cube_t_max = slab.x, slab.y
        
        if cube_t_min > cube_t_max:
            output[i, j] = tm.vec3(0.0, 0.0, 0.0)
        else:
            output[i, j] = tm.vec3(cube_t_max - cube_t_min)
