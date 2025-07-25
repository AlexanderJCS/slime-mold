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


@ti.kernel
def render_3d(output: ti.template(), volume: ti.template()):
    for i, j in output:
        ray = start_ray(i, j)
        output[i, j] = tm.vec3(ray * 0.5 + 0.5)
