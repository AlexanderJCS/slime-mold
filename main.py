import config
import rendering

import colorsys

import numpy as np
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)


agents_grid = ti.field(dtype=ti.f32, shape=config.GRID_SIZE)
agents_grid_temp = ti.field(dtype=ti.f32, shape=config.GRID_SIZE)
render_img = ti.field(dtype=tm.vec3, shape=config.RESOLUTION)


Agent = ti.types.struct(
    position=ti.types.vector(3, ti.f32),
    basis_x=ti.types.vector(3, ti.f32),
    basis_y=ti.types.vector(3, ti.f32),
    basis_z=ti.types.vector(3, ti.f32)
)

agents = Agent.field(shape=(config.AGENT_COUNT,))

sigma = 0.1
RADIUS = int(np.ceil(sigma * 3))

# precompute 1D Gaussian weights
weights_cpu = np.array([np.exp(-0.5 * (i / sigma) ** 2) for i in range(-RADIUS, RADIUS + 1)], dtype=np.float32)
weights_cpu /= np.sum(weights_cpu)

weights = ti.field(dtype=ti.f32, shape=len(weights_cpu))
weights.from_numpy(weights_cpu)


@ti.func
def orthonormalize(x: tm.vec3, y: tm.vec3, z: tm.vec3):
    x = x.normalized()
    y = (y - x*(x.dot(y))).normalized()
    z = x.cross(y)
    
    return x, y, z


@ti.kernel
def gen_agents():
    for i in range(config.AGENT_COUNT):
        # 1) random pos
        pos = tm.vec3(
            ti.random() * config.GRID_SIZE[0],
            ti.random() * config.GRID_SIZE[1],
            ti.random() * config.GRID_SIZE[2]
        )
        agents[i].position = pos

        # 2) random unit quaternion
        u1, u2, u3, u4 = ti.random(), ti.random(), ti.random(), ti.random()
        q = tm.normalize(tm.vec4(
            tm.sqrt(-2.0 * ti.log(u1)) * ti.cos(2*tm.pi * u2),
            tm.sqrt(-2.0 * ti.log(u1)) * ti.sin(2*tm.pi * u2),
            tm.sqrt(-2.0 * ti.log(u3)) * ti.cos(2*tm.pi * u4),
            tm.sqrt(-2.0 * ti.log(u3)) * ti.sin(2*tm.pi * u4)
        ))

        # unpack
        w = q.x
        x = q.y
        y = q.z
        z = q.w

        # 3) convert to orthonormal basis
        mat = tm.mat3(  # Taichi mat3 from quaternion
            1 - 2*(y*y + z*z),  2*(x*y - w*z),    2*(x*z + w*y),
            2*(x*y + w*z),      1 - 2*(x*x + z*z),2*(y*z - w*x),
            2*(x*z - w*y),      2*(y*z + w*x),    1 - 2*(x*x + y*y)
        )
        agents[i].basis_x = tm.vec3(mat[0, 0], mat[0, 1], mat[0, 2])
        agents[i].basis_y = tm.vec3(mat[1, 0], mat[1, 1], mat[1, 2])
        agents[i].basis_z = tm.vec3(mat[2, 0], mat[2, 1], mat[2, 2])


@ti.func
def sense(img: ti.template(), sense_center) -> float:
    # When SENSE_AREA == 1, we only sample the center point
    sense_pos = tm.round(sense_center) % config.GRID_SIZE
    return img[int(sense_pos.x), int(sense_pos.y), int(sense_pos.z)]


@ti.func
def rotate_3d(angle: float, axis: tm.vec3) -> tm.mat3:
    """Returns a rotation matrix for a given angle around a specified axis."""
    a = tm.normalize(axis)
    s = tm.sin(angle)
    c = tm.cos(angle)
    r = 1.0 - c
    
    return tm.mat3(
        a.x * a.x * r + c,
        a.y * a.x * r + a.z * s,
        a.z * a.x * r - a.y * s,
        a.x * a.y * r - a.z * s,
        a.y * a.y * r + c,
        a.z * a.y * r + a.x * s,
        a.x * a.z * r + a.y * s,
        a.y * a.z * r - a.x * s,
        a.z * a.z * r + c
    )


@ti.func
def isclose(a, b, tol=1e-5):
    """Check if two values are close within a tolerance."""
    return abs(a - b) < tol


@ti.kernel
def update_pos(img: ti.template(), sense_angle: float, steer_strength: float, sense_reach: float):
    for i in range(config.AGENT_COUNT):
        forwards = agents[i].basis_z
        
        sense_positions = [tm.vec3(0.0, 0.0, 0.0) for _ in range(4)]
        sense_positions[0] = agents[i].position + forwards * sense_reach
        
        tipped_up_sense = rotate_3d(sense_angle, agents[i].basis_x) @ forwards * sense_reach
        for j in ti.static(range(1, 4)):
            sense_positions[j] = agents[i].position + rotate_3d(2.0 * tm.pi * j / 3.0, agents[i].basis_z) @ tipped_up_sense
        
        sense_samples = [0.0 for _ in range(4)]
        for j in ti.static(range(4)):
            sense_samples[j] = sense(img, sense_positions[j])
        
        max_sample = ti.max(
            sense_samples[0],
            sense_samples[1],
            sense_samples[2],
            sense_samples[3],
        )
        
        selected = 0  # default to no steering
        steer_amount = 0.0
        if (
            sense_samples[1] < sense_samples[0] and
            sense_samples[2] < sense_samples[0] and
            sense_samples[3] < sense_samples[0]
        ):
            selected = int(ti.random() * 3) + 1
            steer_amount = (ti.random() - 0.5) * steer_strength
        elif isclose(max_sample, sense_samples[0]):
            selected = 0
            steer_amount = ti.random() * steer_strength
        elif isclose(max_sample, sense_samples[1]):
            selected = 1
            steer_amount = ti.random() * steer_strength
        elif isclose(max_sample, sense_samples[2]):
            selected = 2
            steer_amount = ti.random() * steer_strength
        elif isclose(max_sample, sense_samples[3]):
            selected = 3
            steer_amount = ti.random() * steer_strength
        
        selected = ti.max(0, ti.min(selected, 3))  # clamp to [0, 3]
        
        if selected != 0:
            steer_tile = sense_angle * steer_amount
            steer_yaw = (2.0 * tm.pi * selected / 3.0) * steer_amount
            
            transform = rotate_3d(steer_yaw, agents[i].basis_z) @ rotate_3d(steer_tile, agents[i].basis_x)
            
            agents[i].basis_x, agents[i].basis_y, agents[i].basis_z = orthonormalize(
                transform @ agents[i].basis_x,
                transform @ agents[i].basis_y,
                transform @ agents[i].basis_z
            )
        
        agents[i].position = (agents[i].position + agents[i].basis_z * config.SPEED) % config.GRID_SIZE


@ti.kernel
def fade(img: ti.template(), strength: float):
    for i, j, k in agents_grid:
        img[i, j, k] *= strength


@ti.kernel
def blur_axis(
    src: ti.template(),
    dst: ti.template(),
    dir_x: ti.f32,
    dir_y: ti.f32,
    dir_z: ti.f32
):
    for i, j, k in dst:
        acc = 0.0
        
        for t in ti.static(range(-RADIUS, RADIUS + 1)):
            x = int(i + t * dir_x)
            y = int(j + t * dir_y)
            z = int(k + t * dir_z)
            # bounds check against your 3D grid size
            if (0 <= x < config.GRID_SIZE[0] and
                0 <= y < config.GRID_SIZE[1] and
                0 <= z < config.GRID_SIZE[2]):
                acc += src[x, y, z] * weights[t + RADIUS]
        dst[i, j, k] = acc


@ti.kernel
def deposit_trail(img: ti.template(), color: float):
    for i in range(config.AGENT_COUNT):
        int_pos = tm.ivec3(tm.round(agents[i].position))
        if all(0 <= int_pos) and all(int_pos < config.GRID_SIZE):
            img[int(int_pos.x), int(int_pos.y), int(int_pos.z)] += color


def smoothstep(t):
    return t**2 * (3.0 - 2.0 * t)
    

def interp_hue(h0, h1, t):
    diff = (h1 - h0 + 0.5) % 1.0 - 0.5
    return (h0 + diff * t) % 1.0


def random_hcl_colormap(
        hue_span_min=0.1,
        hue_span_max=0.2,
        start_black=True
):
    # pick two nearby hues
    hue1 = np.random.rand()
    delta = np.random.uniform(hue_span_min, hue_span_max)
    hue2 = (hue1 + delta) % 1.0

    # 3‑knot: black→hue1→hue2
    xs = np.array([0.0, 0.5, 1.0])
    control_hues = np.array([hue1,   hue1,   hue2])
    control_lights = np.array([0.0,     0.6,     0.8])
    if not start_black:
        control_lights = 1 - control_lights
    control_sats = np.array([0.0,     0.85,    0.85])

    xi = np.linspace(0, 1, config.CMAP_COLORS)

    # build the colormap
    cmap = np.zeros((config.CMAP_COLORS, 3), dtype=np.float32)
    for i, x in enumerate(xi):
        # find which segment we're in (0→0.5 or 0.5→1.0)
        j = 0 if x <= 0.5 else 1
        t = (x - xs[j]) / (xs[j+1] - xs[j])
        t_e = smoothstep(t)

        # interpolate H, L, S
        h = interp_hue(control_hues[j], control_hues[j+1], t)
        # apply gamma to lightness for a subtle pop
        l0, l1 = control_lights[j], control_lights[j+1]
        l = ((1 - t_e) * (l0 ** 1.1) + t_e * (l1 ** 1.1)) ** (1/1.1)
        s0, s1 = control_sats[j], control_sats[j+1]
        s = (1 - t_e) * s0 + t_e * s1

        cmap[i] = colorsys.hls_to_rgb(h, l, s)

    return cmap


def gen_cmap():
    cmap_cpu = random_hcl_colormap(start_black=True)
    cmap = ti.field(dtype=tm.vec3, shape=(config.CMAP_COLORS,))
    cmap.from_numpy(cmap_cpu)
    
    return cmap


@ti.func
def interp_cmap(cmap: ti.template(), value: float) -> tm.vec3:
    """Interpolate between two colors in the colormap based on the value."""
    idx_f = value * (config.CMAP_COLORS - 1)
    idx = int(idx_f)
    frac = idx_f - idx
    c0 = cmap[idx]
    c1 = cmap[ti.min(idx + 1, config.CMAP_COLORS - 1)]
    return tm.mix(c0, c1, frac)


@ti.kernel
def initialize_3d():
    for i, j, k in agents_grid:
        agents_grid[i, j, k] = 0.5


def main():
    gui = ti.GUI("Slime Mold", res=config.RESOLUTION, fast_gui=True)

    gen_agents()

    # Render first frame - since the first iteration takes a while due to JIT compilation
    rendering.render_3d(render_img, agents_grid)
    gui.set_image(render_img)
    gui.show()

    count = 0
    
    ping, pong = agents_grid, agents_grid_temp
    initialize_3d()
    
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        deposit_trail(ping, config.COLOR)
        update_pos(ping, np.radians(60), 1, 15.0)
        fade(ping, 0.93)

        blur_axis(ping, pong, 1.0, 0.0, 0.0)
        ping, pong = pong, ping
        blur_axis(ping, pong, 0.0, 1.0, 0.0)
        ping, pong = pong, ping
        blur_axis(ping, pong, 0.0, 0.0, 1.0)

        rendering.render_3d(render_img, agents_grid)

        gui.set_image(render_img)
        gui.show()

        count -= 1


if __name__ == "__main__":
    main()
