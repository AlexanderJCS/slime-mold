import config
import rendering

import colorsys

import numpy as np
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)


def random_points_in_sphere(n, r=1.0):
    # Random angles
    theta = np.random.uniform(0, 2 * np.pi, n)  # Azimuthal angle
    phi = np.arccos(1 - 2 * np.random.uniform(0, 1, n))  # Polar angle
    # Radii with √³ to ensure uniform volume density
    radius = np.cbrt(np.random.uniform(0, 1, n)) * r

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack((x, y, z))


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

sigma = 0.2
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
    sensor_value = 0.0
    for dx, dy, dz in ti.static(
            [(x, y, z)
             for x in (-1, 0, 1)
             for y in (-1, 0, 1)
             for z in (-1, 0, 1)]):
        sense_pos = tm.round(sense_center + tm.vec3(dx, dy, dz)) % config.GRID_SIZE
        sensor_value += img[int(sense_pos.x), int(sense_pos.y), int(sense_pos.z)]

    return sensor_value


@ti.func
def pitch_yaw_to_vec(angles: tm.vec2) -> tm.vec3:
    cos_pitch = tm.cos(angles[0])
    sin_pitch = tm.sin(angles[0])
    cos_yaw = tm.cos(angles[1])
    sin_yaw = tm.sin(angles[1])
    
    x = cos_pitch * sin_yaw
    y = sin_pitch
    z = cos_pitch * cos_yaw
    return tm.vec3(x, y, z)


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


@ti.kernel
def update_pos(img: ti.template(), sense_angle: float, steer_strength: float, sense_reach: float):
    for i in range(config.AGENT_COUNT):
        # Definining values:
        #  agents[i][0] == x position
        #  agents[i][1] == y position
        #  agents[i][2] == angle
        sense_forward = sense(img, agents[i].position, agents[i].angles, sense_reach)
        sense_pitch_neg = sense(img, agents[i].position, agents[i].angles + tm.vec2(-sense_angle, 0), sense_reach)
        sense_pitch_pos = sense(img, agents[i].position, agents[i].angles + tm.vec2(sense_angle, 0), sense_reach)
        sense_yaw_neg = sense(img, agents[i].position, agents[i].angles + tm.vec2(0, -sense_angle), sense_reach)
        sense_yaw_pos = sense(img, agents[i].position, agents[i].angles + tm.vec2(0, sense_angle), sense_reach)
        
        # Calculate the steering direction based on sensed values
        rand = ti.random()

        if (sense_forward > sense_pitch_neg and
                sense_forward > sense_pitch_pos and
                sense_forward > sense_yaw_neg and
                sense_forward > sense_yaw_pos):
            # no turn
            agents[i].angles += tm.vec2(0, 0)

        # 2) worst in front of all five → random jitter in both pitch & yaw
        elif (sense_forward < sense_pitch_neg and
              sense_forward < sense_pitch_pos and
              sense_forward < sense_yaw_neg and
              sense_forward < sense_yaw_pos):
            jitter = (rand - 0.5) * 2 * steer_strength
            agents[i].angles += tm.vec2(jitter, jitter)

        # 3) otherwise steer by comparing each axis separately
        else:
            # pitch axis
            if sense_pitch_pos > sense_pitch_neg:
                agents[i].angles.x += rand * steer_strength
            elif sense_pitch_neg > sense_pitch_pos:
                agents[i].angles.x -= rand * steer_strength

            # yaw axis
            if sense_yaw_pos > sense_yaw_neg:
                agents[i].angles.y += rand * steer_strength
            elif sense_yaw_neg > sense_yaw_pos:
                agents[i].angles.y -= rand * steer_strength
        
        agents[i].position += tm.normalize(pitch_yaw_to_vec(agents[i].angles)) * config.SPEED
        agents[i].position %= config.GRID_SIZE  # wrap around the grid


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
        # walk along the ray in 3D
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
def render(old_cmap: ti.template(), new_cmap: ti.template(), t: float):
    # cmap is assumed to be a ti.field(dtype=tm.vec3, shape=N)
    
    for i, j in agents_grid:
        value = agents_grid[i, j]
        tonemapped = value / (value + 1)  # Reinhard tonemapping
        old_color = interp_cmap(old_cmap, tonemapped)
        new_color = interp_cmap(new_cmap, tonemapped)
        gui_color = tm.mix(old_color, new_color, t)
        
        render_img[i, j] = gui_color
        

def main():
    gui = ti.GUI("Slime Mold", res=config.RESOLUTION, fast_gui=True)

    gen_agents()

    # Render first frame - since the first iteration takes a while due to JIT compilation
    rendering.render_3d(render_img, agents_grid)
    gui.set_image(render_img)
    gui.show()

    # initial params
    steer_old = steer_new = 2.0
    fade_old = fade_new = 0.97
    angle_old = angle_new = np.radians(90)
    reach_old = reach_new = 20.0

    old_cmap = gen_cmap()
    new_cmap = gen_cmap()

    max_count = 200
    count = 0
    
    ping, pong = agents_grid, agents_grid_temp
    
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        # if count <= 0:
        #     # shift "new"→"old"
        #     steer_old, steer_new = (
        #         steer_new,  np.random.uniform(1.0, 3.0)
        #         if np.random.random() > 0.5
        #         else np.random.uniform(0.2, 0.6)
        #     )
        #
        #     fade_old, fade_new = fade_new,   np.random.uniform(0.97, 0.99)
        #
        #     loop_over = np.random.random() > 0.8
        #     high_reach = np.random.random() > 0.4
        #
        #     angle_old, angle_new = angle_new, np.radians(np.clip(
        #         np.random.uniform(160, 179) if loop_over
        #         else np.random.normal(70, 30),
        #         5, 179
        #     ))
        #
        #     reach_old, reach_new = reach_new, (
        #         np.random.uniform(20, 40) if high_reach
        #         else np.random.uniform(8, 15)
        #     )
        #
        #     old_cmap = new_cmap
        #     new_cmap = gen_cmap()
        #
        #     print(f"New params → steer: {steer_new:.2f}, fade: {fade_new:.3f}, "
        #           f"angle: {np.degrees(angle_new):.1f}°, reach: {reach_new:.1f}")
        #
        #     count = max_count
        #
        # t_raw = (max_count - count) / max_count
        # t = float(np.clip(t_raw, 0.0, 1.0))
        #
        # # interpolate each parameter
        # steer_strength = (1 - t) * steer_old + t * steer_new
        # fade_strength = (1 - t) * fade_old + t * fade_new
        # sense_angle = (1 - t) * angle_old + t * angle_new
        # sense_reach = (1 - t) * reach_old + t * reach_new
        #
        # # simulation steps use the interpolated values
        # update_pos(sense_angle, steer_strength, sense_reach)
        # fade(fade_strength)
        # deposit_trail(config.COLOR)
        # blur(agents_grid, temp, 1.0, 0.0)
        # blur(temp, agents_grid, 0.0, 1.0)
        #
        # # colormap cross‑fade
        # render(old_cmap, new_cmap, t)

        deposit_trail(ping, config.COLOR)
        update_pos(ping, np.radians(45), 2.0, 10.0)
        fade(ping, 0.97)
        
        blur_axis(ping, pong, 1.0, 0.0, 0.0)
        ping, pong = pong, ping
        blur_axis(ping, pong, 0.0, 1.0, 0.0)
        ping, pong = pong, ping
        blur_axis(ping, pong, 0.0, 0.0, 1.0)

        rendering.render_3d(render_img, ping)

        gui.set_image(render_img)
        gui.show()

        count -= 1


if __name__ == "__main__":
    main()
