import colorsys

import numpy as np
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

HEIGHT = 480
SIZE = (HEIGHT * 16 // 9, HEIGHT)
AGENT_COUNT = 5000000
COLOR = 2000 / AGENT_COUNT
SENSE_AREA = 3
CMAP_COLORS = 256  # number of colors in the colormap
SPEED = 2.0


def random_points_in_circle(n, r=1.0):
    # random angles
    theta = np.random.uniform(0, 2 * np.pi, n)
    # radii with √ to ensure uniform area density
    r = np.sqrt(np.random.uniform(0, 1, n)) * r

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))


agents_grid = ti.field(dtype=ti.f32, shape=SIZE)
temp = ti.field(dtype=ti.f32, shape=SIZE)
render_img = ti.field(dtype=tm.vec3, shape=SIZE)

agents_cpu = np.zeros((AGENT_COUNT, 3), dtype=np.float32)
origins = random_points_in_circle(AGENT_COUNT, r=SIZE[0] // 4)
agents_cpu[:, 0] = origins[:, 0] + SIZE[0] // 2
agents_cpu[:, 1] = origins[:, 1] + SIZE[1] // 2
agents_cpu[:, 2] = np.random.random(AGENT_COUNT) * 2 * np.pi

agents = ti.field(dtype=tm.vec3, shape=(AGENT_COUNT,))
agents.from_numpy(agents_cpu)

sigma = 0.1
RADIUS = int(np.ceil(sigma * 3))

# precompute 1D Gaussian weights
weights_cpu = np.array([np.exp(-0.5 * (i / sigma) ** 2) for i in range(-RADIUS, RADIUS + 1)], dtype=np.float32)
weights_cpu /= np.sum(weights_cpu)

weights = ti.field(dtype=ti.f32, shape=len(weights_cpu))
weights.from_numpy(weights_cpu)


@ti.func
def sense(pos: tm.vec2, cos_angle: float, sin_angle: float, sense_reach: float) -> float:
    """Optimized sensing using precomputed sin/cos values."""
    sense_dir = tm.vec2(cos_angle, sin_angle)
    sense_center = pos + sense_dir * sense_reach

    # Define sensing area bounds
    x_start = int(tm.round(sense_center[0] - SENSE_AREA // 2)) % SIZE[0]
    x_end = int(tm.round(sense_center[0] + SENSE_AREA // 2)) % SIZE[0]
    y_start = int(tm.round(sense_center[1] - SENSE_AREA // 2)) % SIZE[1]
    y_end = int(tm.round(sense_center[1] + SENSE_AREA // 2)) % SIZE[1]

    # Accumulate sensor values directly
    sensor_value = 0.0
    for x in range(x_start, x_end + 1):
        for y in range(y_start, y_end + 1):
            sensor_value += agents_grid[x % SIZE[0], y % SIZE[1]]

    return sensor_value


@ti.kernel
def update_pos(sense_angle: float, steer_strength: float, sense_reach: float):
    for i in range(AGENT_COUNT):
        # Definining values:
        #  agents[i][0] == x position
        #  agents[i][1] == y position
        #  agents[i][2] == angle
        current_pos = tm.vec2(agents[i][0], agents[i][1])
        
        # Precompute trigonometric values once per agent
        agent_angle = agents[i][2]
        cos_agent = ti.cos(agent_angle)
        sin_agent = ti.sin(agent_angle)
        cos_left = ti.cos(agent_angle - sense_angle)
        sin_left = ti.sin(agent_angle - sense_angle)
        cos_right = ti.cos(agent_angle + sense_angle)
        sin_right = ti.sin(agent_angle + sense_angle)
        
        # Use precomputed values for sensing
        left_sense = sense(current_pos, cos_left, sin_left, sense_reach)
        forward_sense = sense(current_pos, cos_agent, sin_agent, sense_reach)
        right_sense = sense(current_pos, cos_right, sin_right, sense_reach)

        rand = ti.random()

        if forward_sense > left_sense and forward_sense > right_sense:
            # Move forward
            agents[i][2] += 0.0
        elif forward_sense < left_sense and forward_sense < right_sense:
            agents[i][2] += (rand - 0.5) * steer_strength  # Randomly turn left or right
        elif right_sense > left_sense:
            agents[i][2] -= rand * steer_strength
        elif left_sense > right_sense:
            agents[i][2] += rand * steer_strength
            
        agents[i][0] += cos_agent * SPEED
        agents[i][1] += sin_agent * SPEED

        agents[i][0] %= SIZE[0]
        agents[i][1] %= SIZE[1]


@ti.kernel
def fade(strength: float):
    for i, j in agents_grid:
        agents_grid[i, j] *= strength


@ti.kernel
def blur(
    src: ti.template(),  # either pixels or temp
    dst: ti.template(),  # either temp or pixels
    dir_x: ti.f32,
    dir_y: ti.f32
):
    for i, j in dst:
        acc = 0.0
        # walk along the given direction
        for k in ti.static(range(-RADIUS, RADIUS + 1)):
            x = int(i + k * dir_x)
            y = int(j + k * dir_y)
            if 0 <= x < SIZE[0] and 0 <= y < SIZE[1]:
                acc += src[x, y] * weights[k + RADIUS]
        dst[i, j] = acc


@ti.kernel
def deposit_trail(color: float):
    for i in range(AGENT_COUNT):
        x = int(tm.round(agents[i][0]))
        y = int(tm.round(agents[i][1]))
        if 0 <= x < SIZE[0] and 0 <= y < SIZE[1]:
            agents_grid[x, y] += color


def interp_hue(h0, h1, t):
    d = (h1 - h0 + 0.5) % 1.0 - 0.5
    return (h0 + d * t) % 1.0


def random_hcl_colormap(
        hue_span_min=0.1,
        hue_span_max=0.2,
        start_black=True
):
    # pick two nearby hues
    hue1  = np.random.rand()
    delta = np.random.uniform(hue_span_min, hue_span_max)
    hue2  = (hue1 + delta) % 1.0

    # 3‑knot: black→hue1→hue2
    xs             = np.array([0.0, 0.5, 1.0])
    control_hues   = np.array([hue1,   hue1,   hue2])
    control_lights = np.array([0.0,     0.6,     0.8])
    if not start_black:
        control_lights = 1 - control_lights
    control_sats   = np.array([0.0,     0.85,    0.85])

    xi = np.linspace(0, 1, CMAP_COLORS)

    # easing function for smooth transitions
    def smoothstep(t):
        return t * t * (3.0 - 2.0 * t)

    # minimal‐arc interpolation in hue
    def interp_hue(h0, h1, t):
        diff = (h1 - h0 + 0.5) % 1.0 - 0.5
        return (h0 + diff * t) % 1.0

    # build the colormap
    cmap = np.zeros((CMAP_COLORS, 3), dtype=np.float32)
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
    cmap = ti.field(dtype=tm.vec3, shape=(CMAP_COLORS,))
    cmap.from_numpy(cmap_cpu)
    
    return cmap


@ti.func
def interp_cmap(cmap: ti.template(), value: float) -> tm.vec3:
    """Interpolate between two colors in the colormap based on the value."""
    idx_f = value * (CMAP_COLORS - 1)
    idx = int(idx_f)
    frac = idx_f - idx
    c0 = cmap[idx]
    c1 = cmap[ti.min(idx + 1, CMAP_COLORS - 1)]
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
    gui = ti.GUI("Slime Mold", res=SIZE, fast_gui=True)

    # initial params
    steer_old = steer_new = 2.0
    fade_old  = fade_new  = 0.97
    angle_old = angle_new = np.radians(90)
    reach_old = reach_new = 20.0

    old_cmap = gen_cmap()
    new_cmap = gen_cmap()

    max_count = 200
    count = 0
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        if count <= 0:
            # shift "new"→"old"
            steer_old,  steer_new  = steer_new,  np.random.uniform(0.2, 3.0)
            fade_old,   fade_new   = fade_new,   np.random.uniform(0.97, 0.99)

            loop_over   = np.random.random() > 0.8
            high_reach  = np.random.random() > 0.4

            angle_old, angle_new = angle_new, np.radians(np.clip(
                np.random.uniform(160, 179) if loop_over
                else np.random.normal(70, 30),
                5, 179
            ))

            reach_old, reach_new = reach_new, (
                np.random.uniform(20, 40) if high_reach
                else np.random.uniform(8, 15)
            )

            old_cmap = new_cmap
            new_cmap = gen_cmap()

            print(f"New params → steer: {steer_new:.2f}, fade: {fade_new:.3f}, "
                  f"angle: {np.degrees(angle_new):.1f}°, reach: {reach_new:.1f}")

            count = max_count

        t_raw = (max_count - count) / max_count
        t = float(np.clip(t_raw, 0.0, 1.0))

        # interpolate each parameter
        steer_strength = (1 - t) * steer_old  + t * steer_new
        fade_strength  = (1 - t) * fade_old   + t * fade_new
        sense_angle    = (1 - t) * angle_old  + t * angle_new
        sense_reach    = (1 - t) * reach_old  + t * reach_new

        # simulation steps use the interpolated values
        update_pos(sense_angle, steer_strength, sense_reach)
        fade(fade_strength)
        deposit_trail(COLOR)
        # blur(agents_grid, temp, 1.0, 0.0)
        # blur(temp, agents_grid, 0.0, 1.0)

        # colormap cross‑fade
        render(old_cmap, new_cmap, t)

        gui.set_image(render_img)
        gui.show()

        count -= 1


if __name__ == "__main__":
    main()
