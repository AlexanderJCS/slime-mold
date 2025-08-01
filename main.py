import config
import cmap

import colorsys

import numpy as np
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

import rendering  # import rendering after ti.init


agents_grid = ti.field(dtype=ti.f32, shape=config.GRID_SIZE)
gradient = ti.Vector.field(3, dtype=ti.f32, shape=config.GRID_SIZE)
agents_grid_temp = ti.field(dtype=ti.f32, shape=config.GRID_SIZE)
render_img = ti.field(dtype=tm.vec3, shape=config.RESOLUTION)


Agent = ti.types.struct(
    position=ti.types.vector(3, ti.f32),
    basis_x=ti.types.vector(3, ti.f32),
    basis_y=ti.types.vector(3, ti.f32),
    basis_z=ti.types.vector(3, ti.f32)
)

agents = Agent.field(shape=(config.AGENT_COUNT,))

sigma = 0.6
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
            x = int(i + t * dir_x) % config.GRID_SIZE[0]
            y = int(j + t * dir_y) % config.GRID_SIZE[1]
            z = int(k + t * dir_z) % config.GRID_SIZE[2]
            acc += src[x, y, z] * weights[t + RADIUS]
        dst[i, j, k] = acc


@ti.kernel
def deposit_trail(img: ti.template(), color: float):
    for i in range(config.AGENT_COUNT):
        int_pos = tm.ivec3(tm.round(agents[i].position)) % config.GRID_SIZE
        img[int(int_pos.x), int(int_pos.y), int(int_pos.z)] += color


@ti.kernel
def compute_gradient(img: ti.template(), gradient: ti.template()):
    for i, j, k in img:
        ip = (i + 1) % config.GRID_SIZE[0]
        im = (i - 1 + config.GRID_SIZE[0]) % config.GRID_SIZE[0]
        jp = (j + 1) % config.GRID_SIZE[1]
        jm = (j - 1 + config.GRID_SIZE[1]) % config.GRID_SIZE[1]
        kp = (k + 1) % config.GRID_SIZE[2]
        km = (k - 1 + config.GRID_SIZE[2]) % config.GRID_SIZE[2]

        dx = 0.5 * (img[ip, j, k] - img[im, j, k])
        dy = 0.5 * (img[i, jp, k] - img[i, jm, k])
        dz = 0.5 * (img[i, j, kp] - img[i, j, km])

        gradient[i, j, k] = ti.Vector([dx, dy, dz])


@ti.kernel
def initialize_3d():
    for i, j, k in agents_grid:
        agents_grid[i, j, k] = 0.5


def rotate_camera(angle: float):
    """Rotate the camera around the Y-axis by a small angle."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x = rendering.camera_pos[None].x * cos_a - rendering.camera_pos[None].z * sin_a
    z = rendering.camera_pos[None].x * sin_a + rendering.camera_pos[None].z * cos_a
    y = rendering.camera_pos[None].y  # unchanged
    rendering.camera_pos[None] = tm.vec3(x, y, z)


def lerp(a, b, t):
    """Linear interpolation between a and b by t."""
    return a + (b - a) * t


def main():
    gui = ti.GUI("Slime Mold", res=config.RESOLUTION, fast_gui=True)

    gen_agents()

    # Render first frame - since the first iteration takes a while due to JIT compilation
    # rendering.render_3d(render_img, agents_grid, gradient)
    # gui.set_image(render_img)
    # gui.show()

    count = 300
    
    ping, pong = agents_grid, agents_grid_temp
    initialize_3d()
    
    rendering.camera_pos[None] = tm.vec3(0, 0, 1.5)
    rendering.look_at[None] = tm.vec3(0, 0, 0)
    rendering.fov[None] = np.radians(60)
    
    old_sense_angle = np.radians(60)
    new_sense_angle = np.radians(60)
    old_steer_strength = 1.0
    new_steer_strength = 1.0
    old_sense_reach = 15.0
    new_sense_reach = 15.0
    old_fade_strength = config.FADE_STRENGTH
    new_fade_strength = config.FADE_STRENGTH
    old_cmap = cmap.gen_cmap()
    new_cmap = cmap.gen_cmap()
    
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        if count == 300:
            big_angle = np.random.rand() > 0.8
            big_dist = np.random.rand() > 0.8
            
            # Change parameters every 100 frames
            old_sense_angle, new_sense_angle = new_sense_angle, np.radians(np.random.uniform(90, 120) if big_angle else np.random.uniform(15, 60))
            old_steer_strength, new_steer_strength = new_steer_strength, np.random.uniform(0.8, 2.5)
            old_sense_reach, new_sense_reach = new_sense_reach, np.random.uniform(13.0, 20.0) if big_dist else np.random.uniform(5.0, 10.0)
            old_fade_strength, new_fade_strength = new_fade_strength, np.random.uniform(0.93, 0.98)
            old_cmap, new_cmap = new_cmap, cmap.gen_cmap()
            count = 0
            print(f"New parameters: sense_angle={np.degrees(new_sense_angle):.2f}°, "
                    f"steer_strength={new_steer_strength:.2f}, "
                    f"sense_reach={new_sense_reach:.2f}, "
                    f"fade_strength={new_fade_strength:.2f}")
        
        t = count / 300.0
        sense_angle = lerp(old_sense_angle, new_sense_angle, t)
        steer_strength = lerp(old_steer_strength, new_steer_strength, t)
        sense_reach = lerp(old_sense_reach, new_sense_reach, t)
        fade_strength = lerp(old_fade_strength, new_fade_strength, t)
        
        deposit_trail(ping, config.COLOR)
        update_pos(ping, sense_angle, steer_strength, sense_reach)
        fade(ping, fade_strength)

        blur_axis(ping, pong, 1.0, 0.0, 0.0)
        ping, pong = pong, ping
        blur_axis(ping, pong, 0.0, 1.0, 0.0)
        ping, pong = pong, ping
        blur_axis(ping, pong, 0.0, 0.0, 1.0)

        # make gradient
        compute_gradient(ping, gradient)

        rendering.render_3d(render_img, agents_grid, gradient, old_cmap, new_cmap, t)

        gui.set_image(render_img)
        gui.show()

        count += 1
        rotate_camera(-0.001)


if __name__ == "__main__":
    main()
