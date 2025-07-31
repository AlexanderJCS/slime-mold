import taichi as ti
import taichi.math as tm

import numpy as np

import config
import colorsys


def interp_hue(h0, h1, t):
    diff = (h1 - h0 + 0.5) % 1.0 - 0.5
    return (h0 + diff * t) % 1.0


def smoothstep(t):
    return t**2 * (3.0 - 2.0 * t)


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
    control_hues = np.array([hue1, hue1, hue2])
    control_lights = np.array([0.0, 0.6, 0.8])
    if not start_black:
        control_lights = 1 - control_lights
    control_sats = np.array([0.0, 0.85, 0.85])
    
    xi = np.linspace(0, 1, config.CMAP_COLORS)
    
    # build the colormap
    cmap = np.zeros((config.CMAP_COLORS, 3), dtype=np.float32)
    for i, x in enumerate(xi):
        # find which segment we're in (0→0.5 or 0.5→1.0)
        j = 0 if x <= 0.5 else 1
        t = (x - xs[j]) / (xs[j + 1] - xs[j])
        t_e = smoothstep(t)
        
        # interpolate H, L, S
        h = interp_hue(control_hues[j], control_hues[j + 1], t)
        # apply gamma to lightness for a subtle pop
        l0, l1 = control_lights[j], control_lights[j + 1]
        l = ((1 - t_e) * (l0 ** 1.1) + t_e * (l1 ** 1.1)) ** (1 / 1.1)
        s0, s1 = control_sats[j], control_sats[j + 1]
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
