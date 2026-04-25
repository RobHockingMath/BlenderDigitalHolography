"""
hologram_lightfield_viewer.py

Interactive geometric light-field / hogel hologram viewer.

This version uses the explicit raytracing model:

    for each viewer pixel:
        1. construct a camera ray,
        2. intersect that ray with the hologram plane,
        3. use the hit point to choose exactly one hogel,
        4. use the incident angle to choose exactly one pixel in that hogel.

No spatial interpolation between hogels. No bilinear texture filtering.
The geometry and hogel FOV are read from hogel_params.py.

Expected filename format:

    hogel_iiii_Na_jjjj_Nb.png

Dependencies:

    pip install pygame moderngl pillow numpy

Run:

    python hologram_lightfield_viewer.py
"""

from __future__ import annotations

import importlib
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("Missing dependency: numpy. Install with: pip install numpy") from exc

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Missing dependency: Pillow. Install with: pip install pillow") from exc

try:
    import pygame
except ImportError as exc:
    raise SystemExit("Missing dependency: pygame. Install with: pip install pygame") from exc

try:
    import moderngl
except ImportError as exc:
    raise SystemExit("Missing dependency: moderngl. Install with: pip install moderngl") from exc


# ============================================================
# PARAMETER LOADING
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    params = importlib.import_module("hogel_params")
    params = importlib.reload(params)
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Could not import hogel_params.py. Put hogel_params.py in the same "
        "folder as hologram_lightfield_viewer.py."
    ) from exc


# Shared hologram/capture params.
X0 = float(params.X0)
Y0 = float(params.Y0)
Z0 = float(params.Z0)
REQUESTED_W = float(params.W)
REQUESTED_H = float(params.H)
HOGEL_CELL_SIZE = float(params.h)
INWARD_NORMAL = np.array(params.INWARD_NORMAL, dtype=np.float32)
HOGEL_FOV_X_DEGREES = float(params.CAMERA_FOV_DEGREES)

# Viewer params.
HOGEL_DIR = params.HOGEL_DIR
WINDOW_WIDTH = int(params.WINDOW_WIDTH)
WINDOW_HEIGHT = int(params.WINDOW_HEIGHT)
VIEWER_FOV_Y_DEGREES = float(params.VIEWER_FOV_Y_DEGREES)
NEAR_PLANE = float(params.NEAR_PLANE)
FAR_PLANE = float(params.FAR_PLANE)
INITIAL_EYE = np.array(params.INITIAL_EYE, dtype=np.float32)
INITIAL_LOOK_AT = np.array(params.INITIAL_LOOK_AT, dtype=np.float32)
BASE_MOVE_SPEED = float(params.BASE_MOVE_SPEED)
FAST_MOVE_MULTIPLIER = float(params.FAST_MOVE_MULTIPLIER)
MOUSE_SENSITIVITY = float(params.MOUSE_SENSITIVITY)
INITIAL_HOLOGRAM_YAW_DEGREES = float(params.INITIAL_HOLOGRAM_YAW_DEGREES)
INITIAL_HOLOGRAM_PITCH_DEGREES = float(params.INITIAL_HOLOGRAM_PITCH_DEGREES)
INITIAL_HOLOGRAM_ROLL_DEGREES = float(params.INITIAL_HOLOGRAM_ROLL_DEGREES)
HOLOGRAM_ROTATE_SPEED_DEGREES_PER_SECOND = float(params.HOLOGRAM_ROTATE_SPEED_DEGREES_PER_SECOND)
HOLOGRAM_MOUSE_SENSITIVITY = float(params.HOLOGRAM_MOUSE_SENSITIVITY)
SHOW_CELL_GRID_OVERLAY = bool(params.SHOW_CELL_GRID_OVERLAY)
BACKGROUND_RGB = tuple(float(x) for x in params.BACKGROUND_RGB)
FLIP_HOGEL_IMAGES_TOP_BOTTOM = bool(params.FLIP_HOGEL_IMAGES_TOP_BOTTOM)
LOAD_PROGRESS_EVERY = int(params.LOAD_PROGRESS_EVERY)


# ============================================================
# SHADERS
# ============================================================

VERTEX_SHADER = r"""
#version 330

in vec2 in_pos;
out vec2 v_uv;

void main() {
    v_uv = 0.5 * (in_pos + vec2(1.0, 1.0));
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = r"""
#version 330

uniform sampler2D u_atlas;

// Viewer camera
uniform vec3  u_eye;
uniform vec3  u_cam_forward;
uniform vec3  u_cam_right;
uniform vec3  u_cam_up;
uniform float u_tan_half_view_fov_y;
uniform float u_view_aspect;

// Hologram pose
uniform vec3  u_holo_center;
uniform vec3  u_holo_forward;
uniform vec3  u_holo_right;
uniform vec3  u_holo_up;

// Hologram dimensions / indexing
uniform float u_xmin;
uniform float u_zmin;
uniform float u_h;
uniform float u_screen_width;
uniform float u_screen_height;
uniform int   u_na;
uniform int   u_nb;

// Atlas metadata
uniform float u_hogel_w;
uniform float u_hogel_h;
uniform float u_atlas_w;
uniform float u_atlas_h;
uniform float u_tan_half_hogel_fov_x;
uniform float u_tan_half_hogel_fov_y;

uniform int   u_show_grid;
uniform vec3  u_bg_color;

in vec2 v_uv;
out vec4 f_color;

vec4 sample_hogel_nearest(int i, int j, vec3 ray_dir_world) {
    if (i < 0 || i >= u_na || j < 0 || j >= u_nb) {
        return vec4(u_bg_color, 1.0);
    }

    float d_forward = dot(ray_dir_world, u_holo_forward);
    if (d_forward <= 1.0e-6) {
        return vec4(u_bg_color, 1.0);
    }

    float sx = dot(ray_dir_world, u_holo_right) / (d_forward * u_tan_half_hogel_fov_x);
    float sy = dot(ray_dir_world, u_holo_up)    / (d_forward * u_tan_half_hogel_fov_y);

    float u = 0.5 + 0.5 * sx;
    float v = 0.5 + 0.5 * sy;

    if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) {
        return vec4(u_bg_color, 1.0);
    }

    // Strict nearest-pixel sampling inside the chosen hogel.
    float local_px = floor(clamp(u, 0.0, 0.999999) * u_hogel_w) + 0.5;
    float local_py = floor(clamp(v, 0.0, 0.999999) * u_hogel_h) + 0.5;

    float atlas_px = float(i) * u_hogel_w + local_px;
    float atlas_py = float(j) * u_hogel_h + local_py;

    vec2 atlas_uv = vec2(atlas_px / u_atlas_w, atlas_py / u_atlas_h);
    return texture(u_atlas, atlas_uv);
}

void main() {
    // Construct the viewer ray explicitly from this output pixel.
    float ndc_x = 2.0 * v_uv.x - 1.0;
    float ndc_y = 2.0 * v_uv.y - 1.0;

    float sx = ndc_x * u_view_aspect * u_tan_half_view_fov_y;
    float sy = ndc_y * u_tan_half_view_fov_y;

    vec3 ray_dir_world = normalize(
        u_cam_forward + sx * u_cam_right + sy * u_cam_up
    );

    // Ray-plane intersection:
    //     R(t) = eye + t * ray_dir_world
    //     dot(R(t) - center, holo_forward) = 0
    float denom = dot(ray_dir_world, u_holo_forward);
    if (abs(denom) < 1.0e-8) {
        f_color = vec4(u_bg_color, 1.0);
        return;
    }

    float t = dot(u_holo_center - u_eye, u_holo_forward) / denom;
    if (t <= 0.0) {
        f_color = vec4(u_bg_color, 1.0);
        return;
    }

    vec3 p = u_eye + t * ray_dir_world;
    vec3 rel = p - u_holo_center;

    // Local hologram coordinates in the rotated hologram frame.
    float a = dot(rel, u_holo_right);
    float b = dot(rel, u_holo_up);

    float half_w = 0.5 * u_screen_width;
    float half_h = 0.5 * u_screen_height;

    if (a < -half_w || a > half_w || b < -half_h || b > half_h) {
        f_color = vec4(u_bg_color, 1.0);
        return;
    }

    int i = int(floor((a - u_xmin) / u_h));
    int j = int(floor((b - u_zmin) / u_h));

    vec4 col = sample_hogel_nearest(i, j, ray_dir_world);

    if (u_show_grid != 0) {
        vec2 cell = vec2((a - u_xmin) / u_h, (b - u_zmin) / u_h);
        float gx = min(fract(cell.x), 1.0 - fract(cell.x));
        float gz = min(fract(cell.y), 1.0 - fract(cell.y));

        float wx = max(fwidth(cell.x) * 1.5, 1.0e-4);
        float wz = max(fwidth(cell.y) * 1.5, 1.0e-4);

        float lx = 1.0 - smoothstep(0.0, wx, gx);
        float lz = 1.0 - smoothstep(0.0, wz, gz);
        float line = max(lx, lz);

        col.rgb = mix(col.rgb, vec3(0.0, 0.85, 1.0), 0.35 * line);
    }

    f_color = vec4(col.rgb, 1.0);
}
"""


# ============================================================
# DATA LOADING
# ============================================================

HOGEL_RE = re.compile(
    r"^hogel_(\d+)_(\d+)_(\d+)_(\d+)\.(png|jpg|jpeg|bmp|tif|tiff)$",
    re.IGNORECASE,
)


@dataclass
class HogelFile:
    path: Path
    i: int
    na: int
    j: int
    nb: int


@dataclass
class HogelAtlas:
    directory: Path
    na: int
    nb: int
    hogel_w: int
    hogel_h: int
    atlas_w: int
    atlas_h: int
    atlas_rgba: np.ndarray
    missing_count: int


def parse_hogel_filename(path: Path) -> Optional[HogelFile]:
    m = HOGEL_RE.match(path.name)
    if not m:
        return None

    i = int(m.group(1))
    na = int(m.group(2))
    j = int(m.group(3))
    nb = int(m.group(4))
    return HogelFile(path=path, i=i, na=na, j=j, nb=nb)


def resolve_hogel_dir(path_string: str) -> Path:
    p = Path(path_string).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (SCRIPT_DIR / p).resolve()


def discover_hogel_files(directory: Path) -> Tuple[int, int, Dict[Tuple[int, int], Path]]:
    if not directory.exists():
        raise FileNotFoundError(f"Hogel directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Hogel path is not a directory: {directory}")

    parsed: List[HogelFile] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        hf = parse_hogel_filename(path)
        if hf is not None:
            parsed.append(hf)

    if not parsed:
        raise RuntimeError(
            f"No hogel images found in {directory}. Expected names like "
            f"hogel_0000_0004_0000_0004.png"
        )

    totals = {(hf.na, hf.nb) for hf in parsed}
    if len(totals) != 1:
        raise RuntimeError(
            "Inconsistent Na/Nb totals in filenames. Found: "
            + ", ".join(str(t) for t in sorted(totals))
        )

    na, nb = next(iter(totals))

    files: Dict[Tuple[int, int], Path] = {}
    duplicates = 0

    for hf in parsed:
        if hf.i < 0 or hf.i >= na or hf.j < 0 or hf.j >= nb:
            raise RuntimeError(f"Out-of-range hogel index in filename: {hf.path.name}")

        key = (hf.i, hf.j)
        if key in files:
            duplicates += 1
        files[key] = hf.path

    if duplicates:
        print(f"Warning: found {duplicates} duplicate hogel indices; using the last one seen.")

    return na, nb, files


def validate_hogel_counts_against_params(na: int, nb: int) -> None:
    expected_na = int(math.floor(REQUESTED_W / HOGEL_CELL_SIZE))
    expected_nb = int(math.floor(REQUESTED_H / HOGEL_CELL_SIZE))

    if na != expected_na or nb != expected_nb:
        raise RuntimeError(
            "Hogel filenames do not match hogel_params.py.\n"
            f"  Filenames say:       Na={na}, Nb={nb}\n"
            f"  hogel_params.py says: Na=floor(W/h)={expected_na}, "
            f"Nb=floor(H/h)={expected_nb}\n"
            f"  W={REQUESTED_W}, H={REQUESTED_H}, h={HOGEL_CELL_SIZE}\n"
            "Regenerate the rig/render, or edit hogel_params.py to match the hogel set."
        )


def get_flip_top_bottom_constant():
    if hasattr(Image, "Transpose"):
        return Image.Transpose.FLIP_TOP_BOTTOM
    return Image.FLIP_TOP_BOTTOM


def load_hogel_atlas(directory: str) -> HogelAtlas:
    directory_path = resolve_hogel_dir(directory)
    na, nb, files = discover_hogel_files(directory_path)
    validate_hogel_counts_against_params(na, nb)

    first_path = next(iter(files.values()))
    with Image.open(first_path) as im:
        first = im.convert("RGBA")
        hogel_w, hogel_h = first.size

    atlas_w = na * hogel_w
    atlas_h = nb * hogel_h
    expected_count = na * nb
    found_count = len(files)
    missing_count = expected_count - found_count

    print("")
    print("Hogel set")
    print(f"  directory:     {directory_path}")
    print(f"  Na x Nb:       {na} x {nb} = {expected_count}")
    print(f"  hogel size:    {hogel_w} x {hogel_h}")
    print(f"  atlas size:    {atlas_w} x {atlas_h}")
    print(f"  files found:   {found_count}")
    print(f"  files missing: {missing_count}")
    print("")

    if missing_count > 0:
        print("Warning: missing hogels will appear black.")

    atlas = np.zeros((atlas_h, atlas_w, 4), dtype=np.uint8)

    flip_tb = get_flip_top_bottom_constant()
    loaded = 0
    t0 = time.time()

    for j in range(nb):
        for i in range(na):
            path = files.get((i, j))
            if path is None:
                continue

            with Image.open(path) as im:
                rgba = im.convert("RGBA")

                if rgba.size != (hogel_w, hogel_h):
                    raise RuntimeError(
                        f"Image size mismatch in {path.name}: got {rgba.size}, "
                        f"expected {(hogel_w, hogel_h)}"
                    )

                if FLIP_HOGEL_IMAGES_TOP_BOTTOM:
                    rgba = rgba.transpose(flip_tb)

                arr = np.asarray(rgba, dtype=np.uint8)

            x0 = i * hogel_w
            y0 = j * hogel_h
            atlas[y0 : y0 + hogel_h, x0 : x0 + hogel_w, :] = arr

            loaded += 1
            if loaded % LOAD_PROGRESS_EVERY == 0 or loaded == found_count:
                elapsed = time.time() - t0
                rate = loaded / elapsed if elapsed > 0 else 0.0
                print(
                    f"Loading hogels: {loaded}/{found_count} "
                    f"({100.0 * loaded / found_count:5.1f}%)  "
                    f"{rate:6.1f} img/s",
                    flush=True,
                )

    print("Done loading hogels into CPU atlas.")
    print("")

    return HogelAtlas(
        directory=directory_path,
        na=na,
        nb=nb,
        hogel_w=hogel_w,
        hogel_h=hogel_h,
        atlas_w=atlas_w,
        atlas_h=atlas_h,
        atlas_rgba=atlas,
        missing_count=missing_count,
    )


# ============================================================
# MATH
# ============================================================


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def infer_yaw_pitch_from_direction(direction: np.ndarray) -> Tuple[float, float]:
    d = normalize(direction)
    yaw = math.atan2(float(d[0]), float(d[1]))
    pitch = math.asin(max(-1.0, min(1.0, float(d[2]))))
    return yaw, pitch


def rotation_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float32,
    )


def rotation_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )


def rotation_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def initial_hologram_basis() -> np.ndarray:
    """
    Returns a 3x3 local-to-world basis matrix whose columns are:
        local +X -> hologram right
        local +Y -> hologram inward/forward
        local +Z -> hologram up
    """
    forward = normalize(INWARD_NORMAL)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    if abs(float(np.dot(forward, world_up))) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    right = normalize(np.cross(forward, world_up))
    up = normalize(np.cross(right, forward))

    return np.column_stack((right, forward, up)).astype(np.float32)


class HologramPose:
    def __init__(self, center: np.ndarray):
        self.center = center.astype(np.float32).copy()
        self.base_basis = initial_hologram_basis()
        self.reset()

    def reset(self) -> None:
        self.yaw = math.radians(INITIAL_HOLOGRAM_YAW_DEGREES)
        self.pitch = math.radians(INITIAL_HOLOGRAM_PITCH_DEGREES)
        self.roll = math.radians(INITIAL_HOLOGRAM_ROLL_DEGREES)

    def basis(self) -> np.ndarray:
        return (
            rotation_z(self.yaw)
            @ self.base_basis
            @ rotation_x(self.pitch)
            @ rotation_y(self.roll)
        ).astype(np.float32)

    def axes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        b = self.basis()
        right = b[:, 0].astype(np.float32)
        forward = b[:, 1].astype(np.float32)
        up = b[:, 2].astype(np.float32)
        return right, forward, up

    def handle_mouse_delta(self, dx: float, dy: float) -> None:
        self.yaw += dx * HOLOGRAM_MOUSE_SENSITIVITY
        self.pitch += dy * HOLOGRAM_MOUSE_SENSITIVITY

    def handle_keys(self, keys, dt: float) -> None:
        speed = math.radians(HOLOGRAM_ROTATE_SPEED_DEGREES_PER_SECOND) * dt
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            speed *= FAST_MOVE_MULTIPLIER

        if keys[pygame.K_LEFT]:
            self.yaw -= speed
        if keys[pygame.K_RIGHT]:
            self.yaw += speed
        if keys[pygame.K_UP]:
            self.pitch += speed
        if keys[pygame.K_DOWN]:
            self.pitch -= speed
        if keys[pygame.K_PAGEUP]:
            self.roll += speed
        if keys[pygame.K_PAGEDOWN]:
            self.roll -= speed

    def angles_degrees(self) -> Tuple[float, float, float]:
        return (
            math.degrees(self.yaw),
            math.degrees(self.pitch),
            math.degrees(self.roll),
        )


# ============================================================
# VIEWER CAMERA
# ============================================================


class ViewerCamera:
    def __init__(self, eye: np.ndarray, look_at_point: np.ndarray):
        self.eye = eye.astype(np.float32).copy()
        self.yaw, self.pitch = infer_yaw_pitch_from_direction(look_at_point - eye)

    def forward(self) -> np.ndarray:
        cp = math.cos(self.pitch)
        return normalize(
            np.array(
                [
                    cp * math.sin(self.yaw),
                    cp * math.cos(self.yaw),
                    math.sin(self.pitch),
                ],
                dtype=np.float32,
            )
        )

    def right(self) -> np.ndarray:
        f = self.forward()
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return normalize(np.cross(f, world_up))

    def up(self) -> np.ndarray:
        r = self.right()
        f = self.forward()
        return normalize(np.cross(r, f))

    def handle_mouse_delta(self, dx: float, dy: float) -> None:
        self.yaw += dx * MOUSE_SENSITIVITY
        self.pitch -= dy * MOUSE_SENSITIVITY

        limit = math.radians(89.0)
        self.pitch = max(-limit, min(limit, self.pitch))

    def move(self, keys, dt: float) -> None:
        speed = BASE_MOVE_SPEED
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            speed *= FAST_MOVE_MULTIPLIER

        f = self.forward()
        r = self.right()
        z = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        delta = np.zeros(3, dtype=np.float32)

        if keys[pygame.K_w]:
            delta += f
        if keys[pygame.K_s]:
            delta -= f
        if keys[pygame.K_d]:
            delta += r
        if keys[pygame.K_a]:
            delta -= r
        if keys[pygame.K_e]:
            delta += z
        if keys[pygame.K_q]:
            delta -= z

        if float(np.linalg.norm(delta)) > 0.0:
            self.eye += normalize(delta) * speed * dt

    def reset(self) -> None:
        self.eye = INITIAL_EYE.astype(np.float32).copy()
        self.yaw, self.pitch = infer_yaw_pitch_from_direction(INITIAL_LOOK_AT - INITIAL_EYE)


# ============================================================
# GPU / RENDER SETUP
# ============================================================


def make_fullscreen_quad() -> np.ndarray:
    return np.array(
        [
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0,
        ],
        dtype=np.float32,
    )


def get_context_int(ctx, name: str, default: Optional[int] = None) -> Optional[int]:
    try:
        value = ctx.info.get(name, default)
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def upload_atlas_texture(ctx, atlas: HogelAtlas):
    max_tex = get_context_int(ctx, "GL_MAX_TEXTURE_SIZE")
    if max_tex is not None:
        print(f"GPU GL_MAX_TEXTURE_SIZE: {max_tex}")
        if atlas.atlas_w > max_tex or atlas.atlas_h > max_tex:
            raise RuntimeError(
                f"Atlas is too large for this OpenGL context: "
                f"{atlas.atlas_w} x {atlas.atlas_h}, max texture size {max_tex}. "
                f"Use smaller hogels, fewer hogels, or a tiled/multi-texture version."
            )

    print("Uploading atlas to GPU...")
    t0 = time.time()

    tex = ctx.texture((atlas.atlas_w, atlas.atlas_h), 4, data=atlas.atlas_rgba)

    # Deliberately NEAREST. No OpenGL bilinear interpolation.
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    tex.repeat_x = False
    tex.repeat_y = False

    elapsed = time.time() - t0
    gib = atlas.atlas_rgba.nbytes / (1024.0 ** 3)
    print(f"Uploaded {gib:.3f} GiB atlas in {elapsed:.2f} s.")
    print("")
    return tex


def set_static_uniforms(program, atlas: HogelAtlas, screen_width: float, screen_height: float) -> None:
    h = float(HOGEL_CELL_SIZE)
    xmin = -screen_width / 2.0
    zmin = -screen_height / 2.0

    fov_x = math.radians(HOGEL_FOV_X_DEGREES)
    aspect = atlas.hogel_w / atlas.hogel_h
    fov_y = 2.0 * math.atan(math.tan(fov_x / 2.0) / aspect)

    program["u_atlas"].value = 0
    program["u_xmin"].value = xmin
    program["u_zmin"].value = zmin
    program["u_h"].value = h
    program["u_screen_width"].value = float(screen_width)
    program["u_screen_height"].value = float(screen_height)
    program["u_na"].value = atlas.na
    program["u_nb"].value = atlas.nb
    program["u_hogel_w"].value = float(atlas.hogel_w)
    program["u_hogel_h"].value = float(atlas.hogel_h)
    program["u_atlas_w"].value = float(atlas.atlas_w)
    program["u_atlas_h"].value = float(atlas.atlas_h)
    program["u_tan_half_hogel_fov_x"].value = math.tan(fov_x / 2.0)
    program["u_tan_half_hogel_fov_y"].value = math.tan(fov_y / 2.0)
    program["u_bg_color"].value = BACKGROUND_RGB

    print("Simulator geometry from hogel_params.py")
    print(f"  screen center:       ({X0}, {Y0}, {Z0})")
    print(f"  requested size:      {REQUESTED_W} x {REQUESTED_H}")
    print(f"  active sampled size: {screen_width} x {screen_height}")
    print(f"  cell size:           {h}")
    print(f"  local xmin/zmin:     {xmin}, {zmin}")
    print(f"  inward normal:       {tuple(float(v) for v in normalize(INWARD_NORMAL))}")
    print(f"  hogel FOV x/y:       {math.degrees(fov_x):.3f} / {math.degrees(fov_y):.3f} deg")
    print("  sampling:            nearest hogel, nearest hogel pixel, no interpolation")
    print("")


def set_hologram_pose_uniforms(program, hologram: HologramPose) -> None:
    right, forward, up = hologram.axes()
    program["u_holo_center"].value = tuple(float(v) for v in hologram.center)
    program["u_holo_right"].value = tuple(float(v) for v in right)
    program["u_holo_forward"].value = tuple(float(v) for v in forward)
    program["u_holo_up"].value = tuple(float(v) for v in up)


# ============================================================
# MAIN LOOP
# ============================================================


def print_controls() -> None:
    print("Controls")
    print("  W/S              move viewer forward/back")
    print("  A/D              move viewer left/right")
    print("  Q/E              move viewer down/up")
    print("  Shift            move/rotate faster")
    print("  Right drag       look around with viewer camera")
    print("  Left drag        rotate hologram yaw/pitch")
    print("  Arrow keys       rotate hologram yaw/pitch")
    print("  PageUp/PageDown  roll hologram")
    print("  R                reset viewer camera")
    print("  Home             reset hologram rotation")
    print("  G                toggle cell grid overlay")
    print("  Esc              quit")
    print("")


def main() -> None:
    global SHOW_CELL_GRID_OVERLAY

    print_controls()

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK,
        pygame.GL_CONTEXT_PROFILE_CORE,
    )
    pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)

    flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), flags)
    pygame.display.set_caption("Hogel / Light-Field Hologram Viewer")

    ctx = moderngl.create_context()
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.CULL_FACE)

    atlas = load_hogel_atlas(HOGEL_DIR)

    screen_width = atlas.na * HOGEL_CELL_SIZE
    screen_height = atlas.nb * HOGEL_CELL_SIZE

    atlas_tex = upload_atlas_texture(ctx, atlas)
    atlas_tex.use(location=0)

    program = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
    set_static_uniforms(program, atlas, screen_width, screen_height)

    hologram = HologramPose(np.array([X0, Y0, Z0], dtype=np.float32))
    set_hologram_pose_uniforms(program, hologram)

    quad = make_fullscreen_quad()
    vbo = ctx.buffer(quad.tobytes())
    vao = ctx.vertex_array(program, [(vbo, "2f", "in_pos")])

    camera = ViewerCamera(INITIAL_EYE, INITIAL_LOOK_AT)

    clock = pygame.time.Clock()
    running = True
    left_mouse_down = False
    right_mouse_down = False
    last_caption_update = 0.0

    program["u_show_grid"].value = 1 if SHOW_CELL_GRID_OVERLAY else 0

    print("Viewer started.")
    print("")

    while running:
        dt = clock.tick(120) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    camera.reset()
                elif event.key == pygame.K_HOME:
                    hologram.reset()
                elif event.key == pygame.K_g:
                    SHOW_CELL_GRID_OVERLAY = not SHOW_CELL_GRID_OVERLAY
                    program["u_show_grid"].value = 1 if SHOW_CELL_GRID_OVERLAY else 0
                    print(f"Grid overlay: {SHOW_CELL_GRID_OVERLAY}")

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    left_mouse_down = True
                elif event.button == 3:
                    right_mouse_down = True
                    pygame.event.set_grab(True)
                    pygame.mouse.set_visible(False)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    left_mouse_down = False
                elif event.button == 3:
                    right_mouse_down = False
                    pygame.event.set_grab(False)
                    pygame.mouse.set_visible(True)

            elif event.type == pygame.MOUSEMOTION:
                if right_mouse_down:
                    dx, dy = event.rel
                    camera.handle_mouse_delta(dx, dy)
                elif left_mouse_down:
                    dx, dy = event.rel
                    hologram.handle_mouse_delta(dx, dy)

        keys = pygame.key.get_pressed()
        camera.move(keys, dt)
        hologram.handle_keys(keys, dt)

        width, height = pygame.display.get_surface().get_size()
        width = max(1, width)
        height = max(1, height)
        ctx.viewport = (0, 0, width, height)

        cam_forward = camera.forward()
        cam_right = camera.right()
        cam_up = camera.up()

        program["u_eye"].value = tuple(float(v) for v in camera.eye)
        program["u_cam_forward"].value = tuple(float(v) for v in cam_forward)
        program["u_cam_right"].value = tuple(float(v) for v in cam_right)
        program["u_cam_up"].value = tuple(float(v) for v in cam_up)
        program["u_tan_half_view_fov_y"].value = math.tan(math.radians(VIEWER_FOV_Y_DEGREES) / 2.0)
        program["u_view_aspect"].value = width / height

        set_hologram_pose_uniforms(program, hologram)

        ctx.clear(*BACKGROUND_RGB, 1.0)
        atlas_tex.use(location=0)
        vao.render(moderngl.TRIANGLES)

        pygame.display.flip()

        now = time.time()
        if now - last_caption_update > 0.25:
            fps = clock.get_fps()
            grid = "grid" if SHOW_CELL_GRID_OVERLAY else "no-grid"
            hyaw, hpitch, hroll = hologram.angles_degrees()
            pygame.display.set_caption(
                f"Hogel Viewer (raytraced, nearest) | {fps:5.1f} FPS | {grid} | "
                f"eye=({camera.eye[0]:.2f}, {camera.eye[1]:.2f}, {camera.eye[2]:.2f}) | "
                f"hologram yaw/pitch/roll=({hyaw:.1f}, {hpitch:.1f}, {hroll:.1f})"
            )
            last_caption_update = now

    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("")
        print("ERROR:")
        print(exc)
        print("")
        pygame.quit()
        raise
