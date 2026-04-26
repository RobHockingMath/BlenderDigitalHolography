"""
hologram_lightfield_viewer_fixed_conv8_bilinear_full_tiled.py

Raytraced hogel/light-field viewer.

Important interaction model:
    - The hologram plane is FIXED.
    - The viewer/eye moves around the hologram.
    - The camera always looks at the hologram center.
    - No hologram rotation.
    - Bilinear interpolation across neighboring hogels.
    - Bilinear interpolation within each hogel image, clamped to that hogel.
    - Tiled atlas textures, so large hogel sets can exceed GL_MAX_TEXTURE_SIZE in total.

For each output pixel:
    1. construct a viewer ray,
    2. intersect it with the fixed hologram plane,
    3. hit point determines the four neighboring hogels around the hit location,
    4. incident angle chooses a clamped subpixel location inside each hogel,
    5. each hogel is sampled bilinearly within its own image,
    6. the four neighboring hogels are then bilinearly blended.

Convention:
    fixed to mode 0x8, which you found to be correct.
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
        "folder as this viewer."
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
INITIAL_EYE = np.array(params.INITIAL_EYE, dtype=np.float32)
INITIAL_LOOK_AT = np.array(params.INITIAL_LOOK_AT, dtype=np.float32)
BASE_MOVE_SPEED = float(params.BASE_MOVE_SPEED)
FAST_MOVE_MULTIPLIER = float(params.FAST_MOVE_MULTIPLIER)
MOUSE_SENSITIVITY = float(params.MOUSE_SENSITIVITY)
SHOW_CELL_GRID_OVERLAY = bool(params.SHOW_CELL_GRID_OVERLAY)
BACKGROUND_RGB = tuple(float(x) for x in params.BACKGROUND_RGB)
LOAD_PROGRESS_EVERY = int(params.LOAD_PROGRESS_EVERY)

HOLOGRAM_CENTER = np.array([X0, Y0, Z0], dtype=np.float32)
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float32)


# ============================================================
# CONVENTION MODES
# ============================================================

# Four independent binary flips:
#   bit 0: flip hogel i index
#   bit 1: flip hogel j index
#   bit 2: flip pixel u inside each hogel
#   bit 3: flip pixel v inside each hogel
CONVENTION_MODES = []
for mask in range(16):
    CONVENTION_MODES.append({
        "flip_hogel_i": (mask >> 0) & 1,
        "flip_hogel_j": (mask >> 1) & 1,
        "flip_pixel_u": (mask >> 2) & 1,
        "flip_pixel_v": (mask >> 3) & 1,
    })

FIXED_CONVENTION_INDEX = 8


# ============================================================
# SHADERS
# ============================================================

VERTEX_SHADER = r"""#version 330 core

in vec2 in_pos;
out vec2 v_uv;

void main() {
    v_uv = 0.5 * (in_pos + vec2(1.0, 1.0));
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = r"""#version 330 core

// The hogel set is split into up to 16 atlas page textures.
// Each page is a normal 2D texture small enough to respect GL_MAX_TEXTURE_SIZE.
uniform sampler2D u_page0;
uniform sampler2D u_page1;
uniform sampler2D u_page2;
uniform sampler2D u_page3;
uniform sampler2D u_page4;
uniform sampler2D u_page5;
uniform sampler2D u_page6;
uniform sampler2D u_page7;
uniform sampler2D u_page8;
uniform sampler2D u_page9;
uniform sampler2D u_page10;
uniform sampler2D u_page11;
uniform sampler2D u_page12;
uniform sampler2D u_page13;
uniform sampler2D u_page14;
uniform sampler2D u_page15;

// Viewer camera
uniform vec3  u_eye;
uniform vec3  u_cam_forward;
uniform vec3  u_cam_right;
uniform vec3  u_cam_up;
uniform float u_tan_half_view_fov_y;
uniform float u_view_aspect;

// Fixed hologram pose
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

// Hogel/page metadata
uniform float u_hogel_w;
uniform float u_hogel_h;
uniform int   u_page_hogels_x;
uniform int   u_page_hogels_y;
uniform int   u_pages_x;
uniform float u_page_w;
uniform float u_page_h;
uniform float u_tan_half_hogel_fov_x;
uniform float u_tan_half_hogel_fov_y;

// Convention flags
uniform int u_flip_hogel_i;
uniform int u_flip_hogel_j;
uniform int u_flip_pixel_u;
uniform int u_flip_pixel_v;

uniform int   u_show_grid;
uniform vec3  u_bg_color;

in vec2 v_uv;
out vec4 f_color;

vec4 sample_page(int layer, vec2 uv) {
    if (layer == 0)  return texture(u_page0,  uv);
    if (layer == 1)  return texture(u_page1,  uv);
    if (layer == 2)  return texture(u_page2,  uv);
    if (layer == 3)  return texture(u_page3,  uv);
    if (layer == 4)  return texture(u_page4,  uv);
    if (layer == 5)  return texture(u_page5,  uv);
    if (layer == 6)  return texture(u_page6,  uv);
    if (layer == 7)  return texture(u_page7,  uv);
    if (layer == 8)  return texture(u_page8,  uv);
    if (layer == 9)  return texture(u_page9,  uv);
    if (layer == 10) return texture(u_page10, uv);
    if (layer == 11) return texture(u_page11, uv);
    if (layer == 12) return texture(u_page12, uv);
    if (layer == 13) return texture(u_page13, uv);
    if (layer == 14) return texture(u_page14, uv);
    if (layer == 15) return texture(u_page15, uv);
    return vec4(u_bg_color, 1.0);
}

vec4 sample_texel_from_hogel(int i, int j, int px, int py) {
    // Map global hogel coordinates (i,j) into a tiled atlas page.
    int page_x = i / u_page_hogels_x;
    int page_y = j / u_page_hogels_y;
    int layer = page_y * u_pages_x + page_x;

    int local_i = i - page_x * u_page_hogels_x;
    int local_j = j - page_y * u_page_hogels_y;

    float page_px = float(local_i) * u_hogel_w + float(px) + 0.5;
    float page_py = float(local_j) * u_hogel_h + float(py) + 0.5;

    vec2 page_uv = vec2(page_px / u_page_w, page_py / u_page_h);
    return sample_page(layer, page_uv);
}

vec4 sample_hogel_clamped_bilinear(int i, int j, vec3 ray_dir_world) {
    i = clamp(i, 0, u_na - 1);
    j = clamp(j, 0, u_nb - 1);

    float d_forward = dot(ray_dir_world, u_holo_forward);
    if (d_forward <= 1.0e-6) {
        return vec4(u_bg_color, 1.0);
    }

    float sx = dot(ray_dir_world, u_holo_right) / (d_forward * u_tan_half_hogel_fov_x);
    float sy = dot(ray_dir_world, u_holo_up)    / (d_forward * u_tan_half_hogel_fov_y);

    float u = 0.5 + 0.5 * sx;
    float v = 0.5 + 0.5 * sy;

    if (u_flip_pixel_u != 0) {
        u = 1.0 - u;
    }
    if (u_flip_pixel_v != 0) {
        v = 1.0 - v;
    }

    // Clamp within each hogel image. Do not reject out-of-range angles.
    // Then bilinearly interpolate strictly INSIDE this hogel tile only.
    float x = clamp(u, 0.0, 1.0) * (u_hogel_w - 1.0);
    float y = clamp(v, 0.0, 1.0) * (u_hogel_h - 1.0);

    int x0 = int(floor(x));
    int y0 = int(floor(y));
    int x1 = min(x0 + 1, int(u_hogel_w) - 1);
    int y1 = min(y0 + 1, int(u_hogel_h) - 1);

    float tx = x - floor(x);
    float ty = y - floor(y);

    vec4 c00 = sample_texel_from_hogel(i, j, x0, y0);
    vec4 c10 = sample_texel_from_hogel(i, j, x1, y0);
    vec4 c01 = sample_texel_from_hogel(i, j, x0, y1);
    vec4 c11 = sample_texel_from_hogel(i, j, x1, y1);

    vec4 cx0 = mix(c00, c10, tx);
    vec4 cx1 = mix(c01, c11, tx);
    return mix(cx0, cx1, ty);
}

vec4 sample_hogel_bilinear(float cell_x, float cell_y, vec3 ray_dir_world) {
    // Interpolate between hogel CENTERS, not hogel vertices.
    float gx = cell_x - 0.5;
    float gy = cell_y - 0.5;

    if (u_flip_hogel_i != 0) {
        gx = float(u_na - 1) - gx;
    }
    if (u_flip_hogel_j != 0) {
        gy = float(u_nb - 1) - gy;
    }

    int i0 = int(floor(gx));
    int j0 = int(floor(gy));
    int i1 = i0 + 1;
    int j1 = j0 + 1;

    float tx = gx - floor(gx);
    float ty = gy - floor(gy);

    vec4 c00 = sample_hogel_clamped_bilinear(i0, j0, ray_dir_world);
    vec4 c10 = sample_hogel_clamped_bilinear(i1, j0, ray_dir_world);
    vec4 c01 = sample_hogel_clamped_bilinear(i0, j1, ray_dir_world);
    vec4 c11 = sample_hogel_clamped_bilinear(i1, j1, ray_dir_world);

    vec4 cx0 = mix(c00, c10, tx);
    vec4 cx1 = mix(c01, c11, tx);
    return mix(cx0, cx1, ty);
}

void main() {
    // Construct viewer ray from this output pixel.
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

    // Local hologram coordinates.
    float a = dot(rel, u_holo_right);
    float b = dot(rel, u_holo_up);

    float half_w = 0.5 * u_screen_width;
    float half_h = 0.5 * u_screen_height;

    if (a < -half_w || a > half_w || b < -half_h || b > half_h) {
        f_color = vec4(u_bg_color, 1.0);
        return;
    }

    float cell_x = (a - u_xmin) / u_h;
    float cell_y = (b - u_zmin) / u_h;

    vec4 col = sample_hogel_bilinear(cell_x, cell_y, ray_dir_world);

    if (u_show_grid != 0) {
        vec2 cell = vec2(cell_x, cell_y);
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
class HogelSet:
    directory: Path
    na: int
    nb: int
    hogel_w: int
    hogel_h: int
    files: Dict[Tuple[int, int], Path]
    missing_count: int


@dataclass
class PageInfo:
    page_hogels_x: int
    page_hogels_y: int
    pages_x: int
    pages_y: int
    page_count: int
    page_w: int
    page_h: int


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def resolve_hogel_dir(path_string: str) -> Path:
    p = Path(path_string).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (SCRIPT_DIR / p).resolve()


def parse_hogel_filename(path: Path) -> Optional[HogelFile]:
    m = HOGEL_RE.match(path.name)
    if not m:
        return None
    return HogelFile(
        path=path,
        i=int(m.group(1)),
        na=int(m.group(2)),
        j=int(m.group(3)),
        nb=int(m.group(4)),
    )


def discover_hogel_files(directory: Path) -> Tuple[int, int, Dict[Tuple[int, int], Path]]:
    if not directory.exists():
        raise FileNotFoundError(f"Hogel directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Hogel path is not a directory: {directory}")

    parsed: List[HogelFile] = []
    for path in directory.iterdir():
        if path.is_file():
            hf = parse_hogel_filename(path)
            if hf is not None:
                parsed.append(hf)

    if not parsed:
        raise RuntimeError(
            f"No hogel images found in {directory}. Expected names like "
            f"hogel_0000_0060_0000_0060.png"
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
            f"  Filenames say:        Na={na}, Nb={nb}\n"
            f"  hogel_params.py says: Na={expected_na}, Nb={expected_nb}\n"
            f"  W={REQUESTED_W}, H={REQUESTED_H}, h={HOGEL_CELL_SIZE}"
        )


def load_hogel_set(directory: str) -> HogelSet:
    """
    Read filenames and image size only. Do NOT build a giant CPU atlas here.
    Large sets are uploaded later page-by-page as separate GPU textures.
    """
    directory_path = resolve_hogel_dir(directory)
    na, nb, files = discover_hogel_files(directory_path)
    validate_hogel_counts_against_params(na, nb)

    first_path = next(iter(files.values()))
    with Image.open(first_path) as im:
        hogel_w, hogel_h = im.convert("RGBA").size

    expected_count = na * nb
    found_count = len(files)
    missing_count = expected_count - found_count

    print("")
    print("Hogel set")
    print(f"  directory:     {directory_path}")
    print(f"  Na x Nb:       {na} x {nb} = {expected_count}")
    print(f"  hogel size:    {hogel_w} x {hogel_h}")
    print(f"  files found:   {found_count}")
    print(f"  files missing: {missing_count}")
    print("")

    if missing_count > 0:
        print("Warning: missing hogels will appear black.")

    return HogelSet(
        directory=directory_path,
        na=na,
        nb=nb,
        hogel_w=hogel_w,
        hogel_h=hogel_h,
        files=files,
        missing_count=missing_count,
    )


# ============================================================
# FIXED HOLOGRAM BASIS
# ============================================================

def fixed_hologram_axes() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = normalize(INWARD_NORMAL)
    up_ref = WORLD_UP.copy()
    if abs(float(np.dot(forward, up_ref))) > 0.98:
        up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    right = normalize(np.cross(forward, up_ref))
    up = normalize(np.cross(right, forward))
    return right, forward, up


# ============================================================
# ORBIT VIEWER CAMERA
# ============================================================

class OrbitViewerCamera:
    """
    The hologram is fixed. This camera moves/orbits around the hologram center
    and always looks at the hologram center.
    """
    def __init__(self, eye: np.ndarray, target: np.ndarray):
        self.target = target.astype(np.float32).copy()
        self.reset_eye = eye.astype(np.float32).copy()
        self.reset()

    def reset(self) -> None:
        v = self.reset_eye - self.target
        self.radius = max(0.05, float(np.linalg.norm(v)))

        # Spherical coordinates for vector from target to eye:
        # yaw=0 puts eye on negative Y side for the default setup.
        self.yaw = math.atan2(float(v[0]), float(-v[1]))
        horiz = math.sqrt(float(v[0] * v[0] + v[1] * v[1]))
        self.pitch = math.atan2(float(v[2]), horiz)

    def eye(self) -> np.ndarray:
        cp = math.cos(self.pitch)
        x = self.radius * cp * math.sin(self.yaw)
        y = -self.radius * cp * math.cos(self.yaw)
        z = self.radius * math.sin(self.pitch)
        return (self.target + np.array([x, y, z], dtype=np.float32)).astype(np.float32)

    def forward(self) -> np.ndarray:
        return normalize(self.target - self.eye())

    def right(self) -> np.ndarray:
        return normalize(np.cross(self.forward(), WORLD_UP))

    def up(self) -> np.ndarray:
        return normalize(np.cross(self.right(), self.forward()))

    def orbit_mouse_delta(self, dx: float, dy: float) -> None:
        self.yaw += dx * MOUSE_SENSITIVITY
        self.pitch += dy * MOUSE_SENSITIVITY

        limit = math.radians(89.0)
        self.pitch = max(-limit, min(limit, self.pitch))

    def handle_keys(self, keys, dt: float) -> None:
        speed = BASE_MOVE_SPEED
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            speed *= FAST_MOVE_MULTIPLIER

        angular = speed * 0.35 * dt
        dolly = speed * dt

        # Hex convention keys use A-F, so orbit uses arrows instead.
        if keys[pygame.K_LEFT]:
            self.yaw -= angular
        if keys[pygame.K_RIGHT]:
            self.yaw += angular
        if keys[pygame.K_DOWN]:
            self.pitch -= angular
        if keys[pygame.K_UP]:
            self.pitch += angular

        # Dolly toward/away from the hologram center.
        if keys[pygame.K_w]:
            self.radius = max(0.05, self.radius - dolly)
        if keys[pygame.K_s]:
            self.radius += dolly

        limit = math.radians(89.0)
        self.pitch = max(-limit, min(limit, self.pitch))


# ============================================================
# GPU SETUP
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


MAX_PAGE_TEXTURES = 16


def choose_page_layout(hogel_set: HogelSet, max_texture_size: int, max_pages: int = MAX_PAGE_TEXTURES) -> PageInfo:
    max_hogels_x = max_texture_size // hogel_set.hogel_w
    max_hogels_y = max_texture_size // hogel_set.hogel_h

    if max_hogels_x <= 0 or max_hogels_y <= 0:
        raise RuntimeError(
            f"A single hogel image is too large for this OpenGL context. "
            f"Hogel size is {hogel_set.hogel_w} x {hogel_set.hogel_h}, "
            f"max texture size is {max_texture_size}."
        )

    # Minimum number of pages needed in each direction.
    pages_x = int(math.ceil(hogel_set.na / max_hogels_x))
    pages_y = int(math.ceil(hogel_set.nb / max_hogels_y))

    # Balance the page sizes. For 150 hogels with max 128 per page, this gives
    # pages_x=2 and page_hogels_x=75, not 128+22.
    page_hogels_x = int(math.ceil(hogel_set.na / pages_x))
    page_hogels_y = int(math.ceil(hogel_set.nb / pages_y))

    page_w = page_hogels_x * hogel_set.hogel_w
    page_h = page_hogels_y * hogel_set.hogel_h
    page_count = pages_x * pages_y

    if page_w > max_texture_size or page_h > max_texture_size:
        raise RuntimeError(
            f"Internal tiling error: page size {page_w} x {page_h} exceeds "
            f"GL_MAX_TEXTURE_SIZE={max_texture_size}."
        )

    if page_count > max_pages:
        raise RuntimeError(
            f"This hogel set needs {page_count} atlas pages ({pages_x} x {pages_y}), "
            f"but this viewer currently supports at most {max_pages} page textures. "
            f"Reduce hogel count/resolution or increase MAX_PAGE_TEXTURES and the shader sampler list."
        )

    return PageInfo(
        page_hogels_x=page_hogels_x,
        page_hogels_y=page_hogels_y,
        pages_x=pages_x,
        pages_y=pages_y,
        page_count=page_count,
        page_w=page_w,
        page_h=page_h,
    )


def make_texture_from_rgba(ctx, rgba: np.ndarray):
    tex = ctx.texture((rgba.shape[1], rgba.shape[0]), 4, data=rgba)
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    tex.repeat_x = False
    tex.repeat_y = False
    return tex


def upload_tiled_atlas_textures(ctx, hogel_set: HogelSet) -> Tuple[List[object], PageInfo]:
    max_tex = get_context_int(ctx, "GL_MAX_TEXTURE_SIZE")
    if max_tex is None:
        max_tex = 16384
    print(f"GPU GL_MAX_TEXTURE_SIZE: {max_tex}")

    max_units = get_context_int(ctx, "GL_MAX_TEXTURE_IMAGE_UNITS")
    if max_units is not None:
        print(f"GPU GL_MAX_TEXTURE_IMAGE_UNITS: {max_units}")
        if max_units < MAX_PAGE_TEXTURES:
            raise RuntimeError(
                f"This viewer binds {MAX_PAGE_TEXTURES} texture units, but this OpenGL context "
                f"only reports {max_units}. Reduce MAX_PAGE_TEXTURES and the shader sampler list."
            )

    page_info = choose_page_layout(hogel_set, max_tex)

    print("")
    print("Tiled atlas layout")
    print(f"  pages:          {page_info.pages_x} x {page_info.pages_y} = {page_info.page_count}")
    print(f"  hogels/page:    {page_info.page_hogels_x} x {page_info.page_hogels_y}")
    print(f"  page texture:   {page_info.page_w} x {page_info.page_h}")
    print(f"  total GPU data: {(hogel_set.na * hogel_set.nb * hogel_set.hogel_w * hogel_set.hogel_h * 4) / (1024.0 ** 3):.3f} GiB")
    print("")

    page_textures: List[object] = []
    total_loaded = 0
    total_expected = hogel_set.na * hogel_set.nb - hogel_set.missing_count
    t_all = time.time()

    for page_y in range(page_info.pages_y):
        for page_x in range(page_info.pages_x):
            layer = page_y * page_info.pages_x + page_x
            print(f"Building atlas page {layer + 1}/{page_info.page_count} ({page_x}, {page_y})...", flush=True)

            page_rgba = np.zeros((page_info.page_h, page_info.page_w, 4), dtype=np.uint8)
            page_loaded = 0
            t_page = time.time()

            for local_j in range(page_info.page_hogels_y):
                global_j = page_y * page_info.page_hogels_y + local_j
                if global_j >= hogel_set.nb:
                    continue

                for local_i in range(page_info.page_hogels_x):
                    global_i = page_x * page_info.page_hogels_x + local_i
                    if global_i >= hogel_set.na:
                        continue

                    path = hogel_set.files.get((global_i, global_j))
                    if path is None:
                        continue

                    with Image.open(path) as im:
                        rgba = im.convert("RGBA")
                        if rgba.size != (hogel_set.hogel_w, hogel_set.hogel_h):
                            raise RuntimeError(
                                f"Image size mismatch in {path.name}: got {rgba.size}, "
                                f"expected {(hogel_set.hogel_w, hogel_set.hogel_h)}"
                            )
                        arr = np.asarray(rgba, dtype=np.uint8)

                    x0 = local_i * hogel_set.hogel_w
                    y0 = local_j * hogel_set.hogel_h
                    page_rgba[y0 : y0 + hogel_set.hogel_h, x0 : x0 + hogel_set.hogel_w, :] = arr

                    page_loaded += 1
                    total_loaded += 1

                    if total_loaded % LOAD_PROGRESS_EVERY == 0 or total_loaded == total_expected:
                        elapsed = time.time() - t_all
                        rate = total_loaded / elapsed if elapsed > 0 else 0.0
                        print(
                            f"Loading/uploading hogels: {total_loaded}/{total_expected} "
                            f"({100.0 * total_loaded / max(total_expected, 1):5.1f}%)  "
                            f"{rate:6.1f} img/s",
                            flush=True,
                        )

            print(
                f"Uploading page {layer + 1}/{page_info.page_count}: "
                f"{page_loaded} hogels, {page_rgba.nbytes / (1024.0 ** 3):.3f} GiB...",
                flush=True,
            )
            tex = make_texture_from_rgba(ctx, page_rgba)
            page_textures.append(tex)
            del page_rgba

            elapsed_page = time.time() - t_page
            print(f"Page {layer + 1}/{page_info.page_count} done in {elapsed_page:.2f} s.", flush=True)

    # Bind dummy 1x1 textures to any unused sampler units so all shader sampler
    # uniforms are valid even though only page_count are actually sampled.
    dummy_rgba = np.zeros((1, 1, 4), dtype=np.uint8)
    dummy_rgba[0, 0, 3] = 255
    dummy = make_texture_from_rgba(ctx, dummy_rgba)
    while len(page_textures) < MAX_PAGE_TEXTURES:
        page_textures.append(dummy)

    total_elapsed = time.time() - t_all
    print("")
    print(f"Finished uploading tiled atlas pages in {total_elapsed:.2f} s.")
    print("")

    return page_textures, page_info


def set_static_uniforms(program, hogel_set: HogelSet, screen_width: float, screen_height: float, page_info: PageInfo) -> None:
    h = float(HOGEL_CELL_SIZE)
    xmin = -screen_width / 2.0
    zmin = -screen_height / 2.0

    fov_x = math.radians(HOGEL_FOV_X_DEGREES)
    aspect = hogel_set.hogel_w / hogel_set.hogel_h
    fov_y = 2.0 * math.atan(math.tan(fov_x / 2.0) / aspect)

    right, forward, up = fixed_hologram_axes()

    for idx in range(MAX_PAGE_TEXTURES):
        uniform_name = f"u_page{idx}"
        if uniform_name in program:
            program[uniform_name].value = idx

    program["u_xmin"].value = xmin
    program["u_zmin"].value = zmin
    program["u_h"].value = h
    program["u_screen_width"].value = float(screen_width)
    program["u_screen_height"].value = float(screen_height)
    program["u_na"].value = hogel_set.na
    program["u_nb"].value = hogel_set.nb
    program["u_hogel_w"].value = float(hogel_set.hogel_w)
    program["u_hogel_h"].value = float(hogel_set.hogel_h)
    program["u_page_hogels_x"].value = int(page_info.page_hogels_x)
    program["u_page_hogels_y"].value = int(page_info.page_hogels_y)
    program["u_pages_x"].value = int(page_info.pages_x)
    program["u_page_w"].value = float(page_info.page_w)
    program["u_page_h"].value = float(page_info.page_h)
    program["u_tan_half_hogel_fov_x"].value = math.tan(fov_x / 2.0)
    program["u_tan_half_hogel_fov_y"].value = math.tan(fov_y / 2.0)
    program["u_bg_color"].value = BACKGROUND_RGB

    program["u_holo_center"].value = tuple(float(v) for v in HOLOGRAM_CENTER)
    program["u_holo_right"].value = tuple(float(v) for v in right)
    program["u_holo_forward"].value = tuple(float(v) for v in forward)
    program["u_holo_up"].value = tuple(float(v) for v in up)

    print("Simulator geometry from hogel_params.py")
    print(f"  screen center:       ({X0}, {Y0}, {Z0})")
    print(f"  requested size:      {REQUESTED_W} x {REQUESTED_H}")
    print(f"  active sampled size: {screen_width} x {screen_height}")
    print(f"  cell size:           {h}")
    print(f"  local xmin/zmin:     {xmin}, {zmin}")
    print(f"  inward normal:       {tuple(float(v) for v in forward)}")
    print(f"  hogel right/up:      {tuple(float(v) for v in right)} / {tuple(float(v) for v in up)}")
    print(f"  hogel FOV x/y:       {math.degrees(fov_x):.3f} / {math.degrees(fov_y):.3f} deg")
    print("  hologram pose:       FIXED, viewer orbits/moves")
    print("  sampling:            bilinear across neighboring hogels and bilinear within each hogel image")
    print("  atlas:               tiled 2D page textures")
    print("")


def convention_label(index: int) -> str:
    m = CONVENTION_MODES[index]
    return (
        f"mode 0x{index:X} | "
        f"hogel_i_flip={m['flip_hogel_i']} "
        f"hogel_j_flip={m['flip_hogel_j']} "
        f"pixel_u_flip={m['flip_pixel_u']} "
        f"pixel_v_flip={m['flip_pixel_v']}"
    )


def apply_convention_mode(program, index: int) -> None:
    m = CONVENTION_MODES[index]
    program["u_flip_hogel_i"].value = int(m["flip_hogel_i"])
    program["u_flip_hogel_j"].value = int(m["flip_hogel_j"])
    program["u_flip_pixel_u"].value = int(m["flip_pixel_u"])
    program["u_flip_pixel_v"].value = int(m["flip_pixel_v"])
    print(f"Convention: {convention_label(index)}", flush=True)


def print_fixed_convention() -> None:
    print(f"Fixed convention: {convention_label(FIXED_CONVENTION_INDEX)}")
    print("")


# ============================================================
# MAIN LOOP
# ============================================================

def print_controls() -> None:
    print("Controls")
    print("  Left/right drag  orbit viewer around fixed hologram center")
    print("  Mouse wheel      dolly viewer toward/away from hologram")
    print("  W/S              dolly viewer toward/away from hologram")
    print("  Arrow keys       orbit viewer left/right/up/down")
    print("  Shift            move/orbit faster")
    print("  R                reset viewer")
    print("  G                toggle cell grid overlay")
    print("  convention       fixed to 0x8")
    print("  Esc              quit")
    print("")


def main() -> None:
    global SHOW_CELL_GRID_OVERLAY

    print_controls()
    print_fixed_convention()

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
    pygame.display.set_caption("Hogel / Light-Field Viewer - fixed convention 0x8 - tiled full bilinear")

    ctx = moderngl.create_context()
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.CULL_FACE)

    hogel_set = load_hogel_set(HOGEL_DIR)

    screen_width = hogel_set.na * HOGEL_CELL_SIZE
    screen_height = hogel_set.nb * HOGEL_CELL_SIZE

    page_textures, page_info = upload_tiled_atlas_textures(ctx, hogel_set)
    for unit, tex in enumerate(page_textures):
        tex.use(location=unit)

    program = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
    set_static_uniforms(program, hogel_set, screen_width, screen_height, page_info)

    quad = make_fullscreen_quad()
    vbo = ctx.buffer(quad.tobytes())
    vao = ctx.vertex_array(program, [(vbo, "2f", "in_pos")])

    camera = OrbitViewerCamera(INITIAL_EYE, HOLOGRAM_CENTER)

    current_convention_index = FIXED_CONVENTION_INDEX
    apply_convention_mode(program, current_convention_index)

    clock = pygame.time.Clock()
    running = True
    mouse_orbit_down = False
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
                elif event.key == pygame.K_g:
                    SHOW_CELL_GRID_OVERLAY = not SHOW_CELL_GRID_OVERLAY
                    program["u_show_grid"].value = 1 if SHOW_CELL_GRID_OVERLAY else 0
                    print(f"Grid overlay: {SHOW_CELL_GRID_OVERLAY}")

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button in (1, 3):
                    mouse_orbit_down = True
                    # Do not grab/hide the mouse here. Some Windows/pygame setups
                    # fail to report relative motion reliably while grabbed.
                elif event.button == 4:
                    camera.radius = max(0.05, camera.radius - BASE_MOVE_SPEED * 0.15)
                elif event.button == 5:
                    camera.radius += BASE_MOVE_SPEED * 0.15

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in (1, 3):
                    mouse_orbit_down = False

            elif event.type == pygame.MOUSEMOTION:
                # Use both our state flag and event.buttons. This makes mouse-orbit
                # robust across pygame/SDL backends where button-up/down state can
                # occasionally be weird in OpenGL windows.
                buttons = getattr(event, "buttons", (0, 0, 0))
                if mouse_orbit_down or buttons[0] or buttons[2]:
                    dx, dy = event.rel
                    camera.orbit_mouse_delta(dx, dy)

        keys = pygame.key.get_pressed()
        camera.handle_keys(keys, dt)

        width, height = pygame.display.get_surface().get_size()
        width = max(1, width)
        height = max(1, height)
        ctx.viewport = (0, 0, width, height)

        eye = camera.eye()
        cam_forward = camera.forward()
        cam_right = camera.right()
        cam_up = camera.up()

        program["u_eye"].value = tuple(float(v) for v in eye)
        program["u_cam_forward"].value = tuple(float(v) for v in cam_forward)
        program["u_cam_right"].value = tuple(float(v) for v in cam_right)
        program["u_cam_up"].value = tuple(float(v) for v in cam_up)
        program["u_tan_half_view_fov_y"].value = math.tan(math.radians(VIEWER_FOV_Y_DEGREES) / 2.0)
        program["u_view_aspect"].value = width / height

        ctx.clear(*BACKGROUND_RGB, 1.0)
        for unit, tex in enumerate(page_textures):
            tex.use(location=unit)
        vao.render(moderngl.TRIANGLES)

        pygame.display.flip()

        now = time.time()
        if now - last_caption_update > 0.25:
            fps = clock.get_fps()
            grid = "grid" if SHOW_CELL_GRID_OVERLAY else "no-grid"
            pygame.display.set_caption(
                f"Hogel Viewer | fixed hologram, orbit viewer | tiled full bilinear | {fps:5.1f} FPS | "
                f"{grid} | convention 0x{current_convention_index:X} | "
                f"eye=({eye[0]:.2f}, {eye[1]:.2f}, {eye[2]:.2f}) | "
                f"radius={camera.radius:.2f}"
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
