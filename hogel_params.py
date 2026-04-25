"""
hogel_params.py

Shared parameters for the Blender hogel rig and the standalone light-field
viewer. Edit this file when changing the hologram geometry or capture FOV.

Keep this file in the same folder as:
    create_hogel_lightfield_rig.py
    hologram_lightfield_viewer.py

The render script can remain unchanged, because it reads the rig metadata that
create_hogel_lightfield_rig.py stores in the Blender collection.
"""

# ============================================================
# SHARED HOLOGRAM / LIGHT-FIELD GEOMETRY
# ============================================================

# Hologram plane center.
# Plane lies in the XZ plane at fixed y = Y0 before any viewer-side rotation.
X0 = 0.0
Y0 = -1.5
Z0 = 0.0

# Requested screen dimensions.
# Width is along Blender X.
# Height is along Blender Z.
W = 6.0
H = 6.0

# Hogel cell size.
# The actual hogel grid is:
#   Na = floor(W / h)
#   Nb = floor(H / h)
# and the actual sampled size is:
#   W_sampled = Na * h
#   H_sampled = Nb * h
h = 0.05

# Direction from hologram plane into the scene.
# Current convention: screen at y=-3, object near origin, so inward is +Y.
INWARD_NORMAL = (0.0, 1.0, 0.0)

# Move capture camera a tiny distance inward from the exact plane.
CAMERA_EPSILON = 1.0e-4

# Hogel capture camera FOV in degrees.
# Blender uses this as camera.data.angle. For square hogel renders, this is
# both horizontal and vertical FOV in the viewer convention.
CAMERA_FOV_DEGREES = 70.0


# ============================================================
# BLENDER RIG OBJECT NAMES
# ============================================================

RIG_COLLECTION_NAME = "Hogel_Lightfield_Rig"
CAMERA_OBJECT_NAME = "Hogel_Render_Camera"
CAMERA_DATA_NAME = "Hogel_Render_Camera_Data"

# If True, recreate the rig collection contents when running the rig script.
CLEAR_PREVIOUS_RIG = True


# ============================================================
# VIEWER SETTINGS
# ============================================================

# Folder containing rendered hogel images.
# Relative paths are resolved relative to hologram_lightfield_viewer.py.
HOGEL_DIR = "hogel_renders"

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800

VIEWER_FOV_Y_DEGREES = 50.0
NEAR_PLANE = 0.01
FAR_PLANE = 100.0

# Initial viewer position.
INITIAL_EYE = (0.0, -6.0, 0.0)
INITIAL_LOOK_AT = (X0, Y0, Z0)

BASE_MOVE_SPEED = 2.5
FAST_MOVE_MULTIPLIER = 4.0
MOUSE_SENSITIVITY = 0.003

INITIAL_HOLOGRAM_YAW_DEGREES = 0.0
INITIAL_HOLOGRAM_PITCH_DEGREES = 0.0
INITIAL_HOLOGRAM_ROLL_DEGREES = 0.0
HOLOGRAM_ROTATE_SPEED_DEGREES_PER_SECOND = 60.0
HOLOGRAM_MOUSE_SENSITIVITY = 0.005

SHOW_CELL_GRID_OVERLAY = False
BACKGROUND_RGB = (0.015, 0.015, 0.018)

# Blender/Pillow PNGs are top-left-origin. The OpenGL shader convention below
# treats v=0 as bottom, so this should usually stay True.
FLIP_HOGEL_IMAGES_TOP_BOTTOM = True

LOAD_PROGRESS_EVERY = 25
