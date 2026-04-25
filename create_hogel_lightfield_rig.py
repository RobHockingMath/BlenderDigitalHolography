import importlib
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


# ============================================================
# PARAMETER LOADING
# ============================================================

def import_hogel_params():
    """Import hogel_params.py from the script folder, text-block folder, or .blend folder."""
    candidate_dirs = []

    if "__file__" in globals():
        candidate_dirs.append(Path(__file__).resolve().parent)

    try:
        space = getattr(bpy.context, "space_data", None)
        text = getattr(space, "text", None)
        filepath = getattr(text, "filepath", "") if text is not None else ""
        if filepath:
            candidate_dirs.append(Path(bpy.path.abspath(filepath)).resolve().parent)
    except Exception:
        pass

    if bpy.data.filepath:
        candidate_dirs.append(Path(bpy.data.filepath).resolve().parent)

    candidate_dirs.append(Path.cwd())

    for d in candidate_dirs:
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)

    try:
        module = importlib.import_module("hogel_params")
        return importlib.reload(module)
    except ModuleNotFoundError as exc:
        searched = "\n".join(f"  {d}" for d in candidate_dirs)
        raise RuntimeError(
            "Could not import hogel_params.py. Put hogel_params.py in the same "
            "folder as this script, or in the same folder as the saved .blend file.\n"
            f"Searched:\n{searched}"
        ) from exc


params = import_hogel_params()


# ============================================================
# INTERNAL HELPERS
# ============================================================

def compute_cell_count_floor(length, cell_size, name):
    if cell_size <= 0:
        raise ValueError("cell_size must be positive.")

    value = length / cell_size
    floored = int(math.floor(value))

    if floored <= 0:
        raise ValueError(
            f"{name} / h gives zero cells. "
            f"{name}={length}, h={cell_size}."
        )

    return floored


def get_or_create_collection(name):
    scene = bpy.context.scene
    collection = bpy.data.collections.get(name)

    if collection is None:
        collection = bpy.data.collections.new(name)
        scene.collection.children.link(collection)

    return collection


def clear_collection(collection):
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def link_to_collection(obj, collection):
    collection.objects.link(obj)


def make_transparent_material(name, color, alpha):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (color[0], color[1], color[2], alpha)

    mat.use_nodes = True
    mat.blend_method = 'BLEND'
    mat.use_screen_refraction = False
    mat.show_transparent_back = True

    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")

    if bsdf is not None:
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], alpha)
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = alpha
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.25

    return mat


def make_solid_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def create_hologram_plane(collection, x0, y0, z0, width, height):
    xmin = x0 - width / 2.0
    xmax = x0 + width / 2.0
    zmin = z0 - height / 2.0
    zmax = z0 + height / 2.0

    verts = [
        (xmin, y0, zmin),
        (xmax, y0, zmin),
        (xmax, y0, zmax),
        (xmin, y0, zmax),
    ]

    faces = [(0, 1, 2, 3)]

    mesh = bpy.data.meshes.new("Hologram_Plane_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new("Hologram_Plane_Visible_Not_Rendered", mesh)
    link_to_collection(obj, collection)

    mat = make_transparent_material(
        name="Hologram_Plane_Transparent_Cyan",
        color=(0.0, 0.85, 1.0),
        alpha=0.18,
    )
    obj.data.materials.append(mat)

    obj.hide_render = True
    obj.show_in_front = True

    return obj


def create_hologram_grid(collection, x0, y0, z0, width, height, cell_size):
    na = compute_cell_count_floor(width, cell_size, "W")
    nb = compute_cell_count_floor(height, cell_size, "H")

    sampled_width = na * cell_size
    sampled_height = nb * cell_size

    xmin = x0 - sampled_width / 2.0
    xmax = x0 + sampled_width / 2.0
    zmin = z0 - sampled_height / 2.0
    zmax = z0 + sampled_height / 2.0

    verts = []
    edges = []

    def add_line(p0, p1):
        idx = len(verts)
        verts.append(p0)
        verts.append(p1)
        edges.append((idx, idx + 1))

    # Vertical grid lines: constant x, running along z.
    for i in range(na + 1):
        x = xmin + i * cell_size
        add_line((x, y0, zmin), (x, y0, zmax))

    # Horizontal grid lines: constant z, running along x.
    for j in range(nb + 1):
        z = zmin + j * cell_size
        add_line((xmin, y0, z), (xmax, y0, z))

    mesh = bpy.data.meshes.new("Hologram_Grid_Mesh")
    mesh.from_pydata(verts, edges, [])
    mesh.update()

    obj = bpy.data.objects.new("Hologram_Cell_Grid_Visible_Not_Rendered", mesh)
    link_to_collection(obj, collection)

    mat = make_solid_material(
        name="Hologram_Grid_Cyan",
        color=(0.0, 0.95, 1.0, 1.0),
    )
    obj.data.materials.append(mat)

    obj.hide_render = True
    obj.display_type = 'WIRE'
    obj.show_in_front = True

    return obj


def create_hogel_camera(collection, inward_normal, fov_degrees):
    cam_data = bpy.data.cameras.new(params.CAMERA_DATA_NAME)
    cam_data.type = 'PERSP'
    cam_data.angle = math.radians(fov_degrees)
    cam_data.clip_start = 0.001
    cam_data.clip_end = 1000.0
    cam_data.display_size = 0.2

    cam = bpy.data.objects.new(params.CAMERA_OBJECT_NAME, cam_data)
    link_to_collection(cam, collection)

    set_camera_direction(cam, inward_normal)

    return cam


def set_camera_direction(camera_obj, direction):
    """
    Blender cameras look along their local -Z axis.
    This rotates the camera so local -Z points along 'direction',
    while local +Y remains the camera's up direction as much as possible.
    """
    direction = direction.normalized()
    camera_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def store_rig_metadata(collection, x0, y0, z0, width, height, cell_size, inward_normal, camera):
    na = compute_cell_count_floor(width, cell_size, "W")
    nb = compute_cell_count_floor(height, cell_size, "H")

    collection["rig_version"] = 2
    collection["params_file"] = "hogel_params.py"

    collection["x0"] = float(x0)
    collection["y0"] = float(y0)
    collection["z0"] = float(z0)

    collection["W"] = float(width)
    collection["H"] = float(height)
    collection["h"] = float(cell_size)

    collection["Na"] = int(na)
    collection["Nb"] = int(nb)

    collection["inward_x"] = float(inward_normal.x)
    collection["inward_y"] = float(inward_normal.y)
    collection["inward_z"] = float(inward_normal.z)

    collection["camera_epsilon"] = float(params.CAMERA_EPSILON)
    collection["camera_fov_degrees"] = float(params.CAMERA_FOV_DEGREES)
    collection["camera_name"] = camera.name


# ============================================================
# MAIN
# ============================================================

def main():
    x0 = float(params.X0)
    y0 = float(params.Y0)
    z0 = float(params.Z0)
    width = float(params.W)
    height = float(params.H)
    cell_size = float(params.h)

    if cell_size <= 0:
        raise ValueError("h must be positive.")
    if width <= 0:
        raise ValueError("W must be positive.")
    if height <= 0:
        raise ValueError("H must be positive.")

    na = compute_cell_count_floor(width, cell_size, "W")
    nb = compute_cell_count_floor(height, cell_size, "H")

    inward = Vector(params.INWARD_NORMAL)
    if inward.length == 0:
        raise ValueError("INWARD_NORMAL must be nonzero.")
    inward.normalize()

    rig_collection = get_or_create_collection(params.RIG_COLLECTION_NAME)

    if bool(params.CLEAR_PREVIOUS_RIG):
        clear_collection(rig_collection)

    create_hologram_plane(
        collection=rig_collection,
        x0=x0,
        y0=y0,
        z0=z0,
        width=na * cell_size,
        height=nb * cell_size,
    )

    create_hologram_grid(
        collection=rig_collection,
        x0=x0,
        y0=y0,
        z0=z0,
        width=width,
        height=height,
        cell_size=cell_size,
    )

    camera = create_hogel_camera(
        collection=rig_collection,
        inward_normal=inward,
        fov_degrees=float(params.CAMERA_FOV_DEGREES),
    )

    # Place the camera initially at the center of the hologram plane.
    camera.location = Vector((x0, y0, z0)) + float(params.CAMERA_EPSILON) * inward
    set_camera_direction(camera, inward)
    bpy.context.scene.camera = camera

    store_rig_metadata(
        collection=rig_collection,
        x0=x0,
        y0=y0,
        z0=z0,
        width=width,
        height=height,
        cell_size=cell_size,
        inward_normal=inward,
        camera=camera,
    )

    print("")
    print("Hogel light-field rig created from hogel_params.py.")
    print(f"Hologram plane center: ({x0}, {y0}, {z0})")
    print(f"Requested hologram plane size: W={width}, H={height}")
    print(f"Cell size h={cell_size}")
    print(f"Grid: Na={na}, Nb={nb}")
    print(f"Actual sampled size: W_sampled={na * cell_size}, H_sampled={nb * cell_size}")
    print(f"Camera FOV: {float(params.CAMERA_FOV_DEGREES)} degrees")
    print(f"Inward normal: {tuple(inward)}")
    print("No hogels were rendered. Run render_hogels_from_rig.py when ready.")
    print("")


main()
