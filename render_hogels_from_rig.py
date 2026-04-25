import bpy
import os
import math
import time
from mathutils import Vector

# ============================================================
# USER SETTINGS
# ============================================================

# This must match the collection created by 02_create_hogel_lightfield_rig.py.
RIG_COLLECTION_NAME = "Hogel_Lightfield_Rig"

# Render resolution per hogel image.
RENDER_RES_X = 256
RENDER_RES_Y = 256

# Output folder.
# "//" means relative to the current .blend file.
# If you use "//hogel_renders", save your .blend file before running this script.
OUTPUT_DIR = "//hogel_renders"

# If True, already-rendered hogel PNGs are not overwritten.
# False is better while testing because it overwrites stale renders.
SKIP_EXISTING = False

# ============================================================
# VISUAL DEBUG SETTINGS
# ============================================================

# If True, make the moving camera much easier to see inside Blender while rendering.
# This is for debugging / demonstration, not maximum performance.
VISUAL_DEBUG = True

# Make the camera icon larger and easier to see.
VISUAL_DEBUG_CAMERA_DISPLAY_SIZE = 1.0

# Show the camera name and draw it in front of other objects.
VISUAL_DEBUG_SHOW_CAMERA_NAME = True
VISUAL_DEBUG_SHOW_CAMERA_IN_FRONT = True

# Create temporary non-rendered helper objects:
#   - a bright marker attached to the render camera
#   - a square marker showing the current hogel cell
VISUAL_DEBUG_CREATE_CAMERA_MARKER = True
VISUAL_DEBUG_CREATE_CELL_MARKER = True

# Camera marker appearance.
VISUAL_DEBUG_CAMERA_MARKER_RADIUS = 0.10
VISUAL_DEBUG_CAMERA_MARKER_SEGMENTS = 16
VISUAL_DEBUG_CAMERA_MARKER_RINGS = 8

# Cell marker appearance.
VISUAL_DEBUG_CELL_MARKER_SCALE = 0.92  # fraction of cell size
VISUAL_DEBUG_CELL_MARKER_OFFSET_FACTOR = 0.25  # move slightly off the plane along inward normal

# Force viewport redraws so you can actually see the camera hop.
VISUAL_DEBUG_REDRAW_ITERATIONS = 1

# Optional pause after moving the camera and before each render.
# Set to 0.0 if you want minimal slowdown even in debug mode.
VISUAL_DEBUG_PRE_RENDER_PAUSE_SECONDS = 0.10


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


def require_collection(name):
    collection = bpy.data.collections.get(name)

    if collection is None:
        raise RuntimeError(
            f"Could not find collection '{name}'. "
            "Run 02_create_hogel_lightfield_rig.py first."
        )

    return collection


def require_custom_property(obj, key):
    if key not in obj:
        raise RuntimeError(
            f"The rig collection is missing custom property '{key}'. "
            "Run 02_create_hogel_lightfield_rig.py again to regenerate the rig."
        )

    return obj[key]


def read_rig_metadata(collection):
    x0 = float(require_custom_property(collection, "x0"))
    y0 = float(require_custom_property(collection, "y0"))
    z0 = float(require_custom_property(collection, "z0"))

    width = float(require_custom_property(collection, "W"))
    height = float(require_custom_property(collection, "H"))
    cell_size = float(require_custom_property(collection, "h"))

    na = compute_cell_count_floor(width, cell_size, "W")
    nb = compute_cell_count_floor(height, cell_size, "H")

    inward = Vector((
        float(require_custom_property(collection, "inward_x")),
        float(require_custom_property(collection, "inward_y")),
        float(require_custom_property(collection, "inward_z")),
    ))

    if inward.length == 0:
        raise RuntimeError("The stored inward normal has zero length.")

    inward.normalize()

    camera_epsilon = float(require_custom_property(collection, "camera_epsilon"))
    camera_fov_degrees = float(require_custom_property(collection, "camera_fov_degrees"))
    camera_name = str(require_custom_property(collection, "camera_name"))

    return {
        "x0": x0,
        "y0": y0,
        "z0": z0,
        "W": width,
        "H": height,
        "h": cell_size,
        "Na": na,
        "Nb": nb,
        "inward": inward,
        "camera_epsilon": camera_epsilon,
        "camera_fov_degrees": camera_fov_degrees,
        "camera_name": camera_name,
    }


def require_camera(camera_name):
    camera = bpy.data.objects.get(camera_name)

    if camera is None:
        raise RuntimeError(
            f"Could not find camera '{camera_name}'. "
            "Run 02_create_hogel_lightfield_rig.py first."
        )

    if camera.type != 'CAMERA':
        raise RuntimeError(f"Object '{camera_name}' exists, but it is not a camera.")

    return camera


def set_camera_direction(camera_obj, direction):
    """
    Blender cameras look along their local -Z axis.
    This rotates the camera so local -Z points along 'direction',
    while local +Y remains the camera's up direction as much as possible.
    """
    direction = direction.normalized()
    camera_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def hogel_center(i, j, x0, y0, z0, width, height, cell_size):
    na = compute_cell_count_floor(width, cell_size, "W")
    nb = compute_cell_count_floor(height, cell_size, "H")

    # Since Na and Nb are floor(W/h), floor(H/h), the sampled region is:
    #   sampled_width  = Na * h
    #   sampled_height = Nb * h
    # centered on (x0, y0, z0).
    sampled_width = na * cell_size
    sampled_height = nb * cell_size

    xmin = x0 - sampled_width / 2.0
    zmin = z0 - sampled_height / 2.0

    x = xmin + (i + 0.5) * cell_size
    y = y0
    z = zmin + (j + 0.5) * cell_size

    return Vector((x, y, z))


def resolve_output_dir(output_dir):
    if output_dir.startswith("//") and not bpy.data.filepath:
        raise RuntimeError(
            f"OUTPUT_DIR is '{output_dir}', which is relative to the .blend file, "
            "but the .blend file has not been saved yet. "
            "Either save the .blend file first, or set OUTPUT_DIR to an absolute path, "
            r"for example r'C:\Users\Rob\Desktop\hogel_renders'."
        )

    output_dir_abs = bpy.path.abspath(output_dir)

    try:
        os.makedirs(output_dir_abs, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(
            f"Could not create output directory:\n"
            f"    {output_dir_abs}\n\n"
            "This is usually a Windows permissions/path problem. "
            "Set OUTPUT_DIR to a normal absolute folder such as your Desktop or Documents."
        ) from exc

    return output_dir_abs


def format_seconds(seconds):
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(int(seconds + 0.5), 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {sec:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {sec:02d}s"
    return f"{sec:d}s"


def force_viewport_redraw(iterations=1):
    bpy.context.view_layer.update()

    iterations = max(1, int(iterations))
    for _ in range(iterations):
        try:
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        except RuntimeError:
            pass


def select_only_object(obj):
    try:
        bpy.ops.object.select_all(action='DESELECT')
    except RuntimeError:
        pass

    obj.select_set(True)

    view_layer = bpy.context.view_layer
    view_layer.objects.active = obj


def set_workspace_status(text):
    workspace = bpy.context.workspace
    if workspace is not None:
        workspace.status_text_set(text)


def clear_workspace_status():
    workspace = bpy.context.workspace
    if workspace is not None:
        workspace.status_text_set(None)


def make_emission_material(name, color, strength=2.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in list(nodes):
        nodes.remove(node)

    out_node = nodes.new(type='ShaderNodeOutputMaterial')
    emission = nodes.new(type='ShaderNodeEmission')
    emission.inputs['Color'].default_value = color
    emission.inputs['Strength'].default_value = strength
    links.new(emission.outputs['Emission'], out_node.inputs['Surface'])

    return mat


def create_camera_debug_marker(collection, camera_obj):
    mesh = bpy.data.meshes.new("Hogel_Debug_Camera_Marker_Mesh")
    obj = bpy.data.objects.new("Hogel_Debug_Camera_Marker", mesh)
    collection.objects.link(obj)

    bm_verts = []
    verts = []
    faces = []

    # Use Blender operator to create a UV sphere, then copy its mesh.
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=VISUAL_DEBUG_CAMERA_MARKER_SEGMENTS,
        ring_count=VISUAL_DEBUG_CAMERA_MARKER_RINGS,
        radius=VISUAL_DEBUG_CAMERA_MARKER_RADIUS,
        location=(0.0, 0.0, 0.0),
    )
    temp = bpy.context.object
    temp_mesh = temp.data.copy()
    bpy.data.objects.remove(temp, do_unlink=True)

    obj.data = temp_mesh
    obj.hide_render = True
    obj.show_in_front = True
    obj.parent = camera_obj
    obj.location = (0.0, 0.0, 0.0)

    mat = make_emission_material(
        "Hogel_Debug_Camera_Marker_Material",
        (1.0, 0.2, 0.05, 1.0),
        strength=3.0,
    )
    obj.data.materials.append(mat)

    return obj


def create_cell_debug_marker(collection, cell_size):
    half = 0.5 * cell_size * VISUAL_DEBUG_CELL_MARKER_SCALE

    verts = [
        (-half, 0.0, -half),
        ( half, 0.0, -half),
        ( half, 0.0,  half),
        (-half, 0.0,  half),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
    ]

    mesh = bpy.data.meshes.new("Hogel_Debug_Cell_Marker_Mesh")
    mesh.from_pydata(verts, edges, [])
    mesh.update()

    obj = bpy.data.objects.new("Hogel_Debug_Cell_Marker", mesh)
    collection.objects.link(obj)

    obj.hide_render = True
    obj.show_in_front = True
    obj.display_type = 'WIRE'

    mat = make_emission_material(
        "Hogel_Debug_Cell_Marker_Material",
        (1.0, 1.0, 0.0, 1.0),
        strength=3.0,
    )
    obj.data.materials.append(mat)

    return obj


def remove_object_if_not_none(obj):
    if obj is not None and obj.name in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)


def setup_visual_debug(rig_collection, camera_obj, rig):
    state = {
        "enabled": VISUAL_DEBUG,
        "camera_debug_marker": None,
        "cell_debug_marker": None,
        "camera_display_size_original": camera_obj.data.display_size,
        "camera_show_name_original": getattr(camera_obj, "show_name", False),
        "camera_show_in_front_original": getattr(camera_obj, "show_in_front", False),
        "selected_objects_original": list(bpy.context.selected_objects),
        "active_object_original": bpy.context.view_layer.objects.active,
    }

    if not VISUAL_DEBUG:
        return state

    camera_obj.data.display_size = VISUAL_DEBUG_CAMERA_DISPLAY_SIZE
    if hasattr(camera_obj, "show_name"):
        camera_obj.show_name = VISUAL_DEBUG_SHOW_CAMERA_NAME
    if hasattr(camera_obj, "show_in_front"):
        camera_obj.show_in_front = VISUAL_DEBUG_SHOW_CAMERA_IN_FRONT

    select_only_object(camera_obj)

    if VISUAL_DEBUG_CREATE_CAMERA_MARKER:
        state["camera_debug_marker"] = create_camera_debug_marker(rig_collection, camera_obj)

    if VISUAL_DEBUG_CREATE_CELL_MARKER:
        state["cell_debug_marker"] = create_cell_debug_marker(rig_collection, rig["h"])

    force_viewport_redraw(VISUAL_DEBUG_REDRAW_ITERATIONS)
    return state


def restore_visual_debug(camera_obj, state):
    if not state.get("enabled", False):
        return

    remove_object_if_not_none(state.get("camera_debug_marker"))
    remove_object_if_not_none(state.get("cell_debug_marker"))

    camera_obj.data.display_size = state["camera_display_size_original"]
    if hasattr(camera_obj, "show_name"):
        camera_obj.show_name = state["camera_show_name_original"]
    if hasattr(camera_obj, "show_in_front"):
        camera_obj.show_in_front = state["camera_show_in_front_original"]

    try:
        bpy.ops.object.select_all(action='DESELECT')
    except RuntimeError:
        pass

    for obj in state["selected_objects_original"]:
        if obj is not None and obj.name in bpy.data.objects:
            obj.select_set(True)

    original_active = state["active_object_original"]
    if original_active is not None and original_active.name in bpy.data.objects:
        bpy.context.view_layer.objects.active = original_active

    force_viewport_redraw(VISUAL_DEBUG_REDRAW_ITERATIONS)


def update_visual_debug_positions(state, center, inward, camera_obj, camera_epsilon):
    if not state.get("enabled", False):
        return

    select_only_object(camera_obj)

    cell_marker = state.get("cell_debug_marker")
    if cell_marker is not None:
        offset = VISUAL_DEBUG_CELL_MARKER_OFFSET_FACTOR * camera_epsilon * inward
        cell_marker.location = center + offset
        cell_marker.rotation_euler = inward.to_track_quat('Y', 'Z').to_euler()

    force_viewport_redraw(VISUAL_DEBUG_REDRAW_ITERATIONS)

    if VISUAL_DEBUG_PRE_RENDER_PAUSE_SECONDS > 0.0:
        time.sleep(VISUAL_DEBUG_PRE_RENDER_PAUSE_SECONDS)
        force_viewport_redraw(VISUAL_DEBUG_REDRAW_ITERATIONS)


def render_hogels(camera_obj, rig_collection, rig):
    scene = bpy.context.scene
    wm = bpy.context.window_manager

    x0 = rig["x0"]
    y0 = rig["y0"]
    z0 = rig["z0"]
    width = rig["W"]
    height = rig["H"]
    cell_size = rig["h"]
    na = rig["Na"]
    nb = rig["Nb"]
    inward = rig["inward"]
    camera_epsilon = rig["camera_epsilon"]
    camera_fov_degrees = rig["camera_fov_degrees"]

    output_dir_abs = resolve_output_dir(OUTPUT_DIR)

    scene.camera = camera_obj
    scene.render.resolution_x = RENDER_RES_X
    scene.render.resolution_y = RENDER_RES_Y
    scene.render.image_settings.file_format = 'PNG'

    # Keep the render camera consistent with the rig settings.
    camera_obj.data.angle = math.radians(camera_fov_degrees)
    camera_obj.data.clip_start = 0.001
    camera_obj.data.clip_end = 1000.0

    total = na * nb
    count = 0
    rendered_count = 0
    skipped_count = 0
    start_time = time.time()

    print("")
    print("Starting hogel render pass")
    print(f"Grid: Na={na}, Nb={nb}, total={total} hogels")
    print(f"Output directory: {output_dir_abs}")
    print(f"Skip existing: {SKIP_EXISTING}")
    print(f"Visual debug: {VISUAL_DEBUG}")
    print("")

    visual_state = setup_visual_debug(rig_collection, camera_obj, rig)
    wm.progress_begin(0, total)

    try:
        for j in range(nb):
            for i in range(na):
                count += 1

                center = hogel_center(i, j, x0, y0, z0, width, height, cell_size)
                camera_obj.location = center + camera_epsilon * inward
                set_camera_direction(camera_obj, inward)

                filename = f"hogel_{i:04d}_{na:04d}_{j:04d}_{nb:04d}.png"
                filepath = os.path.join(output_dir_abs, filename)

                elapsed = time.time() - start_time
                avg_time = elapsed / max(count - 1, 1) if count > 1 else 0.0
                remaining = avg_time * (total - count + 1)
                percent = 100.0 * count / total if total > 0 else 100.0

                status_prefix = (
                    f"Hogel render {count}/{total} ({percent:6.2f}%) | "
                    f"cell=({i+1}/{na}, {j+1}/{nb}) | ETA {format_seconds(remaining)}"
                )

                if SKIP_EXISTING and os.path.exists(filepath):
                    skipped_count += 1
                    set_workspace_status(status_prefix + f" | skipping {filename}")

                    if VISUAL_DEBUG:
                        update_visual_debug_positions(
                            visual_state,
                            center,
                            inward,
                            camera_obj,
                            camera_epsilon,
                        )

                    print(
                        f"[{count:04d}/{total:04d}] "
                        f"{percent:6.2f}%  "
                        f"ETA {format_seconds(remaining):>10s}  "
                        f"Skipping existing {filename}",
                        flush=True,
                    )

                    wm.progress_update(count)
                    continue

                scene.render.filepath = filepath
                set_workspace_status(status_prefix + f" | rendering {filename}")

                print(
                    f"[{count:04d}/{total:04d}] "
                    f"Rendering {filename}  "
                    f"ETA before render {format_seconds(remaining)}",
                    flush=True,
                )

                if VISUAL_DEBUG:
                    update_visual_debug_positions(
                        visual_state,
                        center,
                        inward,
                        camera_obj,
                        camera_epsilon,
                    )
                else:
                    bpy.context.view_layer.update()

                bpy.ops.render.render(write_still=True)

                rendered_count += 1

                elapsed = time.time() - start_time
                avg_time = elapsed / count if count > 0 else 0.0
                remaining = avg_time * (total - count)
                percent = 100.0 * count / total if total > 0 else 100.0

                set_workspace_status(
                    f"Hogel render {count}/{total} ({percent:6.2f}%) | "
                    f"cell=({i+1}/{na}, {j+1}/{nb}) | ETA {format_seconds(remaining)} | rendered {filename}"
                )

                print(
                    f"[{count:04d}/{total:04d}] "
                    f"{percent:6.2f}%  "
                    f"ETA {format_seconds(remaining):>10s}  "
                    f"Rendered {filename}",
                    flush=True,
                )

                wm.progress_update(count)

    finally:
        wm.progress_end()
        clear_workspace_status()

        # Return the camera to the center of the hologram plane when finished.
        camera_obj.location = Vector((x0, y0, z0)) + camera_epsilon * inward
        set_camera_direction(camera_obj, inward)
        bpy.context.view_layer.update()

        restore_visual_debug(camera_obj, visual_state)

    total_elapsed = time.time() - start_time

    print("")
    print("Done rendering hogels.")
    print(f"Rendered: {rendered_count}")
    print(f"Skipped:  {skipped_count}")
    print(f"Total elapsed time: {format_seconds(total_elapsed)}")
    print("")


# ============================================================
# MAIN
# ============================================================

def main():
    rig_collection = require_collection(RIG_COLLECTION_NAME)
    rig = read_rig_metadata(rig_collection)
    camera = require_camera(rig["camera_name"])

    render_hogels(camera, rig_collection, rig)


main()
