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


def render_hogels(camera_obj, rig):
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
    print("")

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

                if SKIP_EXISTING and os.path.exists(filepath):
                    skipped_count += 1

                    elapsed = time.time() - start_time
                    avg_time = elapsed / count if count > 0 else 0.0
                    remaining = avg_time * (total - count)
                    percent = 100.0 * count / total if total > 0 else 100.0

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

                elapsed_before = time.time() - start_time
                avg_before = elapsed_before / max(count - 1, 1)
                remaining_before = avg_before * (total - count + 1)

                print(
                    f"[{count:04d}/{total:04d}] "
                    f"Rendering {filename}  "
                    f"ETA before render {format_seconds(remaining_before)}",
                    flush=True,
                )

                bpy.context.view_layer.update()
                bpy.ops.render.render(write_still=True)

                rendered_count += 1

                elapsed = time.time() - start_time
                avg_time = elapsed / count if count > 0 else 0.0
                remaining = avg_time * (total - count)
                percent = 100.0 * count / total if total > 0 else 100.0

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

    # Return the camera to the center of the hologram plane when finished.
    camera_obj.location = Vector((x0, y0, z0)) + camera_epsilon * inward
    set_camera_direction(camera_obj, inward)
    bpy.context.view_layer.update()

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

    render_hogels(camera, rig)


main()
