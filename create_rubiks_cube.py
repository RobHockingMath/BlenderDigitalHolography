import bpy
from mathutils import Vector

# ============================================================
# Helpers
# ============================================================

def look_at(obj, target):
    """
    Rotate obj so its local -Z axis points at target.
    Useful for cameras and area lights.
    """
    target = Vector(target)
    direction = target - obj.location

    if direction.length == 0:
        return

    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def make_principled_material(name, base_color, roughness=0.4, metallic=0.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = base_color
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic

    mat.diffuse_color = base_color
    return mat


def assign_cubelet_materials(obj, ix, iy, iz, mats):
    """
    Assign materials to the cubelet faces based on which outer layer the
    cubelet belongs to.

    Material slot order:
        0: black plastic
        1: +X  (red)
        2: -X  (orange)
        3: +Y  (green)
        4: -Y  (blue)
        5: +Z  (white)
        6: -Z  (yellow)
    """
    obj.data.materials.clear()

    ordered = [
        mats["black"],
        mats["pos_x"],
        mats["neg_x"],
        mats["pos_y"],
        mats["neg_y"],
        mats["pos_z"],
        mats["neg_z"],
    ]

    for mat in ordered:
        obj.data.materials.append(mat)

    for poly in obj.data.polygons:
        n = poly.normal
        slot = 0  # default: black plastic

        if n.x > 0.9 and ix == 1:
            slot = 1
        elif n.x < -0.9 and ix == -1:
            slot = 2
        elif n.y > 0.9 and iy == 1:
            slot = 3
        elif n.y < -0.9 and iy == -1:
            slot = 4
        elif n.z > 0.9 and iz == 1:
            slot = 5
        elif n.z < -0.9 and iz == -1:
            slot = 6

        poly.material_index = slot


# ============================================================
# Clear the default scene
# ============================================================

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()


# ============================================================
# Materials for the Rubik's cube
# ============================================================

materials = {
    # black plastic body
    "black": make_principled_material(
        "Rubiks_Black",
        (0.03, 0.03, 0.03, 1.0),
        roughness=0.55,
        metallic=0.0,
    ),

    # sticker colors
    "pos_x": make_principled_material(
        "Rubiks_Red",
        (0.80, 0.05, 0.05, 1.0),
        roughness=0.35,
        metallic=0.0,
    ),
    "neg_x": make_principled_material(
        "Rubiks_Orange",
        (1.00, 0.42, 0.02, 1.0),
        roughness=0.35,
        metallic=0.0,
    ),
    "pos_y": make_principled_material(
        "Rubiks_Green",
        (0.00, 0.55, 0.12, 1.0),
        roughness=0.35,
        metallic=0.0,
    ),
    "neg_y": make_principled_material(
        "Rubiks_Blue",
        (0.02, 0.22, 0.85, 1.0),
        roughness=0.35,
        metallic=0.0,
    ),
    "pos_z": make_principled_material(
        "Rubiks_White",
        (0.95, 0.95, 0.95, 1.0),
        roughness=0.35,
        metallic=0.0,
    ),
    "neg_z": make_principled_material(
        "Rubiks_Yellow",
        (0.95, 0.80, 0.02, 1.0),
        roughness=0.35,
        metallic=0.0,
    ),
}


# ============================================================
# Build the Rubik's cube at the origin
# ============================================================

# These choices make the whole cube roughly fit inside a radius-1 scale.
CUBELET_SIZE = 0.62
CUBELET_PITCH = 0.66

# Optional parent empty so the whole thing is easy to move later.
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0.0, 0.0))
rubiks_root = bpy.context.object
rubiks_root.name = "Rubiks_Cube"

for ix in (-1, 0, 1):
    for iy in (-1, 0, 1):
        for iz in (-1, 0, 1):
            location = (
                ix * CUBELET_PITCH,
                iy * CUBELET_PITCH,
                iz * CUBELET_PITCH,
            )

            bpy.ops.mesh.primitive_cube_add(
                size=CUBELET_SIZE,
                location=location,
            )

            cubelet = bpy.context.object
            cubelet.name = f"Cubelet_{ix+1}_{iy+1}_{iz+1}"

            assign_cubelet_materials(cubelet, ix, iy, iz, materials)

            # Parent to the Rubik's cube root
            cubelet.parent = rubiks_root

            # Slight bevel for nicer edges
            bevel = cubelet.modifiers.new(name="Bevel", type='BEVEL')
            bevel.width = 0.03
            bevel.segments = 3
            bevel.limit_method = 'NONE'

# ============================================================
# Add a light placed with the hogel rig in mind
# ============================================================

# Keep the light in the same position as before.
# The hogel rig plane is at y = -3 and looks inward along +Y,
# so this lights the face of the cube seen by the hogel cameras.
LIGHT_LOCATION = (0.0, -2.4, 3.0)
LIGHT_TARGET = (0.0, 0.0, 0.0)

bpy.ops.object.light_add(
    type='AREA',
    location=LIGHT_LOCATION
)

light = bpy.context.object
light.name = "Large_Soft_Key_Light_For_Hogel_Rig"
light.data.energy = 900.0
light.data.size = 4.0

look_at(light, LIGHT_TARGET)


# ============================================================
# Set a simple world background
# ============================================================

world = bpy.context.scene.world
if world is not None:
    world.color = (0.02, 0.02, 0.025)


print("Created a Rubik's cube at the origin with the same area light as before.")