"""
GN Interior Tool - exterior-aware interior generator for GTA MLO.

Reads an exterior building shell (even a messy non-manifold game asset) and,
per floor, extracts the outer wall outline, insets it 20 cm, and builds an
interior shell that CONFORMS to the real building - unlike a blind room tool.

  * Auto footprint per floor via horizontal cross-section (robust on
    non-manifold shells - bisect + weld gives clean closed loops).
  * L-shaped / arbitrary outlines supported (rooms follow the polygon).
  * Fixed room height (2.95 m) with an enforced gap between storeys.
  * Floor levels defined by picking a facade edge (reads its Z).

Designed to sit alongside the Room Tool: this makes the exterior-correct
envelope; you carve partitions/doors inside it.
"""

bl_info = {
    "name": "GN Interior Tool (MLO)",
    "author": "GN's Studio + Claude",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport > Sidebar (N) > GN Interior",
    "description": "Exterior-aware interior shell + per-floor boundaries for GTA MLO.",
    "category": "Object",
}

import bpy
import bmesh
import math
import json
import gpu
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
from mathutils import Vector, Matrix
from mathutils.geometry import intersect_line_plane
from bpy.props import (FloatProperty, IntProperty, StringProperty, BoolProperty,
                       PointerProperty, CollectionProperty)
from bpy.types import Operator, Panel, PropertyGroup

BOUND_COLL = "GN_Boundaries"
INT_COLL = "GN_Interiors"
ROOM_COLL = "GN_Rooms"
CUTTER_COLL = "GN_Cutters"
DOORFRAME_COLL = "GN_DoorFrames"


# ===========================================================================
# geometry core
# ===========================================================================
def _eval_bmesh(obj, context):
    """World-space bmesh of the evaluated (modifier-applied) exterior."""
    deps = context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(deps)
    me = ev.to_mesh()
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.transform(obj.matrix_world)
    ev.to_mesh_clear()
    return bm


def _poly_area(pts):
    a = 0.0
    n = len(pts)
    for i in range(n):
        a += pts[i].x * pts[(i + 1) % n].y - pts[(i + 1) % n].x * pts[i].y
    return a * 0.5


def section_loops(src, zc, weld=0.02):
    """Cross-section the shell at height zc -> list of ordered XY loops (Vectors)."""
    bm = src.copy()
    res = bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
                                 dist=1e-5, plane_co=(0, 0, zc), plane_no=(0, 0, 1))
    cut = [e for e in res['geom_cut'] if isinstance(e, bmesh.types.BMEdge)]
    bm2 = bmesh.new()
    vmap = {}
    for e in cut:
        vv = []
        for v in e.verts:
            k = (round(v.co.x, 3), round(v.co.y, 3))
            if k not in vmap:
                vmap[k] = bm2.verts.new((v.co.x, v.co.y, 0.0))
            vv.append(vmap[k])
        if vv[0] != vv[1]:
            try:
                bm2.edges.new(vv)
            except ValueError:
                pass
    bmesh.ops.remove_doubles(bm2, verts=bm2.verts[:], dist=weld)
    bm2.verts.ensure_lookup_table()
    visited = set()
    loops = []
    for v in bm2.verts:
        if v in visited or len(v.link_edges) != 2:
            continue
        lp = []
        cur = v
        prev = None
        while cur is not None and cur not in visited:
            visited.add(cur)
            lp.append(Vector((cur.co.x, cur.co.y)))
            nxt = None
            for e in cur.link_edges:
                ov = e.other_vert(cur)
                if ov is not prev and ov not in visited:
                    nxt = ov
                    break
            prev = cur
            cur = nxt
        if len(lp) >= 3:
            loops.append(lp)
    bm.free()
    bm2.free()
    return loops


def outer_footprint(src, zc, min_area=1.0):
    """Largest closed loop at height zc, forced CCW. None if nothing solid."""
    loops = [l for l in section_loops(src, zc) if abs(_poly_area(l)) > min_area]
    if not loops:
        return None
    loops.sort(key=lambda l: abs(_poly_area(l)), reverse=True)
    poly = loops[0]
    if _poly_area(poly) < 0:
        poly = poly[::-1]
    return poly


def _weld_loop(poly, mind):
    """Drop consecutive points closer than mind (removes slivers)."""
    if mind <= 0 or len(poly) < 4:
        return poly
    out = [poly[0]]
    for p in poly[1:]:
        if (p - out[-1]).length > mind:
            out.append(p)
    if len(out) > 2 and (out[0] - out[-1]).length <= mind:
        out.pop()
    return out if len(out) >= 3 else poly


def _dp(pts, tol):
    """Douglas-Peucker on an open polyline (list of 2D Vectors)."""
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    L = ab.length
    dmax = 0.0
    idx = 0
    for i in range(1, len(pts) - 1):
        if L < 1e-9:
            d = (pts[i] - a).length
        else:
            d = abs((pts[i] - a).cross(ab)) / L   # perpendicular distance
        if d > dmax:
            dmax = d
            idx = i
    if dmax > tol:
        left = _dp(pts[:idx + 1], tol)
        right = _dp(pts[idx:], tol)
        return left[:-1] + right
    return [a, b]


def simplify_loop(poly, tol=0.08):
    """Shape-preserving cleanup: weld slivers + Douglas-Peucker at distance tol.

    Removes window-reveal wiggles and near-duplicate points while keeping real
    corners, so an orthogonal L stays an L. tol is in metres; 0 = off.
    """
    poly = _weld_loop(poly, max(tol * 0.4, 0.005))
    if tol <= 0 or len(poly) < 4:
        return poly
    # split the closed loop at its two farthest-apart anchors, DP each chain
    a = poly[0]
    far = max(range(len(poly)), key=lambda i: (poly[i] - a).length_squared)
    c1 = poly[:far + 1]
    c2 = poly[far:] + [poly[0]]
    s1 = _dp(c1, tol)
    s2 = _dp(c2, tol)
    out = s1[:-1] + s2[:-1]
    return out if len(out) >= 3 else poly


def _line_x(a, b, c, d):
    den = (a.x - b.x) * (c.y - d.y) - (a.y - b.y) * (c.x - d.x)
    if abs(den) < 1e-7:
        return None
    t = ((a.x - c.x) * (c.y - d.y) - (a.y - c.y) * (c.x - d.x)) / den
    return Vector((a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)))


def inset_loop(poly, d):
    """Inset a CCW polygon inward by d (edge offset + intersect). Arbitrary/concave OK."""
    n = len(poly)
    offs = []
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        e = b - a
        L = max(e.length, 1e-6)
        nrm = Vector((-e.y / L, e.x / L))  # left normal = inward for CCW
        offs.append((a + nrm * d, b + nrm * d))
    out = []
    for i in range(n):
        e0 = offs[(i - 1) % n]
        e1 = offs[i]
        p = _line_x(e0[0], e0[1], e1[0], e1[1])
        out.append(p if p else offs[i][0])
    return out


def _get_coll(name):
    c = bpy.data.collections.get(name)
    if c is None:
        c = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(c)
    elif name not in [ch.name for ch in bpy.context.scene.collection.children]:
        bpy.context.scene.collection.children.link(c)
    return c


def _clear_coll(name):
    c = bpy.data.collections.get(name)
    if c:
        for ob in list(c.objects):
            bpy.data.objects.remove(ob, do_unlink=True)


# ===========================================================================
# properties
# ===========================================================================
class GN_FloorLevel(PropertyGroup):
    z: FloatProperty(name="Base Z", default=0.0, unit='LENGTH')
    top: FloatProperty(name="Top Z", default=0.0, unit='LENGTH',
        description="Ceiling Z. 0 = auto (base + Room Height)")
    bound_json: StringProperty(default="")   # inset boundary polygon [[x,y],...]


class GN_Room(PropertyGroup):
    floor_index: IntProperty(default=0)
    poly_json: StringProperty(default="[]")  # footprint [[x,y],...]
    uid: IntProperty(default=0)              # stable id (survives rebuilds)


class GN_Opening(PropertyGroup):
    cx: FloatProperty()
    cy: FloatProperty()
    nx: FloatProperty()
    ny: FloatProperty()
    hw: FloatProperty()     # half-width along the wall
    sill: FloatProperty()   # bottom Z
    top: FloatProperty()    # top Z
    uid: IntProperty(default=0)
    is_door: BoolProperty(default=False)


class GN_DoorPreset(PropertyGroup):
    name: StringProperty(default="Door")
    width: FloatProperty(name="Width", default=0.9, min=0.2, max=4.0, unit='LENGTH')
    height: FloatProperty(name="Height", default=2.0, min=0.5, max=5.0, unit='LENGTH')
    mesh_object: PointerProperty(name="Frame Mesh", type=bpy.types.Object,
        description="Optional door mesh placed as a linked instance at each opening")


class GN_IntProps(PropertyGroup):
    exterior: PointerProperty(name="Exterior Shell", type=bpy.types.Object,
        description="The exterior building shell to read")
    wall_margin: FloatProperty(name="Wall Margin", default=0.20, min=0.0, max=2.0,
        unit='LENGTH', description="Inset from exterior walls to interior walls")
    room_height: FloatProperty(name="Room Height", default=2.95, min=0.5, max=10.0,
        unit='LENGTH', description="Fixed interior clear height per floor")
    floor_gap: FloatProperty(name="Floor Gap", default=0.10, min=0.0, max=2.0,
        unit='LENGTH', description="Minimum gap between a ceiling and the floor above")
    sample_offset: FloatProperty(name="Sample Height", default=1.0, min=0.05, max=5.0,
        unit='LENGTH', description="Height above each floor base to cut the outline "
        "(pick a solid wall band, between windows)")
    cleanup: FloatProperty(name="Cleanup", default=0.08, min=0.0, max=0.5,
        unit='LENGTH', description="Simplify the outline: remove wiggles/slivers "
        "smaller than this (metres). Keeps real corners. 0 = exact outline")
    floors: CollectionProperty(type=GN_FloorLevel)
    floor_index: IntProperty(default=0)
    active_floor: IntProperty(name="Draw on Floor", default=0, min=0,
        description="Which floor new rooms are drawn on")
    snap: FloatProperty(name="Grid Snap", default=0.10, min=0.0, max=1.0,
        unit='LENGTH', description="Round drawn room corners to this grid (0 = off)")
    rooms: CollectionProperty(type=GN_Room)
    room_index: IntProperty(default=0)
    uid_counter: IntProperty(default=1)
    openings: CollectionProperty(type=GN_Opening)
    reveal: BoolProperty(name="Reveal Jambs", default=True,
        description="Cap the opening sides so windows/doors have depth (the 20cm reveal)")
    door_presets: CollectionProperty(type=GN_DoorPreset)
    active_door_preset: IntProperty(default=0)
    partition: FloatProperty(name="Partition Wall", default=0.10, min=0.0, max=1.0,
        unit='LENGTH', description="Gap left between two rooms when splitting "
        "(the interior partition wall thickness)")


# ===========================================================================
# operators
# ===========================================================================
class GN_OT_set_exterior(Operator):
    bl_idname = "gn_int.set_exterior"
    bl_label = "Use Active as Exterior"
    bl_description = "Set the active object as the exterior shell"

    def execute(self, context):
        ob = context.active_object
        if not ob or ob.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object")
            return {'CANCELLED'}
        context.scene.gn_int.exterior = ob
        return {'FINISHED'}


class GN_OT_add_floor_sel(Operator):
    bl_idname = "gn_int.add_floor_sel"
    bl_label = "Add Floor from Selected Edge"
    bl_description = ("Add a floor level at the Z of the selected geometry "
                      "(select a facade edge in Edit Mode)")

    def execute(self, context):
        ob = context.active_object
        if not ob or ob.type != 'MESH' or ob.mode != 'EDIT':
            self.report({'ERROR'}, "Enter Edit Mode and select an edge/verts on the facade")
            return {'CANCELLED'}
        s = context.scene.gn_int
        bm = bmesh.from_edit_mesh(ob.data)
        sel = [v for v in bm.verts if v.select]
        if not sel:
            self.report({'ERROR'}, "No vertices selected")
            return {'CANCELLED'}
        zs = sorted((ob.matrix_world @ v.co).z for v in sel)
        # two distinct height clusters -> bottom + top (custom height); else base only
        base = zs[0]
        top = 0.0
        if zs[-1] - zs[0] > 0.15:      # spans two edges at different heights
            base = zs[0]
            top = zs[-1]
        it = s.floors.add()
        it.z = round(base, 3)
        it.top = round(top, 3)
        _sort_floors(s)
        h = (top - base) if top > 0 else s.room_height
        self.report({'INFO'}, f"Added floor at z={base:.2f} (height {h:.2f} m)")
        return {'FINISHED'}


class GN_OT_add_floor_ground(Operator):
    bl_idname = "gn_int.add_floor_ground"
    bl_label = "Add Ground Floor"
    bl_description = "Add a floor level at the base of the exterior shell"

    def execute(self, context):
        s = context.scene.gn_int
        ex = s.exterior
        if not ex:
            self.report({'ERROR'}, "Set an exterior shell first")
            return {'CANCELLED'}
        zmin = min((ex.matrix_world @ v.co).z for v in ex.data.vertices)
        it = s.floors.add()
        it.z = round(zmin, 3)
        _sort_floors(s)
        return {'FINISHED'}


class GN_OT_remove_floor(Operator):
    bl_idname = "gn_int.remove_floor"
    bl_label = "Remove Floor"
    index: IntProperty()

    def execute(self, context):
        s = context.scene.gn_int
        if 0 <= self.index < len(s.floors):
            s.floors.remove(self.index)
        return {'FINISHED'}


def _sort_floors(s):
    data = sorted(((f.z, f.top) for f in s.floors), key=lambda t: t[0])
    s.floors.clear()
    for z, top in data:
        it = s.floors.add()
        it.z = z
        it.top = top


def _floor_tops(context):
    """Return list of (base_z, room_top_z, next_base_or_None) honoring height+gap.

    Per-floor top override wins; otherwise base + global Room Height.
    """
    s = context.scene.gn_int
    floors = sorted(((f.z, f.top) for f in s.floors), key=lambda t: t[0])
    out = []
    for i, (b, custom_top) in enumerate(floors):
        nxt = floors[i + 1][0] if i + 1 < len(floors) else None
        top = custom_top if custom_top > b + 0.1 else b + s.room_height
        out.append((b, top, nxt))
    return out


class GN_OT_gen_boundaries(Operator):
    bl_idname = "gn_int.gen_boundaries"
    bl_label = "Generate Boundaries"
    bl_description = "Create a per-floor boundary outline (exterior inset by the margin)"

    def execute(self, context):
        s = context.scene.gn_int
        ex = s.exterior
        if not ex:
            self.report({'ERROR'}, "Set an exterior shell")
            return {'CANCELLED'}
        if not s.floors:
            self.report({'ERROR'}, "Add at least one floor level")
            return {'CANCELLED'}
        src = _eval_bmesh(ex, context)
        _clear_coll(BOUND_COLL)
        coll = _get_coll(BOUND_COLL)
        made = 0
        for i, f in enumerate(_floor_tops(context)):
            base, top, nxt = f
            poly = outer_footprint(src, base + s.sample_offset)
            if not poly:
                self.report({'WARNING'}, f"Floor {i+1}: no solid outline at z={base+s.sample_offset:.2f}")
                continue
            poly = simplify_loop(poly, s.cleanup)
            ip = _weld_loop(inset_loop(poly, s.wall_margin), max(s.cleanup*0.4, 0.005))
            _make_loop_object(coll, f"GN_Bound_Floor{i+1}", ip, base)
            if i < len(s.floors):
                s.floors[i].bound_json = json.dumps([[round(p.x, 4), round(p.y, 4)] for p in ip])
            made += 1
        src.free()
        self.report({'INFO'}, f"Generated {made} floor boundaries")
        return {'FINISHED'}


class GN_OT_clear(Operator):
    bl_idname = "gn_int.clear"
    bl_label = "Clear Generated"
    bl_description = "Delete generated boundaries and interior shells"

    def execute(self, context):
        _clear_coll(BOUND_COLL)
        _clear_coll(INT_COLL)
        return {'FINISHED'}


# ---- mesh builders --------------------------------------------------------
def _make_loop_object(coll, name, poly_xy, z):
    bm = bmesh.new()
    vs = [bm.verts.new((p.x, p.y, z)) for p in poly_xy]
    for i in range(len(vs)):
        try:
            bm.edges.new((vs[i], vs[(i + 1) % len(vs)]))
        except ValueError:
            pass
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    ob.show_in_front = True
    ob.display_type = 'WIRE'
    return ob


def _wall_spans(A, B, base_z, ceil_z, openings):
    """Return (d, L, wn, spans) for wall A->B; spans = (x0,x1,z0,z1) holes on it."""
    d = B - A
    L = d.length
    if L < 1e-6:
        return None
    d = d / L
    wn = Vector((-d.y, d.x))            # wall inward normal (CCW polygon)
    spans = []
    for op in openings or []:
        nrm = Vector((op.nx, op.ny))
        if nrm.length > 1e-6 and abs((nrm.normalized()).dot(wn)) < 0.8:
            continue
        rel = Vector((op.cx, op.cy)) - A
        x = rel.dot(d)
        if abs(rel.dot(wn)) > 0.45 or x < -op.hw or x > L + op.hw:
            continue
        x0 = max(0.0, x - op.hw)
        x1 = min(L, x + op.hw)
        z0 = max(base_z, op.sill)
        z1 = min(ceil_z, op.top)
        if x1 - x0 > 0.02 and z1 - z0 > 0.02:
            spans.append((x0, x1, z0, z1))
    return d, L, wn, spans


def _build_wall(bm, A, B, base_z, ceil_z, openings, reveal_depth):
    """Build wall A->B (floor..ceil) with rectangular holes + optional reveal jambs."""
    r = _wall_spans(A, B, base_z, ceil_z, openings)
    if not r:
        return
    d, L, wn, spans = r
    xs = sorted(set([0.0, L] + [v for sp in spans for v in (sp[0], sp[1])]))
    zs = sorted(set([base_z, ceil_z] + [v for sp in spans for v in (sp[2], sp[3])]))

    def in_hole(cx, cz):
        for (x0, x1, z0, z1) in spans:
            if x0 - 1e-6 < cx < x1 + 1e-6 and z0 - 1e-6 < cz < z1 + 1e-6:
                return True
        return False

    def W(x, z):
        return bm.verts.new((A.x + d.x * x, A.y + d.y * x, z))
    for i in range(len(xs) - 1):
        for j in range(len(zs) - 1):
            if in_hole((xs[i] + xs[i + 1]) * 0.5, (zs[j] + zs[j + 1]) * 0.5):
                continue
            bm.faces.new((W(xs[i], zs[j]), W(xs[i + 1], zs[j]),
                          W(xs[i + 1], zs[j + 1]), W(xs[i], zs[j + 1])))
    # reveal jambs: extrude the opening rim outward (toward the exterior) by reveal_depth
    if reveal_depth > 1e-4:
        o = -wn * reveal_depth

        def P(x, z, out):
            base = Vector((A.x + d.x * x, A.y + d.y * x, z))
            if out:
                base = base + Vector((o.x, o.y, 0.0))
            return bm.verts.new((base.x, base.y, z))
        for (x0, x1, z0, z1) in spans:
            bm.faces.new((P(x0, z0, 0), P(x1, z0, 0), P(x1, z0, 1), P(x0, z0, 1)))  # bottom
            bm.faces.new((P(x0, z1, 0), P(x1, z1, 0), P(x1, z1, 1), P(x0, z1, 1)))  # top
            bm.faces.new((P(x0, z0, 0), P(x0, z1, 0), P(x0, z1, 1), P(x0, z0, 1)))  # left
            bm.faces.new((P(x1, z0, 0), P(x1, z1, 0), P(x1, z1, 1), P(x1, z0, 1)))  # right


def _build_shell(coll, name, poly_xy, base_z, ceil_z, openings=None, reveal_depth=0.0):
    bm = bmesh.new()
    n = len(poly_xy)
    # floor/ceiling loops carry the opening cut points too, so their edges weld to the
    # wall subdivisions (no T-junctions where an opening meets floor/ceiling)
    floor_loop, ceil_loop = [], []
    for k in range(n):
        A = poly_xy[k]
        B = poly_xy[(k + 1) % n]
        r = _wall_spans(A, B, base_z, ceil_z, openings)
        if not r:
            floor_loop.append((A.x, A.y, base_z))
            ceil_loop.append((A.x, A.y, ceil_z))
            continue
        d, L, wn, spans = r
        xs = sorted(set([0.0, L] + [v for sp in spans for v in (sp[0], sp[1])]))
        for x in xs[:-1]:              # corner + interior cuts (exclude L = next corner)
            floor_loop.append((A.x + d.x * x, A.y + d.y * x, base_z))
            ceil_loop.append((A.x + d.x * x, A.y + d.y * x, ceil_z))
    fv = [bm.verts.new(p) for p in floor_loop]
    cv = [bm.verts.new(p) for p in ceil_loop]
    try:
        bm.faces.new(fv)               # floor
    except ValueError:
        pass
    try:
        bm.faces.new(cv[::-1])         # ceiling
    except ValueError:
        pass
    for k in range(n):
        _build_wall(bm, poly_xy[k], poly_xy[(k + 1) % n], base_z, ceil_z,
                    openings, reveal_depth)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-4)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    for f in bm.faces:                 # face inward
        f.normal_flip()
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob


# ===========================================================================
# rooms
# ===========================================================================
def _pt_in_poly(p, poly):
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        pi = poly[i]
        pj = poly[j]
        if ((pi.y > p.y) != (pj.y > p.y)) and \
           (p.x < (pj.x - pi.x) * (p.y - pi.y) / ((pj.y - pi.y) or 1e-9) + pi.x):
            inside = not inside
        j = i
    return inside


def _nearest_on_seg(p, a, b):
    ab = b - a
    d = ab.dot(ab)
    t = 0.0
    if d > 1e-12:
        t = max(0.0, min(1.0, (p - a).dot(ab) / d))
    return a + ab * t


def _clamp_to_poly(p, poly):
    if not poly or _pt_in_poly(p, poly):
        return p
    best = None
    bd = 1e18
    n = len(poly)
    for i in range(n):
        q = _nearest_on_seg(p, poly[i], poly[(i + 1) % n])
        dd = (q - p).length_squared
        if dd < bd:
            bd = dd
            best = q
    return best if best else p


def _snap_pt(p, g):
    if g <= 0:
        return p
    return Vector((round(p.x / g) * g, round(p.y / g) * g))


def _floor_by_index(context, idx):
    """Return (base_z, ceil_z, boundary_poly_or_None) for floor idx."""
    tops = _floor_tops(context)
    if not (0 <= idx < len(tops)):
        return None
    base, top, _ = tops[idx]
    s = context.scene.gn_int
    bnd = None
    if idx < len(s.floors) and s.floors[idx].bound_json:
        try:
            bnd = [Vector(pt) for pt in json.loads(s.floors[idx].bound_json)]
        except Exception:
            bnd = None
    return (base, top, bnd)


def _new_uid(s):
    u = s.uid_counter if s.uid_counter else 1
    s.uid_counter = u + 1
    return u


def create_room(context, poly_xy, floor_idx):
    s = context.scene.gn_int
    fl = _floor_by_index(context, floor_idx)
    if not fl:
        return None
    base, top, _ = fl
    rec = s.rooms.add()
    rec.floor_index = floor_idx
    rec.uid = _new_uid(s)
    rec.poly_json = json.dumps([[round(p.x, 4), round(p.y, 4)] for p in poly_xy])
    coll = _get_coll(ROOM_COLL)
    ob = _build_shell(coll, f"GN_Room_F{floor_idx+1}_{rec.uid}", poly_xy, base, top,
                      s.openings, s.wall_margin if s.reveal else 0.0)
    ob["gn_room_uid"] = rec.uid
    return rec


def rebuild_rooms(context):
    """Rebuild all room shells, PRESERVING each room's visibility (keyed by uid)."""
    s = context.scene.gn_int
    # snapshot hide state by uid before clearing
    prev = {}
    coll = bpy.data.collections.get(ROOM_COLL)
    if coll:
        for ob in coll.objects:
            u = ob.get("gn_room_uid")
            if u is not None:
                prev[u] = (ob.hide_get(), ob.hide_render, ob.hide_select)
    _clear_coll(ROOM_COLL)
    coll = _get_coll(ROOM_COLL)
    for r in s.rooms:
        if not r.uid:
            r.uid = _new_uid(s)
        fl = _floor_by_index(context, r.floor_index)
        if not fl:
            continue
        base, top, _ = fl
        try:
            poly = [Vector(pt) for pt in json.loads(r.poly_json)]
        except Exception:
            continue
        if len(poly) >= 3:
            ob = _build_shell(coll, f"GN_Room_F{r.floor_index+1}_{r.uid}", poly,
                              base, top, s.openings, s.wall_margin if s.reveal else 0.0)
            ob["gn_room_uid"] = r.uid
            if r.uid in prev:                 # restore visibility
                hv, hr, hs = prev[r.uid]
                ob.hide_set(hv)
                ob.hide_render = hr
                ob.hide_select = hs
    _dump_scene(context.scene)                # keep the reload-survival backup current


def _clip_halfplane(poly, C, n):
    """Sutherland-Hodgman clip: keep the part of poly where (p-C).n >= 0."""
    out = []
    N = len(poly)
    for i in range(N):
        cur = poly[i]
        nxt = poly[(i + 1) % N]
        dcur = (cur - C).dot(n)
        dnxt = (nxt - C).dot(n)
        if dcur >= 0:
            out.append(cur)
        if (dcur >= 0) != (dnxt >= 0):
            denom = dcur - dnxt
            if abs(denom) > 1e-12:
                t = dcur / denom
                out.append(cur + (nxt - cur) * t)
    return out


def split_polygon(poly, A, B, gap):
    """Split poly along line A->B, pushing the two halves apart by `gap`.
    Returns (poly_pos, poly_neg) - either may be [] if empty/degenerate."""
    d = (B - A)
    if d.length < 1e-6:
        return poly, []
    d.normalize()
    n = Vector((-d.y, d.x))          # normal to the cut line
    h = gap * 0.5
    pos = _clip_halfplane(poly, A + n * h, n)
    neg = _clip_halfplane(poly, A - n * h, -n)
    pos = pos if len(pos) >= 3 and abs(_poly_area(pos)) > 0.05 else []
    neg = neg if len(neg) >= 3 and abs(_poly_area(neg)) > 0.05 else []
    return pos, neg


def seed_rooms_from_boundaries(context):
    """Make one room per floor = that floor's whole boundary polygon."""
    s = context.scene.gn_int
    s.rooms.clear()
    for i, f in enumerate(s.floors):
        if not f.bound_json:
            continue
        rec = s.rooms.add()
        rec.floor_index = i
        rec.uid = _new_uid(s)
        rec.poly_json = f.bound_json
    rebuild_rooms(context)


def split_room_record(context, room_idx, A, B):
    """Split room[room_idx] by line A-B with the partition gap. Returns count made."""
    s = context.scene.gn_int
    if not (0 <= room_idx < len(s.rooms)):
        return 0
    rec = s.rooms[room_idx]
    fidx = rec.floor_index
    try:
        poly = [Vector(p) for p in json.loads(rec.poly_json)]
    except Exception:
        return 0
    pos, neg = split_polygon(poly, A, B, s.partition)
    if not pos or not neg:
        return 0
    s.rooms.remove(room_idx)
    for part in (pos, neg):
        r = s.rooms.add()
        r.floor_index = fidx
        r.uid = _new_uid(s)
        r.poly_json = json.dumps([[round(p.x, 4), round(p.y, 4)] for p in part])
    rebuild_rooms(context)
    return 2


def _room_at_point(context, floor_idx, pt):
    """Index of the room on floor_idx whose footprint contains pt, else -1."""
    s = context.scene.gn_int
    for i, r in enumerate(s.rooms):
        if r.floor_index != floor_idx:
            continue
        try:
            poly = [Vector(p) for p in json.loads(r.poly_json)]
        except Exception:
            continue
        if _pt_in_poly(pt, poly):
            return i
    return -1


def _plane_hit(context, event, z):
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return None
    co = (event.mouse_region_x, event.mouse_region_y)
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, co)
    vec = view3d_utils.region_2d_to_vector_3d(region, rv3d, co)
    hit = intersect_line_plane(origin, origin + vec, Vector((0, 0, z)), Vector((0, 0, 1)))
    return hit


def _draw_room_overlay(self, context):
    if self.p0 is None or self.p1 is None:
        return
    a, b, z = self.p0, self.p1, self.base_z
    pts = [(a.x, a.y, z), (b.x, a.y, z), (b.x, b.y, z), (a.x, b.y, z)]
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": pts})
    gpu.state.line_width_set(2.0)
    gpu.state.blend_set('ALPHA')
    shader.bind()
    shader.uniform_float("color", (1.0, 0.6, 0.1, 1.0))
    batch.draw(shader)
    gpu.state.line_width_set(1.0)


class GN_OT_draw_room(Operator):
    bl_idname = "gn_int.draw_room"
    bl_label = "Draw Room"
    bl_description = ("Click two opposite corners on the active floor to create a "
                      "rectangular room (clamped to the boundary)")

    def invoke(self, context, event):
        s = context.scene.gn_int
        fl = _floor_by_index(context, s.active_floor)
        if not fl or fl[2] is None:
            self.report({'ERROR'}, "Generate boundaries first (need a floor boundary)")
            return {'CANCELLED'}
        self.base_z = fl[0]
        self.bound = fl[2]
        self.p0 = None
        self.p1 = None
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_room_overlay, (self, context), 'WINDOW', 'POST_VIEW')
        context.window_manager.modal_handler_add(self)
        context.area.header_text_set("Draw Room: click first corner, then opposite corner  |  Esc/RMB cancel")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        context.area.tag_redraw()
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
        if event.type == 'MOUSEMOVE':
            hit = _plane_hit(context, event, self.base_z)
            if hit:
                snapped = _snap_pt(Vector((hit.x, hit.y)), context.scene.gn_int.snap)
                if self.p0 is None:
                    self.p1 = snapped   # show cursor crosshair-ish (single corner preview off)
                    self.p1 = None
                else:
                    self.p1 = snapped
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            hit = _plane_hit(context, event, self.base_z)
            if hit is None:
                return {'RUNNING_MODAL'}
            snapped = _snap_pt(Vector((hit.x, hit.y)), context.scene.gn_int.snap)
            if self.p0 is None:
                self.p0 = snapped
            else:
                self.p1 = snapped
                self._commit(context)
                return self._end(context, True)
            return {'RUNNING_MODAL'}
        if event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
            return self._end(context, False)
        return {'RUNNING_MODAL'}

    def _commit(self, context):
        s = context.scene.gn_int
        a, b = self.p0, self.p1
        x1, x2 = sorted((a.x, b.x))
        y1, y2 = sorted((a.y, b.y))
        if abs(x2 - x1) < 0.05 or abs(y2 - y1) < 0.05:
            return
        poly = [Vector((x1, y1)), Vector((x2, y1)), Vector((x2, y2)), Vector((x1, y2))]
        poly = [_clamp_to_poly(p, self.bound) for p in poly]
        create_room(context, poly, s.active_floor)

    def _end(self, context, ok):
        bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
        context.area.header_text_set(None)
        context.area.tag_redraw()
        return {'FINISHED'} if ok else {'CANCELLED'}


class GN_OT_add_room(Operator):
    bl_idname = "gn_int.add_room"
    bl_label = "Add Room (center)"
    bl_description = "Add a default room in the middle of the active floor (no drawing)"

    def execute(self, context):
        s = context.scene.gn_int
        fl = _floor_by_index(context, s.active_floor)
        if not fl or fl[2] is None:
            self.report({'ERROR'}, "Generate boundaries first")
            return {'CANCELLED'}
        bnd = fl[2]
        xs = [p.x for p in bnd]
        ys = [p.y for p in bnd]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        w = (max(xs) - min(xs)) * 0.3
        h = (max(ys) - min(ys)) * 0.3
        poly = [Vector((cx - w, cy - h)), Vector((cx + w, cy - h)),
                Vector((cx + w, cy + h)), Vector((cx - w, cy + h))]
        poly = [_clamp_to_poly(p, bnd) for p in poly]
        create_room(context, poly, s.active_floor)
        self.report({'INFO'}, f"Added room on floor {s.active_floor+1}")
        return {'FINISHED'}


class GN_OT_remove_room(Operator):
    bl_idname = "gn_int.remove_room"
    bl_label = "Remove Room"
    index: IntProperty()

    def execute(self, context):
        s = context.scene.gn_int
        if 0 <= self.index < len(s.rooms):
            s.rooms.remove(self.index)
            rebuild_rooms(context)
        return {'FINISHED'}


class GN_OT_rebuild_rooms(Operator):
    bl_idname = "gn_int.rebuild_rooms"
    bl_label = "Rebuild Rooms"
    bl_description = "Rebuild all room shells from stored footprints"

    def execute(self, context):
        rebuild_rooms(context)
        return {'FINISHED'}


class GN_OT_clear_rooms(Operator):
    bl_idname = "gn_int.clear_rooms"
    bl_label = "Clear Rooms"

    def execute(self, context):
        context.scene.gn_int.rooms.clear()
        _clear_coll(ROOM_COLL)
        return {'FINISHED'}


class GN_OT_seed_rooms(Operator):
    bl_idname = "gn_int.seed_rooms"
    bl_label = "Rooms = Envelope"
    bl_description = ("Start room-making: one room per floor = that floor's whole "
                      "envelope. Then split it into rooms")

    def execute(self, context):
        if not context.scene.gn_int.floors:
            self.report({'ERROR'}, "Generate boundaries first")
            return {'CANCELLED'}
        seed_rooms_from_boundaries(context)
        n = len(context.scene.gn_int.rooms)
        if n == 0:
            self.report({'WARNING'}, "No boundaries stored - run Generate Boundaries")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Seeded {n} room(s) from the envelope")
        return {'FINISHED'}


def _draw_cut_overlay(self, context):
    if self.p0 is None or self.p1 is None:
        return
    z = self.base_z
    pts = [(self.p0.x, self.p0.y, z), (self.p1.x, self.p1.y, z)]
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'LINES', {"pos": pts})
    gpu.state.line_width_set(3.0)
    gpu.state.blend_set('ALPHA')
    shader.bind()
    shader.uniform_float("color", (1.0, 0.25, 0.25, 1.0))
    batch.draw(shader)
    gpu.state.line_width_set(1.0)


class GN_OT_split_room(Operator):
    bl_idname = "gn_int.split_room"
    bl_label = "Split Room"
    bl_description = ("Click two points across a room (wall to opposite wall) to "
                      "cut it into two rooms with a partition-wall gap")

    def invoke(self, context, event):
        s = context.scene.gn_int
        fl = _floor_by_index(context, s.active_floor)
        if not fl:
            self.report({'ERROR'}, "No active floor")
            return {'CANCELLED'}
        if not any(r.floor_index == s.active_floor for r in s.rooms):
            self.report({'ERROR'}, "No rooms on this floor - click 'Rooms = Envelope' first")
            return {'CANCELLED'}
        self.base_z = fl[0]
        self.p0 = None
        self.p1 = None
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_cut_overlay, (self, context), 'WINDOW', 'POST_VIEW')
        context.window_manager.modal_handler_add(self)
        context.area.header_text_set("Split: click on one wall, then the opposite wall  |  Esc/RMB cancel")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        context.area.tag_redraw()
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
        if event.type == 'MOUSEMOVE' and self.p0 is not None:
            hit = _plane_hit(context, event, self.base_z)
            if hit:
                self.p1 = _snap_pt(Vector((hit.x, hit.y)), context.scene.gn_int.snap)
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            hit = _plane_hit(context, event, self.base_z)
            if hit is None:
                return {'RUNNING_MODAL'}
            p = _snap_pt(Vector((hit.x, hit.y)), context.scene.gn_int.snap)
            if self.p0 is None:
                self.p0 = p
            else:
                self.p1 = p
                self._commit(context)
                return self._end(context, True)
            return {'RUNNING_MODAL'}
        if event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
            return self._end(context, False)
        return {'RUNNING_MODAL'}

    def _commit(self, context):
        s = context.scene.gn_int
        mid = (self.p0 + self.p1) * 0.5
        ridx = _room_at_point(context, s.active_floor, mid)
        if ridx < 0:
            self.report({'WARNING'}, "Cut midpoint not inside a room")
            return
        made = split_room_record(context, ridx, self.p0, self.p1)
        if made:
            self.report({'INFO'}, "Room split")
        else:
            self.report({'WARNING'}, "Split failed (line must cross the room)")

    def _end(self, context, ok):
        bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
        context.area.header_text_set(None)
        context.area.tag_redraw()
        return {'FINISHED'} if ok else {'CANCELLED'}


class GN_OT_split_edges(Operator):
    bl_idname = "gn_int.split_edges"
    bl_label = "Split at Selected Edges"
    bl_description = ("Edit Mode: select two edges (one on each of two opposite "
                      "walls) of a room, then run this to cut the room between them")

    @classmethod
    def poll(cls, context):
        ob = context.edit_object
        return ob is not None and ob.type == 'MESH'

    def execute(self, context):
        ob = context.edit_object
        bm = bmesh.from_edit_mesh(ob.data)
        mw = ob.matrix_world
        pts = []
        for e in bm.edges:
            if e.select:
                for v in e.verts:
                    w = mw @ v.co
                    pts.append(Vector((w.x, w.y)))
        if not pts:                       # fall back to selected verts
            for v in bm.verts:
                if v.select:
                    w = mw @ v.co
                    pts.append(Vector((w.x, w.y)))
        # collapse near-coincident points (a vertical wall edge -> one XY point)
        uniq = []
        for p in pts:
            if not any((p - q).length < 0.05 for q in uniq):
                uniq.append(p)
        if len(uniq) < 2:
            self.report({'ERROR'}, "Select two edges on opposite walls")
            return {'CANCELLED'}
        # cut line = the two farthest-apart selected points
        A, B = uniq[0], uniq[1]
        bd = (A - B).length_squared
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                dd = (uniq[i] - uniq[j]).length_squared
                if dd > bd:
                    bd = dd
                    A, B = uniq[i], uniq[j]
        # which room? use the object's uid tag, else the room under the cut midpoint
        ruid = ob.get("gn_room_uid")
        mid = (A + B) * 0.5
        bpy.ops.object.mode_set(mode='OBJECT')
        s = context.scene.gn_int
        ridx = -1
        if ruid is not None:
            ridx = next((i for i, r in enumerate(s.rooms) if r.uid == ruid), -1)
        if ridx < 0:
            for i, r in enumerate(s.rooms):
                try:
                    poly = [Vector(pt) for pt in json.loads(r.poly_json)]
                except Exception:
                    continue
                if _pt_in_poly(mid, poly):
                    ridx = i
                    break
        if ridx < 0:
            self.report({'WARNING'}, "Could not resolve which room these edges belong to")
            return {'CANCELLED'}
        if split_room_record(context, ridx, A, B):
            self.report({'INFO'}, "Room split between the selected edges")
            return {'FINISHED'}
        self.report({'WARNING'}, "Split failed - the two edges must be on opposite walls")
        return {'CANCELLED'}


# ===========================================================================
# window / door openings (boolean projection)
# ===========================================================================
def _piece_frame(obj):
    """Return (center, u, v, n, half_w, half_h) for a window/door piece.
    Works for a flat plane OR a real 3D mesh: n = area-weighted face normal,
    u/v = in-plane axes, extents from the projected bounds."""
    me = obj.data
    if not me.vertices:
        return None
    mw = obj.matrix_world
    verts = [mw @ v.co for v in me.vertices]
    n = Vector((0.0, 0.0, 0.0))
    rot = mw.to_3x3()
    for p in me.polygons:
        n += (rot @ p.normal) * max(p.area, 1e-6)
    if n.length < 1e-6:
        n = Vector((0.0, 1.0, 0.0))
    n.normalize()
    up = Vector((0.0, 0.0, 1.0))
    if abs(n.dot(up)) > 0.95:            # near-horizontal opening (rare)
        u = Vector((1.0, 0.0, 0.0))
    else:
        u = n.cross(up).normalized()
    v = u.cross(n).normalized()
    us = [p.dot(u) for p in verts]
    vs = [p.dot(v) for p in verts]
    ns = [p.dot(n) for p in verts]
    cu = (min(us) + max(us)) * 0.5
    cv = (min(vs) + max(vs)) * 0.5
    cn = (min(ns) + max(ns)) * 0.5
    center = u * cu + v * cv + n * cn
    return center, u, v, n, (max(us) - min(us)) * 0.5, (max(vs) - min(vs)) * 0.5


class GN_OT_project_openings(Operator):
    bl_idname = "gn_int.project_openings"
    bl_label = "Project Openings"
    bl_description = ("Cut openings into the room walls from the selected window/door "
                      "pieces (planes or meshes). Openings survive room splits")

    def execute(self, context):
        s = context.scene.gn_int
        rooms = bpy.data.collections.get(ROOM_COLL)
        room_objs = set(rooms.objects) if rooms else set()
        pieces = [o for o in context.selected_objects
                  if o.type == 'MESH' and o not in room_objs]
        if not pieces:
            self.report({'ERROR'}, "Select the separated window/door pieces first")
            return {'CANCELLED'}
        made = 0
        for o in pieces:
            fr = _piece_frame(o)
            if not fr:
                continue
            center, u, v, n, hw, hh = fr
            op = s.openings.add()
            op.cx = center.x
            op.cy = center.y
            op.nx = n.x
            op.ny = n.y
            op.hw = hw + 0.01
            op.sill = center.z - hh
            op.top = center.z + hh
            made += 1
        rebuild_rooms(context)
        self.report({'INFO'}, f"Projected {made} opening(s) into the rooms")
        return {'FINISHED'}


class GN_OT_clear_openings(Operator):
    bl_idname = "gn_int.clear_openings"
    bl_label = "Clear Openings"

    def execute(self, context):
        context.scene.gn_int.openings.clear()
        _clear_coll(DOORFRAME_COLL)
        rebuild_rooms(context)
        return {'FINISHED'}


# ===========================================================================
# doors (Room-Tool-style edit mode)
# ===========================================================================
def _active_door(s):
    """Return (width, height, mesh_object) from the active door preset."""
    if 0 <= s.active_door_preset < len(s.door_presets):
        p = s.door_presets[s.active_door_preset]
        return p.width, p.height, p.mesh_object
    return 0.9, 2.0, None


def _room_base(obj):
    return min((v.co.z for v in obj.data.vertices), default=0.0)


def _door_under(s, xy, z):
    for i, op in enumerate(s.openings):
        if not op.is_door:
            continue
        c = Vector((op.cx, op.cy))
        n = Vector((op.nx, op.ny))
        along = Vector((-n.y, n.x))
        d = xy - c
        if abs(d.dot(along)) < op.hw and abs(d.dot(n)) < 0.35 and \
           op.sill - 0.05 < z < op.top + 0.05:
            return i
    return -1


def _place_door_mesh(op, mesh_src, width, height, base):
    if mesh_src is None or not mesh_src.data:
        return
    coll = _get_coll(DOORFRAME_COLL)
    name = f"GN_DoorFrame_{op.uid}"
    inst = bpy.data.objects.get(name)
    if inst is None:
        inst = bpy.data.objects.new(name, mesh_src.data)
        coll.objects.link(inst)
    else:
        inst.data = mesh_src.data
    n = Vector((op.nx, op.ny, 0.0)).normalized()
    along = Vector((-n.y, n.x, 0.0))
    up = Vector((0.0, 0.0, 1.0))
    center = Vector((op.cx, op.cy, base))
    M = Matrix(((along.x, n.x, up.x, center.x),
                (along.y, n.y, up.y, center.y),
                (along.z, n.z, up.z, center.z),
                (0.0, 0.0, 0.0, 1.0)))
    dim = mesh_src.dimensions
    sx = width / dim.x if dim.x > 1e-4 else 1.0
    sz = height / dim.z if dim.z > 1e-4 else 1.0
    inst.matrix_world = M @ Matrix.Diagonal((sx, 1.0, sz)).to_4x4()


def _remove_door_mesh(uid):
    inst = bpy.data.objects.get(f"GN_DoorFrame_{uid}")
    if inst:
        bpy.data.objects.remove(inst, do_unlink=True)


def add_door(context, xy, normal, base):
    s = context.scene.gn_int
    w, h, mesh = _active_door(s)
    op = s.openings.add()
    op.uid = _new_uid(s)
    op.is_door = True
    op.cx, op.cy = xy.x, xy.y
    op.nx, op.ny = normal.x, normal.y
    op.hw = w * 0.5
    op.sill = base
    op.top = base + h
    rebuild_rooms(context)
    _place_door_mesh(op, mesh, w, h, base)


def remove_door(context, idx):
    s = context.scene.gn_int
    _remove_door_mesh(s.openings[idx].uid)
    s.openings.remove(idx)
    rebuild_rooms(context)


def _wall_under_cursor(context, event, max_dist=0.8):
    """Snap to the nearest room WALL under the cursor (projected on the floor).
    Works for partition walls even though they're hidden inside closed room boxes.
    Returns (xy_on_wall, normal_xy, base_z) or None."""
    s = context.scene.gn_int
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return None
    co = (event.mouse_region_x, event.mouse_region_y)
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, co)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, co)
    best = None
    bestd = max_dist
    for r in s.rooms:
        fl = _floor_by_index(context, r.floor_index)
        if not fl:
            continue
        base = fl[0]
        p = intersect_line_plane(origin, origin + direction,
                                 Vector((0, 0, base)), Vector((0, 0, 1)))
        if p is None:
            continue
        xy = Vector((p.x, p.y))
        try:
            poly = [Vector(pt) for pt in json.loads(r.poly_json)]
        except Exception:
            continue
        nn = len(poly)
        for k in range(nn):
            A = poly[k]
            B = poly[(k + 1) % nn]
            q = _nearest_on_seg(xy, A, B)
            d = (q - xy).length
            if d < bestd:
                bestd = d
                e = B - A
                L = max(e.length, 1e-6)
                best = (q, Vector((-e.y / L, e.x / L)), base)
    return best


def _draw_door_ghost(self, context):
    if not self.hover:
        return
    obj, xy, n, base = self.hover
    s = context.scene.gn_int
    w, h, _ = _active_door(s)
    along = Vector((-n.y, n.x))
    a = xy - along * (w * 0.5)
    b = xy + along * (w * 0.5)
    z0, z1 = base, base + h
    pts = [(a.x, a.y, z0), (b.x, b.y, z0), (b.x, b.y, z1), (a.x, a.y, z1)]
    color = (1.0, 0.3, 0.3, 1.0) if self.remove_hover else (0.1, 1.0, 0.3, 1.0)
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(2.5)
    b2 = batch_for_shader(shader, 'LINE_LOOP', {"pos": pts})
    shader.bind()
    shader.uniform_float("color", color)
    b2.draw(shader)
    gpu.state.line_width_set(1.0)


class GN_OT_door_edit(Operator):
    bl_idname = "gn_int.door_edit"
    bl_label = "Door Edit Mode"
    bl_description = ("Hover a wall to preview a door; LMB = add (cuts both rooms), "
                      "LMB on a door = remove, Tab = next preset, Esc/RMB = exit")

    def invoke(self, context, event):
        s = context.scene.gn_int
        if not s.door_presets:
            p = s.door_presets.add()
            p.name = "Door"
        self.hover = None
        self.remove_hover = False
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_door_ghost, (self, context), 'WINDOW', 'POST_VIEW')
        context.window_manager.modal_handler_add(self)
        context.area.header_text_set("Door Edit: LMB add · LMB on door remove · Tab preset · Esc exit")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        context.area.tag_redraw()
        s = context.scene.gn_int
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
        if event.type == 'MOUSEMOVE':
            w = _wall_under_cursor(context, event)
            if w:
                xy, n, base = w
                self.hover = (None, xy, n, base)
                self.remove_hover = _door_under(s, xy, base + 0.1) >= 0
            else:
                self.hover = None
                self.remove_hover = False
            return {'RUNNING_MODAL'}
        if event.type == 'TAB' and event.value == 'PRESS':
            if s.door_presets:
                s.active_door_preset = (s.active_door_preset + 1) % len(s.door_presets)
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            w = _wall_under_cursor(context, event)
            if w:
                xy, n, base = w
                di = _door_under(s, xy, base + 0.1)
                if di >= 0:
                    remove_door(context, di)
                else:
                    add_door(context, xy, n, base)
            return {'RUNNING_MODAL'}
        if event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            context.area.header_text_set(None)
            context.area.tag_redraw()
            return {'FINISHED'}
        return {'RUNNING_MODAL'}


class GN_OT_door_preset_add(Operator):
    bl_idname = "gn_int.door_preset_add"
    bl_label = "Add Door Preset"

    def execute(self, context):
        s = context.scene.gn_int
        p = s.door_presets.add()
        p.name = f"Door {len(s.door_presets)}"
        s.active_door_preset = len(s.door_presets) - 1
        return {'FINISHED'}


class GN_OT_door_preset_remove(Operator):
    bl_idname = "gn_int.door_preset_remove"
    bl_label = "Remove Door Preset"

    def execute(self, context):
        s = context.scene.gn_int
        if s.door_presets:
            s.door_presets.remove(s.active_door_preset)
            s.active_door_preset = max(0, s.active_door_preset - 1)
        return {'FINISHED'}


# ===========================================================================
# UI
# ===========================================================================
class GN_UL_door_presets(bpy.types.UIList):
    def draw_item(self, ctx, layout, data, item, icon, adata, aprop, index=0, flt=0):
        layout.prop(item, "name", text="", emboss=False, icon='MESH_DATA')
        layout.label(text=f"{item.width:.2f}x{item.height:.2f}")


class GN_UL_floors(bpy.types.UIList):
    def draw_item(self, ctx, layout, data, item, icon, adata, aprop, index=0, flt=0):
        s = ctx.scene.gn_int
        h = (item.top - item.z) if item.top > item.z + 0.1 else s.room_height
        row = layout.row(align=True)
        row.label(text=f"Floor {index+1}", icon='DECORATE')
        row.label(text=f"z {item.z:.2f}  h {h:.2f}m")
        op = row.operator("gn_int.remove_floor", text="", icon='X')
        op.index = index


class GN_PT_interior(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GN Interior"
    bl_label = "Interior Tool"

    def draw(self, context):
        s = context.scene.gn_int
        layout = self.layout

        box = layout.box()
        row = box.row(align=True)
        row.prop(s, "exterior", text="Shell")
        row.operator("gn_int.set_exterior", text="", icon='EYEDROPPER')

        col = layout.column(align=True)
        col.prop(s, "wall_margin")
        col.prop(s, "room_height")
        col.prop(s, "floor_gap")
        col.prop(s, "sample_offset")
        col.prop(s, "cleanup")

        box = layout.box()
        box.label(text="Floor levels:", icon='ALIGN_JUSTIFY')
        box.template_list("GN_UL_floors", "", s, "floors", s, "floor_index", rows=3)
        r = box.row(align=True)
        r.operator("gn_int.add_floor_ground", icon='TRIA_DOWN_BAR')
        r.operator("gn_int.add_floor_sel", text="From Edge", icon='EDGESEL')

        layout.separator()
        layout.operator("gn_int.gen_boundaries", icon='MESH_GRID')
        layout.operator("gn_int.clear", icon='TRASH')

        box = layout.box()
        box.label(text="Rooms:", icon='MOD_BUILD')
        row = box.row(align=True)
        row.prop(s, "active_floor")
        row.prop(s, "snap")
        box.prop(s, "partition")
        box.operator("gn_int.seed_rooms", icon='MESH_PLANE')
        box.operator("gn_int.split_edges", icon='MOD_BEVEL')
        box.label(text="Edit Mode: pick 2 wall edges, then Split", icon='INFO')
        box.label(text=f"{len(s.rooms)} room(s)")
        row = box.row(align=True)
        row.operator("gn_int.rebuild_rooms", icon='FILE_REFRESH')
        row.operator("gn_int.clear_rooms", icon='TRASH')

        box = layout.box()
        box.label(text="Openings:", icon='MOD_BOOLEAN')
        box.prop(s, "reveal")
        box.operator("gn_int.project_openings", icon='SELECT_DIFFERENCE')
        box.label(text="Select window/door pieces, then project", icon='INFO')
        box.operator("gn_int.clear_openings", icon='TRASH')

        box = layout.box()
        box.label(text="Doors (between rooms):", icon='MESH_DATA')
        row = box.row()
        row.template_list("GN_UL_door_presets", "", s, "door_presets",
                          s, "active_door_preset", rows=2)
        col = row.column(align=True)
        col.operator("gn_int.door_preset_add", text="", icon='ADD')
        col.operator("gn_int.door_preset_remove", text="", icon='REMOVE')
        if 0 <= s.active_door_preset < len(s.door_presets):
            dp = s.door_presets[s.active_door_preset]
            box.prop(dp, "width")
            box.prop(dp, "height")
            box.prop(dp, "mesh_object")
        box.operator("gn_int.door_edit", icon='GREASEPENCIL')


# ===========================================================================
_classes = (
    GN_FloorLevel, GN_Room, GN_Opening, GN_DoorPreset, GN_IntProps,
    GN_OT_set_exterior, GN_OT_add_floor_sel, GN_OT_add_floor_ground,
    GN_OT_remove_floor, GN_OT_gen_boundaries, GN_OT_clear,
    GN_OT_draw_room, GN_OT_add_room, GN_OT_remove_room, GN_OT_rebuild_rooms,
    GN_OT_clear_rooms, GN_OT_seed_rooms, GN_OT_split_room, GN_OT_split_edges,
    GN_OT_project_openings, GN_OT_clear_openings,
    GN_OT_door_edit, GN_OT_door_preset_add, GN_OT_door_preset_remove,
    GN_UL_door_presets, GN_UL_floors, GN_PT_interior,
)


_SETTINGS_KEYS = ("wall_margin", "room_height", "floor_gap", "sample_offset",
                  "cleanup", "partition", "reveal", "uid_counter",
                  "active_floor", "snap", "active_door_preset")


def _dump_scene(scene):
    """Serialize interior data to a plain ID-property that survives add-on reload."""
    if not hasattr(scene, "gn_int"):
        return
    s = scene.gn_int
    data = {
        "settings": {k: getattr(s, k) for k in _SETTINGS_KEYS},
        "exterior": s.exterior.name if s.exterior else "",
        "floors": [{"z": f.z, "top": f.top, "bound_json": f.bound_json} for f in s.floors],
        "rooms": [{"floor_index": r.floor_index, "uid": r.uid,
                   "poly_json": r.poly_json} for r in s.rooms],
        "openings": [{k: getattr(o, k) for k in
                      ("cx", "cy", "nx", "ny", "hw", "sill", "top", "uid", "is_door")}
                     for o in s.openings],
        "door_presets": [{"name": p.name, "width": p.width, "height": p.height,
                          "mesh": p.mesh_object.name if p.mesh_object else ""}
                         for p in s.door_presets],
    }
    scene["gn_int_backup"] = json.dumps(data)


def _restore_scene(scene):
    raw = scene.get("gn_int_backup")
    if not raw:
        return
    try:
        data = json.loads(raw)
    except Exception:
        return
    s = scene.gn_int
    for k, v in data.get("settings", {}).items():
        try:
            setattr(s, k, v)
        except Exception:
            pass
    exn = data.get("exterior")
    if exn and exn in bpy.data.objects:
        s.exterior = bpy.data.objects[exn]
    s.floors.clear()
    for f in data.get("floors", []):
        it = s.floors.add()
        it.z = f["z"]; it.top = f["top"]; it.bound_json = f.get("bound_json", "")
    s.rooms.clear()
    for r in data.get("rooms", []):
        it = s.rooms.add()
        it.floor_index = r["floor_index"]; it.uid = r["uid"]; it.poly_json = r["poly_json"]
    s.openings.clear()
    for o in data.get("openings", []):
        it = s.openings.add()
        for k, v in o.items():
            setattr(it, k, v)
    s.door_presets.clear()
    for p in data.get("door_presets", []):
        it = s.door_presets.add()
        it.name = p.get("name", "Door")
        it.width = p.get("width", 0.9)
        it.height = p.get("height", 2.0)
        mn = p.get("mesh", "")
        if mn and mn in bpy.data.objects:
            it.mesh_object = bpy.data.objects[mn]


def _deferred_restore():
    # runs after enable completes (bpy.data is accessible here, unlike register())
    for scene in bpy.data.scenes:
        try:
            s = scene.gn_int
            if len(s.rooms) == 0 and len(s.floors) == 0 and scene.get("gn_int_backup"):
                _restore_scene(scene)           # only when a reload wiped the data
        except Exception as e:
            print("[GN Interior] restore failed:", e)
    return None                                 # don't repeat


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.gn_int = PointerProperty(type=GN_IntProps)
    bpy.app.timers.register(_deferred_restore, first_interval=0.0)


def unregister():
    try:
        for scene in bpy.data.scenes:           # best-effort backup before delete
            _dump_scene(scene)
    except Exception:
        pass
    del bpy.types.Scene.gn_int
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
