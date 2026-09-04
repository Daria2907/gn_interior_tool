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
import re
import numpy as np
import gpu
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
from mathutils import Vector, Matrix
from mathutils.geometry import intersect_line_plane
from bpy.props import (FloatProperty, IntProperty, StringProperty, BoolProperty,
                       PointerProperty, CollectionProperty, EnumProperty)
from bpy.types import Operator, Panel, PropertyGroup

BOUND_COLL = "GN_Boundaries"
INT_COLL = "GN_Interiors"
ROOM_COLL = "GN_Rooms"
CUTTER_COLL = "GN_Cutters"
DOORFRAME_COLL = "GN_DoorFrames"
WINDOWFRAME_COLL = "GN_WindowFrames"
THRESHOLD_COLL = "GN_Thresholds"
OPENING_PIECES_COLL = "Openings"
STAIR_COLL = "GN_Stairs"


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


# ===========================================================================
# raster silhouette footprint  (robust on fragmented / holey / junky shells)
#
# rasterize wall cross-section -> seal gaps (morphological close) -> flood the
# outside -> the enclosed region is the building -> trace one contour ->
# simplify -> rectilinearize (straighten walls, drop corner chamfers).
# Handles non-manifold shells, holes, stray window/stair geometry, fragments.
# ===========================================================================
def _cut_segments(src, zc):
    """Cross-section the shell at height zc -> list of ((x0,y0),(x1,y1))."""
    bm = src.copy()
    res = bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
                                 dist=1e-5, plane_co=(0, 0, zc), plane_no=(0, 0, 1))
    segs = []
    for e in res['geom_cut']:
        if isinstance(e, bmesh.types.BMEdge):
            a = e.verts[0].co
            b = e.verts[1].co
            segs.append(((a.x, a.y), (b.x, b.y)))
    bm.free()
    return segs


def _rast_line(grid, a, b, minx, miny, cell):
    ax = (a[0] - minx) / cell; ay = (a[1] - miny) / cell
    bx = (b[0] - minx) / cell; by = (b[1] - miny) / cell
    n = int(max(abs(bx - ax), abs(by - ay))) + 1
    H, W = grid.shape
    for i in range(n + 1):
        t = i / n
        c = int(ax + (bx - ax) * t); r = int(ay + (by - ay) * t)
        if 0 <= r < H and 0 <= c < W:
            grid[r, c] = True


def _rast_dilate(mask, r):
    out = mask.copy()
    for _ in range(r):
        acc = out.copy()
        acc[1:, :] |= out[:-1, :]; acc[:-1, :] |= out[1:, :]
        acc[:, 1:] |= out[:, :-1]; acc[:, :-1] |= out[:, 1:]
        out = acc
    return out


def _rast_erode(mask, r):
    return ~_rast_dilate(~mask, r)


def _rast_flood_border(free):
    H, W = free.shape
    out = np.zeros_like(free)
    stack = []
    for c in range(W):
        if free[0, c]: stack.append((0, c))
        if free[H - 1, c]: stack.append((H - 1, c))
    for r in range(H):
        if free[r, 0]: stack.append((r, 0))
        if free[r, W - 1]: stack.append((r, W - 1))
    while stack:
        r, c = stack.pop()
        if out[r, c] or not free[r, c]:
            continue
        out[r, c] = True
        if r > 0: stack.append((r - 1, c))
        if r < H - 1: stack.append((r + 1, c))
        if c > 0: stack.append((r, c - 1))
        if c < W - 1: stack.append((r, c + 1))
    return out


def _rast_components(mask):
    H, W = mask.shape
    seen = np.zeros_like(mask)
    comps = []
    for sr in range(H):
        for sc in range(W):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            stack = [(sr, sc)]; cells = []
            while stack:
                r, c = stack.pop()
                if seen[r, c] or not mask[r, c]:
                    continue
                seen[r, c] = True; cells.append((r, c))
                if r > 0: stack.append((r - 1, c))
                if r < H - 1: stack.append((r + 1, c))
                if c > 0: stack.append((r, c - 1))
                if c < W - 1: stack.append((r, c + 1))
            comps.append((len(cells), cells))
    return comps


def _rast_mask_from_cells(shape, cells):
    m = np.zeros(shape, bool)
    for r, c in cells:
        m[r, c] = True
    return m


def _rast_fill_holes(mask):
    bg = ~mask
    outside = _rast_flood_border(bg)
    return mask | (bg & ~outside)


def _rast_trace(mask):
    H, W = mask.shape
    start = None
    for r in range(H):
        for c in range(W):
            if mask[r, c]:
                start = (r, c); break
        if start:
            break
    if start is None:
        return []
    nbr = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

    def solid(r, c):
        return 0 <= r < H and 0 <= c < W and mask[r, c]

    contour = [start]; cur = start; b = 6; count = 0; maxc = H * W * 4
    while count < maxc:
        count += 1; found = False
        for k in range(8):
            d = (b + 1 + k) % 8
            nr = cur[0] + nbr[d][0]; nc = cur[1] + nbr[d][1]
            if solid(nr, nc):
                b = (d + 4) % 8; cur = (nr, nc); found = True; break
        if not found:
            break
        if cur == start and len(contour) > 2:
            break
        contour.append(cur)
    return contour


def _r_dp(pts, tol):
    if len(pts) < 3:
        return pts
    ax, ay = pts[0]; bx, by = pts[-1]
    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy)
    dmax, idx = 0.0, 0
    for i in range(1, len(pts) - 1):
        px, py = pts[i]
        if L < 1e-9:
            d = math.hypot(px - ax, py - ay)
        else:
            d = abs((px - ax) * dy - (py - ay) * dx) / L
        if d > dmax:
            dmax, idx = d, i
    if dmax > tol:
        return _r_dp(pts[:idx + 1], tol)[:-1] + _r_dp(pts[idx:], tol)
    return [pts[0], pts[-1]]


def _r_dp_closed(poly, tol):
    if len(poly) < 4:
        return poly
    a = poly[0]
    far = max(range(len(poly)),
              key=lambda i: (poly[i][0] - a[0]) ** 2 + (poly[i][1] - a[1]) ** 2)
    s1 = _r_dp(poly[:far + 1], tol)
    s2 = _r_dp(poly[far:] + [poly[0]], tol)
    out = s1[:-1] + s2[:-1]
    return out if len(out) >= 3 else poly


def _r_area(pts):
    a = 0.0; n = len(pts)
    for i in range(n):
        a += pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
    return a * 0.5


def _line_isect(p0, d0, p1, d1):
    if d0 is None or d1 is None:
        return None
    den = d0[0] * d1[1] - d0[1] * d1[0]
    if abs(den) < 1e-9:
        return None
    dx = p1[0] - p0[0]; dy = p1[1] - p0[1]
    t = (dx * d1[1] - dy * d1[0]) / den
    return (p0[0] + d0[0] * t, p0[1] + d0[1] * t)


def _dom_orient(poly):
    """Length-weighted dominant orientation (radians, mod 90deg)."""
    sx = sy = 0.0; n = len(poly)
    for i in range(n):
        ax, ay = poly[i]; bx, by = poly[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy)
        if L < 1e-6:
            continue
        a = math.atan2(dy, dx) % (math.pi / 2)
        sx += L * math.cos(4 * a); sy += L * math.sin(4 * a)
    if sx == 0 and sy == 0:
        return 0.0
    return (math.atan2(sy, sx) / 4.0) % (math.pi / 2)


def regularize(poly, ang_tol_deg=20.0, min_edge=0.20, corner_max=0.7, allow45=False):
    """Straighten walls to the dominant axis and drop short off-grid corner
    chamfers so corners are the two long walls meeting directly. Genuine long
    angled walls keep their true angle."""
    n = len(poly)
    if n < 4:
        return poly
    th0 = _dom_orient(poly)
    tol = math.radians(ang_tol_deg)
    grid = [th0, th0 + math.pi / 2]
    if allow45:
        grid += [th0 + math.pi / 4, th0 + 3 * math.pi / 4]
    edges = []
    for i in range(n):
        ax, ay = poly[i]; bx, by = poly[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy)
        if L < 1e-6:
            continue
        a = math.atan2(dy, dx); best = None; bestd = 1e9
        for g in grid:
            for gg in (g, g + math.pi):
                d = abs(((a - gg + math.pi) % (2 * math.pi)) - math.pi)
                if d < bestd:
                    bestd = d; best = gg
        if bestd <= tol:
            d_unit = (math.cos(best), math.sin(best)); snp = True
        else:
            d_unit = (dx / L, dy / L); snp = False
        edges.append((ax, ay, bx, by, d_unit[0], d_unit[1], L, snp))
    m = len(edges)
    if m < 4:
        return poly

    def same_dir(e, f):
        cross = e[4] * f[5] - e[5] * f[4]
        dot = e[4] * f[4] + e[5] * f[5]
        return abs(cross) < 1e-4 and dot > 0.5

    start = 0
    for i in range(m):
        if not same_dir(edges[i - 1], edges[i]):
            start = i; break
    runs = []; i = 0; order = [(start + k) % m for k in range(m)]
    while i < m:
        j = order[i]; dirx, diry = edges[j][4], edges[j][5]
        wx = wy = wsum = 0.0; length = 0.0; k = i
        while k < m and same_dir(edges[order[k]], edges[j]):
            e = edges[order[k]]
            mx = (e[0] + e[2]) / 2; my = (e[1] + e[3]) / 2
            wx += mx * e[6]; wy += my * e[6]; wsum += e[6]; length += e[6]; k += 1
        runs.append([(wx / wsum, wy / wsum), (dirx, diry), length, edges[j][7]])
        i = k

    def keep(r):
        _, _, length, snp = r
        return length >= min_edge if snp else length > corner_max

    surv = [r for r in runs if keep(r)]
    if len(surv) < 4:
        surv = [r for r in runs if r[2] >= min_edge] or runs

    def parallel_same_way(a, b):
        (ax, ay), (bx, by) = a[1], b[1]
        cross = ax * by - ay * bx
        dot = ax * bx + ay * by
        return abs(cross) < 1e-4 and dot > 0.5

    def merge_runs(a, b):
        wsum = a[2] + b[2]
        mx = (a[0][0] * a[2] + b[0][0] * b[2]) / wsum
        my = (a[0][1] * a[2] + b[0][1] * b[2]) / wsum
        return [(mx, my), a[1], wsum, a[3] or b[3]]

    # Dropping a short corner chamfer leaves two flanking runs that are
    # PARALLEL (a raster staircase along one straight wall), not
    # perpendicular (a real corner) -- those can't be joined by
    # intersection, so without this they'd stay linked point-to-point as a
    # fake jog. Merge any such adjacent parallel runs into one.
    changed = True
    while changed and len(surv) > 1:
        changed = False
        ns = len(surv)
        for i in range(ns):
            if parallel_same_way(surv[i - 1], surv[i]):
                merged = merge_runs(surv[i - 1], surv[i])
                if i == 0:
                    surv = [merged] + surv[1:-1]
                else:
                    surv = surv[:i - 1] + [merged] + surv[i + 1:]
                changed = True
                break
    out = []
    for i in range(len(surv)):
        p = _line_isect(surv[i - 1][0], surv[i - 1][1], surv[i][0], surv[i][1])
        out.append(p if p else surv[i][0])
    cleaned = [out[0]]
    for p in out[1:]:
        if math.hypot(p[0] - cleaned[-1][0], p[1] - cleaned[-1][1]) > 1e-3:
            cleaned.append(p)
    if len(cleaned) > 2 and math.hypot(cleaned[0][0] - cleaned[-1][0],
                                       cleaned[0][1] - cleaned[-1][1]) <= 1e-3:
        cleaned.pop()
    cleaned = _despike(cleaned)
    return cleaned if len(cleaned) >= 3 else poly


def _turn_angle(a, b, c):
    """Deviation (deg) from straight at b: 0 = straight, 90 = corner, 180 = reversal."""
    v1x, v1y = b[0] - a[0], b[1] - a[1]
    v2x, v2y = c[0] - b[0], c[1] - b[1]
    L1 = math.hypot(v1x, v1y); L2 = math.hypot(v2x, v2y)
    if L1 < 1e-9 or L2 < 1e-9:
        return 0.0
    dot = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (L1 * L2)))
    return math.degrees(math.acos(dot))


def _despike(poly, straight_deg=6.0, spike_deg=150.0, tooth_deg=30.0, tooth_len=1.5):
    """Drop vertices that are redundant (nearly straight), spikes (the path
    nearly reverses -- thin slivers), or raster staircase "teeth" (a wall
    traced at a slight angle to the pixel grid produces an alternating
    ~45/90deg sawtooth of SHORT edges -- individual turn angles are too
    moderate to be caught as spikes, but both edges at the tooth are short).
    Rectilinear footprints only turn ~90deg, so any of these is an artifact."""
    poly = list(poly)
    changed = True
    while changed and len(poly) > 3:
        changed = False
        n = len(poly)
        for i in range(n):
            a, b, c = poly[(i - 1) % n], poly[i], poly[(i + 1) % n]
            t = _turn_angle(a, b, c)
            len1 = math.hypot(b[0] - a[0], b[1] - a[1])
            len2 = math.hypot(c[0] - b[0], c[1] - b[1])
            is_tooth = t < tooth_deg and len1 < tooth_len and len2 < tooth_len
            if t < straight_deg or t > spike_deg or is_tooth:
                del poly[i]; changed = True; break
    return poly


def raster_footprint(segs, tol=0.15, cell=0.05, seed=None, bridge=0.5,
                     square=True, ang_tol_deg=20.0, allow45=False):
    """segs -> ONE clean CCW footprint polygon [(x,y),...] (world units), or None.

    tol    = detail size to ignore (metres).
    bridge = max wall gap/hole to seal (metres).
    seed   = (x, y) interior point to disambiguate which region is the building.
    square = rectilinearize the result (straight walls, crisp corners).
    """
    if not segs:
        return None
    xs = [p[0] for s in segs for p in s]
    ys = [p[1] for s in segs for p in s]
    R = max(1, int(round(bridge * 0.5 / cell)))
    pad = (R + 3) * cell
    minx, miny = min(xs) - pad, min(ys) - pad
    maxx, maxy = max(xs) + pad, max(ys) + pad
    W = int((maxx - minx) / cell) + 1
    H = int((maxy - miny) / cell) + 1
    if W * H > 6_000_000:               # safety: keep grids sane on huge shells
        cell = math.sqrt((maxx - minx) * (maxy - miny) / 4_000_000.0)
        R = max(1, int(round(bridge * 0.5 / cell)))
        W = int((maxx - minx) / cell) + 1
        H = int((maxy - miny) / cell) + 1
    wall = np.zeros((H, W), bool)
    for a, b in segs:
        _rast_line(wall, a, b, minx, miny, cell)
    wall = _rast_dilate(wall, R)
    outside = _rast_flood_border(~wall)
    solid = ~outside
    solid = _rast_erode(solid, R)
    solid = _rast_fill_holes(solid)
    comps = _rast_components(solid)
    if not comps:
        return None
    chosen = None
    if seed is not None:
        sc = int((seed[0] - minx) / cell); sr = int((seed[1] - miny) / cell)
        for size, cells in comps:
            if (sr, sc) in set(cells):
                chosen = cells; break
    if chosen is None:
        comps.sort(key=lambda x: x[0], reverse=True)
        chosen = comps[0][1]
    region = _rast_fill_holes(_rast_mask_from_cells((H, W), chosen))
    contour = _rast_trace(region)
    if len(contour) < 3:
        return None
    poly = [(minx + (c + 0.5) * cell, miny + (rr + 0.5) * cell) for rr, c in contour]
    poly = _r_dp_closed(poly, tol)
    if square:
        poly = regularize(poly, ang_tol_deg=ang_tol_deg,
                          min_edge=max(tol, 0.15), corner_max=max(tol * 4, 0.7),
                          allow45=allow45)
    if _r_area(poly) < 0:
        poly = poly[::-1]
    return poly


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


def _clear_named(coll, name):
    """Remove any object in coll whose name matches (base or .NNN suffix)."""
    for ob in list(coll.objects):
        if ob.name == name or ob.name.startswith(name + "."):
            bpy.data.objects.remove(ob, do_unlink=True)


# ===========================================================================
# properties
# ===========================================================================
class GN_CustomEmptyItem(PropertyGroup):
    """One entry in the user's custom empty list (Add Empties)."""
    name: StringProperty(name="Name", default="custom")
    enabled: BoolProperty(name="Enabled", default=True)


class GN_FloorLevel(PropertyGroup):
    z: FloatProperty(name="Base Z", default=0.0, unit='LENGTH')
    top: FloatProperty(name="Top Z", default=0.0, unit='LENGTH',
        description="Ceiling Z (only used when Custom Top is on; otherwise "
        "base + Room Height)")
    top_is_custom: BoolProperty(default=False,
        description="Use this floor's explicit Top Z instead of base + Room Height")
    bound_json: StringProperty(default="")   # inset boundary polygon [[x,y],...]
    lock: BoolProperty(name="Lock", default=False,
        description="Lock this floor's boundary: Generate Boundaries skips it so "
        "hand edits are preserved")


class GN_Room(PropertyGroup):
    floor_index: IntProperty(default=0)
    poly_json: StringProperty(default="[]")  # footprint [[x,y],...]
    uid: IntProperty(default=0)              # stable id (survives rebuilds)
    z_offset: FloatProperty(default=0.0, unit='LENGTH',
        description="Vertical shift from the floor's own base/top (for a "
        "half-floor / mezzanine room). Set by grabbing and moving the room "
        "object in Z -- rebuild_rooms detects and re-applies it")


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
    projected: BoolProperty(default=False)   # from an exterior piece (no threshold)


class GN_DoorPreset(PropertyGroup):
    name: StringProperty(default="Door")
    width: FloatProperty(name="Width", default=0.9, min=0.2, max=4.0, unit='LENGTH')
    height: FloatProperty(name="Height", default=2.0, min=0.5, max=5.0, unit='LENGTH')
    mesh_object: PointerProperty(name="Frame Mesh", type=bpy.types.Object,
        description="Optional door mesh placed as a linked instance at each opening")


class GN_WindowPreset(PropertyGroup):
    name: StringProperty(default="Window")
    width: FloatProperty(name="Width", default=1.0, min=0.2, max=6.0, unit='LENGTH')
    height: FloatProperty(name="Height", default=1.2, min=0.2, max=5.0, unit='LENGTH')
    sill: FloatProperty(name="Sill Height", default=0.9, min=0.0, max=4.0, unit='LENGTH',
        description="Height of the window bottom above the floor")
    mesh_object: PointerProperty(name="Frame Mesh", type=bpy.types.Object,
        description="Optional window mesh placed as a linked instance at each opening")


_SUSPEND_CB = False


class _SceneCtx:
    """Minimal context-like shim (only .scene) for calling context-taking
    helpers from a deferred bpy.app.timers callback, where no real bpy
    Context is available."""
    __slots__ = ("scene",)

    def __init__(self, scene):
        self.scene = scene


def _cb_threshold(self, context):
    """Live-update threshold strips when the toggle/size changes.

    Deferred via a timer: rebuilding objects directly inside a property
    update callback can crash Blender if the callback fires while undo/redo
    is being processed (property restores during undo can trigger update
    callbacks). Scheduling it for the next event-loop tick avoids that
    reentrancy entirely -- same pattern as this add-on's reload-recovery.
    """
    if _SUSPEND_CB:
        return
    scene_name = context.scene.name

    def _do():
        scene = bpy.data.scenes.get(scene_name)
        if not scene or not hasattr(scene, "gn_int") or _SUSPEND_CB:
            return None
        try:
            _refresh_thresholds(_SceneCtx(scene))
        except Exception as e:
            print("[GN Interior] threshold update:", e)
        return None

    bpy.app.timers.register(_do, first_interval=0.0)


def _cb_stair_settings(self, context):
    """Live-rebuild the selected stair(s) when Step Height/Depth changes.
    _rebuild_stair_mesh is defined later in the file (near the rest of the
    stairs feature) -- fine, since a function body only resolves names at
    CALL time, and by then the whole module has finished loading.

    Deferred via a timer for the same reentrancy reason as _cb_threshold
    above: rebuilding objects directly inside a property update callback can
    crash Blender if it fires while undo/redo is being processed.
    """
    if _SUSPEND_CB:
        return
    names = [ob.name for ob in context.selected_objects
             if ob.type == 'MESH' and "gn_stair_data" in ob]
    if not names:
        return
    height, depth = self.stair_step_height, self.stair_step_depth

    def _do():
        for nm in names:
            ob = bpy.data.objects.get(nm)
            if ob:
                try:
                    _rebuild_stair_mesh(ob, height, depth)
                except Exception as e:
                    print("[GN Interior] stair update:", e)
        return None

    bpy.app.timers.register(_do, first_interval=0.0)


def _cb_mlo_name(self, context):
    """Keep Timecycle prefilled with the MLO Name as it's typed, until the
    user edits Timecycle to something else (then they're independent).
    Deferred via a timer for the same undo/redo reentrancy reason as
    _cb_threshold above."""
    if _SUSPEND_CB or not self.timecycle_auto:
        return
    scene_name = context.scene.name
    name = self.mlo_name

    def _do():
        scene = bpy.data.scenes.get(scene_name)
        if not scene or not hasattr(scene, "gn_int"):
            return None
        s = scene.gn_int
        if s.timecycle_auto:
            s.timecycle_name = name
        return None

    bpy.app.timers.register(_do, first_interval=0.0)


def _cb_timecycle_name(self, context):
    """Editing Timecycle by hand de-links it from MLO Name (re-links if it's
    typed back to match). Deferred via a timer, same reentrancy reason."""
    if _SUSPEND_CB:
        return
    scene_name = context.scene.name
    matches = (self.timecycle_name == self.mlo_name)

    def _do():
        scene = bpy.data.scenes.get(scene_name)
        if not scene or not hasattr(scene, "gn_int"):
            return None
        scene.gn_int.timecycle_auto = matches
        return None

    bpy.app.timers.register(_do, first_interval=0.0)


def _cb_floor_select(self, context):
    """Selecting a floor in the list selects its boundary + rooms in the
    scene, so the Outliner and viewport highlight what you're looking at.

    Deferred via a timer: mutating object selection directly inside a
    property update callback can crash Blender if the callback fires while
    undo/redo is being processed (property restores during undo can trigger
    update callbacks). Scheduling it for the next event-loop tick avoids that
    reentrancy entirely -- same pattern as this add-on's reload-recovery.
    """
    scene_name = context.scene.name
    idx = context.scene.gn_int.floor_index

    def _do():
        scene = bpy.data.scenes.get(scene_name)
        if not scene or not hasattr(scene, "gn_int"):
            return None
        s = scene.gn_int
        if s.floor_index != idx or not (0 <= idx < len(s.floors)):
            return None
        try:
            for ob in bpy.context.selectable_objects:
                ob.select_set(False)
        except Exception:
            pass
        floor = s.floors[idx]
        picked = []
        bnd = _boundary_object_for_floor(floor)
        if bnd:
            picked.append(bnd)
        room_coll = bpy.data.collections.get(ROOM_COLL)
        if room_coll:
            room_uids = {r.uid for r in s.rooms if r.floor_index == idx}
            for ob in room_coll.objects:
                if ob.get("gn_room_uid") in room_uids:
                    picked.append(ob)
        for ob in picked:
            try:
                ob.select_set(True)
            except Exception:
                pass
        if picked:
            try:
                bpy.context.view_layer.objects.active = picked[0]
            except Exception:
                pass
        try:
            for w in bpy.context.window_manager.windows:
                for a in w.screen.areas:
                    if a.type == 'VIEW_3D':
                        a.tag_redraw()
        except Exception:
            pass
        return None

    bpy.app.timers.register(_do, first_interval=0.0)


def _cb_portal_select(self, context):
    """Selecting a portal in the list selects that object, so the Outliner
    and viewport highlight what you're looking at. Deferred via a timer for
    the same undo-reentrancy reason as _cb_floor_select."""
    scene_name = context.scene.name
    idx = context.scene.gn_int.portal_index

    def _do():
        scene = bpy.data.scenes.get(scene_name)
        if not scene or not hasattr(scene, "gn_int"):
            return None
        s = scene.gn_int
        if s.portal_index != idx:
            return None
        coll = _mlo_portals_collection(scene)
        if not coll or not (0 <= idx < len(coll.objects)):
            return None
        try:
            for ob in bpy.context.selectable_objects:
                ob.select_set(False)
        except Exception:
            pass
        target = coll.objects[idx]
        try:
            target.select_set(True)
            bpy.context.view_layer.objects.active = target
        except Exception:
            pass
        try:
            for w in bpy.context.window_manager.windows:
                for a in w.screen.areas:
                    if a.type == 'VIEW_3D':
                        a.tag_redraw()
        except Exception:
            pass
        return None

    bpy.app.timers.register(_do, first_interval=0.0)


def _cb_redraw(self, context):
    """Redraw viewports so the selected-opening highlight updates."""
    try:
        for w in context.window_manager.windows:
            for a in w.screen.areas:
                if a.type == 'VIEW_3D':
                    a.tag_redraw()
    except Exception:
        pass


_ROOM_NAME_RE = re.compile(r'^r\d+$', re.IGNORECASE)


def _gn_iter_room_collections(context):
    """Yield r<digits> collections under the active MLO (int_<mlo_name>),
    falling back to any matching collections in the file if the MLO isn't
    found (so pickers still work before Build MLO Collections has run)."""
    try:
        name = context.scene.gn_int.mlo_name.strip()
    except Exception:
        name = ""
    mlo = bpy.data.collections.get(f"int_{name}") if name else None
    if mlo is not None:
        rooms = [c for c in mlo.children if _ROOM_NAME_RE.match(c.name)]
        if rooms:
            return sorted(rooms, key=lambda c: c.name)
    return sorted((c for c in bpy.data.collections if _ROOM_NAME_RE.match(c.name)),
                  key=lambda c: c.name)


def _gn_room_enum_items(self, context):
    items = [(c.name, c.name, f"Room collection {c.name}")
            for c in _gn_iter_room_collections(context)]
    if not items:
        items.append(("NONE", "-- no rooms found --", "Build MLO Collections first"))
    return items


def _gn_room_enum_items_with_all(self, context):
    rooms = _gn_iter_room_collections(context)
    items = [("ALL", "All Rooms", "Add to every room")]
    items.extend((c.name, c.name, f"Room collection {c.name}") for c in rooms)
    return items


def _gn_room_number_from_name(room_name):
    """'r01' -> '01' (keeps the zero-padding, unlike the portal-naming token)."""
    m = re.search(r'(\d+)$', room_name)
    return m.group(1) if m else room_name


class GN_IntProps(PropertyGroup):
    exterior: PointerProperty(name="Exterior Shell", type=bpy.types.Object,
        description="The exterior building shell to read")
    wall_margin: FloatProperty(name="Wall Margin", default=0.20, min=0.0, max=2.0,
        unit='LENGTH', description="Inset from exterior walls to interior walls")
    room_height: FloatProperty(name="Room Height", default=2.95, min=0.5, max=10.0,
        unit='LENGTH', description="Fixed interior clear height per floor")
    sample_offset: FloatProperty(name="Sample Height", default=1.0, min=0.05, max=5.0,
        unit='LENGTH', description="Height above each floor base to cut the outline "
        "(pick a solid wall band, between windows)")
    cleanup: FloatProperty(name="Cleanup", default=0.08, min=0.0, max=0.5,
        unit='LENGTH', description="Simplify the outline: remove wiggles/slivers "
        "smaller than this (metres). Keeps real corners. 0 = exact outline")
    detail_tol: FloatProperty(name="Ignore Details <", default=0.15, min=0.0, max=1.0,
        unit='LENGTH', description="Boundary: ignore wall detail smaller than this "
        "(window reveals, tiny jogs). Bigger = simpler outline")
    bridge: FloatProperty(name="Bridge Gaps <", default=0.5, min=0.0, max=3.0,
        unit='LENGTH', description="Boundary: seal holes and gaps in the shell up "
        "to this size (non-manifold buildings, missing walls)")
    square: BoolProperty(name="Square Walls", default=True,
        description="Boundary: straighten walls to the building's main axis and "
        "make corners crisp (recommended). Off = follow the raw outline")
    ang_tol: FloatProperty(name="Square Angle", default=20.0, min=0.0, max=45.0,
        description="A wall within this many degrees of the main axis is "
        "straightened to it; further off, it keeps its real angle")
    allow45: BoolProperty(name="Allow 45°", default=False,
        description="Also snap walls to 45° diagonals (for buildings with "
        "diagonal wings)")
    uv_scale: FloatProperty(name="UV Scale", default=2.0, min=0.05, max=20.0,
        unit='LENGTH', description="Cube-UV texture size: a texture tiles every "
        "this many metres")
    mlo_name: StringProperty(name="MLO Name", default="",
        description="Interior name -> collection 'int_<name>', shell empty "
        "'<name>_shell'. Run this once, after rooms/doors are finished",
        update=_cb_mlo_name)
    timecycle_name: StringProperty(name="Timecycle", default="",
        description="RageKit room timecycle name (optional). Prefilled from "
        "MLO Name until you edit it yourself", update=_cb_timecycle_name)
    timecycle_auto: BoolProperty(default=True, options={'HIDDEN'},
        description="Internal: Timecycle is still following MLO Name")

    # ── Add Empties ─────────────────────────────────────────────────────
    empty_decals: BoolProperty(name="Decals", default=False)
    empty_details: BoolProperty(name="Details", default=False)
    empty_proxy: BoolProperty(name="Proxy", default=False)
    empty_visuals: BoolProperty(name="Visuals", default=False)
    empty_lights: BoolProperty(name="Lights", default=False)
    empty_custom_name: StringProperty(name="Custom Name",
        description="Name for a custom empty type", default="")
    empty_target_room: EnumProperty(name="Target Room",
        description="Room collection to add empties into", items=_gn_room_enum_items_with_all)
    custom_empties: CollectionProperty(type=GN_CustomEmptyItem)

    # ── Smart Rename ────────────────────────────────────────────────────
    sr_room: EnumProperty(name="Room",
        description="Room to use in the generated name", items=_gn_room_enum_items)
    sr_category: EnumProperty(name="Category", description="Object category",
        items=[('decals', "Decals", ""), ('details', "Details", ""),
              ('proxy', "Proxy", ""), ('visuals', "Visuals", ""),
              ('lights', "Lights", ""), ('custom', "Custom...", "")],
        default='decals')
    sr_category_custom: StringProperty(name="Custom Category",
        description="Type a custom category name", default="")
    sr_merge: BoolProperty(name="Merge after rename",
        description="Apply modifiers, apply scale, rename UV to 'UVMap 0', "
        "then join all selected objects", default=False)

    # ── Create Asset ────────────────────────────────────────────────────
    asset_type: EnumProperty(name="Asset Type",
        items=[('DOOR', "Door", "Armature root, no .bvh"),
              ('REGULAR', "Regular", "Empty root, with .bvh")],
        default='REGULAR')
    auto_collision: BoolProperty(name="Auto Collision",
        description="Also build a .poly_mesh collision copy with guessed "
        "collision materials per slot", default=True)

    # ── Build MLO popup: what to include ───────────────────────────────
    build_main: BoolProperty(name="Main", default=True,
        description="Shell empty + per-room shell meshes")
    build_room_colls: BoolProperty(name="Room Collections", default=True,
        description="r01, r02... collections with RageKit room defaults")
    build_prop_colls: BoolProperty(name="Prop Collections", default=True,
        description="Props_r0N sub-collection per room")
    build_asset_colls: BoolProperty(name="Asset Collections", default=True,
        description="Assets_r0N sub-collection per room")
    build_shell_collision: BoolProperty(name="Shell Collision", default=False,
        description="Auto-guessed collision materials (needs Sollumz) -- use "
        "Manual Setup afterward to review/change the per-material mapping")
    build_portals: BoolProperty(name="Auto Make Portals", default=False,
        description="One portal per opening, named by the rooms it borders")
    build_empties: BoolProperty(name="Add Empties", default=False,
        description="Add the ticked presets below (Manual Setup > Add Empties) "
        "to every room")

    floors: CollectionProperty(type=GN_FloorLevel)
    new_floor_z: FloatProperty(name="Z", default=0.0, unit='LENGTH',
        description="Base Z for the next floor added via 'Add Floor at Z'")
    floor_index: IntProperty(default=0, update=_cb_floor_select)
    active_floor: IntProperty(name="Draw on Floor", default=0, min=0,
        description="Which floor new rooms are drawn on")
    snap: FloatProperty(name="Grid Snap", default=0.10, min=0.0, max=1.0,
        unit='LENGTH', description="Round drawn room corners to this grid (0 = off)")
    rooms: CollectionProperty(type=GN_Room)
    room_index: IntProperty(default=0)
    uid_counter: IntProperty(default=1)
    openings: CollectionProperty(type=GN_Opening)
    opening_index: IntProperty(default=0, update=_cb_redraw)
    portal_index: IntProperty(default=0, update=_cb_portal_select)
    opening_filter: EnumProperty(name="Filter", default='ALL',
        items=[('ALL', "All", "Show all openings"),
               ('DOOR', "Doors", "Show only doors"),
               ('WINDOW', "Windows", "Show only windows")])
    reveal: BoolProperty(name="Reveal Jambs", default=True,
        description="Cap the opening sides so windows/doors have depth (the 20cm reveal)")
    door_presets: CollectionProperty(type=GN_DoorPreset)
    active_door_preset: IntProperty(default=0)
    window_presets: CollectionProperty(type=GN_WindowPreset)
    active_window_preset: IntProperty(default=0)
    add_threshold: BoolProperty(name="Door Threshold", default=False,
        description="Place a low floor strip across the bottom of each door opening",
        update=_cb_threshold)
    threshold_height: FloatProperty(name="Threshold Height", default=0.008,
        min=0.0, max=0.3, unit='LENGTH', update=_cb_threshold)
    threshold_depth: FloatProperty(name="Threshold Depth", default=0.05,
        min=0.005, max=0.5, unit='LENGTH',
        description="How far the threshold reaches into the room from the doorway",
        update=_cb_threshold)
    threshold_flip: BoolProperty(name="Flip Side", default=False,
        description="Reverse the direction the threshold extends / the normal", update=_cb_threshold)
    threshold_offset: FloatProperty(name="Offset", default=0.0, min=-0.5, max=0.5,
        unit='LENGTH', description="Slide the threshold (and its pivot) across the "
        "doorway. 0 = door line; negative = toward the next room", update=_cb_threshold)
    partition: FloatProperty(name="Partition Wall", default=0.10, min=0.0, max=1.0,
        unit='LENGTH', description="Gap left between two rooms when splitting "
        "(the interior partition wall thickness)")
    stair_step_height: FloatProperty(name="Step Height", default=0.18,
        min=0.02, max=0.5, unit='LENGTH',
        description="Target riser height per step (actual may come out lower "
        "to divide the run evenly). Also live-updates any selected, "
        "already-created stairs",
        update=_cb_stair_settings)
    stair_step_depth: FloatProperty(name="Step Depth", default=0.28,
        min=0.05, max=1.0, unit='LENGTH',
        description="Target tread depth per step (actual may come out lower "
        "to divide the run evenly). Also live-updates any selected, "
        "already-created stairs",
        update=_cb_stair_settings)


# ===========================================================================
# operators
# ===========================================================================
class GN_OT_set_exterior(Operator):
    bl_idname = "gn_int.set_exterior"
    bl_options = {'REGISTER', 'UNDO'}
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
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Add Floor from Selected Edge"
    bl_description = ("Add a floor level at the Z of the selected geometry "
                      "(select a facade edge in Edit Mode). Height is always "
                      "Room Height - use the floor list's Top field for a custom "
                      "height on a specific floor")

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
        base = min((ob.matrix_world @ v.co).z for v in sel)
        it = s.floors.add()
        it.z = round(base, 3)
        _sort_floors(s)
        self.report({'INFO'}, f"Added floor at z={base:.2f} (height = Room Height, "
                              f"{s.room_height:.2f} m)")
        return {'FINISHED'}




class GN_OT_add_floor_z(Operator):
    bl_idname = "gn_int.add_floor_z"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Add Floor at Z"
    bl_description = "Add a floor level at the typed Z height, no geometry needed"

    def execute(self, context):
        s = context.scene.gn_int
        it = s.floors.add()
        it.z = round(s.new_floor_z, 3)
        _sort_floors(s)
        self.report({'INFO'}, f"Added floor at z={s.new_floor_z:.2f}")
        return {'FINISHED'}


def _draw_slice_overlay(self, context):
    if not self._segs:
        return
    z = self._z
    pts = []
    for a, b in self._segs:
        pts.append((a[0], a[1], z))
        pts.append((b[0], b[1], z))
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'LINES', {"pos": pts})
    gpu.state.line_width_set(2.5)
    gpu.state.blend_set('ALPHA')
    shader.bind()
    shader.uniform_float("color", (0.25, 0.85, 1.0, 1.0))
    batch.draw(shader)
    gpu.state.line_width_set(1.0)


class GN_OT_pick_floor_z(Operator):
    bl_idname = "gn_int.pick_floor_z"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Slice for Floor Height"
    bl_description = ("Drag to slide a live cross-section up/down through the "
                      "exterior and see the real wall outline at each height, "
                      "then click to add a floor there. The exterior mesh is "
                      "never modified - only a throwaway copy is cut for preview")

    def invoke(self, context, event):
        s = context.scene.gn_int
        ex = s.exterior
        if not ex:
            self.report({'ERROR'}, "Set an exterior shell first")
            return {'CANCELLED'}
        self._src = _eval_bmesh(ex, context)
        if s.floors:
            top_floor = max(s.floors, key=lambda f: f.z)
            start = (top_floor.top if top_floor.top_is_custom else
                     top_floor.z + s.room_height)
        else:
            start = min((ex.matrix_world @ v.co).z for v in ex.data.vertices)
        self._z = start
        self._segs = _cut_segments(self._src, self._z)
        self._last_mouse_y = event.mouse_y
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_slice_overlay, (self, context), 'WINDOW', 'POST_VIEW')
        context.window_manager.modal_handler_add(self)
        self._update_header(context)
        return {'RUNNING_MODAL'}

    def _update_header(self, context):
        context.area.header_text_set(
            f"Slice for Floor Height:  z = {self._z:.3f} m   |   "
            "move mouse: slide  ·  Shift: fine  ·  wheel: nudge 2cm  ·  "
            "Click/Enter: confirm  ·  Esc: cancel")

    def _retag(self, context):
        self._segs = _cut_segments(self._src, self._z)
        if context.area:
            context.area.tag_redraw()
        self._update_header(context)

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            dy = event.mouse_y - self._last_mouse_y
            self._last_mouse_y = event.mouse_y
            sens = 0.001 if event.shift else 0.01
            self._z += dy * sens
            self._retag(context)
            return {'RUNNING_MODAL'}
        if event.type == 'WHEELUPMOUSE' and event.value == 'PRESS':
            self._z += 0.02
            self._retag(context)
            return {'RUNNING_MODAL'}
        if event.type == 'WHEELDOWNMOUSE' and event.value == 'PRESS':
            self._z -= 0.02
            self._retag(context)
            return {'RUNNING_MODAL'}
        if event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            return self._end(context, True)
        if event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
            return self._end(context, False)
        return {'RUNNING_MODAL'}

    def _end(self, context, confirmed):
        bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
        context.area.header_text_set(None)
        z = self._z
        self._src.free()
        if context.area:
            context.area.tag_redraw()
        if not confirmed:
            return {'CANCELLED'}
        s = context.scene.gn_int
        it = s.floors.add()
        it.z = round(z, 3)
        _sort_floors(s)
        self.report({'INFO'}, f"Added floor at z={z:.3f}")
        return {'FINISHED'}


class GN_OT_remove_floor(Operator):
    bl_idname = "gn_int.remove_floor"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Remove Floor"
    index: IntProperty()

    def execute(self, context):
        s = context.scene.gn_int
        if 0 <= self.index < len(s.floors):
            s.floors.remove(self.index)
        return {'FINISHED'}


def _sort_floors(s):
    data = sorted(((f.z, f.top, f.top_is_custom, f.bound_json, f.lock)
                   for f in s.floors), key=lambda t: t[0])
    s.floors.clear()
    for z, top, top_is_custom, bound_json, lock in data:
        it = s.floors.add()
        it.z = z
        it.top = top
        it.top_is_custom = top_is_custom
        it.bound_json = bound_json
        it.lock = lock


def _floor_tops(context):
    """Return list of (base_z, room_top_z, next_base_or_None) honoring height+gap.

    Per-floor top override wins; otherwise base + global Room Height.
    """
    s = context.scene.gn_int
    floors = sorted(((f.z, f.top, f.top_is_custom) for f in s.floors),
                    key=lambda t: t[0])
    out = []
    for i, (b, custom_top, is_custom) in enumerate(floors):
        nxt = floors[i + 1][0] if i + 1 < len(floors) else None
        top = custom_top if is_custom else b + s.room_height
        out.append((b, top, nxt))
    return out


class GN_OT_gen_boundaries(Operator):
    bl_idname = "gn_int.gen_boundaries"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Generate Boundaries"
    bl_description = ("Create the boundary outline for the SELECTED floor only "
                      "(exterior inset by the margin). Other floors are untouched")

    def execute(self, context):
        s = context.scene.gn_int
        ex = s.exterior
        if not ex:
            self.report({'ERROR'}, "Set an exterior shell")
            return {'CANCELLED'}
        if not (0 <= s.floor_index < len(s.floors)):
            self.report({'ERROR'}, "Select a floor in the list")
            return {'CANCELLED'}
        target = s.floors[s.floor_index]
        if target.lock and target.bound_json:
            self.report({'INFO'}, "Floor is locked - boundary kept as-is")
            return {'CANCELLED'}
        # find this floor's (base, top) among the z-sorted, gap-honoring list
        tops = _floor_tops(context)
        sorted_idx = sorted(range(len(s.floors)), key=lambda k: s.floors[k].z)
        pos = sorted_idx.index(s.floor_index)
        base, top, nxt = tops[pos]

        src = _eval_bmesh(ex, context)
        coll = _get_coll(BOUND_COLL)
        zc = base + s.sample_offset
        segs = _cut_segments(src, zc)
        poly_t = raster_footprint(
            segs, tol=s.detail_tol, cell=0.05, seed=None, bridge=s.bridge,
            square=s.square, ang_tol_deg=s.ang_tol, allow45=s.allow45)
        fell_back = False
        if not poly_t or len(poly_t) < 3:
            old = outer_footprint(src, zc)
            if old:
                poly_t = [(p.x, p.y) for p in simplify_loop(old, s.cleanup)]
                fell_back = True
        src.free()
        if not poly_t or len(poly_t) < 3:
            self.report({'WARNING'}, f"No outline found at z={zc:.2f}")
            return {'CANCELLED'}
        poly = [Vector(p) for p in poly_t]
        ip = _weld_loop(inset_loop(poly, s.wall_margin),
                        max(s.detail_tol * 0.4, 0.005))
        _clear_named(coll, f"GN_Bound_Floor{pos+1}")
        _make_loop_object(coll, f"GN_Bound_Floor{pos+1}", ip, base)
        target.bound_json = json.dumps(
            [[round(p.x, 4), round(p.y, 4)] for p in ip])
        msg = f"Generated boundary for Floor {s.floor_index+1}"
        if fell_back:
            msg += " (fell back to legacy method)"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class GN_OT_clear(Operator):
    bl_idname = "gn_int.clear"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Clear Generated"
    bl_description = "Delete generated boundaries and interior shells"

    def execute(self, context):
        _clear_coll(BOUND_COLL)
        _clear_coll(INT_COLL)
        return {'FINISHED'}


class GN_OT_clean_interior(Operator):
    bl_idname = "gn_int.clean_interior"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Clean Interior Tool (New Building)"
    bl_description = ("Reset for a different building: clears the exterior "
                      "reference, floors, rooms, openings, and generated "
                      "boundaries, and the MLO name fields. Does NOT touch any "
                      "MLO collections already built (int_<name> etc) -- use "
                      "Clean MLO for that first if you want those gone too")

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        s = context.scene.gn_int
        s.exterior = None
        s.floors.clear()
        s.rooms.clear()
        s.openings.clear()
        s.mlo_name = ""
        s.timecycle_name = ""
        s.timecycle_auto = True
        _clear_coll(BOUND_COLL)
        _clear_coll(ROOM_COLL)
        _clear_coll(DOORFRAME_COLL)
        _clear_coll(WINDOWFRAME_COLL)
        _clear_coll(THRESHOLD_COLL)
        _clear_coll(INT_COLL)
        self.report({'INFO'}, "Interior Tool reset - set a new exterior shell to start")
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


def _ordered_loop_xy(ob):
    """Walk ob's edge loop in order -> world-space XY polygon [(x,y),...]."""
    me = ob.data
    if len(me.vertices) < 3:
        return None
    adj = {}
    for e in me.edges:
        a, b = e.vertices[0], e.vertices[1]
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    if not adj:
        return None
    start = next(iter(adj))
    order = [start]
    prev = None
    cur = start
    while True:
        nxts = [v for v in adj.get(cur, []) if v != prev]
        if not nxts:
            break
        nxt = nxts[0]
        if nxt == start:
            break
        order.append(nxt)
        prev, cur = cur, nxt
        if len(order) > len(me.vertices):
            break
    mat = ob.matrix_world
    return [((mat @ me.vertices[i].co).x, (mat @ me.vertices[i].co).y) for i in order]


def _boundary_object_for_floor(floor, tol=0.05):
    """Find the GN_Bound_Floor* object for this floor, matched by Z (not name)
    so it's correct even if floors were added out of Z order."""
    coll = bpy.data.collections.get(BOUND_COLL)
    if not coll:
        return None
    best_ob = None
    best_dz = 1e9
    for ob in coll.objects:
        me = ob.data
        if not me or len(me.vertices) < 3:
            continue
        z0 = (ob.matrix_world @ me.vertices[0].co).z
        dz = abs(z0 - floor.z)
        if dz < best_dz:
            best_dz = dz
            best_ob = ob
    return best_ob if best_ob is not None and best_dz < tol else None


def _read_boundary_poly(floor):
    """Read the CURRENT boundary mesh for this floor (picks up hand edits made
    in Edit Mode, which never touch floor.bound_json). Falls back to the
    stored bound_json if no live boundary object is found."""
    best_ob = _boundary_object_for_floor(floor)
    if best_ob is not None:
        poly = _ordered_loop_xy(best_ob)
        if poly and len(poly) >= 3:
            return poly
    if floor.bound_json:
        try:
            return [tuple(p) for p in json.loads(floor.bound_json)]
        except Exception:
            pass
    return None


# surface material slots (index order used by _build_wall / _build_shell)
_SURF_MATS = (("GN_Wall", (0.62, 0.62, 0.62, 1.0)),      # 0
              ("GN_Floor", (0.55, 0.45, 0.35, 1.0)),     # 1
              ("GN_Ceiling", (0.85, 0.85, 0.85, 1.0)),   # 2
              ("GN_Reveal", (0.40, 0.40, 0.42, 1.0)))    # 3
MAT_WALL, MAT_FLOOR, MAT_CEIL, MAT_REVEAL = 0, 1, 2, 3


def _get_mat(name, color):
    m = bpy.data.materials.get(name)
    if m is None:
        m = bpy.data.materials.new(name)
        m.diffuse_color = color
        m.use_nodes = True
        try:
            m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = color
        except Exception:
            pass
    return m


def _assign_surface_materials(ob):
    ob.data.materials.clear()
    for name, color in _SURF_MATS:
        ob.data.materials.append(_get_mat(name, color))


def _box_uv(bm, scale, matrix=None):
    """World-scale cube/box projection: each face is projected onto the axis-plane
    it faces, so a `scale`-metre texture tiles every `scale` metres everywhere."""
    scale = max(scale, 1e-4)
    uvl = bm.loops.layers.uv.verify()
    for f in bm.faces:
        nrm = f.normal
        ax = max(range(3), key=lambda i: abs(nrm[i]))   # dominant axis
        for loop in f.loops:
            co = loop.vert.co if matrix is None else (matrix @ loop.vert.co)
            if ax == 2:
                u, v = co.x, co.y
            elif ax == 0:
                u, v = co.y, co.z
            else:
                u, v = co.x, co.z
            loop[uvl].uv = (u / scale, v / scale)


def _wall_spans(A, B, base_z, ceil_z, openings, win_reveal=0.0, door_reveal=0.0):
    """Return (d, L, wn, spans) for wall A->B; spans = (x0,x1,z0,z1,reveal)."""
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
            # a door sits between two rooms (~partition gap apart) -> each side
            # reveals only to the gap midpoint so the two jambs meet, no overlap
            rd = door_reveal if op.is_door else win_reveal
            spans.append((x0, x1, z0, z1, rd))
    return d, L, wn, spans


def _build_wall(bm, A, B, base_z, ceil_z, openings, win_reveal, door_reveal):
    """Build wall A->B (floor..ceil) with rectangular holes + per-opening reveal jambs."""
    r = _wall_spans(A, B, base_z, ceil_z, openings, win_reveal, door_reveal)
    if not r:
        return
    d, L, wn, spans = r
    xs = sorted(set([0.0, L] + [v for sp in spans for v in (sp[0], sp[1])]))
    zs = sorted(set([base_z, ceil_z] + [v for sp in spans for v in (sp[2], sp[3])]))

    def in_hole(cx, cz):
        for (x0, x1, z0, z1, _rd) in spans:
            if x0 - 1e-6 < cx < x1 + 1e-6 and z0 - 1e-6 < cz < z1 + 1e-6:
                return True
        return False

    def W(x, z):
        return bm.verts.new((A.x + d.x * x, A.y + d.y * x, z))
    for i in range(len(xs) - 1):
        for j in range(len(zs) - 1):
            if in_hole((xs[i] + xs[i + 1]) * 0.5, (zs[j] + zs[j + 1]) * 0.5):
                continue
            f = bm.faces.new((W(xs[i], zs[j]), W(xs[i + 1], zs[j]),
                              W(xs[i + 1], zs[j + 1]), W(xs[i], zs[j + 1])))
            f.material_index = MAT_WALL
    # reveal jambs: extrude each opening's rim outward by its own reveal depth
    for (x0, x1, z0, z1, rd) in spans:
        if rd <= 1e-4:
            continue
        o = -wn * rd

        def P(x, z, out):
            bx, by = A.x + d.x * x, A.y + d.y * x
            if out:
                bx += o.x
                by += o.y
            return bm.verts.new((bx, by, z))
        for quad in ((P(x0, z0, 0), P(x1, z0, 0), P(x1, z0, 1), P(x0, z0, 1)),   # bottom
                     (P(x0, z1, 0), P(x1, z1, 0), P(x1, z1, 1), P(x0, z1, 1)),   # top
                     (P(x0, z0, 0), P(x0, z1, 0), P(x0, z1, 1), P(x0, z0, 1)),   # left
                     (P(x1, z0, 0), P(x1, z1, 0), P(x1, z1, 1), P(x1, z0, 1))):  # right
            bm.faces.new(quad).material_index = MAT_REVEAL


def _build_shell(coll, name, poly_xy, base_z, ceil_z, openings=None,
                 win_reveal=0.0, door_reveal=0.0, uv_scale=2.0):
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
        bm.faces.new(fv).material_index = MAT_FLOOR
    except ValueError:
        pass
    try:
        bm.faces.new(cv[::-1]).material_index = MAT_CEIL
    except ValueError:
        pass
    for k in range(n):
        _build_wall(bm, poly_xy[k], poly_xy[(k + 1) % n], base_z, ceil_z,
                    openings, win_reveal, door_reveal)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-4)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    for f in bm.faces:                 # face inward
        f.normal_flip()
    _box_uv(bm, uv_scale)              # world-scale cube UVs (verts still in world coords)
    # put the object ORIGIN at the room's centre (of its bounds) instead of world 0
    co = [v.co for v in bm.verts]
    if co:
        cx = (min(v.x for v in co) + max(v.x for v in co)) * 0.5
        cy = (min(v.y for v in co) + max(v.y for v in co)) * 0.5
        cz = (min(v.z for v in co) + max(v.z for v in co)) * 0.5
        bmesh.ops.translate(bm, verts=bm.verts, vec=(-cx, -cy, -cz))
    else:
        cx = cy = cz = 0.0
    me = bpy.data.meshes.new(name)
    for name_col in _SURF_MATS:            # slots present BEFORE to_mesh so indices stay valid
        me.materials.append(_get_mat(name_col[0], name_col[1]))
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    ob.location = (cx, cy, cz)
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
    ob = _build_shell(coll, f"r{len(s.rooms):02d}", poly_xy, base, top,
                      s.openings,
                      (s.wall_margin if s.reveal else 0.0),
                      (s.partition * 0.5 if s.reveal else 0.0), s.uv_scale)
    ob["gn_room_uid"] = rec.uid
    return rec


def rebuild_rooms(context):
    """Rebuild all room shells, PRESERVING each room's visibility (keyed by uid)
    and Z offset (a manual grab-and-move in Z, e.g. for a half-floor/mezzanine
    room -- detected from how far the object's centre currently sits from its
    floor's own base/top, then re-applied so openings still cut at the right
    height on the next rebuild)."""
    s = context.scene.gn_int
    # snapshot hide state AND detected z_offset by uid before clearing
    prev = {}
    coll = bpy.data.collections.get(ROOM_COLL)
    if coll:
        for ob in coll.objects:
            u = ob.get("gn_room_uid")
            if u is not None:
                prev[u] = (ob.hide_get(), ob.hide_render, ob.hide_select, ob.location.z)
    for r in s.rooms:
        if r.uid in prev:
            fl = _floor_by_index(context, r.floor_index)
            if fl:
                base, top, _ = fl
                r.z_offset = prev[r.uid][3] - (base + top) * 0.5
    _clear_coll(ROOM_COLL)
    coll = _get_coll(ROOM_COLL)
    for i, r in enumerate(s.rooms):
        if not r.uid:
            r.uid = _new_uid(s)
        fl = _floor_by_index(context, r.floor_index)
        if not fl:
            continue
        base, top, _ = fl
        base += r.z_offset; top += r.z_offset
        try:
            poly = [Vector(pt) for pt in json.loads(r.poly_json)]
        except Exception:
            continue
        if len(poly) >= 3:
            ob = _build_shell(coll, f"r{i+1:02d}", poly,
                              base, top, s.openings,
                              (s.wall_margin if s.reveal else 0.0),
                              (s.partition * 0.5 if s.reveal else 0.0), s.uv_scale)
            ob["gn_room_uid"] = r.uid
            if r.uid in prev:                 # restore visibility
                hv, hr, hs, _z = prev[r.uid]
                ob.hide_set(hv)
                ob.hide_render = hr
                ob.hide_select = hs
    _refresh_thresholds(context)              # door threshold strips
    try:
        context.view_layer.update()           # sync matrix_world for centred origins
    except Exception:
        pass
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


def _nearest_edge_param(poly, p):
    """Closest point on poly's boundary to p -> (edge_index, point_on_edge)."""
    n = len(poly)
    best = None
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        L2 = e.length_squared
        t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, (p - a).dot(e) / L2))
        proj = a + e * t
        d = (p - proj).length
        if best is None or d < best[0]:
            best = (d, i, proj)
    return best[1], best[2]


def _walk_arc(poly, i0, p0, i1, p1):
    """CCW walk along poly's boundary from p0 (on edge i0) forward to p1 (on
    edge i1). Returns points including p0 and p1, not closing the loop."""
    n = len(poly)
    out = [p0]
    i = i0
    while i != i1:
        i = (i + 1) % n
        out.append(poly[i])
    out.append(p1)
    return out


def _line_x_clamped(a, b, c, d, ref, max_dist):
    """Like _line_x, but returns None (instead of a wild point) if the
    intersection lands farther than max_dist from `ref`. Two lines meeting
    at a shallow angle can intersect arbitrarily far away -- this guards
    against that runaway case (a degenerate multi-hundred-metre sliver)."""
    p = _line_x(a, b, c, d)
    if p is None or (p - ref).length > max_dist:
        return None
    return p


def _offset_line_to_edge(poly, edge_i, seg_a, seg_b, d):
    """Offset the line through seg_a->seg_b sideways by d, then intersect it
    with poly's edge `edge_i` -- so the cut's wall-entry point slides along
    that wall instead of floating off it. Falls back to the nearest point on
    the edge SEGMENT (not its infinite line) if the two lines are parallel or
    meet at a shallow angle that would send the intersection far away."""
    e = seg_b - seg_a
    L = max(e.length, 1e-6)
    nrm = Vector((-e.y / L, e.x / L))
    a2, b2 = seg_a + nrm * d, seg_b + nrm * d
    ea, eb = poly[edge_i], poly[(edge_i + 1) % len(poly)]
    ref = seg_a + nrm * d
    max_dist = max(abs(d) * 8, (eb - ea).length * 2)
    p = _line_x_clamped(a2, b2, ea, eb, ref, max_dist)
    if p is not None:
        return p
    edge_dir = eb - ea
    L2 = edge_dir.length_squared
    t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, (ref - ea).dot(edge_dir) / L2))
    return ea + edge_dir * t


def _offset_polyline_mitered(pts, d):
    """Offset an OPEN polyline sideways by d; interior vertices mitered via
    line-intersection (same technique as inset_loop), guarded against a
    shallow-angle bend sending the miter point far away (falls back to a
    plain unmitered offset point in that case). Endpoints are handled
    separately by the caller via _offset_line_to_edge."""
    segs = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        e = b - a
        L = max(e.length, 1e-6)
        nrm = Vector((-e.y / L, e.x / L))
        segs.append((a + nrm * d, b + nrm * d))
    out = [segs[0][0]]
    for i in range(len(segs) - 1):
        ref = pts[i + 1]
        max_dist = max(abs(d) * 8, (pts[i + 1] - pts[i]).length,
                       (pts[i + 2] - pts[i + 1]).length)
        p = _line_x_clamped(segs[i][0], segs[i][1], segs[i + 1][0], segs[i + 1][1],
                            ref, max_dist)
        out.append(p if p is not None else segs[i][1])
    out.append(segs[-1][1])
    return out


def split_polygon_path(poly, path, gap):
    """Split poly (CCW) by an open polyline `path` (>=2 pts, e.g. an L-shaped
    cut) whose ends are snapped onto poly's boundary on two DIFFERENT edges.
    Returns the two pieces pushed apart by `gap`, mitered at interior bends.
    Either may be [] if the cut is degenerate."""
    if len(path) < 2:
        return [], []
    i0, _ = _nearest_edge_param(poly, path[0])
    i1, _ = _nearest_edge_param(poly, path[-1])
    if i0 == i1:
        return [], []   # both ends on the same wall -- not supported yet
    h = gap * 0.5

    def offset_chord(d):
        start = _offset_line_to_edge(poly, i0, path[0], path[1], d)
        end = _offset_line_to_edge(poly, i1, path[-2], path[-1], d)
        if len(path) > 2:
            mid = _offset_polyline_mitered(path, d)
            return [start] + mid[1:-1] + [end]
        return [start, end]

    chord_pos = offset_chord(h)
    chord_neg = offset_chord(-h)

    piece_a = (_walk_arc(poly, i0, chord_neg[0], i1, chord_neg[-1])
              + list(reversed(chord_neg[1:-1])))
    piece_b = (_walk_arc(poly, i1, chord_pos[-1], i0, chord_pos[0])
              + chord_pos[1:-1])

    piece_a = piece_a if len(piece_a) >= 3 and abs(_poly_area(piece_a)) > 0.05 else []
    piece_b = piece_b if len(piece_b) >= 3 and abs(_poly_area(piece_b)) > 0.05 else []
    return piece_a, piece_b


def split_room_record_path(context, room_idx, path):
    """Split room[room_idx] by an open polyline (bend cut) with the
    partition gap. Returns count made (0 or 2)."""
    s = context.scene.gn_int
    if not (0 <= room_idx < len(s.rooms)):
        return 0
    rec = s.rooms[room_idx]
    fidx = rec.floor_index
    try:
        poly = [Vector(p) for p in json.loads(rec.poly_json)]
    except Exception:
        return 0
    a, b = split_polygon_path(poly, path, s.partition)
    if not a or not b:
        return 0
    s.rooms.remove(room_idx)
    for part in (a, b):
        r = s.rooms.add()
        r.floor_index = fidx
        r.uid = _new_uid(s)
        r.poly_json = json.dumps([[round(p.x, 4), round(p.y, 4)] for p in part])
    rebuild_rooms(context)
    return 2


def seed_rooms_from_boundaries(context, floor_idx=None):
    """Make one room = the SELECTED floor's whole boundary polygon (all floors
    if floor_idx is None). Rooms on OTHER floors -- including any splits
    already made there -- are left completely untouched.

    Reads the LIVE boundary mesh (picks up hand edits) rather than the
    bound_json snapshot from when Generate Boundaries last ran.
    Returns the number of floors seeded.
    """
    s = context.scene.gn_int
    targets = {floor_idx} if floor_idx is not None else set(range(len(s.floors)))
    keep = [(r.floor_index, r.poly_json, r.uid) for r in s.rooms
            if r.floor_index not in targets]
    s.rooms.clear()
    for fi, poly_json, uid in keep:
        rec = s.rooms.add()
        rec.floor_index = fi
        rec.poly_json = poly_json
        rec.uid = uid
    made = 0
    for i in sorted(targets):
        if not (0 <= i < len(s.floors)):
            continue
        f = s.floors[i]
        poly = _read_boundary_poly(f)
        if not poly:
            continue
        rec = s.rooms.add()
        rec.floor_index = i
        rec.uid = _new_uid(s)
        rec.poly_json = json.dumps([[round(p[0], 4), round(p[1], 4)] for p in poly])
        f.bound_json = rec.poly_json   # keep the snapshot in sync with the edit
        made += 1
    rebuild_rooms(context)
    return made


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
    bl_options = {'REGISTER', 'UNDO'}
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
    bl_options = {'REGISTER', 'UNDO'}
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
    bl_options = {'REGISTER', 'UNDO'}
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
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Rebuild Rooms"
    bl_description = "Rebuild all room shells from stored footprints"

    def execute(self, context):
        rebuild_rooms(context)
        return {'FINISHED'}


class GN_OT_reunwrap(Operator):
    bl_idname = "gn_int.reunwrap"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Re-Cube-Unwrap"
    bl_description = ("Box-project world-scale UVs onto the selected room meshes "
                      "(or all rooms) — run after resizing to fix UVs")

    def execute(self, context):
        s = context.scene.gn_int
        rooms = bpy.data.collections.get(ROOM_COLL)
        targets = [o for o in context.selected_objects if o.type == 'MESH']
        if not targets and rooms:
            targets = list(rooms.objects)
        n = 0
        for ob in targets:
            bm = bmesh.new()
            bm.from_mesh(ob.data)
            _box_uv(bm, s.uv_scale, ob.matrix_world)   # world coords -> tiling survives scaling
            bm.to_mesh(ob.data)
            bm.free()
            ob.data.update()
            n += 1
        self.report({'INFO'}, f"Re-unwrapped {n} object(s)")
        return {'FINISHED'}


# ===========================================================================
# MLO setup (RageKit ytyp/room collections) -- run once, after rooms/doors
# are finished. Mirrors the conventions in the user's scene_organizer.py.
# ===========================================================================
def _mlo_free_collection_name(name):
    existing = bpy.data.collections.get(name)
    if existing is None:
        return
    for scene in bpy.data.scenes:
        if existing in list(scene.collection.children):
            scene.collection.children.unlink(existing)
    for col in list(bpy.data.collections):
        if existing.name in [c.name for c in col.children]:
            try:
                col.children.unlink(existing)
            except Exception:
                pass
    bpy.data.collections.remove(existing)


def _mlo_ensure_scene_collection(name, scene):
    for child in scene.collection.children:
        if child.name == name:
            return child
    _mlo_free_collection_name(name)
    col = bpy.data.collections.new(name)
    scene.collection.children.link(col)
    return col


def _mlo_make_collection(name, parent):
    for child in parent.children:
        if child.name == name:
            return child
    _mlo_free_collection_name(name)
    col = bpy.data.collections.new(name)
    parent.children.link(col)
    return col


def _mlo_apply_ytyp(col):
    try:
        col.ragequit_type = 'ytyp'
        col.ragequit_ytyp.lodDist = 200
        col.ragequit_ytyp.hdTextureDist = 100
    except Exception:
        pass


def _mlo_apply_room_defaults(col, timecycle_name):
    try:
        col.ragequit_type = 'room'
        room = col.ragequit_room
        room.timecycleName = timecycle_name
        room.floorId = 0
        room.bounds_mode = 'AUTO'
    except Exception:
        pass


def _mlo_apply_collection_types(main_col):
    special = {'Collisions': 'collision', 'Portals': 'portals', 'Assets': 'assets'}
    for child in main_col.children:
        try:
            child.ragequit_type = special.get(child.name, 'room')
            if child.ragequit_type == 'room':
                for sub in child.children:
                    if sub.name.startswith("Props_"):
                        sub.ragequit_type = 'room_props'
        except Exception:
            pass


def _mlo_room_token_from_shell_name(shell_name, mlo_name):
    """'<mlo_name>_<room_token>_shell.model' or '..._shell_NN.model' -> room_token.
    None if shell_name doesn't match that pattern (leave unrecognised objects alone)."""
    prefix = f"{mlo_name}_"
    if not shell_name.startswith(prefix):
        return None
    rest = shell_name[len(prefix):]
    if rest.endswith("_shell.model"):
        return rest[:-len("_shell.model")]
    m = re.match(r'^(.*)_shell_\d+\.model$', rest)
    return m.group(1) if m else None


def _mlo_delete_empty_tree(col):
    """Delete col and its sub-collections, but ONLY where genuinely empty (no
    objects, and every child was itself removable). A collection still holding
    objects -- props the user placed, say -- is left in place. Returns
    (kept_names, collections_removed)."""
    kept = []
    col_count = 0
    for child in list(col.children):
        c_kept, cc = _mlo_delete_empty_tree(child)
        kept.extend(c_kept)
        col_count += cc
    if col.objects or col.children:
        kept.append(col.name)
        return kept, col_count
    try:
        bpy.data.collections.remove(col)
        col_count += 1
    except Exception:
        kept.append(col.name)
    return kept, col_count


class GN_OT_clean_mlo(Operator):
    bl_idname = "gn_int.clean_mlo"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Clean MLO"
    bl_description = ("Undo the MLO build: move each room shell mesh in Main "
                      "back into GN_Rooms under its original name, delete the "
                      "shell empty, then remove the int_<name> collection "
                      "scaffolding. Any collection still holding real content "
                      "(props/assets you placed) is left in place, not deleted")

    def execute(self, context):
        s = context.scene.gn_int
        name = s.mlo_name.strip()
        if not name:
            self.report({'ERROR'}, "Set an MLO Name first")
            return {'CANCELLED'}
        col_name = f"int_{name}"
        main_col = bpy.data.collections.get(col_name)
        if main_col is None:
            self.report({'INFO'}, f"'{col_name}' doesn't exist - nothing to clean")
            return {'CANCELLED'}

        main_room = None
        for child in main_col.children:
            if child.name == "Main":
                main_room = child
                break

        restored = 0
        if main_room:
            room_coll = _get_coll(ROOM_COLL)
            for ob in list(main_room.objects):
                if ob.type == 'EMPTY' and ob.name == f"{name}_shell":
                    bpy.data.objects.remove(ob, do_unlink=True)
                    continue
                if ob.type != 'MESH':
                    continue
                token = _mlo_room_token_from_shell_name(ob.name, name)
                if token is None:
                    continue   # not a recognised room shell -- leave it alone
                world_mat = ob.matrix_world.copy()
                ob.parent = None
                ob.matrix_world = world_mat
                if bpy.data.objects.get(token) in (None, ob):
                    ob.name = token
                    if ob.data:
                        ob.data.name = token
                for col in list(ob.users_collection):
                    col.objects.unlink(ob)
                room_coll.objects.link(ob)
                restored += 1

        kept, col_count = _mlo_delete_empty_tree(main_col)
        msg = f"Restored {restored} room(s) to GN_Rooms, removed {col_count} empty collection(s)"
        if kept:
            msg += f" - kept (not empty): {', '.join(kept)}"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class GN_OT_build_mlo(Operator):
    bl_idname = "gn_int.build_mlo"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Build MLO"
    bl_description = ("Create (or update) the int_<name> collection structure. "
                      "Pick what to include -- anything left unticked can be "
                      "added later from Manual Setup below")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=280)

    def draw(self, context):
        s = context.scene.gn_int
        col = self.layout.column(align=True)
        col.prop(s, "build_main")
        col.prop(s, "build_room_colls")
        sub = col.column(align=True)
        sub.enabled = s.build_room_colls
        sub.prop(s, "build_prop_colls")
        sub.prop(s, "build_asset_colls")
        self.layout.separator()
        col2 = self.layout.column(align=True)
        col2.prop(s, "build_shell_collision")
        col2.prop(s, "build_portals")
        col2.prop(s, "build_empties")

    def execute(self, context):
        s = context.scene.gn_int
        name = s.mlo_name.strip()
        if not name:
            self.report({'ERROR'}, "Set an MLO Name first")
            return {'CANCELLED'}
        needs_rooms = (s.build_main or s.build_room_colls
                      or s.build_prop_colls or s.build_asset_colls)
        room_coll = bpy.data.collections.get(ROOM_COLL)
        if needs_rooms and (not room_coll or not room_coll.objects):
            self.report({'ERROR'}, "No built rooms - run Make Floor Walls first")
            return {'CANCELLED'}

        main_col = _mlo_ensure_scene_collection(f"int_{name}", context.scene)
        _mlo_apply_ytyp(main_col)
        try:
            _mlo_make_collection("Collisions", main_col).ragequit_type = 'collision'
        except Exception:
            pass
        try:
            _mlo_make_collection("Portals", main_col).ragequit_type = 'portals'
        except Exception:
            pass
        assets = _mlo_make_collection("Assets", main_col)
        try:
            assets.ragequit_type = 'assets'
        except Exception:
            pass
        doors = _mlo_make_collection("Doors", assets)   # nested INSIDE Assets
        try:
            doors.ragequit_type = 'assets'
        except Exception:
            pass

        empty = None
        main_room = None
        if s.build_main:
            main_room = _mlo_make_collection("Main", main_col)
            _mlo_apply_room_defaults(main_room, s.timecycle_name)
            empty_name = f"{name}_shell"
            empty = bpy.data.objects.get(empty_name)
            if empty is None:
                empty = bpy.data.objects.new(empty_name, None)
                empty.empty_display_type = 'PLAIN_AXES'
                empty.location = (0.0, 0.0, 0.0)
            if empty.name not in main_room.objects:
                main_room.objects.link(empty)
            for col in list(empty.users_collection):
                if col is not main_room:
                    col.objects.unlink(empty)

        # each built room: optionally get an r0N collection (+ Props_/Assets_
        # subfolders), and optionally have its shell mesh renamed and moved
        # into Main (matching int_gn_hq_triad / int_gn_legion_bank)
        moved = 0
        replaced = 0
        built_room_tokens = []
        for ob in list(room_coll.objects) if room_coll else []:
            room_token = ob.name
            built_room_tokens.append(room_token)

            if s.build_room_colls:
                rcol = _mlo_make_collection(room_token, main_col)
                _mlo_apply_room_defaults(rcol, s.timecycle_name)
                if s.build_prop_colls:
                    _mlo_make_collection(f"Props_{room_token}", rcol)
                if s.build_asset_colls:
                    try:
                        _mlo_make_collection(f"Assets_{room_token}", rcol).ragequit_type = 'none'
                    except Exception:
                        pass

            if not s.build_main:
                continue

            new_name = f"{name}_{room_token}_shell.model"
            existing = bpy.data.objects.get(new_name)
            if existing is not None and existing is not ob and existing.name in main_room.objects:
                # a room rebuilt after an earlier MLO build (same room_token,
                # new mesh) -- the old shell is stale, replace it cleanly
                bpy.data.objects.remove(existing, do_unlink=True)
                replaced += 1
                existing = None
            if existing is not None and existing is not ob:
                counter = 1
                while bpy.data.objects.get(
                        f"{name}_{room_token}_shell_{counter:02d}.model") not in (None, ob):
                    counter += 1
                new_name = f"{name}_{room_token}_shell_{counter:02d}.model"
            ob.name = new_name
            if ob.data:
                ob.data.name = new_name

            world_mat = ob.matrix_world.copy()
            for col in list(ob.users_collection):
                col.objects.unlink(ob)
            main_room.objects.link(ob)
            ob.parent = empty
            ob.matrix_parent_inverse = Matrix.Identity(4)
            ob.matrix_world = world_mat
            moved += 1

        _mlo_apply_collection_types(main_col)
        msg = f"Built int_{name}"
        if s.build_main:
            msg += f", {moved} room shell(s)"
            if replaced:
                msg += f" ({replaced} replaced)"

        if s.build_shell_collision:
            try:
                from Sollumz.ybn.collision_materials import collisionmats as coll_mats
            except ImportError:
                coll_mats = None
            shell_empty, room_meshes = _find_mlo_shell_data(name)
            mat_mapping = {}
            if shell_empty is not None:
                seen = set()
                for meshes in room_meshes.values():
                    for o in meshes:
                        for mat in o.data.materials:
                            if mat and mat.name not in seen:
                                seen.add(mat.name)
                                idx = _guess_collision_material_index(mat.name)
                                if idx > 0:
                                    mat_mapping[mat.name] = idx
            created, err = _do_build_shell_collision(context, name, mat_mapping)
            if err:
                msg += f", collision SKIPPED ({err})"
            else:
                msg += f", collision for {created} room(s)"

        if s.build_portals:
            try:
                bpy.ops.gn_int.create_portals()
                msg += ", portals"
            except Exception as e:
                msg += f", portals SKIPPED ({e})"

        if s.build_empties:
            to_create = [p for p in PRESET_EMPTIES if getattr(s, _PRESET_ATTR[p], False)]
            for item in s.custom_empties:
                if item.enabled and item.name.strip():
                    to_create.append(item.name.strip())
            n_empties = 0
            target_tokens = built_room_tokens or [c.name for c in _gn_iter_room_collections(context)]
            if to_create:
                for room_token in target_tokens:
                    rcol = bpy.data.collections.get(room_token)
                    if rcol is None:
                        continue
                    for t in to_create:
                        _gn_create_empty_in_room(rcol, name, room_token, t)
                        n_empties += 1
            msg += f", {n_empties} empt(y/ies)"

        self.report({'INFO'}, msg)
        return {'FINISHED'}


def _mlo_room_tokens_all(context):
    """Every room we know about: rooms still in GN_Rooms (not yet moved into
    Main) plus any r0N collection that already exists under the MLO -- so
    these standalone buttons work both for brand-new rooms and for
    refreshing/extending ones a Build MLO pass already processed."""
    tokens = set()
    room_coll = bpy.data.collections.get(ROOM_COLL)
    if room_coll:
        tokens.update(ob.name for ob in room_coll.objects)
    tokens.update(c.name for c in _gn_iter_room_collections(context))
    return sorted(tokens)


class GN_OT_add_room_collections(Operator):
    bl_idname = "gn_int.add_room_collections"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Add Room Collections"
    bl_description = ("Create (or refresh) an r0N collection with RageKit "
                      "room defaults for every known room")

    def execute(self, context):
        s = context.scene.gn_int
        name = s.mlo_name.strip()
        if not name:
            self.report({'ERROR'}, "Set an MLO Name first")
            return {'CANCELLED'}
        tokens = _mlo_room_tokens_all(context)
        if not tokens:
            self.report({'WARNING'}, "No rooms found - build rooms or run Build MLO first")
            return {'CANCELLED'}
        main_col = _mlo_ensure_scene_collection(f"int_{name}", context.scene)
        for token in tokens:
            rcol = _mlo_make_collection(token, main_col)
            _mlo_apply_room_defaults(rcol, s.timecycle_name)
        self.report({'INFO'}, f"Room collection(s) ready for {len(tokens)} room(s)")
        return {'FINISHED'}


class GN_OT_add_prop_collections(Operator):
    bl_idname = "gn_int.add_prop_collections"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Add Prop Collections"
    bl_description = "Add a Props_r0N sub-collection to every existing room collection"

    def execute(self, context):
        rooms = _gn_iter_room_collections(context)
        if not rooms:
            self.report({'ERROR'}, "No room collections found - run Add Room Collections first")
            return {'CANCELLED'}
        for rcol in rooms:
            _mlo_make_collection(f"Props_{rcol.name}", rcol)
        self.report({'INFO'}, f"Props_ sub-collection ready for {len(rooms)} room(s)")
        return {'FINISHED'}


class GN_OT_add_asset_collections(Operator):
    bl_idname = "gn_int.add_asset_collections"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Add Asset Collections"
    bl_description = "Add an Assets_r0N sub-collection to every existing room collection"

    def execute(self, context):
        rooms = _gn_iter_room_collections(context)
        if not rooms:
            self.report({'ERROR'}, "No room collections found - run Add Room Collections first")
            return {'CANCELLED'}
        for rcol in rooms:
            try:
                _mlo_make_collection(f"Assets_{rcol.name}", rcol).ragequit_type = 'none'
            except Exception:
                pass
        self.report({'INFO'}, f"Assets_ sub-collection ready for {len(rooms)} room(s)")
        return {'FINISHED'}


# ===========================================================================
# Shell collision (Sollumz BOUND_COMPOSITE -> Shell.BVH -> R##_Shell.poly_mesh)
# -- same hierarchy/behaviour as the user's scene_organizer.py
# SO_OT_CreateShellCollision, adapted to read this tool's own Main-collection
# shell naming (<name>_r0N_shell.model under <name>_shell). Run after
# Build MLO Collections, since it reads the shells that step produces.
# ===========================================================================
def _guess_collision_material_index(hint):
    """Keyword-match a material/room name to a Sollumz collision material
    index. Falls back to 0 (DEFAULT) if Sollumz is unavailable or nothing
    matches. Identical keyword list to scene_organizer.py for consistency."""
    try:
        from Sollumz.ybn.collision_materials import collisionmats
    except ImportError:
        return 0
    name_to_idx = {m.name.upper(): i for i, m in enumerate(collisionmats)}
    keyword_map = [
        (["garage"], "METAL_GARAGE_DOOR"),
        (["glass"], "GLASS_OPAQUE"),
        (["wood", "timber", "plank", "oak", "pine",
          "maple", "cedar", "mahogany", "walnut"], "WOOD_SOLID_LARGE"),
        (["metal", "steel", "iron", "alum",
          "copper", "brass", "zinc"], "METAL_SOLID_LARGE"),
        (["rubber", "tyre", "tire"], "RUBBER"),
        (["plastic", "pvc", "resin", "fibreglass", "fiberglass"], "PLASTIC"),
        (["brick", "tile", "terrac"], "BRICK"),
        (["stone", "marble", "granite", "rock", "slate"], "STONE"),
        (["concrete", "cement"], "CONCRETE"),
    ]
    h = hint.lower()
    for keywords, mat_name in keyword_map:
        for kw in keywords:
            if kw in h:
                idx = name_to_idx.get(mat_name)
                if idx is not None:
                    return idx
    return 0


class GN_CollMatSearchItem(PropertyGroup):
    """One entry in the searchable collision-material list (name only)."""
    pass   # 'name' is the built-in PropertyGroup attribute used by prop_search


class GN_ShellCollMappingItem(PropertyGroup):
    orig_mat_name: StringProperty(name="Original Material", default="")
    coll_mat_name: StringProperty(name="Collision Material", default="Keep Original")


class GN_UL_shell_coll_mappings(bpy.types.UIList):
    def draw_item(self, ctx, layout, data, item, icon, adata, aprop, index=0, flt=0):
        row = layout.row(align=True)
        row.label(text=item.orig_mat_name)
        row.prop_search(item, "coll_mat_name", ctx.scene, "gn_coll_mat_search_items", text="")


def _find_mlo_shell_data(mlo_name):
    """Return (shell_empty, {room_token: [mesh_objs]}) matching
    GN_OT_build_mlo's own <name>_r0N_shell(.model|_NN.model) naming, or
    (None, {}) if the shell empty isn't found."""
    shell_empty = bpy.data.objects.get(f"{mlo_name}_shell")
    if shell_empty is None:
        return None, {}
    pat = re.compile(rf'^{re.escape(mlo_name)}_(r\d+)_shell(?:_\d+)?\.model$', re.IGNORECASE)
    room_meshes = {}
    for obj in shell_empty.children_recursive:
        if obj.type != 'MESH':
            continue
        m = pat.match(obj.name)
        if m:
            room_meshes.setdefault(m.group(1).lower(), []).append(obj)
    return shell_empty, room_meshes


class GN_OT_create_shell_collision(Operator):
    bl_idname = "gn_int.create_shell_collision"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Create Shell Collision"
    bl_description = ("Build a Sollumz collision hierarchy (BOUND_COMPOSITE -> "
                      "Shell.BVH -> R0N_Shell.poly_mesh) from the room shells "
                      "in Main. Run after Build MLO Collections")

    @classmethod
    def poll(cls, context):
        try:
            from Sollumz.sollumz_properties import SollumType   # noqa: F401
            return True
        except ImportError:
            return False

    def invoke(self, context, event):
        s = context.scene.gn_int
        name = s.mlo_name.strip()
        if not name:
            self.report({'ERROR'}, "Set an MLO Name first")
            return {'CANCELLED'}
        shell_empty, room_meshes = _find_mlo_shell_data(name)
        if shell_empty is None:
            self.report({'ERROR'}, f"Shell empty '{name}_shell' not found - "
                                    "run Build MLO Collections first")
            return {'CANCELLED'}
        if not room_meshes:
            self.report({'ERROR'}, "No room shell meshes found under the shell empty")
            return {'CANCELLED'}

        unique_mats = []
        for meshes in room_meshes.values():
            for obj in meshes:
                for mat in obj.data.materials:
                    if mat and mat not in unique_mats:
                        unique_mats.append(mat)

        try:
            from Sollumz.ybn.collision_materials import collisionmats as coll_mats
        except ImportError:
            coll_mats = None

        search_items = context.scene.gn_coll_mat_search_items
        search_items.clear()
        keep = search_items.add()
        keep.name = "Keep Original"
        if coll_mats:
            for m in coll_mats:
                si = search_items.add()
                si.name = m.name
        else:
            si = search_items.add()
            si.name = "DEFAULT"

        mappings = context.scene.gn_shell_coll_mappings
        mappings.clear()
        for mat in unique_mats:
            item = mappings.add()
            item.orig_mat_name = mat.name
            guessed = _guess_collision_material_index(mat.name)
            if guessed > 0 and coll_mats and guessed < len(coll_mats):
                item.coll_mat_name = coll_mats[guessed].name
            else:
                item.coll_mat_name = "Keep Original"

        return context.window_manager.invoke_props_dialog(self, width=520)

    def draw(self, context):
        layout = self.layout
        mappings = context.scene.gn_shell_coll_mappings
        layout.label(text="Assign collision materials to each shell material:", icon='INFO')
        layout.label(text="'Keep Original' leaves the slot unchanged.", icon='BLANK1')
        layout.separator()
        if not mappings:
            layout.label(text="No materials found on shell meshes.", icon='ERROR')
            return
        layout.template_list("GN_UL_shell_coll_mappings", "", context.scene,
                             "gn_shell_coll_mappings", context.scene,
                             "gn_shell_coll_active_idx", rows=6, maxrows=12)

    def execute(self, context):
        try:
            from Sollumz.sollumz_properties import SollumType   # noqa: F401
        except ImportError:
            self.report({'ERROR'}, "Sollumz addon not found")
            return {'CANCELLED'}

        s = context.scene.gn_int
        name = s.mlo_name.strip()
        if not name:
            self.report({'ERROR'}, "Set an MLO Name first")
            return {'CANCELLED'}

        try:
            from Sollumz.ybn.collision_materials import collisionmats as coll_mats
        except ImportError:
            coll_mats = None
        name_to_idx = {m.name: i for i, m in enumerate(coll_mats)} if coll_mats else {}
        mat_mapping = {}
        for item in context.scene.gn_shell_coll_mappings:
            if item.coll_mat_name in ('', 'Keep Original'):
                continue
            idx = name_to_idx.get(item.coll_mat_name)
            if idx is not None:
                mat_mapping[item.orig_mat_name] = idx

        created, err = _do_build_shell_collision(context, name, mat_mapping)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        self.report({'INFO'}, f"Shell collision created for {created} room(s)")
        return {'FINISHED'}


def _do_build_shell_collision(context, name, mat_mapping):
    """Core of Create Shell Collision, callable without the interactive
    per-material dialog (e.g. from the Build MLO bulk popup, which passes an
    auto-guessed mat_mapping instead). Returns (created_count, error_or_None)."""
    try:
        from Sollumz.sollumz_properties import SollumType
    except ImportError:
        return 0, "Sollumz addon not found"

    shell_empty, room_meshes = _find_mlo_shell_data(name)
    if shell_empty is None or not room_meshes:
        return 0, "Shell meshes not found"

    try:
        from Sollumz.ybn.collision_materials import collisionmats as coll_mats
    except ImportError:
        coll_mats = None

    if True:
        main_col = bpy.data.collections.get(f"int_{name}")
        coll_col = None
        if main_col:
            for child in main_col.children:
                if child.name == "Collisions":
                    coll_col = child
                    break
        if coll_col is None:
            coll_col = _mlo_ensure_scene_collection("Collisions", context.scene)

        IDENT = Matrix.Identity(4)

        root_name = f"int_{name}"
        root_obj = bpy.data.objects.get(root_name)
        if root_obj is None or root_obj.type != 'EMPTY':
            root_obj = bpy.data.objects.new(root_name + "_bound", None)
        root_obj.empty_display_type = 'PLAIN_AXES'
        root_obj.empty_display_size = 0.1
        root_obj.sollum_type = SollumType.BOUND_COMPOSITE
        root_obj.matrix_world = IDENT.copy()
        if root_obj.name not in coll_col.objects:
            coll_col.objects.link(root_obj)

        bvh_name = "Shell.BVH"
        bvh_obj = bpy.data.objects.get(bvh_name)
        if bvh_obj is None:
            bvh_obj = bpy.data.objects.new(bvh_name, None)
        bvh_obj.empty_display_type = 'PLAIN_AXES'
        bvh_obj.empty_display_size = 0.1
        bvh_obj.sollum_type = SollumType.BOUND_GEOMETRYBVH
        bvh_obj.parent = root_obj
        bvh_obj.matrix_parent_inverse = IDENT.copy()
        bvh_obj.location = (0, 0, 0)
        if bvh_obj.name not in coll_col.objects:
            coll_col.objects.link(bvh_obj)

        try:
            from Sollumz.ybn.collision_materials import create_collision_material_from_index
            from Sollumz.sollumz_properties import MaterialType as _MatType
        except ImportError:
            create_collision_material_from_index = None
            _MatType = None

        created = 0
        for room in sorted(room_meshes):
            room_upper = room.upper()
            poly_name = f"{room_upper}_Shell.poly_mesh"
            src_objects = room_meshes[room]

            merged_mats = []
            mat_idx_maps = []
            for src in src_objects:
                idx_map = {}
                for old_i, mat in enumerate(src.data.materials):
                    if mat not in merged_mats:
                        merged_mats.append(mat)
                    idx_map[old_i] = merged_mats.index(mat)
                mat_idx_maps.append(idx_map)

            bm = bmesh.new()
            for src, idx_map in zip(src_objects, mat_idx_maps):
                tmp = src.data.copy()
                tmp.transform(src.matrix_world)
                for face in tmp.polygons:
                    face.material_index = idx_map.get(face.material_index, 0)
                bm.from_mesh(tmp)
                bpy.data.meshes.remove(tmp)

            poly_data = bpy.data.meshes.new(poly_name)
            bm.to_mesh(poly_data)
            bm.free()
            poly_data.name = poly_name
            for mat in merged_mats:
                poly_data.materials.append(mat)

            old_obj = bpy.data.objects.get(poly_name)
            if old_obj:
                bpy.data.objects.remove(old_obj, do_unlink=True)

            poly_obj = bpy.data.objects.new(poly_name, poly_data)
            poly_obj.sollum_type = SollumType.BOUND_POLY_TRIANGLE
            poly_obj.parent = bvh_obj
            poly_obj.matrix_parent_inverse = IDENT.copy()
            poly_obj.matrix_world = IDENT.copy()
            coll_col.objects.link(poly_obj)

            for i in range(len(poly_obj.data.materials)):
                slot_mat = poly_obj.data.materials[i]
                if slot_mat is None:
                    continue
                col_idx = mat_mapping.get(slot_mat.name)
                if col_idx is None:
                    continue
                col_mat_name = coll_mats[col_idx].name if coll_mats and col_idx < len(coll_mats) else "DEFAULT"
                existing = bpy.data.materials.get(col_mat_name)
                if (existing is not None and _MatType is not None
                        and hasattr(existing, 'sollum_type')
                        and existing.sollum_type == _MatType.COLLISION):
                    col_mat = existing
                elif create_collision_material_from_index is not None:
                    col_mat = create_collision_material_from_index(col_idx)
                else:
                    col_mat = existing or bpy.data.materials.new(col_mat_name)
                poly_obj.data.materials[i] = col_mat

            created += 1

    return created, None


# ===========================================================================
# Portals -- one magenta quad per opening, named "<room> - <room>" (or
# "<room> - limbo" facing the exterior). Same visual/RageKit convention as
# scene_organizer.py's Create/Rename Portal, but fully automatic: this tool
# already knows each opening's position and can look up which room(s) it
# borders, where scene_organizer needs the user to pick rooms by hand.
# ===========================================================================
def _mlo_apply_archetype_defaults(obj, set_static=False):
    try:
        arch = obj.ragequit_archetype
        arch.lodDist = 200
        arch.hdTextureDist = 100
        arch.add_to_mlo = True
        if set_static:
            arch.flag_static = True
    except Exception:
        pass


def _mlo_apply_sollumz_lod_defaults(obj):
    try:
        dp = obj.drawable_properties
        dp.lod_dist_high = 9998.0
        dp.lod_dist_med = 9998.0
        dp.lod_dist_low = 9998.0
        dp.lod_dist_vlow = 9998.0
    except Exception:
        pass


def _mlo_get_portal_material():
    """Shared 'Portal' material (solid magenta), matching scene_organizer.py."""
    mat = bpy.data.materials.get("Portal")
    if mat is None:
        mat = bpy.data.materials.new("Portal")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (1.0, 0.0, 1.0, 1.0)
        mat.diffuse_color = (1.0, 0.0, 1.0, 1.0)
    mat.use_backface_culling = False
    if hasattr(mat, 'use_backface_culling_shadow'):
        mat.use_backface_culling_shadow = False
    return mat


def _force_obj_name(obj, name, keep=None):
    """Set obj.name to exactly `name`. If another object already holds that
    name, move it ASIDE with a .NNN suffix rather than deleting it -- nothing
    is ever removed, so existing scene objects can never disappear."""
    existing = bpy.data.objects.get(name)
    if existing and existing is not obj and existing is not keep:
        existing.name = f"{name}.001"
    obj.name = name


def _force_data_name(data, name):
    """Set a data-block's name to exactly `name`, moving any existing
    occupant aside with a .NNN suffix instead of deleting it."""
    for col in (bpy.data.meshes, bpy.data.armatures, bpy.data.curves,
               bpy.data.metaballs, bpy.data.lattices):
        conflict = col.get(name)
        if conflict and conflict is not data:
            conflict.name = f"{name}.001"
            break
    data.name = name


def _mlo_room_token(room_obj_name):
    """'r01' -> '1', 'r08' -> '8'. Anything else (e.g. 'Main') unchanged."""
    if room_obj_name[:1] in ('r', 'R'):
        try:
            return str(int(room_obj_name[1:]))
        except ValueError:
            pass
    return room_obj_name


def _floor_idx_for_z(context, z):
    """Which floor (index into s.floors) a world Z falls inside, else the
    nearest one by base Z."""
    tops = _floor_tops(context)
    for i, (base, top, nxt) in enumerate(tops):
        if base - 0.05 <= z <= top + 0.05:
            return i
    if tops:
        return min(range(len(tops)), key=lambda i: abs(tops[i][0] - z))
    return None


def _room_token_for_uid(context, uid):
    """Numeric portal token ('1', '2', ...) for the room with this uid,
    finding its shell object wherever it currently lives -- still the raw
    'r0N' object in GN_Rooms (before Build MLO), or already moved/renamed by
    Build MLO to '<mlo_name>_r0N_shell.model' (inside Main). Search is
    global by the gn_room_uid custom prop rather than one fixed collection,
    since Build MLO relocates the object."""
    s = context.scene.gn_int
    mlo_name = s.mlo_name.strip()
    for ob in bpy.data.objects:
        if ob.get("gn_room_uid") == uid:
            if mlo_name:
                t = _mlo_room_token_from_shell_name(ob.name, mlo_name)
                if t is not None:
                    return _mlo_room_token(t)
            return _mlo_room_token(ob.name)
    return None


def _wall_room_for_opening(context, o, want_sign):
    """The room bordering this opening on one particular side, found by the
    SAME wall-segment alignment/proximity/Z-overlap test
    _opening_matches_any_wall uses to decide a wall actually cut this hole
    (rather than probing a point in space and checking polygon containment,
    which is fragile right next to a jog/notch in the room's boundary).
    Generalized to search every room (not just the opening's own nominal
    floor) so a manually z_offset half-floor/mezzanine room still matches.

    want_sign: +1 = the wall whose OWN inward normal points the same way as
    the opening's normal (so the opening's normal points INTO that room --
    it's the 'to' side); -1 = the opposite (the 'from' side, behind it).
    Picks the closest-aligned match if more than one room's wall qualifies.
    Returns a room token, or 'limbo' if nothing matches on that side."""
    s = context.scene.gn_int
    nrm = Vector((o.nx, o.ny))
    if nrm.length < 1e-6:
        return "limbo"
    nrm = nrm.normalized()
    best = None
    for r in s.rooms:
        fl = _floor_by_index(context, r.floor_index)
        if not fl:
            continue
        base_z, ceil_z, _ = fl
        base_z += r.z_offset; ceil_z += r.z_offset
        try:
            poly = [Vector(p) for p in json.loads(r.poly_json)]
        except Exception:
            continue
        n = len(poly)
        for i in range(n):
            A, B = poly[i], poly[(i + 1) % n]
            d = B - A
            L = d.length
            if L < 1e-6:
                continue
            d = d / L
            wn = Vector((-d.y, d.x))
            align = nrm.dot(wn)
            if abs(align) < 0.8 or (align > 0) != (want_sign > 0):
                continue
            rel = Vector((o.cx, o.cy)) - A
            x = rel.dot(d)
            lat = abs(rel.dot(wn))
            if lat > 0.45 or x < -o.hw or x > L + o.hw:
                continue
            z0 = max(base_z, o.sill); z1 = min(ceil_z, o.top)
            if z1 - z0 <= 0.02:
                continue
            if best is None or lat < best[0]:
                best = (lat, r)
    if best is None:
        return "limbo"
    token = _room_token_for_uid(context, best[1].uid)
    return token if token is not None else "limbo"


def _rooms_for_opening(context, o):
    """Which two rooms an opening borders. A side with no matching room
    (facing outdoors) is labelled 'limbo' -- e.g. '1 - limbo' -- matching
    the real portal-naming convention."""
    token_to = _wall_room_for_opening(context, o, +1)
    token_from = _wall_room_for_opening(context, o, -1)
    return token_to, token_from


def _mlo_portals_collection(scene):
    """The Portals collection nested under int_<mlo_name>, or None if the
    MLO / that collection doesn't exist yet."""
    s = scene.gn_int
    name = s.mlo_name.strip()
    if not name:
        return None
    main_col = bpy.data.collections.get(f"int_{name}")
    if not main_col:
        return None
    for c in main_col.children:
        if c.name == "Portals":
            return c
    return None


class GN_UL_portals(bpy.types.UIList):
    def draw_item(self, ctx, layout, data, item, icon, adata, aprop, index=0, flt=0):
        row = layout.row(align=True)
        row.prop(item, "name", text="", icon='OUTLINER_OB_LIGHTPROBE', emboss=False)


class GN_OT_remove_portal(Operator):
    bl_idname = "gn_int.remove_portal"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Delete Selected Portal"
    bl_description = "Delete the selected portal object"

    def execute(self, context):
        s = context.scene.gn_int
        coll = _mlo_portals_collection(context.scene)
        if not coll or not (0 <= s.portal_index < len(coll.objects)):
            self.report({'WARNING'}, "No portal selected")
            return {'CANCELLED'}
        ob = coll.objects[s.portal_index]
        bpy.data.objects.remove(ob, do_unlink=True)
        s.portal_index = max(0, min(s.portal_index, len(coll.objects) - 1))
        return {'FINISHED'}


class GN_OT_clear_portals(Operator):
    bl_idname = "gn_int.clear_portals"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Clear All Portals"
    bl_description = "Delete every portal (they're regenerated fresh by Create Portals)"

    def execute(self, context):
        coll = _mlo_portals_collection(context.scene)
        if not coll or not coll.objects:
            self.report({'WARNING'}, "No portals to clear")
            return {'CANCELLED'}
        n = len(coll.objects)
        for ob in list(coll.objects):
            bpy.data.objects.remove(ob, do_unlink=True)
        context.scene.gn_int.portal_index = 0
        self.report({'INFO'}, f"Cleared {n} portal(s)")
        return {'FINISHED'}


class GN_OT_flip_portal(Operator):
    bl_idname = "gn_int.flip_portal"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Flip Selected Portal"
    bl_description = "Flip the face normal of the selected portal"

    def execute(self, context):
        s = context.scene.gn_int
        coll = _mlo_portals_collection(context.scene)
        if not coll or not (0 <= s.portal_index < len(coll.objects)):
            self.report({'WARNING'}, "No portal selected")
            return {'CANCELLED'}
        ob = coll.objects[s.portal_index]
        if ob.type != 'MESH':
            self.report({'WARNING'}, "Selected portal has no mesh")
            return {'CANCELLED'}
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        for f in bm.faces:
            f.normal_flip()
        bm.normal_update()
        bm.to_mesh(ob.data)
        bm.free()
        ob.data.update()
        # keep the name's "from - to" order matching the now-flipped normal
        parts = ob.name.split(" - ", 1)
        if len(parts) == 2:
            ob.name = f"{parts[1]} - {parts[0]}"
        self.report({'INFO'}, f"Flipped '{ob.name}'")
        return {'FINISHED'}


def _spawn_portal(coll, world_positions, name, face_dir=None):
    """One Portal quad from 4 world corners, origin at the quad's centre.
    face_dir (world-space Vector, optional): the face normal is flipped to
    point this way -- so 'A - B' consistently means the normal points
    toward B. Falls back to the old always-+Y winding when face_dir is
    None (matches scene_organizer.py's _spawn_portal)."""
    mat = _mlo_get_portal_material()
    center = sum(world_positions, Vector((0.0, 0.0, 0.0))) / len(world_positions)
    local_positions = [p - center for p in world_positions]
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bm_verts = [bm.verts.new(p) for p in local_positions]
    face = bm.faces.new(bm_verts)
    bm.normal_update()
    flip = (face.normal.dot(face_dir) < 0.0) if face_dir is not None else (face.normal.y < 0.0)
    if flip:
        face.normal_flip()
        bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    ob.location = center
    coll.objects.link(ob)
    me.materials.append(mat)
    _mlo_apply_archetype_defaults(ob)
    _mlo_apply_sollumz_lod_defaults(ob)
    return ob


class GN_OT_create_portals(Operator):
    bl_idname = "gn_int.create_portals"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Create Portals"
    bl_description = ("Create a Portal quad at every opening, named "
                      "'<room> - <room>' (or '<room> - limbo' facing the "
                      "exterior). Safe to re-run: a portal for an opening "
                      "that's already been done is replaced, not duplicated")

    def execute(self, context):
        s = context.scene.gn_int
        name = s.mlo_name.strip()
        if not name:
            self.report({'ERROR'}, "Set an MLO Name first")
            return {'CANCELLED'}
        if not s.openings:
            self.report({'ERROR'}, "No openings - run Project Openings first")
            return {'CANCELLED'}

        portals_col = _mlo_portals_collection(context.scene)
        if portals_col is None:
            portals_col = _mlo_ensure_scene_collection("Portals", context.scene)

        existing = {ob.get("gn_opening_uid"): ob for ob in portals_col.objects
                   if ob.get("gn_opening_uid") is not None}

        made = 0
        skipped = 0
        for o in s.openings:
            nrm = Vector((o.nx, o.ny))
            if nrm.length < 1e-6 or o.hw < 0.01 or o.top <= o.sill:
                skipped += 1
                continue
            tangent = Vector((-nrm.y, nrm.x)).normalized()
            c = Vector((o.cx, o.cy))
            # token_to = the room the opening's normal (o.nx, o.ny) points
            # toward; token_from = the room behind it. Name is "from - to"
            # so it reads in the direction the portal actually faces.
            token_to, token_from = _rooms_for_opening(context, o)
            portal_name = f"{token_from} - {token_to}"

            # A room's wall reveal (Boundary cleanup > win/door reveal) is
            # extruded OUTWARD from this same hole rectangle, away from that
            # room, by _build_wall -- so at a room/limbo boundary the portal
            # (sitting at the raw hole) reads as recessed behind that jamb.
            # Push it out to the jamb's outer rim, toward whichever side is
            # limbo. A room-to-room opening needs no shift: both sides'
            # jambs already meet exactly at the raw centre.
            rd = 0.0
            if s.reveal:
                rd = (s.partition * 0.5) if o.is_door else s.wall_margin
            shift = Vector((0.0, 0.0))
            if rd > 1e-4:
                if token_to == "limbo" and token_from != "limbo":
                    shift = nrm * rd
                elif token_from == "limbo" and token_to != "limbo":
                    shift = -nrm * rd
            c = c + shift
            corners = [
                Vector(((c - tangent * o.hw).x, (c - tangent * o.hw).y, o.sill)),
                Vector(((c + tangent * o.hw).x, (c + tangent * o.hw).y, o.sill)),
                Vector(((c + tangent * o.hw).x, (c + tangent * o.hw).y, o.top)),
                Vector(((c - tangent * o.hw).x, (c - tangent * o.hw).y, o.top)),
            ]

            old = existing.pop(o.uid, None)
            if old:
                bpy.data.objects.remove(old, do_unlink=True)

            ob = _spawn_portal(portals_col, corners, portal_name,
                               face_dir=Vector((o.nx, o.ny, 0.0)))
            ob["gn_opening_uid"] = o.uid
            made += 1

        removed = 0
        for ob in existing.values():
            bpy.data.objects.remove(ob, do_unlink=True)
            removed += 1

        msg = f"Created {made} portal(s)"
        if skipped:
            msg += f", {skipped} skipped (degenerate)"
        if removed:
            msg += f", {removed} stale removed"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


# ===========================================================================
# Add Empties -- decals/details/proxy/visuals/lights (+ custom types) placed
# in a room collection, named "<mlo>_<room_num>_<type>". Pivot = the room
# object's own origin (already centred by _build_shell), matching
# scene_organizer.py's Add Empties but without its separate pivot-override
# system, which this tool doesn't need.
# ===========================================================================
PRESET_EMPTIES = ["decals", "details", "proxy", "visuals", "lights"]
_PRESET_ATTR = {"decals": "empty_decals", "details": "empty_details",
                "proxy": "empty_proxy", "visuals": "empty_visuals",
                "lights": "empty_lights"}


def _mlo_apply_empty_defaults(empty):
    try:
        from Sollumz.sollumz_properties import SollumType
        empty.sollum_type = SollumType.DRAWABLE
    except Exception:
        pass
    _mlo_apply_archetype_defaults(empty, set_static=True)
    _mlo_apply_sollumz_lod_defaults(empty)


def _gn_create_empty_in_room(room_col, mlo_name, room_name, empty_type_name):
    room_num = _gn_room_number_from_name(room_name)
    obj_name = f"{mlo_name}_{room_num}_{empty_type_name}"
    pivot = (0.0, 0.0, 0.0)
    room_ob = bpy.data.objects.get(room_name)
    if room_ob is None:
        rc = bpy.data.collections.get(ROOM_COLL)
        if rc:
            room_ob = rc.objects.get(room_name)
    if room_ob is not None:
        pivot = tuple(room_ob.matrix_world.translation)

    existing = bpy.data.objects.get(obj_name)
    if existing is not None:
        if room_col.name not in [c.name for c in existing.users_collection]:
            room_col.objects.link(existing)
        return existing

    empty = bpy.data.objects.new(obj_name, None)
    empty.empty_display_type = 'PLAIN_AXES'
    empty.location = pivot
    room_col.objects.link(empty)
    _mlo_apply_empty_defaults(empty)
    return empty


class GN_OT_add_empties(Operator):
    bl_idname = "gn_int.add_empties"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Add Selected Empties"
    bl_description = ("Add enabled empties to the selected room collection "
                      "(or every room, if 'All Rooms' is picked). Pivot = "
                      "that room object's own origin")

    def execute(self, context):
        s = context.scene.gn_int
        name = s.mlo_name.strip()
        if not name:
            self.report({'ERROR'}, "Set an MLO Name first")
            return {'CANCELLED'}

        if s.empty_target_room == 'ALL':
            room_cols = _gn_iter_room_collections(context)
            if not room_cols:
                self.report({'ERROR'}, "No room collections found - run Build MLO Collections first")
                return {'CANCELLED'}
        else:
            room_col = bpy.data.collections.get(s.empty_target_room) if s.empty_target_room else None
            if room_col is None or s.empty_target_room in ("NONE", ""):
                rooms = _gn_iter_room_collections(context)
                if not rooms:
                    self.report({'ERROR'}, "No room collections found - run Build MLO Collections first")
                    return {'CANCELLED'}
                room_col = rooms[0]
                try:
                    s.empty_target_room = room_col.name
                except Exception:
                    pass
            room_cols = [room_col]

        to_create = [p for p in PRESET_EMPTIES if getattr(s, _PRESET_ATTR[p], False)]
        for item in s.custom_empties:
            if item.enabled and item.name.strip():
                to_create.append(item.name.strip())
        if not to_create:
            self.report({'WARNING'}, "Tick at least one preset or custom empty above")
            return {'CANCELLED'}

        total = 0
        for rc in room_cols:
            for t in to_create:
                _gn_create_empty_in_room(rc, name, rc.name, t)
                total += 1
        rooms_desc = "every room" if s.empty_target_room == 'ALL' else f"'{room_cols[0].name}'"
        self.report({'INFO'}, f"Added {total} empty(s) to {rooms_desc}")
        return {'FINISHED'}


class GN_OT_add_custom_empty(Operator):
    bl_idname = "gn_int.add_custom_empty"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Add Custom Empty Type"
    bl_description = "Add a custom empty type to the list"

    def execute(self, context):
        s = context.scene.gn_int
        nm = s.empty_custom_name.strip()
        if not nm:
            self.report({'WARNING'}, "Enter a name in Custom Name first")
            return {'CANCELLED'}
        if nm in [e.name for e in s.custom_empties]:
            self.report({'WARNING'}, f"'{nm}' is already in the list")
            return {'CANCELLED'}
        item = s.custom_empties.add()
        item.name = nm
        s.empty_custom_name = ""
        return {'FINISHED'}


class GN_OT_remove_custom_empty(Operator):
    bl_idname = "gn_int.remove_custom_empty"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Remove"
    bl_description = "Remove this custom empty type"
    index: IntProperty()

    def execute(self, context):
        s = context.scene.gn_int
        if 0 <= self.index < len(s.custom_empties):
            s.custom_empties.remove(self.index)
        return {'FINISHED'}


# ===========================================================================
# Smart Rename -- rename selected mesh objects to "<mlo>_<room>_<category>"
# (Blender auto-dedups .001, .002...), optionally applying modifiers/scale,
# renaming the UV map to "UVMap 0", and joining into one object.
# ===========================================================================
class GN_OT_smart_rename(Operator):
    bl_idname = "gn_int.smart_rename"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Rename Selected"
    bl_description = ("Rename selected mesh objects to <mlo>_<room>_<category>. "
                      "Optionally applies modifiers/scale, renames UV to "
                      "'UVMap 0', then merges into one object")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=280)

    def draw(self, context):
        s = context.scene.gn_int
        col = self.layout.column(align=True)
        col.label(text=f"MLO: {s.mlo_name or '-- not set --'}", icon='SCENE_DATA')
        col.prop(s, "sr_room", text="Room")
        col.prop(s, "sr_category", text="Category")
        if s.sr_category == 'custom':
            col.prop(s, "sr_category_custom", text="Custom")
        self.layout.prop(s, "sr_merge")

    def execute(self, context):
        s = context.scene.gn_int
        name = s.mlo_name.strip()
        if not name:
            self.report({'ERROR'}, "Set an MLO Name first")
            return {'CANCELLED'}
        room = s.sr_room
        if room in ('NONE', ''):
            self.report({'WARNING'}, "No room selected - build rooms first")
            return {'CANCELLED'}
        category = s.sr_category
        if category == 'custom':
            category = s.sr_category_custom.strip()
            if not category:
                self.report({'WARNING'}, "Enter a custom category")
                return {'CANCELLED'}

        objects = sorted([o for o in context.selected_objects if o.type == 'MESH'],
                         key=lambda o: o.name)
        if not objects:
            self.report({'WARNING'}, "No mesh objects selected")
            return {'CANCELLED'}

        base = f"{name}_{room}_{category}"
        for i, obj in enumerate(objects):
            obj.name = f"__gn_tmp_{i:04d}__"
        for obj in objects:
            obj.name = base

        if s.sr_merge:
            for obj in objects:
                for mod in list(obj.modifiers):
                    try:
                        with context.temp_override(object=obj,
                                                    selected_editable_objects=[obj],
                                                    active_object=obj):
                            bpy.ops.object.modifier_apply(modifier=mod.name)
                    except Exception:
                        pass
                try:
                    with context.temp_override(selected_editable_objects=[obj],
                                                active_object=obj, object=obj):
                        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
                except Exception:
                    pass
                if obj.data and obj.data.uv_layers:
                    obj.data.uv_layers[0].name = "UVMap 0"

            all_matching = [o for o in bpy.data.objects
                           if o.name == base or re.match(rf'^{re.escape(base)}\.\d+$', o.name)]
            if len(all_matching) > 1:
                target = bpy.data.objects.get(base) or all_matching[0]
                scene_col = context.scene.collection
                for o in all_matching:
                    try:
                        scene_col.objects.link(o)
                    except RuntimeError:
                        pass
                try:
                    with context.temp_override(
                            window=context.window, scene=context.scene,
                            view_layer=context.view_layer,
                            selected_objects=all_matching,
                            selected_editable_objects=all_matching,
                            active_object=target, object=target):
                        bpy.ops.object.join()
                    target.name = base
                    self.report({'INFO'},
                                f"Renamed and merged {len(all_matching)} object(s) -> '{base}'")
                    return {'FINISHED'}
                except RuntimeError as e:
                    self.report({'WARNING'}, f"Join failed: {e}")

        self.report({'INFO'}, f"Renamed {len(objects)} object(s) -> '{base}'")
        return {'FINISHED'}


# ===========================================================================
# Create Asset -- Sollumz asset hierarchy for the selected mesh, placed at
# world origin in Assets (or Assets/Doors for door assets). Same structure
# as scene_organizer.py's Create Asset: Door -> Armature, Regular -> Empty,
# each DRAWABLE -> .col (BOUND_COMPOSITE) -> [.bvh ->] .poly_mesh (only when
# Auto Collision is on) + .model (the original mesh, moved into the
# hierarchy, keeping its own name). Matching instances already placed in
# Props_*/Assets_*/Doors* collections are relinked to the shared mesh.
# ===========================================================================
class GN_OT_create_asset(Operator):
    bl_idname = "gn_int.create_asset"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Create Asset"
    bl_description = ("Build a Sollumz asset hierarchy for the selected mesh. "
                      "Select mesh, set options, click Create. The asset "
                      "keeps the selected object's name")

    @classmethod
    def poll(cls, context):
        try:
            from Sollumz.sollumz_properties import SollumType   # noqa: F401
        except ImportError:
            return False
        return (context.mode == 'OBJECT' and context.active_object is not None
                and context.active_object.type == 'MESH')

    def execute(self, context):
        try:
            from Sollumz.sollumz_properties import SollumType
        except ImportError:
            self.report({'ERROR'}, "Sollumz addon not found")
            return {'CANCELLED'}

        s = context.scene.gn_int
        name = s.mlo_name.strip()
        obj = context.active_object
        asset_type = s.asset_type
        auto_collision = s.auto_collision

        base = obj.name.split('.', 1)[0].strip()
        if obj.data.users > 1:
            obj.data = obj.data.copy()
        shared_data = obj.data
        _force_data_name(shared_data, base)

        name_bvh = f"{base}.bvh"
        name_poly = f"{base}.poly_mesh"
        name_model = f"{base}.model"

        main_col = bpy.data.collections.get(f"int_{name}") if name else None
        assets_col = None
        if main_col:
            for c in main_col.children:
                if c.name == "Assets":
                    assets_col = c
                    break
        if assets_col is None:
            assets_col = _mlo_ensure_scene_collection("Assets", context.scene)
        if asset_type == 'DOOR':
            dest_col = None
            for c in assets_col.children:
                if c.name == "Doors":
                    dest_col = c
                    break
            if dest_col is None:
                dest_col = _mlo_make_collection("Doors", assets_col)
        else:
            dest_col = assets_col

        IDENT = Matrix.Identity(4)

        if asset_type == 'DOOR':
            arm_data = bpy.data.armatures.new(base)
            root_obj = bpy.data.objects.new(base, arm_data)
        else:
            root_obj = bpy.data.objects.new(base, None)
            root_obj.empty_display_type = 'PLAIN_AXES'
            root_obj.empty_display_size = 0.1
        _force_obj_name(root_obj, base)
        root_obj.sollum_type = SollumType.DRAWABLE
        root_obj.matrix_world = IDENT.copy()
        dest_col.objects.link(root_obj)
        _mlo_apply_archetype_defaults(root_obj, set_static=True)
        _mlo_apply_sollumz_lod_defaults(root_obj)

        col_name = f"{base}.col"
        col_obj = bpy.data.objects.new(col_name, None)
        _force_obj_name(col_obj, col_name)
        col_obj.empty_display_type = 'PLAIN_AXES'
        col_obj.empty_display_size = 0.1
        col_obj.sollum_type = SollumType.BOUND_COMPOSITE
        col_obj.parent = root_obj
        col_obj.matrix_parent_inverse = IDENT.copy()
        col_obj.location = (0, 0, 0)
        dest_col.objects.link(col_obj)

        door_rot_fix = None
        if asset_type == 'DOOR':
            vcos = obj.data.vertices
            if vcos:
                spread_x = max(v.co.x for v in vcos) - min(v.co.x for v in vcos)
                spread_y = max(v.co.y for v in vcos) - min(v.co.y for v in vcos)
                if spread_y > spread_x:
                    rot_fix = Matrix.Rotation(math.pi / 2, 3, 'Z')
                    for v in vcos:
                        v.co = rot_fix @ v.co
                    obj.data.update()
                    door_rot_fix = Matrix.Rotation(math.pi / 2, 4, 'Z')

        if auto_collision:
            if asset_type == 'DOOR':
                poly_parent = col_obj
            else:
                bvh_obj = bpy.data.objects.new(name_bvh, None)
                _force_obj_name(bvh_obj, name_bvh)
                bvh_obj.empty_display_type = 'PLAIN_AXES'
                bvh_obj.empty_display_size = 0.1
                bvh_obj.sollum_type = SollumType.BOUND_GEOMETRYBVH
                bvh_obj.parent = col_obj
                bvh_obj.matrix_parent_inverse = IDENT.copy()
                bvh_obj.location = (0, 0, 0)
                dest_col.objects.link(bvh_obj)
                poly_parent = bvh_obj

            poly_data = obj.data.copy()
            _force_data_name(poly_data, name_poly)
            poly_obj = bpy.data.objects.new(name_poly, poly_data)
            _force_obj_name(poly_obj, name_poly)
            poly_obj.sollum_type = SollumType.BOUND_POLY_TRIANGLE
            poly_obj.parent = poly_parent
            poly_obj.matrix_parent_inverse = IDENT.copy()
            poly_obj.location = (0, 0, 0)
            dest_col.objects.link(poly_obj)

            try:
                from Sollumz.ybn.collision_materials import (
                    collisionmats as coll_mats,
                    create_collision_material_from_index as mk_col_mat)
                from Sollumz.sollumz_properties import MaterialType as MatType
            except ImportError:
                coll_mats = None
                mk_col_mat = None
                MatType = None

            orig_slot_names = [m.name if m else "" for m in poly_obj.data.materials] or [""]
            poly_obj.data.materials.clear()
            for slot_name in orig_slot_names:
                hint = f"{base} {slot_name}".strip()
                col_idx = _guess_collision_material_index(hint)
                col_mat_name = (coll_mats[col_idx].name
                               if coll_mats and col_idx < len(coll_mats) else "DEFAULT")
                existing = bpy.data.materials.get(col_mat_name)
                if (existing is not None and MatType is not None
                        and hasattr(existing, 'sollum_type')
                        and existing.sollum_type == MatType.COLLISION):
                    col_mat = existing
                elif mk_col_mat is not None:
                    col_mat = mk_col_mat(col_idx)
                else:
                    col_mat = existing or bpy.data.materials.new(col_mat_name)
                poly_obj.data.materials.append(col_mat)

        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        dest_col.objects.link(obj)
        _force_obj_name(obj, name_model)
        obj.sollum_type = SollumType.DRAWABLE_MODEL
        obj.parent = root_obj
        obj.matrix_parent_inverse = IDENT.copy()
        obj.matrix_world = IDENT.copy()
        _mlo_apply_archetype_defaults(obj, set_static=True)
        _mlo_apply_sollumz_lod_defaults(obj)

        def _strip_dedup(n):
            return re.sub(r'\.\d+$', '', n)

        def _is_instance_col(c):
            n = c.name.lower()
            return (n.startswith("props_") or n.startswith("prop_")
                    or n.startswith("assets_") or n.startswith("doors"))

        props_renamed = 0
        for pobj in bpy.data.objects:
            if pobj.type != 'MESH' or pobj is obj:
                continue
            if not any(_is_instance_col(c) for c in pobj.users_collection):
                continue
            name_match = _strip_dedup(pobj.name) == base
            data_match = pobj.data is shared_data
            if name_match or data_match:
                pobj.name = base
                pobj.data = shared_data
                if door_rot_fix is not None:
                    pobj.matrix_world = pobj.matrix_world @ door_rot_fix.inverted()
                props_renamed += 1

        kind = "door" if asset_type == 'DOOR' else "regular"
        coll = "auto" if auto_collision else "manual"
        extra = f" Renamed {props_renamed} prop instance(s)." if props_renamed else ""
        self.report({'INFO'}, f"Created {kind} asset '{base}' with {coll} collision.{extra}")
        return {'FINISHED'}


class GN_OT_clear_rooms(Operator):
    bl_idname = "gn_int.clear_rooms"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Clear Rooms"

    def execute(self, context):
        context.scene.gn_int.rooms.clear()
        _clear_coll(ROOM_COLL)
        return {'FINISHED'}


class GN_OT_seed_rooms(Operator):
    bl_idname = "gn_int.seed_rooms"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Make Floor Walls"
    bl_description = ("Reset the SELECTED floor to one room = its whole envelope, "
                      "ready to split. Other floors (and any splits already made "
                      "there) are left untouched")

    def execute(self, context):
        s = context.scene.gn_int
        if not s.floors:
            self.report({'ERROR'}, "Generate boundaries first")
            return {'CANCELLED'}
        if not (0 <= s.floor_index < len(s.floors)):
            self.report({'ERROR'}, "Select a floor in the list")
            return {'CANCELLED'}
        made = seed_rooms_from_boundaries(context, s.floor_index)
        if made == 0:
            self.report({'WARNING'}, "No boundary for this floor - run Generate Boundaries")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Floor {s.floor_index+1} reset to its envelope")
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
    bl_options = {'REGISTER', 'UNDO'}
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


def _draw_path_overlay(self, context):
    all_pts = self.pts + ([self.cur] if self.cur is not None else [])
    if len(all_pts) < 2:
        return
    z = self.base_z
    line_pts = []
    for i in range(len(all_pts) - 1):
        a, b = all_pts[i], all_pts[i + 1]
        line_pts.append((a.x, a.y, z))
        line_pts.append((b.x, b.y, z))
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'LINES', {"pos": line_pts})
    gpu.state.line_width_set(3.0)
    gpu.state.blend_set('ALPHA')
    shader.bind()
    shader.uniform_float("color", (1.0, 0.25, 0.85, 1.0))
    batch.draw(shader)
    gpu.state.line_width_set(1.0)


class GN_OT_split_room_path(Operator):
    bl_idname = "gn_int.split_room_path"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Split Room (Path)"
    bl_description = ("Click a bent cut across a room: a wall point, any "
                      "number of bend points, then a final wall point on a "
                      "DIFFERENT wall. Enter to finish, Esc/RMB to cancel")

    def invoke(self, context, event):
        s = context.scene.gn_int
        self.floor_idx = s.floor_index   # the floor selected in the Floors list
        fl = _floor_by_index(context, self.floor_idx)
        if not fl:
            self.report({'ERROR'}, "Select a floor in the Floors list")
            return {'CANCELLED'}
        if not any(r.floor_index == self.floor_idx for r in s.rooms):
            self.report({'ERROR'}, "No rooms on this floor - click 'Make Floor Walls' first")
            return {'CANCELLED'}
        self.base_z = fl[0]
        self.pts = []
        self.cur = None
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_path_overlay, (self, context), 'WINDOW', 'POST_VIEW')
        context.window_manager.modal_handler_add(self)
        context.area.header_text_set(
            "Split (Path): click wall point, bend point(s), then opposite wall "
            "point  |  Enter: finish  |  Esc/RMB: cancel")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        context.area.tag_redraw()
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
        if event.type == 'MOUSEMOVE':
            hit = _plane_hit(context, event, self.base_z)
            if hit:
                self.cur = _snap_pt(Vector((hit.x, hit.y)), context.scene.gn_int.snap)
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            hit = _plane_hit(context, event, self.base_z)
            if hit is None:
                return {'RUNNING_MODAL'}
            self.pts.append(_snap_pt(Vector((hit.x, hit.y)), context.scene.gn_int.snap))
            return {'RUNNING_MODAL'}
        if event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            if len(self.pts) < 2:
                self.report({'WARNING'}, "Need at least 2 points (start + end wall)")
                return {'RUNNING_MODAL'}
            self._commit(context)
            return self._end(context, True)
        if event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
            return self._end(context, False)
        return {'RUNNING_MODAL'}

    def _commit(self, context):
        mid = sum(self.pts, Vector((0.0, 0.0))) / len(self.pts)
        ridx = _room_at_point(context, self.floor_idx, mid)
        if ridx < 0:
            self.report({'WARNING'}, "Path midpoint not inside a room")
            return
        made = split_room_record_path(context, ridx, self.pts)
        if made:
            self.report({'INFO'}, "Room split")
        else:
            self.report({'WARNING'},
                        "Split failed (path must cross the room, ends on two "
                        "different walls)")

    def _end(self, context, ok):
        bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
        context.area.header_text_set(None)
        context.area.tag_redraw()
        return {'FINISHED'} if ok else {'CANCELLED'}


class GN_OT_split_edges(Operator):
    bl_idname = "gn_int.split_edges"
    bl_options = {'REGISTER', 'UNDO'}
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
# stairs (solid flight built between two picked edges)
# ===========================================================================
def _selected_edge_endpoints(context):
    """World-space (v0, v1) pairs for every selected edge across every mesh
    object currently in Edit Mode (supports multi-object edit)."""
    out = []
    for ob in context.objects_in_mode:
        if ob.type != 'MESH':
            continue
        bm = bmesh.from_edit_mesh(ob.data)
        mw = ob.matrix_world
        for e in bm.edges:
            if e.select:
                a, b = e.verts
                out.append((mw @ a.co, mw @ b.co))
    return out


def _build_stairs_between_edges(b0, b1, t0, t1, step_height, step_depth):
    """Solid staircase (treads, risers, closed sides, soffit) running from
    edge (b0,b1) up to edge (t0,t1). Each edge is assumed roughly level (flat
    at its own Z); the two edges need not be parallel or the same length --
    the sides taper linearly between them. Returns (verts, faces) in world
    space, or None if the edges are too close in height or in the travel
    direction to form a run."""
    b_mid = (b0 + b1) / 2; t_mid = (t0 + t1) / 2
    if t_mid.z < b_mid.z:
        b0, b1, t0, t1 = t0, t1, b0, b1
        b_mid, t_mid = t_mid, b_mid
    dz = t_mid.z - b_mid.z
    travel = (Vector((t_mid.x, t_mid.y)) - Vector((b_mid.x, b_mid.y))).length
    if dz < 0.05 or travel < 0.05:
        return None
    # pair up b-side / t-side verts so the run doesn't twist
    d_same = (b0 - t0).length + (b1 - t1).length
    d_swap = (b0 - t1).length + (b1 - t0).length
    if d_swap < d_same:
        t0, t1 = t1, t0
    z_bot, z_top = b_mid.z, t_mid.z

    n = max(2, math.ceil(dz / max(step_height, 0.02)),
                math.ceil(travel / max(step_depth, 0.02)))
    rise = dz / n

    def side_xy(bp, tp, t):
        x = bp.x + (tp.x - bp.x) * t
        y = bp.y + (tp.y - bp.y) * t
        return x, y

    verts, faces = [], []

    def quad(a, b, c, d_):
        base = len(verts)
        verts.extend([a, b, c, d_])
        faces.append((base, base + 1, base + 2, base + 3))

    def tri(a, b, c):
        base = len(verts)
        verts.extend([a, b, c])
        faces.append((base, base + 1, base + 2))

    for i in range(n):
        tf, tb = i / n, (i + 1) / n
        zl, zh = z_bot + i * rise, z_bot + (i + 1) * rise
        x0f, y0f = side_xy(b0, t0, tf); x1f, y1f = side_xy(b1, t1, tf)
        x0b, y0b = side_xy(b0, t0, tb); x1b, y1b = side_xy(b1, t1, tb)
        # tread (top of the step)
        quad((x0b, y0b, zh), (x1b, y1b, zh), (x1f, y1f, zh), (x0f, y0f, zh))
        # riser (front face of the step)
        quad((x0f, y0f, zl), (x1f, y1f, zl), (x1f, y1f, zh), (x0f, y0f, zh))
        # side wedges: close the gap between the stepped profile and the
        # straight incline underneath, on both sides
        tri((x0f, y0f, zh), (x0f, y0f, zl), (x0b, y0b, zh))
        tri((x1f, y1f, zh), (x1f, y1f, zl), (x1b, y1b, zh))

    # soffit -- the single straight incline closing the underside
    quad((b0.x, b0.y, z_bot), (b1.x, b1.y, z_bot), (t1.x, t1.y, z_top), (t0.x, t0.y, z_top))
    return verts, faces


def _apply_stair_mesh(ob, verts, faces):
    """Write world-space verts/faces (from _build_stairs_between_edges) into
    ob's existing mesh data, converting through the object's CURRENT
    matrix_world (so this still works if the stair object has been moved or
    rotated since it was created)."""
    mw_inv = ob.matrix_world.inverted()
    bm = bmesh.new()
    bverts = [bm.verts.new(mw_inv @ Vector(v)) for v in verts]
    for f in faces:
        try:
            bm.faces.new([bverts[i] for i in f])
        except ValueError:
            pass
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()


def _rebuild_stair_mesh(ob, height, depth):
    """Re-run the stair build for an EXISTING stair object at new
    height/depth settings, using the original two edges stored on it at
    creation time. Returns True if ob was a stair and got rebuilt."""
    raw = ob.get("gn_stair_data")
    if not raw:
        return False
    try:
        d = json.loads(raw)
        b0, b1 = Vector(d["b0"]), Vector(d["b1"])
        t0, t1 = Vector(d["t0"]), Vector(d["t1"])
    except Exception:
        return False
    built = _build_stairs_between_edges(b0, b1, t0, t1, height, depth)
    if not built:
        return False
    _apply_stair_mesh(ob, *built)
    return True


class GN_OT_create_stairs(Operator):
    bl_idname = "gn_int.create_stairs"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Create Stairs"
    bl_description = ("Edit Mode: select two edges (one at the bottom, one at "
                      "the top -- can be on two different objects, e.g. two "
                      "floor boundaries) and build a solid flight of stairs "
                      "between them")

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def execute(self, context):
        s = context.scene.gn_int
        pairs = _selected_edge_endpoints(context)
        if len(pairs) != 2:
            self.report({'ERROR'}, f"Select exactly 2 edges (one top, one bottom) -- {len(pairs)} selected")
            return {'CANCELLED'}
        (b0, b1), (t0, t1) = pairs
        built = _build_stairs_between_edges(b0, b1, t0, t1,
                                            s.stair_step_height, s.stair_step_depth)
        if not built:
            self.report({'ERROR'}, "The two edges are too close together (in height or distance) to build a run")
            return {'CANCELLED'}
        bpy.ops.object.mode_set(mode='OBJECT')
        coll = _get_coll(STAIR_COLL)
        name = f"GN_Stairs.{len(coll.objects):03d}"
        me = bpy.data.meshes.new(name)
        ob = bpy.data.objects.new(name, me)
        coll.objects.link(ob)
        _apply_stair_mesh(ob, *built)
        ob["gn_stair_data"] = json.dumps({
            "b0": list(b0), "b1": list(b1), "t0": list(t0), "t1": list(t1)})
        context.view_layer.objects.active = ob
        bpy.ops.object.select_all(action='DESELECT')
        ob.select_set(True)
        self.report({'INFO'}, f"Created {name}")
        return {'FINISHED'}


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


def _collect_into(objs, coll):
    for o in objs:
        for c in list(o.users_collection):
            if c is not coll:
                c.objects.unlink(o)
        if coll not in o.users_collection:
            coll.objects.link(o)


class GN_OT_split_opening_pieces(Operator):
    bl_idname = "gn_int.split_opening_pieces"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Duplicate & Split Selected"
    bl_description = ("Duplicate the selected window/door geometry, split it into one "
                      "object per disconnected piece, and collect them in the "
                      "'Openings' collection -- ready to select-all and Project. "
                      "Works on a face selection in Edit Mode, or on whole objects "
                      "in Object Mode; the source mesh is left untouched")

    @classmethod
    def poll(cls, context):
        if context.mode == 'EDIT_MESH':
            return context.active_object is not None
        return any(o.type == 'MESH' for o in context.selected_objects)

    def execute(self, context):
        coll = _get_coll(OPENING_PIECES_COLL)

        if context.mode == 'EDIT_MESH':
            ob = context.active_object
            bm = bmesh.from_edit_mesh(ob.data)
            if not any(f.select for f in bm.faces):
                self.report({'ERROR'}, "Select the window/door faces to split first")
                return {'CANCELLED'}
            before = set(bpy.data.objects.keys())
            bpy.ops.mesh.duplicate()
            bpy.ops.mesh.separate(type='SELECTED')
            bpy.ops.object.mode_set(mode='OBJECT')
            new_objs = [o for k, o in bpy.data.objects.items() if k not in before]
        else:
            srcs = [o for o in context.selected_objects if o.type == 'MESH']
            if not srcs:
                self.report({'ERROR'}, "Select the window/door mesh(es) to split first")
                return {'CANCELLED'}
            dups = []
            for src in srcs:
                dup = src.copy()
                dup.data = src.data.copy()
                for c in src.users_collection:
                    c.objects.link(dup)
                dups.append(dup)
            bpy.ops.object.select_all(action='DESELECT')
            for o in dups:
                o.select_set(True)
            context.view_layer.objects.active = dups[-1]
            new_objs = dups

        # split whatever we now have into one object per disconnected piece
        before = set(bpy.data.objects.keys())
        bpy.ops.object.select_all(action='DESELECT')
        for o in new_objs:
            o.select_set(True)
        context.view_layer.objects.active = new_objs[-1]
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.separate(type='LOOSE')
        bpy.ops.object.mode_set(mode='OBJECT')
        result = [o for o in context.selected_objects]
        result += [o for k, o in bpy.data.objects.items()
                  if k not in before and o not in result]

        _collect_into(result, coll)
        bpy.ops.object.select_all(action='DESELECT')
        for o in result:
            o.select_set(True)
        if result:
            context.view_layer.objects.active = result[-1]
        self.report({'INFO'}, f"Split into {len(result)} opening piece(s) in '{OPENING_PIECES_COLL}'")
        return {'FINISHED'}


class GN_OT_project_openings(Operator):
    bl_idname = "gn_int.project_openings"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Project Selected Pieces"
    bl_description = ("Cut openings into the room walls from the SELECTED window/door "
                      "pieces only (planes or meshes). Additive; skips duplicates")

    def execute(self, context):
        s = context.scene.gn_int
        rooms = bpy.data.collections.get(ROOM_COLL)
        room_objs = set(rooms.objects) if rooms else set()
        pieces = [o for o in context.selected_objects
                  if o.type == 'MESH' and o not in room_objs]
        if not pieces:
            self.report({'ERROR'}, "Select the window/door pieces to project first")
            return {'CANCELLED'}
        made = skipped = 0
        for o in pieces:
            fr = _piece_frame(o)
            if not fr:
                continue
            center, u, v, n, hw, hh = fr
            sill, top = center.z - hh, center.z + hh
            # skip if an opening already exists at ~this spot (avoid duplicates)
            if any(abs(e.cx - center.x) < 0.1 and abs(e.cy - center.y) < 0.1
                   and abs(e.sill - sill) < 0.15 for e in s.openings):
                skipped += 1
                continue
            op = s.openings.add()
            op.uid = _new_uid(s)
            op.cx, op.cy = center.x, center.y
            op.nx, op.ny = n.x, n.y
            op.hw = hw + 0.01
            op.sill = sill
            op.top = top
            op.projected = True             # exterior piece -> no threshold
            # a piece that reaches (near) the ACTUAL floor of whichever room(s)
            # it borders is a door; otherwise a window. Checked against those
            # specific rooms' own base + z_offset, not any room in the scene --
            # a half-floor/mezzanine room's floor can coincidentally line up
            # with some OTHER unrelated room's nominal base
            bordering = set(_rooms_for_opening(context, op)) - {"limbo"}
            op.is_door = any(
                abs(sill - (fl[0] + r.z_offset)) < 0.3
                for r in s.rooms
                if _room_token_for_uid(context, r.uid) in bordering
                and (fl := _floor_by_index(context, r.floor_index)))
            made += 1
        rebuild_rooms(context)
        msg = f"Projected {made} selected opening(s)"
        if skipped:
            msg += f" ({skipped} already existed)"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


def _opening_matches_any_wall(context, o):
    """True if opening o still lines up with a wall edge of some CURRENT room
    on some floor -- same normal/distance/span test _wall_spans uses to punch
    the hole. False means the room it was placed on has since been split,
    resized, or removed out from under it."""
    s = context.scene.gn_int
    tops = _floor_tops(context)
    nrm = Vector((o.nx, o.ny))
    for r in s.rooms:
        if not (0 <= r.floor_index < len(tops)):
            continue
        base_z, ceil_z, _ = tops[r.floor_index]
        try:
            poly = [Vector(p) for p in json.loads(r.poly_json)]
        except Exception:
            continue
        n = len(poly)
        for i in range(n):
            A, B = poly[i], poly[(i + 1) % n]
            d = B - A
            L = d.length
            if L < 1e-6:
                continue
            d = d / L
            wn = Vector((-d.y, d.x))
            if nrm.length > 1e-6 and abs(nrm.normalized().dot(wn)) < 0.8:
                continue
            rel = Vector((o.cx, o.cy)) - A
            x = rel.dot(d)
            if abs(rel.dot(wn)) > 0.45 or x < -o.hw or x > L + o.hw:
                continue
            z0 = max(base_z, o.sill)
            z1 = min(ceil_z, o.top)
            if z1 - z0 > 0.02:
                return True
    return False


class GN_OT_clean_stale_openings(Operator):
    bl_idname = "gn_int.clean_stale_openings"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Delete Stale Openings"
    bl_description = ("Delete openings that no longer line up with any current "
                      "room wall (left behind after a room was split, resized, "
                      "or its boundary was regenerated). Also removes their "
                      "door frames/thresholds, then rebuilds rooms")

    def execute(self, context):
        s = context.scene.gn_int
        stale_idx = [i for i, o in enumerate(s.openings)
                    if not _opening_matches_any_wall(context, o)]
        if not stale_idx:
            self.report({'INFO'}, "No stale openings found")
            return {'CANCELLED'}
        for i in reversed(stale_idx):
            uid = s.openings[i].uid
            _remove_frame_mesh(uid)
            _remove_threshold(uid)
            s.openings.remove(i)
        rebuild_rooms(context)
        self.report({'INFO'}, f"Deleted {len(stale_idx)} stale opening(s)")
        return {'FINISHED'}


class GN_OT_clear_openings(Operator):
    bl_idname = "gn_int.clear_openings"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Clear Openings"

    def execute(self, context):
        context.scene.gn_int.openings.clear()
        _clear_coll(DOORFRAME_COLL)
        _clear_coll(WINDOWFRAME_COLL)
        _clear_coll(THRESHOLD_COLL)
        rebuild_rooms(context)
        return {'FINISHED'}


# ===========================================================================
# doors (Room-Tool-style edit mode)
# ===========================================================================
def _active_preset(s, kind):
    """Return (width, height, sill, mesh_object) for the active door/window preset."""
    if kind == 'DOOR':
        if 0 <= s.active_door_preset < len(s.door_presets):
            p = s.door_presets[s.active_door_preset]
            return p.width, p.height, 0.0, p.mesh_object
        return 0.9, 2.0, 0.0, None
    if 0 <= s.active_window_preset < len(s.window_presets):
        p = s.window_presets[s.active_window_preset]
        return p.width, p.height, p.sill, p.mesh_object
    return 1.0, 1.2, 0.9, None


def _room_base(obj):
    return min((v.co.z for v in obj.data.vertices), default=0.0)


def _opening_under(s, xy, z, want_door):
    for i, op in enumerate(s.openings):
        if op.is_door != want_door:
            continue
        n = Vector((op.nx, op.ny))
        along = Vector((-n.y, n.x))
        d = xy - Vector((op.cx, op.cy))
        if abs(d.dot(along)) < op.hw and abs(d.dot(n)) < 0.35 and \
           op.sill - 0.05 < z < op.top + 0.05:
            return i
    return -1


def _frame_coll(kind):
    return DOORFRAME_COLL if kind == 'DOOR' else WINDOWFRAME_COLL


def _place_frame_mesh(op, mesh_src, width, height, kind):
    """Instance a door/window frame mesh into the opening, oriented to the wall.
    Detects the mesh's own axes from its bounding box (Z=height, the larger
    horizontal extent=width, the smaller=depth), so any panel-like mesh orients
    correctly regardless of how it was modelled. Origin-agnostic (uses bbox)."""
    if mesh_src is None or not mesh_src.data:
        return
    coll = _get_coll(_frame_coll(kind))
    name = f"GN_Frame_{op.uid}"
    inst = bpy.data.objects.get(name)
    if inst is None:
        inst = bpy.data.objects.new(name, mesh_src.data)
        coll.objects.link(inst)
    else:
        inst.data = mesh_src.data
    bb = [Vector(c) for c in mesh_src.bound_box]     # local-space corners
    xs = [c.x for c in bb]; ys = [c.y for c in bb]; zs = [c.z for c in bb]
    dx, dy, dz = max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)
    cxl, cyl = (min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5
    zbot = min(zs)

    n = Vector((op.nx, op.ny, 0.0)).normalized()
    along = Vector((-n.y, n.x, 0.0))
    up = Vector((0.0, 0.0, 1.0))
    sz = height / dz if dz > 1e-4 else 1.0
    if dx >= dy:                                     # width = local X, depth = local Y
        colX = along * (width / dx if dx > 1e-4 else 1.0)
        colY = n
    else:                                            # width = local Y, depth = local X
        colX = n
        colY = along * (width / dy if dy > 1e-4 else 1.0)
    colZ = up * sz

    def _mat(cX, cY, cZ):
        return Matrix(((cX.x, cY.x, cZ.x, 0.0),
                       (cX.y, cY.y, cZ.y, 0.0),
                       (cX.z, cY.z, cZ.z, 0.0),
                       (0.0, 0.0, 0.0, 1.0)))
    R = _mat(colX, colY, colZ)
    if R.to_3x3().determinant() < 0:                 # keep right-handed (no mirrored normals)
        if dx >= dy:
            colY = -colY
        else:
            colX = -colX
        R = _mat(colX, colY, colZ)
    # place bbox centre (width/depth) at the opening centre, bottom at the sill
    R.translation = Vector((op.cx, op.cy, op.sill)) - R.to_3x3() @ Vector((cxl, cyl, zbot))
    inst.matrix_world = R


def _remove_frame_mesh(uid):
    for c in (DOORFRAME_COLL, WINDOWFRAME_COLL):
        inst = bpy.data.objects.get(f"GN_Frame_{uid}")
        if inst:
            bpy.data.objects.remove(inst, do_unlink=True)
            return


def _remove_threshold(uid):
    ob = bpy.data.objects.get(f"GN_Threshold_{uid}")
    if ob:
        bpy.data.objects.remove(ob, do_unlink=True)


def _refresh_thresholds(context):
    """Rebuild threshold strips for all door openings (or clear them if disabled).
    The strip sits inside ONE room (the side the door normal points to, flippable),
    not bridging both. It spans from the gap centre into that room by threshold_depth."""
    s = context.scene.gn_int
    _clear_coll(THRESHOLD_COLL)
    if not s.add_threshold:
        return
    coll = _get_coll(THRESHOLD_COLL)
    for op in s.openings:
        if not op.is_door or op.projected:      # only interior doors get a threshold
            continue
        n = Vector((op.nx, op.ny, 0.0)).normalized()   # points into the room it was placed in
        along = Vector((-n.y, n.x, 0.0))
        up = Vector((0.0, 0.0, 1.0))
        hw = op.hw
        gap = max(s.partition, 0.0)                     # distance across the doorway
        depth = gap * 0.5 + max(s.threshold_depth, 0.0)  # from the boundary into the room
        h = max(s.threshold_height, 1e-4)
        # ORIGIN sits on the boundary where the two rooms meet (centre of the gap);
        # the strip lies on ONE side of it, so the pivot is on the strip's EDGE.
        ox = op.cx - n.x * (gap * 0.5)
        oy = op.cy - n.y * (gap * 0.5)
        bm = bmesh.new()
        vlo = [bm.verts.new((-hw, 0.0, 0.0)), bm.verts.new((hw, 0.0, 0.0)),
               bm.verts.new((hw, depth, 0.0)), bm.verts.new((-hw, depth, 0.0))]
        vhi = [bm.verts.new((v.co.x, v.co.y, h)) for v in vlo]
        bm.faces.new(vlo)
        bm.faces.new(vhi[::-1])
        for k in range(4):
            bm.faces.new((vlo[k], vlo[(k + 1) % 4], vhi[(k + 1) % 4], vhi[k]))
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        me = bpy.data.meshes.new(f"GN_Threshold_{op.uid}")
        bm.to_mesh(me); bm.free()
        ob = bpy.data.objects.new(f"GN_Threshold_{op.uid}", me)
        coll.objects.link(ob)
        ob.matrix_world = Matrix((
            (along.x, n.x, up.x, ox),
            (along.y, n.y, up.y, oy),
            (along.z, n.z, up.z, op.sill),
            (0.0, 0.0, 0.0, 1.0)))


def add_opening(context, xy, normal, base, kind):
    s = context.scene.gn_int
    w, h, sill, mesh = _active_preset(s, kind)
    op = s.openings.add()
    op.uid = _new_uid(s)
    op.is_door = (kind == 'DOOR')
    op.cx, op.cy = xy.x, xy.y
    op.nx, op.ny = normal.x, normal.y
    op.hw = w * 0.5
    op.sill = base + sill
    op.top = base + sill + h
    rebuild_rooms(context)                      # also refreshes thresholds
    _place_frame_mesh(op, mesh, w, h, kind)


def remove_opening(context, idx):
    s = context.scene.gn_int
    uid = s.openings[idx].uid
    _remove_frame_mesh(uid)
    _remove_threshold(uid)
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


def _draw_opening_ghost(self, context):
    try:
        if not getattr(self, "hover", None):
            return
        _, xy, n, base = self.hover
        s = context.scene.gn_int
        w, h, sill, _ = _active_preset(s, self.kind)
        along = Vector((-n.y, n.x))
        a = xy - along * (w * 0.5)
        b = xy + along * (w * 0.5)
        z0, z1 = base + sill, base + sill + h
        pts = [(a.x, a.y, z0), (b.x, b.y, z0), (b.x, b.y, z1), (a.x, a.y, z1)]
        color = (1.0, 0.3, 0.3, 1.0) if self.remove_hover else (0.1, 1.0, 0.3, 1.0)
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(2.5)
        batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": pts})
        shader.bind()
        shader.uniform_float("color", color)
        batch.draw(shader)
        gpu.state.line_width_set(1.0)
    except Exception:
        pass


def _presets_for(s, kind):
    return (s.door_presets, "active_door_preset") if kind == 'DOOR' \
        else (s.window_presets, "active_window_preset")


class GN_OT_opening_edit(Operator):
    bl_idname = "gn_int.opening_edit"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Opening Edit Mode"
    bl_description = ("Hover a wall to preview; LMB = add, LMB on an opening = remove, "
                      "Tab = next preset, Esc/RMB = exit")
    kind: EnumProperty(items=[('DOOR', "Door", ""), ('WINDOW', "Window", "")],
                       default='DOOR', options={'HIDDEN'})

    def invoke(self, context, event):
        s = context.scene.gn_int
        presets, _ = _presets_for(s, self.kind)
        if not presets:
            p = presets.add()
            p.name = self.kind.title()
        self.hover = None
        self.remove_hover = False
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_opening_ghost, (self, context), 'WINDOW', 'POST_VIEW')
        context.window_manager.modal_handler_add(self)
        context.area.header_text_set(
            f"{self.kind.title()} Edit: LMB add · LMB on one to remove · Tab preset · Esc exit")
        return {'RUNNING_MODAL'}

    def _cleanup(self, context):
        try:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
        except Exception:
            pass
        if context.area:
            context.area.header_text_set(None)
            context.area.tag_redraw()

    def modal(self, context, event):
        try:
            if context.area:
                context.area.tag_redraw()
            s = context.scene.gn_int
            want_door = (self.kind == 'DOOR')
            if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
                return {'PASS_THROUGH'}
            if event.type == 'MOUSEMOVE':
                w = _wall_under_cursor(context, event)
                if w:
                    xy, n, base = w
                    self.hover = (None, xy, n, base)
                    self.remove_hover = _opening_under(s, xy, base + 0.1, want_door) >= 0
                else:
                    self.hover = None
                    self.remove_hover = False
                return {'RUNNING_MODAL'}
            if event.type == 'TAB' and event.value == 'PRESS':
                presets, attr = _presets_for(s, self.kind)
                if presets:
                    setattr(s, attr, (getattr(s, attr) + 1) % len(presets))
                return {'RUNNING_MODAL'}
            if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
                w = _wall_under_cursor(context, event)
                if w:
                    xy, n, base = w
                    idx = _opening_under(s, xy, base + 0.1, want_door)
                    if idx >= 0:
                        remove_opening(context, idx)
                    else:
                        add_opening(context, xy, n, base, self.kind)
                return {'RUNNING_MODAL'}
            if event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
                self._cleanup(context)
                return {'FINISHED'}
            return {'RUNNING_MODAL'}
        except Exception as e:
            print("[GN Interior] opening_edit error:", e)
            self._cleanup(context)
            return {'CANCELLED'}


class GN_OT_preset_add(Operator):
    bl_idname = "gn_int.preset_add"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Add Preset"
    kind: EnumProperty(items=[('DOOR', "Door", ""), ('WINDOW', "Window", "")],
                       default='DOOR', options={'HIDDEN'})

    def execute(self, context):
        s = context.scene.gn_int
        presets, attr = _presets_for(s, self.kind)
        p = presets.add()
        p.name = f"{self.kind.title()} {len(presets)}"
        setattr(s, attr, len(presets) - 1)
        return {'FINISHED'}


class GN_OT_preset_remove(Operator):
    bl_idname = "gn_int.preset_remove"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Remove Preset"
    kind: EnumProperty(items=[('DOOR', "Door", ""), ('WINDOW', "Window", "")],
                       default='DOOR', options={'HIDDEN'})

    def execute(self, context):
        s = context.scene.gn_int
        presets, attr = _presets_for(s, self.kind)
        if presets:
            presets.remove(getattr(s, attr))
            setattr(s, attr, max(0, getattr(s, attr) - 1))
        return {'FINISHED'}


class GN_OT_remove_opening(Operator):
    bl_idname = "gn_int.remove_opening"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Delete Opening"
    bl_description = "Delete the selected opening (its hole, frame mesh and threshold)"
    index: IntProperty(default=-1)

    def execute(self, context):
        s = context.scene.gn_int
        i = self.index if self.index >= 0 else s.opening_index
        if 0 <= i < len(s.openings):
            remove_opening(context, i)
            s.opening_index = min(i, len(s.openings) - 1)
            return {'FINISHED'}
        self.report({'WARNING'}, "No opening selected")
        return {'CANCELLED'}


# ===========================================================================
# UI
# ===========================================================================
class GN_UL_door_presets(bpy.types.UIList):
    def draw_item(self, ctx, layout, data, item, icon, adata, aprop, index=0, flt=0):
        layout.prop(item, "name", text="", emboss=False, icon='MESH_DATA')
        layout.label(text=f"{item.width:.2f}x{item.height:.2f}")


class GN_UL_openings(bpy.types.UIList):
    def draw_item(self, ctx, layout, data, item, icon, adata, aprop, index=0, flt=0):
        row = layout.row(align=True)
        is_door = item.is_door
        row.label(text=("Door" if is_door else "Window"),
                  icon='MOD_BEVEL' if is_door else 'MOD_LATTICE')
        row.label(text=f"({item.cx:.1f}, {item.cy:.1f})  w{item.hw*2:.2f}")
        op = row.operator("gn_int.remove_opening", text="", icon='X')
        op.index = index

    def filter_items(self, ctx, data, propname):
        items = getattr(data, propname)
        flt = [self.bitflag_filter_item] * len(items)
        mode = ctx.scene.gn_int.opening_filter
        if mode != 'ALL':
            want_door = (mode == 'DOOR')
            for i, it in enumerate(items):
                if it.is_door != want_door:
                    flt[i] &= ~self.bitflag_filter_item
        return flt, []


class GN_UL_floors(bpy.types.UIList):
    def draw_item(self, ctx, layout, data, item, icon, adata, aprop, index=0, flt=0):
        s = ctx.scene.gn_int
        h = (item.top - item.z) if item.top_is_custom else s.room_height
        row = layout.row(align=True)
        row.label(text=f"Floor {index+1}", icon='DECORATE')
        row.label(text=f"z {item.z:.2f}  h {h:.2f}m")
        # lock: keep a hand-edited boundary (Generate skips locked floors)
        row.prop(item, "lock", text="",
                 icon='LOCKED' if item.lock else 'UNLOCKED', emboss=False)
        op = row.operator("gn_int.remove_floor", text="", icon='TRASH')
        op.index = index


class _PanelBase:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GN Interior"


class GN_PT_interior(_PanelBase, Panel):
    bl_label = "Interior Tool"

    def draw(self, context):
        s = context.scene.gn_int
        row = self.layout.row(align=True)
        row.prop(s, "exterior", text="Shell")
        row.operator("gn_int.set_exterior", text="", icon='EYEDROPPER')
        row = self.layout.row()
        row.alert = True
        row.operator("gn_int.clean_interior", icon='TRASH')


class GN_PT_setup(_PanelBase, Panel):
    bl_parent_id = "GN_PT_interior"
    bl_label = "Room Setup"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        s = context.scene.gn_int
        col = self.layout.column(align=True)
        col.prop(s, "room_height")
        col.prop(s, "uv_scale")


class GN_PT_floors(_PanelBase, Panel):
    bl_parent_id = "GN_PT_interior"
    bl_label = "Floors & Boundaries"

    def draw(self, context):
        s = context.scene.gn_int
        layout = self.layout
        layout.template_list("GN_UL_floors", "", s, "floors", s, "floor_index", rows=3)
        if 0 <= s.floor_index < len(s.floors):
            f = s.floors[s.floor_index]
            row = layout.row(align=True)
            row.prop(f, "top_is_custom", text=f"Custom height (Floor {s.floor_index+1})")
            sub = row.row(align=True)
            sub.enabled = f.top_is_custom
            sub.prop(f, "top", text="Top Z")
        layout.operator("gn_int.add_floor_sel", text="From Edge", icon='EDGESEL')
        layout.operator("gn_int.pick_floor_z", text="Slice for Floor Height",
                        icon='EMPTY_SINGLE_ARROW')
        r2 = layout.row(align=True)
        r2.prop(s, "new_floor_z")
        r2.operator("gn_int.add_floor_z", text="Add at Z")

        box = layout.box()
        box.label(text="Boundary cleanup", icon='MOD_BEVEL')
        col = box.column(align=True)
        col.prop(s, "detail_tol")
        col.prop(s, "bridge")
        col.prop(s, "wall_margin")
        col.prop(s, "sample_offset")
        box.label(text="Lock a floor to keep hand-edited boundaries", icon='INFO')

        layout.operator("gn_int.gen_boundaries", icon='MESH_GRID')
        layout.operator("gn_int.seed_rooms", icon='MESH_PLANE')
        layout.operator("gn_int.clear", icon='TRASH')


class GN_PT_rooms(_PanelBase, Panel):
    bl_parent_id = "GN_PT_interior"
    bl_label = "Rooms"

    def draw(self, context):
        s = context.scene.gn_int
        layout = self.layout
        layout.prop(s, "partition")
        layout.operator("gn_int.split_edges", icon='MOD_BEVEL')
        layout.label(text="Edit Mode: pick 2 wall edges, then Split", icon='INFO')
        layout.operator("gn_int.split_room_path", icon='GP_MULTIFRAME_EDITING')
        layout.label(text="Or click a bent path (L-shape) across a room", icon='INFO')
        layout.label(text=f"{len(s.rooms)} room(s)")
        row = layout.row(align=True)
        row.operator("gn_int.rebuild_rooms", icon='FILE_REFRESH')
        row.operator("gn_int.clear_rooms", icon='TRASH')
        layout.operator("gn_int.reunwrap", icon='UV')


class GN_PT_stairs(_PanelBase, Panel):
    bl_parent_id = "GN_PT_interior"
    bl_label = "Stairs"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        s = context.scene.gn_int
        layout = self.layout
        col = layout.column(align=True)
        col.prop(s, "stair_step_height")
        col.prop(s, "stair_step_depth")
        layout.operator("gn_int.create_stairs", icon='MOD_ARRAY')
        layout.label(text="Edit Mode: pick 1 edge at the bottom, 1 at the", icon='INFO')
        layout.label(text="top (can be on 2 different objects), then run")


class GN_PT_openings(_PanelBase, Panel):
    bl_parent_id = "GN_PT_interior"
    bl_label = "Openings (project)"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        s = context.scene.gn_int
        layout = self.layout
        layout.operator("gn_int.split_opening_pieces", icon='MOD_EXPLODE')
        layout.label(text="Select window/door faces or objects, then split", icon='INFO')
        layout.separator()
        layout.prop(s, "reveal")
        layout.operator("gn_int.project_openings", icon='SELECT_DIFFERENCE')
        layout.label(text="Select window/door pieces, then project", icon='INFO')
        if len(s.openings):
            layout.label(text=f"{len(s.openings)} opening(s) — X deletes one:")
            layout.prop(s, "opening_filter", expand=True)
            layout.template_list("GN_UL_openings", "", s, "openings",
                                 s, "opening_index", rows=4)
        row = layout.row(align=True)
        row.operator("gn_int.remove_opening", text="Delete Selected", icon='X').index = -1
        row.operator("gn_int.clear_openings", text="Clear All", icon='TRASH')
        layout.operator("gn_int.clean_stale_openings", icon='ORPHAN_DATA')


class GN_PT_doors(_PanelBase, Panel):
    bl_parent_id = "GN_PT_interior"
    bl_label = "Doors"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        s = context.scene.gn_int
        layout = self.layout
        row = layout.row()
        row.template_list("GN_UL_door_presets", "doors", s, "door_presets",
                          s, "active_door_preset", rows=2)
        col = row.column(align=True)
        col.operator("gn_int.preset_add", text="", icon='ADD').kind = 'DOOR'
        col.operator("gn_int.preset_remove", text="", icon='REMOVE').kind = 'DOOR'
        if 0 <= s.active_door_preset < len(s.door_presets):
            dp = s.door_presets[s.active_door_preset]
            layout.prop(dp, "width")
            layout.prop(dp, "height")
            layout.prop(dp, "mesh_object")
        layout.operator("gn_int.opening_edit", text="Door Edit Mode",
                        icon='GREASEPENCIL').kind = 'DOOR'
        layout.prop(s, "add_threshold")
        if s.add_threshold:
            r = layout.row(align=True)
            r.prop(s, "threshold_height", text="H")
            r.prop(s, "threshold_depth", text="D")


class GN_PT_windows(_PanelBase, Panel):
    bl_parent_id = "GN_PT_interior"
    bl_label = "Windows"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        s = context.scene.gn_int
        layout = self.layout
        row = layout.row()
        row.template_list("GN_UL_door_presets", "wins", s, "window_presets",
                          s, "active_window_preset", rows=2)
        col = row.column(align=True)
        col.operator("gn_int.preset_add", text="", icon='ADD').kind = 'WINDOW'
        col.operator("gn_int.preset_remove", text="", icon='REMOVE').kind = 'WINDOW'
        if 0 <= s.active_window_preset < len(s.window_presets):
            wp = s.window_presets[s.active_window_preset]
            layout.prop(wp, "width")
            layout.prop(wp, "height")
            layout.prop(wp, "sill")
            layout.prop(wp, "mesh_object")
        layout.operator("gn_int.opening_edit", text="Window Edit Mode",
                        icon='GREASEPENCIL').kind = 'WINDOW'


class GN_PT_mlo(_PanelBase, Panel):
    bl_parent_id = "GN_PT_interior"
    bl_label = "MLO Setup"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        s = context.scene.gn_int
        layout = self.layout
        layout.label(text="Run once, after rooms/doors are finished", icon='INFO')
        layout.prop(s, "mlo_name")
        layout.prop(s, "timecycle_name")
        layout.operator("gn_int.build_mlo", icon='OUTLINER_COLLECTION')
        row = layout.row()
        row.alert = True
        row.operator("gn_int.clean_mlo", icon='TRASH')


class GN_PT_manual_setup(_PanelBase, Panel):
    bl_parent_id = "GN_PT_mlo"
    bl_label = "Manual Setup"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.label(text="Add anything that wasn't included in Build MLO", icon='INFO')
        layout.operator("gn_int.add_room_collections", icon='OUTLINER_COLLECTION')
        layout.operator("gn_int.add_prop_collections", icon='OUTLINER_OB_GROUP_INSTANCE')
        layout.operator("gn_int.add_asset_collections", icon='ASSET_MANAGER')
        layout.separator()
        layout.operator("gn_int.create_shell_collision", icon='MESH_ICOSPHERE')
        if GN_OT_create_shell_collision.poll(context) is False:
            layout.label(text="Needs the Sollumz add-on", icon='ERROR')
        layout.operator("gn_int.create_portals", icon='OUTLINER_OB_LIGHTPROBE')
        portals_col = _mlo_portals_collection(context.scene)
        if portals_col and portals_col.objects:
            layout.label(text=f"{len(portals_col.objects)} portal(s):")
            layout.template_list("GN_UL_portals", "", portals_col, "objects",
                                 context.scene.gn_int, "portal_index", rows=4)
            row = layout.row(align=True)
            row.operator("gn_int.flip_portal", icon='ARROW_LEFTRIGHT')
            row.operator("gn_int.remove_portal", icon='X')
            layout.operator("gn_int.clear_portals", icon='TRASH')
            layout.label(text="Click a portal's name in the list to rename it", icon='INFO')


class GN_PT_add_empties(_PanelBase, Panel):
    bl_parent_id = "GN_PT_manual_setup"
    bl_label = "Add Empties"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        s = context.scene.gn_int
        layout = self.layout
        layout.prop(s, "empty_target_room", icon='OUTLINER_COLLECTION')
        layout.separator()
        box = layout.box()
        box.label(text="Presets:", icon='EMPTY_AXIS')
        grid = box.grid_flow(row_major=True, columns=2, even_columns=True, align=True)
        for preset in PRESET_EMPTIES:
            grid.prop(s, _PRESET_ATTR[preset], toggle=True)
        layout.separator()
        box = layout.box()
        box.label(text="Custom:", icon='ADD')
        for i, item in enumerate(s.custom_empties):
            row = box.row(align=True)
            row.prop(item, "enabled", text="")
            row.prop(item, "name", text="")
            op = row.operator("gn_int.remove_custom_empty", text="", icon='X')
            op.index = i
        row = box.row(align=True)
        row.prop(s, "empty_custom_name", text="")
        row.operator("gn_int.add_custom_empty", text="", icon='ADD')
        layout.separator()
        layout.operator("gn_int.add_empties", icon='EMPTY_AXIS')


class GN_PT_smart_rename(_PanelBase, Panel):
    bl_parent_id = "GN_PT_manual_setup"
    bl_label = "Smart Rename"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        s = context.scene.gn_int
        layout = self.layout
        box = layout.box()
        row = box.row()
        row.label(text="MLO:", icon='SCENE_DATA')
        row.label(text=s.mlo_name if s.mlo_name else "-- not set --")
        box.prop(s, "sr_room", icon='OUTLINER_COLLECTION')
        box.prop(s, "sr_category", icon='FILTER')
        if s.sr_category == 'custom':
            box.prop(s, "sr_category_custom", text="", icon='GREASEPENCIL')
        box.separator()
        cat = s.sr_category_custom.strip() if s.sr_category == 'custom' else s.sr_category
        room = s.sr_room if s.sr_room != 'NONE' else '???'
        mlo = s.mlo_name if s.mlo_name else '???'
        box.label(text=f"-> {mlo}_{room}_{cat}", icon='INFO')
        layout.separator()
        layout.prop(s, "sr_merge", icon='AUTOMERGE_ON')
        layout.separator()
        layout.operator("gn_int.smart_rename", icon='SORTALPHA')


class GN_PT_create_asset(_PanelBase, Panel):
    bl_parent_id = "GN_PT_manual_setup"
    bl_label = "Create Asset"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        s = context.scene.gn_int
        layout = self.layout
        layout.label(text="Select mesh, set options, click Create.", icon='INFO')
        layout.label(text="Asset keeps the selected object's name.", icon='INFO')
        layout.prop(s, "asset_type", expand=True)
        layout.prop(s, "auto_collision")
        layout.operator("gn_int.create_asset", icon='ADD')
        if GN_OT_create_asset.poll(context) is False:
            layout.label(text="Needs the Sollumz add-on + a selected mesh", icon='ERROR')


# ===========================================================================
_classes = (
    GN_FloorLevel, GN_Room, GN_Opening, GN_DoorPreset, GN_WindowPreset,
    GN_CustomEmptyItem, GN_IntProps,
    GN_CollMatSearchItem, GN_ShellCollMappingItem,
    GN_OT_set_exterior, GN_OT_add_floor_sel,
    GN_OT_add_floor_z, GN_OT_pick_floor_z,
    GN_OT_remove_floor, GN_OT_gen_boundaries, GN_OT_clear, GN_OT_clean_interior,
    GN_OT_draw_room, GN_OT_add_room, GN_OT_remove_room, GN_OT_rebuild_rooms,
    GN_OT_clear_rooms, GN_OT_seed_rooms, GN_OT_split_room, GN_OT_split_room_path,
    GN_OT_split_edges, GN_OT_create_stairs,
    GN_OT_reunwrap, GN_OT_build_mlo, GN_OT_clean_mlo,
    GN_OT_add_room_collections, GN_OT_add_prop_collections, GN_OT_add_asset_collections,
    GN_OT_create_shell_collision,
    GN_OT_create_portals, GN_OT_remove_portal, GN_OT_flip_portal, GN_OT_clear_portals,
    GN_OT_add_empties, GN_OT_add_custom_empty, GN_OT_remove_custom_empty,
    GN_OT_smart_rename, GN_OT_create_asset,
    GN_OT_split_opening_pieces, GN_OT_project_openings, GN_OT_clear_openings,
    GN_OT_opening_edit, GN_OT_preset_add, GN_OT_preset_remove, GN_OT_remove_opening,
    GN_OT_clean_stale_openings,
    GN_UL_door_presets, GN_UL_openings, GN_UL_floors, GN_UL_shell_coll_mappings,
    GN_UL_portals,
    GN_PT_interior, GN_PT_setup, GN_PT_floors, GN_PT_rooms, GN_PT_stairs,
    GN_PT_openings, GN_PT_doors, GN_PT_windows, GN_PT_mlo, GN_PT_manual_setup,
    GN_PT_add_empties, GN_PT_smart_rename, GN_PT_create_asset,
)


_SETTINGS_KEYS = ("wall_margin", "room_height", "sample_offset",
                  "cleanup", "detail_tol", "bridge", "square", "ang_tol",
                  "allow45", "partition", "reveal", "uid_counter", "uv_scale",
                  "active_floor", "snap", "active_door_preset",
                  "active_window_preset", "add_threshold", "threshold_height",
                  "threshold_depth", "threshold_flip", "threshold_offset",
                  "mlo_name", "timecycle_name", "timecycle_auto",
                  "build_main", "build_room_colls", "build_prop_colls",
                  "build_asset_colls", "build_shell_collision", "build_portals",
                  "build_empties", "stair_step_height", "stair_step_depth")


def _dump_scene(scene):
    """Serialize interior data to a plain ID-property that survives add-on reload."""
    if not hasattr(scene, "gn_int"):
        return
    s = scene.gn_int
    data = {
        "settings": {k: getattr(s, k) for k in _SETTINGS_KEYS},
        "exterior": s.exterior.name if s.exterior else "",
        "floors": [{"z": f.z, "top": f.top, "top_is_custom": f.top_is_custom,
                    "bound_json": f.bound_json, "lock": f.lock} for f in s.floors],
        "rooms": [{"floor_index": r.floor_index, "uid": r.uid,
                   "poly_json": r.poly_json} for r in s.rooms],
        "openings": [{k: getattr(o, k) for k in
                      ("cx", "cy", "nx", "ny", "hw", "sill", "top", "uid",
                       "is_door", "projected")}
                     for o in s.openings],
        "door_presets": [{"name": p.name, "width": p.width, "height": p.height,
                          "mesh": p.mesh_object.name if p.mesh_object else ""}
                         for p in s.door_presets],
        "window_presets": [{"name": p.name, "width": p.width, "height": p.height,
                            "sill": p.sill,
                            "mesh": p.mesh_object.name if p.mesh_object else ""}
                           for p in s.window_presets],
    }
    scene["gn_int_backup"] = json.dumps(data)


def _restore_scene(scene):
    global _SUSPEND_CB
    raw = scene.get("gn_int_backup")
    if not raw:
        return
    try:
        data = json.loads(raw)
    except Exception:
        return
    s = scene.gn_int
    _SUSPEND_CB = True                       # don't fire update callbacks during restore
    try:
        for k, v in data.get("settings", {}).items():
            try:
                setattr(s, k, v)
            except Exception:
                pass
    finally:
        _SUSPEND_CB = False
    exn = data.get("exterior")
    if exn and exn in bpy.data.objects:
        s.exterior = bpy.data.objects[exn]
    s.floors.clear()
    for f in data.get("floors", []):
        it = s.floors.add()
        it.z = f["z"]; it.top = f["top"]; it.bound_json = f.get("bound_json", "")
        it.top_is_custom = f.get("top_is_custom", False)
        it.lock = f.get("lock", False)
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
    s.window_presets.clear()
    for p in data.get("window_presets", []):
        it = s.window_presets.add()
        it.name = p.get("name", "Window")
        it.width = p.get("width", 1.0)
        it.height = p.get("height", 1.2)
        it.sill = p.get("sill", 0.9)
        mn = p.get("mesh", "")
        if mn and mn in bpy.data.objects:
            it.mesh_object = bpy.data.objects[mn]


_HL_HANDLE = None


def _draw_opening_highlight():
    """Persistent overlay: outline the opening currently selected in the list."""
    try:
        s = bpy.context.scene.gn_int
        if not (0 <= s.opening_index < len(s.openings)):
            return
        op = s.openings[s.opening_index]
        n = Vector((op.nx, op.ny))
        along = Vector((-n.y, n.x))
        c = Vector((op.cx, op.cy))
        a = c - along * op.hw
        b = c + along * op.hw
        z0, z1 = op.sill, op.top
        pts = [(a.x, a.y, z0), (b.x, b.y, z0), (b.x, b.y, z1), (a.x, a.y, z1)]
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(3.0)
        batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": pts})
        shader.bind()
        shader.uniform_float("color", (1.0, 0.85, 0.1, 1.0))
        batch.draw(shader)
        gpu.state.line_width_set(1.0)
    except Exception:
        pass


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
    global _HL_HANDLE
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.gn_int = PointerProperty(type=GN_IntProps)
    bpy.types.Scene.gn_shell_coll_active_idx = IntProperty(default=0)
    bpy.types.Scene.gn_coll_mat_search_items = CollectionProperty(
        type=GN_CollMatSearchItem,
        description="Searchable list of Sollumz collision material names")
    bpy.types.Scene.gn_shell_coll_mappings = CollectionProperty(
        type=GN_ShellCollMappingItem,
        description="Temporary material->collision mapping for Create Shell Collision")
    bpy.app.timers.register(_deferred_restore, first_interval=0.0)
    if _HL_HANDLE is None:
        _HL_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
            _draw_opening_highlight, (), 'WINDOW', 'POST_VIEW')


def unregister():
    global _HL_HANDLE
    if _HL_HANDLE is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_HL_HANDLE, 'WINDOW')
        except Exception:
            pass
        _HL_HANDLE = None
    try:
        for scene in bpy.data.scenes:           # best-effort backup before delete
            _dump_scene(scene)
    except Exception:
        pass
    del bpy.types.Scene.gn_int
    del bpy.types.Scene.gn_shell_coll_active_idx
    del bpy.types.Scene.gn_coll_mat_search_items
    del bpy.types.Scene.gn_shell_coll_mappings
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
