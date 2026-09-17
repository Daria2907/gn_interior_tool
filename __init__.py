"""
UltimateMLO - exterior-aware interior generator for GTA MLO.

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
    "name": "UltimateMLO",
    "author": "GN's Studio + Claude",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport > Sidebar (N) > UltimateMLO",
    "description": "Exterior-aware interior shell + per-floor Floor Map for GTA MLO.",
    "category": "Object",
}

import bpy
import bmesh
import math
import json
import time
import shutil
import os
import re
import numpy as np
import gpu
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
from mathutils import Vector, Matrix
from mathutils.geometry import intersect_line_plane
from bpy.props import (FloatProperty, IntProperty, StringProperty, BoolProperty,
                       PointerProperty, CollectionProperty, EnumProperty)
from bpy.types import Operator, Panel, PropertyGroup, AddonPreferences

BOUND_COLL = "GN_FloorMap"
INT_COLL = "GN_Interiors"
ROOM_COLL = "GN_Rooms"
CUTTER_COLL = "GN_Cutters"
DOORFRAME_COLL = "GN_DoorFrames"
WINDOWFRAME_COLL = "GN_WindowFrames"
THRESHOLD_COLL = "GN_Thresholds"
OPENING_PIECES_COLL = "Openings"
STAIR_COLL = "GN_Stairs"
WINLIB_COLL = "GN_WindowLibrary"
CURTAIN_COLL = "GN_Curtains"
# wall hole is cut this much SMALLER than the placed window frame on every
# side, so the frame overlaps the rough cut edge (like real window casing
# over a rough opening) instead of leaving a visible sliver of the hole
FRAME_OVERLAP = 0.01
# slack when testing whether a window fits an opening -- float noise, not
# architecture, and a half-millimetre shortfall shouldn't rule a window out
FIT_TOL = 0.002


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


def _shell_dom_orient(ob):
    """Length-weighted dominant orientation (radians, mod 90deg) of the WHOLE
    exterior shell, in world space -- computed once from the building as a
    whole rather than per-floor from each floor's own cross-section, so
    every floor's boundary snaps to the exact same grid instead of each one
    independently estimating a slightly different angle (which is what made
    two floors' boundaries come out a degree or so out of parallel)."""
    me = ob.data
    mw = ob.matrix_world
    sx = sy = 0.0
    for e in me.edges:
        a = mw @ me.vertices[e.vertices[0]].co
        b = mw @ me.vertices[e.vertices[1]].co
        dx, dy = b.x - a.x, b.y - a.y
        L = math.hypot(dx, dy)
        if L < 1e-6:
            continue
        ang = math.atan2(dy, dx) % (math.pi / 2)
        sx += L * math.cos(4 * ang); sy += L * math.sin(4 * ang)
    if sx == 0 and sy == 0:
        return 0.0
    return (math.atan2(sy, sx) / 4.0) % (math.pi / 2)


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


def regularize(poly, ang_tol_deg=20.0, min_edge=0.20, corner_max=0.7, allow45=False, th0=None):
    """Straighten walls to the dominant axis and drop short off-grid corner
    chamfers so corners are the two long walls meeting directly. Genuine long
    angled walls keep their true angle. th0 (radians) can be passed in to use
    a shared building-wide orientation instead of estimating one from just
    this polygon -- keeps multiple floors' boundaries mutually parallel."""
    n = len(poly)
    if n < 4:
        return poly
    if th0 is None:
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

    def fit_dir(points, fallback):
        # true best-fit direction through every point of a run (principal
        # axis of their spread), not just the first edge's own angle -- a
        # long wall walked as many small raster-quantized segments was
        # otherwise coming out tilted by whatever that one edge's noise was,
        # visibly off the wall's real line even after "snapping".
        pts = np.array(points, dtype=np.float64)
        d = pts - pts.mean(axis=0)
        cov = d.T @ d
        if not np.all(np.isfinite(cov)):
            return fallback
        evals, evecs = np.linalg.eigh(cov)
        v = evecs[:, int(np.argmax(evals))]
        vx, vy = float(v[0]), float(v[1])
        if vx * fallback[0] + vy * fallback[1] < 0:
            vx, vy = -vx, -vy
        return (vx, vy)

    start = 0
    for i in range(m):
        if not same_dir(edges[i - 1], edges[i]):
            start = i; break
    runs = []; i = 0; order = [(start + k) % m for k in range(m)]
    while i < m:
        j = order[i]; dirx, diry = edges[j][4], edges[j][5]
        wx = wy = wsum = 0.0; length = 0.0; k = i
        pts_for_fit = []
        while k < m and same_dir(edges[order[k]], edges[j]):
            e = edges[order[k]]
            mx = (e[0] + e[2]) / 2; my = (e[1] + e[3]) / 2
            wx += mx * e[6]; wy += my * e[6]; wsum += e[6]; length += e[6]; k += 1
            pts_for_fit.append((e[0], e[1])); pts_for_fit.append((e[2], e[3]))
        snp = edges[j][7]
        if len(pts_for_fit) >= 3:
            dirx, diry = fit_dir(pts_for_fit, (dirx, diry))
            a = math.atan2(diry, dirx); bestd = 1e9; best = None
            for g in grid:
                for gg in (g, g + math.pi):
                    dd = abs(((a - gg + math.pi) % (2 * math.pi)) - math.pi)
                    if dd < bestd:
                        bestd = dd; best = gg
            if bestd <= tol:
                dirx, diry = math.cos(best), math.sin(best); snp = True
        runs.append([(wx / wsum, wy / wsum), (dirx, diry), length, snp])
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


def _rectify_diagonals(poly, th0, ang_tol_deg=2.0, max_len=3.0, min_len=0.2):
    """Replace any short edge that isn't aligned to the building's grid with
    a right-angle step along that grid instead of leaving it as a diagonal
    cut -- these are raster-staircase artifacts through a real notch/step in
    the wall, not genuine angled architecture (which is long and gets left
    alone via max_len). Keeps the notch's shape and position, just made of
    two straight, grid-aligned segments instead of one diagonal one.
    Edges shorter than min_len are pure noise below the level of any real
    feature -- welded away instead of turned into an even-more-visible tiny
    right-angle tooth."""
    ux, uy = math.cos(th0), math.sin(th0)
    vx, vy = -uy, ux
    tol = math.radians(ang_tol_deg)
    n = len(poly)
    out = []
    skip_next = False
    for i in range(n):
        a = poly[i]; b = poly[(i + 1) % n]
        if skip_next:
            skip_next = False
        else:
            out.append(a)
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < 1e-6 or L > max_len:
            continue
        if L < min_len:
            # too short to be a real feature whether or not it happens to
            # already be grid-aligned -- e.g. a stray residual left where
            # two surviving runs' corner reconstruction didn't quite meet.
            # Weld it away rather than keep it as a visible tiny tooth.
            skip_next = True
            continue
        ang = math.atan2(dy, dx)
        bestd = min(abs(((ang - (th0 + k * math.pi / 2) + math.pi) % (2 * math.pi)) - math.pi)
                    for k in range(4))
        if bestd <= tol:
            continue
        du = dx * ux + dy * uy
        corner = (a[0] + du * ux, a[1] + du * uy)
        out.append(corner)
    return out if len(out) >= 3 else poly


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
                     square=True, ang_tol_deg=20.0, allow45=False, th0=None):
    """segs -> ONE clean CCW footprint polygon [(x,y),...] (world units), or None.

    tol    = detail size to ignore (metres).
    bridge = max wall gap/hole to seal (metres).
    seed   = (x, y) interior point to disambiguate which region is the building.
    square = rectilinearize the result (straight walls, crisp corners).
    th0    = shared building orientation (radians) to snap to -- keeps
             multiple floors mutually parallel; None = estimate from this
             floor's own cross-section only.
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
                          allow45=allow45, th0=th0)
        if not allow45 and len(poly) >= 4:
            # welding away one tiny edge can leave a fresh tiny residual
            # where its neighbours now meet -- iterate to a fixed point
            fixed_th0 = th0 if th0 is not None else _dom_orient(poly)
            for _ in range(5):
                nxt = _rectify_diagonals(poly, fixed_th0)
                if len(nxt) == len(poly):
                    break
                poly = nxt
            poly = nxt
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


def _remove_coll_if_empty(name):
    """Delete the collection entirely once it has nothing left in it, instead
    of leaving an empty husk sitting in the Outliner forever (GN_Thresholds/
    GN_DoorFrames/GN_WindowFrames only ever get individual objects removed
    from them as doors/windows/openings are deleted, never the collection
    itself)."""
    c = bpy.data.collections.get(name)
    if c and not c.objects and not c.children:
        try:
            bpy.data.collections.remove(c)
        except Exception:
            pass


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
    lock: BoolProperty(name="Saved as Final", default=False,
        description="This floor's map was saved as final (via 'Save Floor "
        "Map as Final' after hand-correcting it) -- Generate Floor Map will "
        "ask for confirmation before overwriting it. Set automatically, not "
        "meant to be toggled by hand")


class GN_Room(PropertyGroup):
    floor_index: IntProperty(default=0)
    poly_json: StringProperty(default="[]")  # footprint [[x,y],...]
    uid: IntProperty(default=0)              # stable id (survives rebuilds)
    z_offset: FloatProperty(default=0.0, unit='LENGTH',
        description="Vertical shift from the floor's own base/top (for a "
        "half-floor / mezzanine room). Set by grabbing and moving the room "
        "object in Z -- rebuild_rooms detects and re-applies it")
    lock: BoolProperty(default=False,
        description="Keep this room's mesh exactly as hand-edited -- "
        "rebuild_rooms() skips it entirely instead of rebuilding it from "
        "poly_json (which would discard a manual resize). Its door/window "
        "holes also stop updating while locked")
    openings_expanded: BoolProperty(default=True,
        description="Expand this room's group in the Openings panel")


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
    win_w: FloatProperty(default=0.0)   # actual placed window frame size, if any
    win_h: FloatProperty(default=0.0)   # (0 = no window placed -- curtains fall back to hw*2/top-sill)
    win_allow_curtain: BoolProperty(default=True)   # from the placed window preset's own flags
    win_allow_blinds: BoolProperty(default=True)
    # the ORIGINAL exterior piece's own bounds -- the space a swapped-in
    # window still has to fit inside. hw/sill/top can't answer that any more:
    # they're sized to the window actually placed, not the piece it came
    # from. 0 = unconstrained (a manually placed window has no piece).
    avail_w: FloatProperty(default=0.0)
    avail_h: FloatProperty(default=0.0)
    win_key: StringProperty(default="")   # which candidate is placed here
    win_rotated: BoolProperty(default=False)   # placed sideways (can_rotate)


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
    library_key: StringProperty(default="",
        description="Catalog key this preset was pulled from the shared "
        "Window Library with, if any -- empty for hand-made presets")
    can_rotate: BoolProperty(default=False,
        description="Can be used both horizontal and vertical -- auto-fit "
        "also tries it sideways and uses whichever orientation fits better")
    allow_curtain: BoolProperty(default=True,
        description="This window supports having a curtain added to it")
    allow_blinds: BoolProperty(default=True,
        description="This window supports having blinds added to it")
    starts_at_floor: BoolProperty(default=False,
        description="Starts at the room's floor instead of the scene's "
        "default window sill height (e.g. a French window/floor-length "
        "window) -- only affects manual placement")


class GN_WindowLibItem(PropertyGroup):
    """One catalog entry mirrored from the shared Window Library's JSON
    sidecar into the scene, purely for the browse-list UI -- rescan_window_
    library() rebuilds this from disk, nothing here is authoritative."""
    name: StringProperty()
    key: StringProperty()
    object_name: StringProperty()
    width: FloatProperty()
    height: FloatProperty()
    sill: FloatProperty()
    category: StringProperty()
    can_rotate: BoolProperty(default=False)
    allow_curtain: BoolProperty(default=True)
    allow_blinds: BoolProperty(default=True)
    starts_at_floor: BoolProperty(default=False)


class GN_CurtainPreset(PropertyGroup):
    """A curtain/blind pulled from the shared library. No width/height/sill
    fields (unlike GN_WindowPreset) -- sizing is fully dynamic per opening,
    computed by overhanging whatever window is actually placed there."""
    name: StringProperty(default="Curtain")
    mesh_object: PointerProperty(name="Mesh", type=bpy.types.Object,
        description="Curtain/blind mesh placed as a linked instance, "
        "overhanging the placed window")
    library_key: StringProperty(default="")
    category: StringProperty(default="",
        description="'curtain' or 'blinds', copied from the library entry -- "
        "used to check the placed window's own allow_curtain/allow_blinds")


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
            print("[UltimateMLO] threshold update:", e)
        return None

    bpy.app.timers.register(_do, first_interval=0.0)


def _cb_stair_settings(self, context):
    """Live-rebuild the selected stair(s) when Step Height/Depth/Nosing
    changes. _rebuild_stair_mesh is defined later in the file (near the rest
    of the stairs feature) -- fine, since a function body only resolves
    names at CALL time, and by then the whole module has finished loading.

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
    height, depth, nosing = self.stair_step_height, self.stair_step_depth, self.stair_nosing

    def _do():
        for nm in names:
            ob = bpy.data.objects.get(nm)
            if ob:
                try:
                    _rebuild_stair_mesh(ob, height, depth, nosing)
                except Exception as e:
                    print("[UltimateMLO] stair update:", e)
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
        # if the viewport's active object already belongs to this floor, this
        # sync came from the REVERSE direction (a viewport/Outliner click, via
        # _gn_active_object_changed) -- don't deselect-and-reselect, or a
        # shift-click multi-selection gets clobbered on every second click
        active = bpy.context.view_layer.objects.active
        if active is not None and _floor_index_for_object(active, s) == idx:
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


def _cb_room_select(self, context):
    """Selecting a room in the list selects its object in the scene -- same
    deferred-timer pattern as _cb_floor_select, for the same reentrancy
    reason."""
    scene_name = context.scene.name
    idx = context.scene.gn_int.room_index

    def _do():
        scene = bpy.data.scenes.get(scene_name)
        if not scene or not hasattr(scene, "gn_int"):
            return None
        s = scene.gn_int
        if s.room_index != idx or not (0 <= idx < len(s.rooms)):
            return None
        # reverse-direction sync landed here (see _cb_floor_select) -- don't
        # clobber a shift-click multi-selection
        active = bpy.context.view_layer.objects.active
        if active is not None and _room_index_for_object(active, s) == idx:
            return None
        uid = s.rooms[idx].uid
        room_coll = bpy.data.collections.get(ROOM_COLL)
        ob = None
        if room_coll:
            ob = next((o for o in room_coll.objects if o.get("gn_room_uid") == uid), None)
        if ob is None:
            # no built wall yet -- the room is just a face on the shared
            # Floor Map object, so fall back to selecting that instead
            fidx = s.rooms[idx].floor_index
            if 0 <= fidx < len(s.floors):
                ob = _boundary_object_for_floor(s.floors[fidx])
        if ob is None:
            return None
        try:
            for o in bpy.context.selectable_objects:
                o.select_set(False)
            ob.select_set(True)
            bpy.context.view_layer.objects.active = ob
            for w in bpy.context.window_manager.windows:
                for a in w.screen.areas:
                    if a.type == 'VIEW_3D':
                        a.tag_redraw()
        except Exception:
            pass
        return None

    bpy.app.timers.register(_do, first_interval=0.0)


def _floor_index_for_object(ob, s):
    """Reverse of _cb_floor_select's own lookup -- given the object that just
    became active (e.g. from a viewport click), return which floor index it
    belongs to, or None if it isn't a floor's boundary or one of its rooms."""
    if ob is None:
        return None
    me = ob.data
    if ob.name.startswith("GN_FloorMap_Floor") and me and len(me.vertices):
        z0 = (ob.matrix_world @ me.vertices[0].co).z
        best_i, best_d = None, 1e9
        for i, f in enumerate(s.floors):
            d = abs(f.z - z0)
            if d < best_d:
                best_d = d; best_i = i
        return best_i if best_d < 0.5 else None
    uid = ob.get("gn_room_uid")
    if uid is not None:
        for r in s.rooms:
            if r.uid == uid:
                return r.floor_index
    return None


def _room_index_for_object(ob, s):
    """Reverse of _cb_room_select's own lookup -- global index into s.rooms
    for the object that just became active, or None if it isn't a room."""
    if ob is None:
        return None
    uid = ob.get("gn_room_uid")
    if uid is None:
        return None
    for i, r in enumerate(s.rooms):
        if r.uid == uid:
            return i
    return None


def _opening_index_for_object(ob, s):
    """Index into s.openings for the object that just became active, parsed
    from the GN_Frame_<uid> / GN_Threshold_<uid> naming _cb_opening_select
    itself uses to find these objects -- there's no custom-prop tag on them,
    just the name-encoded uid (matching _remove_frame_mesh/_remove_threshold)."""
    if ob is None:
        return None
    for prefix in ("GN_Frame_", "GN_Threshold_"):
        if ob.name.startswith(prefix):
            try:
                uid = int(ob.name[len(prefix):].split(".")[0])
            except ValueError:
                return None
            for i, op in enumerate(s.openings):
                if op.uid == uid:
                    return i
    return None


def _portal_index_for_object(ob, scene):
    """Index into the MLO portals collection for the object that just became
    active, or None if it isn't a portal."""
    if ob is None:
        return None
    coll = _mlo_portals_collection(scene)
    if not coll:
        return None
    for i, o in enumerate(coll.objects):
        if o == ob:
            return i
    return None


_GN_MSGBUS_OWNER = object()


def _sync_lists_from_active(scene):
    """Point each list at whatever object is active: a floor's boundary ->
    Floors, a room/wall -> Rooms, an opening's frame/threshold -> Openings,
    a portal -> Portals. The reverse direction of each _cb_*_select callback.
    Called from BOTH the msgbus subscription and the depsgraph handler --
    msgbus alone doesn't fire for every selection change, which left the
    Openings list (and so the window-swap target) pointing at the wrong
    window after clicking a frame in the viewport."""
    if not scene or not hasattr(scene, "gn_int"):
        return
    s = scene.gn_int
    vl = bpy.context.view_layer
    ob = vl.objects.active if vl else None
    if ob is None:
        return
    fidx = _floor_index_for_object(ob, s)
    if fidx is not None and fidx != s.floor_index and 0 <= fidx < len(s.floors):
        s.floor_index = fidx
    ridx = _room_index_for_object(ob, s)
    if ridx is not None and ridx != s.room_index and 0 <= ridx < len(s.rooms):
        s.room_index = ridx
    oidx = _opening_index_for_object(ob, s)
    if oidx is not None and oidx != s.opening_index and 0 <= oidx < len(s.openings):
        s.opening_index = oidx
    pidx = _portal_index_for_object(ob, scene)
    if pidx is not None and pidx != s.portal_index:
        s.portal_index = pidx


def _gn_active_object_changed():
    """msgbus callback -- deferred via a timer for the same undo/reentrancy
    reason as every other callback in this file."""
    def _do():
        _sync_lists_from_active(bpy.context.scene)
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
        # reverse-direction sync landed here (see _cb_floor_select) -- don't
        # clobber a shift-click multi-selection
        active = bpy.context.view_layer.objects.active
        if active is not None and _portal_index_for_object(active, scene) == idx:
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


def _cb_opening_select(self, context):
    """Selecting an opening in the list also selects its frame/threshold
    object (if either exists) so the Outliner highlights it too -- openings
    only ever had a custom-drawn viewport highlight before, with no real
    Outliner/scene selection. Same deferred-timer pattern as the other
    _cb_*_select callbacks."""
    _cb_redraw(self, context)
    scene_name = context.scene.name
    idx = context.scene.gn_int.opening_index

    def _do():
        scene = bpy.data.scenes.get(scene_name)
        if not scene or not hasattr(scene, "gn_int"):
            return None
        s = scene.gn_int
        if s.opening_index != idx or not (0 <= idx < len(s.openings)):
            return None
        uid = s.openings[idx].uid
        picked = [ob for ob in (bpy.data.objects.get(f"GN_Frame_{uid}"),
                                bpy.data.objects.get(f"GN_Threshold_{uid}")) if ob]
        if not picked:
            return None
        # reverse-direction sync landed here (see _cb_floor_select) -- don't
        # clobber a shift-click multi-selection
        active = bpy.context.view_layer.objects.active
        if active is not None and active in picked:
            return None
        try:
            for ob in bpy.context.selectable_objects:
                ob.select_set(False)
            for ob in picked:
                ob.select_set(True)
            bpy.context.view_layer.objects.active = picked[0]
        except Exception:
            pass
        return None

    bpy.app.timers.register(_do, first_interval=0.0)


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


def _winlib_index_update(self, context):
    """Clicking a row in the Window Library list (template_list's own
    click-to-select) picks it immediately -- no separate 'Pick' click.
    Windows only: pulls the entry into the project if it isn't already
    there and makes it the active window for Window Edit Mode, same as
    GN_OT_win_lib_add_to_project. Curtains/blinds are left alone here --
    they stay an explicit opt-in via their own Add button, since a window
    doesn't automatically get dressing just because it's now selected in
    a shared list. Called through the module namespace (not a direct name
    reference) so this can sit above _pull_library_entry_into_project's
    own definition without an ordering problem -- the name only needs to
    resolve once this actually fires, by which point the module is fully
    loaded."""
    s = self
    if not (0 <= s.winlib_index < len(s.winlib_items)):
        return
    entry = s.winlib_items[s.winlib_index]
    if entry.category in _CURTAIN_CATEGORIES:
        # picking a curtain only makes it the ACTIVE one -- putting it on a
        # window is still the separate, explicit Add Curtain step
        existing_idx = next((i for i, p in enumerate(s.curtain_presets)
                             if p.library_key == entry.key), -1)
        if existing_idx >= 0:
            s.active_curtain_preset = existing_idx
        else:
            _pull_library_entry_into_project(context, entry)
        return
    # a window opening selected? clicking a library row swaps THAT window for
    # this one (the operator reports if it doesn't fit the exterior opening).
    # Either way the row also becomes the active window for placing new ones.
    sel = (s.openings[s.opening_index]
           if 0 <= s.opening_index < len(s.openings) else None)
    if sel is not None and not sel.is_door:
        # deferred: this runs during a UI property update, where an operator
        # that rebuilds meshes can't safely run inline (same timer pattern as
        # the _cb_*_select callbacks)
        key, oi = entry.key, s.opening_index

        def _do_swap():
            try:
                bpy.ops.gn_int.swap_window(key=key, index=oi)
            except Exception as e:
                print("[UltimateMLO] swap failed:", e)
            return None
        bpy.app.timers.register(_do_swap, first_interval=0.0)
    existing_idx = next((i for i, p in enumerate(s.window_presets)
                         if p.library_key == entry.key), -1)
    if existing_idx >= 0:
        s.active_window_preset = existing_idx
        return
    _pull_library_entry_into_project(context, entry)


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
        unit='LENGTH', description="Floor Map: ignore wall detail smaller than this "
        "(window reveals, tiny jogs). Bigger = simpler outline")
    bridge: FloatProperty(name="Bridge Gaps <", default=0.5, min=0.0, max=3.0,
        unit='LENGTH', description="Floor Map: seal holes and gaps in the shell up "
        "to this size (non-manifold buildings, missing walls)")
    square: BoolProperty(name="Square Walls", default=True,
        description="Floor Map: straighten walls to the building's main axis and "
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
    timecycle_name: StringProperty(name="Timecycle", default="int_gasstation",
        description="RageKit room timecycle name. Defaults to int_gasstation "
        "until you edit it yourself", update=_cb_timecycle_name)
    timecycle_auto: BoolProperty(default=False, options={'HIDDEN'},
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
    room_index: IntProperty(default=0, update=_cb_room_select)
    uid_counter: IntProperty(default=1)
    openings: CollectionProperty(type=GN_Opening)
    opening_index: IntProperty(default=0, update=_cb_opening_select)
    portal_index: IntProperty(default=0, update=_cb_portal_select)
    show_portal_list: BoolProperty(default=True)
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
    win_edit_mode: BoolProperty(name="Window Edit Mode", default=False,
        description="Show a move/scale gizmo on the selected window so you can "
        "slide it along the wall, raise/lower it, and scale it uniformly")
    default_window_sill: FloatProperty(name="Default Window Sill", default=0.9,
        min=0.0, max=4.0, unit='LENGTH',
        description="Standard height manually-placed windows start at, unless "
        "the active window is flagged 'Starts at Floor' in the library")
    winlib_items: CollectionProperty(type=GN_WindowLibItem)
    winlib_index: IntProperty(default=0, update=_winlib_index_update)
    curtain_presets: CollectionProperty(type=GN_CurtainPreset)
    active_curtain_preset: IntProperty(default=0)
    curtain_side_overhang: FloatProperty(name="Side Overhang", default=0.15,
        min=0.0, max=1.0, unit='LENGTH',
        description="How far the curtain/blind extends past each side of the placed window")
    curtain_top_overhang: FloatProperty(name="Top Overhang", default=0.15,
        min=0.0, max=1.0, unit='LENGTH',
        description="How far the curtain/blind extends above the placed window")
    curtain_bottom_drop: FloatProperty(name="Bottom Drop", default=0.0,
        min=0.0, max=3.0, unit='LENGTH',
        description="How far below the window sill the curtain/blind extends "
        "(0 = stops at the sill, larger = floor-length)")
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
    stair_nosing: FloatProperty(name="Nosing", default=0.0,
        min=0.0, max=0.1, unit='LENGTH',
        description="Rounded overhang at the front of each tread, recessing "
        "the riser to match (0 = square edge, flush riser). Also "
        "live-updates any selected, already-created stairs",
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
                      "(select a facade edge in Edit Mode) and immediately "
                      "generate its floor map too, if a Shell is set. Height is "
                      "always Room Height - use the floor list's Top field for "
                      "a custom height on a specific floor")

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
        new_z = it.z
        _sort_floors(s)
        _select_floor_by_z(s, new_z)
        msg = f"Added floor at z={base:.2f} (height = Room Height, {s.room_height:.2f} m)"
        if s.exterior:
            bpy.ops.gn_int.gen_boundaries()
            msg += " + generated floor map"
        self.report({'INFO'}, msg)
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
        new_z = it.z
        _sort_floors(s)
        _select_floor_by_z(s, new_z)
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
                      "then click to add a floor there and immediately generate "
                      "its floor map too. The exterior mesh is never modified - "
                      "only a throwaway copy is cut for preview")

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
        # world XY the slice height is measured at -- the exterior's own
        # bounding-box center. Only used as the fixed reference column for
        # turning the mouse cursor into a Z height (see modal()); doesn't
        # affect the slice itself, which always spans the full cross-section.
        corners = [ex.matrix_world @ Vector(c) for c in ex.bound_box]
        self._ref_xy = (sum(c.x for c in corners) / 8, sum(c.y for c in corners) / 8)
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
            # slice height follows the cursor directly: cast the mouse
            # position through the view onto a vertical line at the
            # exterior's XY center, at the slice's current depth -- so the
            # drawn line tracks under the mouse instead of drifting off via
            # an accumulated relative delta (which is what made the slice
            # end up far from the cursor before).
            region, rv3d = context.region, context.region_data
            coord = (event.mouse_region_x, event.mouse_region_y)
            depth = Vector((self._ref_xy[0], self._ref_xy[1], self._z))
            loc = view3d_utils.region_2d_to_location_3d(region, rv3d, coord, depth)
            if event.shift:
                # fine control: only take a fraction of the cursor's move
                # since last frame, instead of snapping straight to it
                dy = event.mouse_y - self._last_mouse_y
                self._z += dy * 0.001
            else:
                self._z = loc.z
            self._last_mouse_y = event.mouse_y
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
        new_z = it.z
        _sort_floors(s)
        _select_floor_by_z(s, new_z)
        msg = f"Added floor at z={z:.3f}"
        if s.exterior:
            bpy.ops.gn_int.gen_boundaries()
            msg += " + generated floor map"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class GN_OT_remove_floor(Operator):
    bl_idname = "gn_int.remove_floor"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Remove Floor"
    index: IntProperty()

    def execute(self, context):
        s = context.scene.gn_int
        if 0 <= self.index < len(s.floors):
            bnd = _boundary_object_for_floor(s.floors[self.index])
            if bnd:
                bpy.data.objects.remove(bnd, do_unlink=True)
            s.floors.remove(self.index)
        return {'FINISHED'}


def _select_floor_by_z(s, z, tol=1e-3):
    """Point floor_index at the floor closest to z (after _sort_floors has
    re-shuffled indices) so a newly added floor ends up selected in the list
    instead of leaving whatever was selected before."""
    if not s.floors:
        return
    best_i = min(range(len(s.floors)), key=lambda i: abs(s.floors[i].z - z))
    if abs(s.floors[best_i].z - z) <= max(tol, 0.01):
        s.floor_index = best_i


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
    bl_label = "Generate Floor Map"
    bl_description = ("Create the Floor Map outline for the SELECTED floor only "
                      "(exterior inset by the margin). Other floors are untouched")

    def invoke(self, context, event):
        s = context.scene.gn_int
        if 0 <= s.floor_index < len(s.floors) and s.floors[s.floor_index].lock:
            return context.window_manager.invoke_confirm(
                self, event,
                message=f"Floor {s.floor_index + 1}'s map was saved as final -- "
                        "regenerating it will discard your corrections. Continue?")
        return self.execute(context)

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
        # find this floor's (base, top) among the z-sorted, gap-honoring list
        tops = _floor_tops(context)
        sorted_idx = sorted(range(len(s.floors)), key=lambda k: s.floors[k].z)
        pos = sorted_idx.index(s.floor_index)
        base, top, nxt = tops[pos]

        src = _eval_bmesh(ex, context)
        coll = _get_coll(BOUND_COLL)
        zc = base + s.sample_offset
        segs = _cut_segments(src, zc)
        # one shared orientation for the whole building, not re-estimated
        # per floor -- otherwise each floor's cross-section can land a
        # fraction of a degree apart and their boundaries stop being
        # mutually parallel even though they're straight individually.
        th0 = _shell_dom_orient(ex) if s.square else None
        poly_t = raster_footprint(
            segs, tol=s.detail_tol, cell=0.05, seed=None, bridge=s.bridge,
            square=s.square, ang_tol_deg=s.ang_tol, allow45=s.allow45, th0=th0)
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
        _clear_named(coll, f"GN_FloorMap_Floor{pos+1}")
        _make_face_object(coll, f"GN_FloorMap_Floor{pos+1}", ip, base)
        target.bound_json = json.dumps(
            [[round(p.x, 4), round(p.y, 4)] for p in ip])
        target.lock = False   # this floor is no longer the saved-as-final shape
        # a fresh Floor Map should already be one splittable room -- no extra
        # "Reset Room Outline" click needed just to get started
        seed_rooms_from_boundaries(context, s.floor_index)
        msg = f"Generated floor map for Floor {s.floor_index+1}"
        if fell_back:
            msg += " (fell back to legacy method)"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class GN_OT_save_floor_map_final(Operator):
    bl_idname = "gn_int.save_floor_map_final"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Save Floor Map as Final"
    bl_description = ("Capture the selected floor's Floor Map exactly as it "
                      "currently is (after hand-correcting it in Edit Mode) "
                      "as the saved-as-final shape. Generate Floor Map will "
                      "then ask before overwriting it. Only works before the "
                      "floor has been split into rooms -- reset the room "
                      "outline first if it already has")

    def execute(self, context):
        s = context.scene.gn_int
        if not (0 <= s.floor_index < len(s.floors)):
            self.report({'ERROR'}, "Select a floor in the list")
            return {'CANCELLED'}
        target = s.floors[s.floor_index]
        fm_ob = _boundary_object_for_floor(target)
        if fm_ob is None:
            self.report({'ERROR'}, "No Floor Map found for this floor")
            return {'CANCELLED'}
        me = fm_ob.data
        if len(me.polygons) != 1:
            self.report({'ERROR'},
                "Floor Map has been split into rooms -- click 'Reset Room "
                "Outline' first, then correct and save the whole-floor shape")
            return {'CANCELLED'}
        mat = fm_ob.matrix_world
        poly = me.polygons[0]
        pts = [mat @ me.vertices[vi].co for vi in poly.vertices]
        if len(pts) < 3:
            self.report({'ERROR'}, "Floor Map has no valid outline")
            return {'CANCELLED'}
        new_json = json.dumps([[round(p.x, 4), round(p.y, 4)] for p in pts])
        target.bound_json = new_json
        target.lock = True
        # the floor's already-seeded room (Generate Floor Map auto-seeds one)
        # holds its OWN separate copy of the polygon -- update it too, or
        # the next split reads the stale pre-correction shape and the fix
        # is lost the moment you start cutting rooms
        updated_rooms = 0
        for r in s.rooms:
            if r.floor_index == s.floor_index:
                r.poly_json = new_json
                updated_rooms += 1
        self.report({'INFO'},
            f"Floor {s.floor_index + 1}'s map saved as final"
            + (f" ({updated_rooms} room outline synced)" if updated_rooms else ""))
        return {'FINISHED'}


class GN_OT_clear(Operator):
    bl_idname = "gn_int.clear"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Clear Generated"
    bl_description = "Delete generated floor maps and interior shells"

    def execute(self, context):
        _clear_coll(BOUND_COLL)
        _clear_coll(INT_COLL)
        return {'FINISHED'}


class GN_OT_clean_interior(Operator):
    bl_idname = "gn_int.clean_interior"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Clean Interior Tool (New Building)"
    bl_description = ("Reset for a different building: clears the exterior "
                      "reference, floors, rooms, openings, stairs, generated "
                      "floor maps, and the MLO name fields. Does NOT touch any "
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
        s.timecycle_name = "int_gasstation"
        s.timecycle_auto = False
        _clear_coll(BOUND_COLL)
        _clear_coll(ROOM_COLL)
        _clear_coll(DOORFRAME_COLL)
        _clear_coll(WINDOWFRAME_COLL)
        _clear_coll(THRESHOLD_COLL)
        _clear_coll(INT_COLL)
        _clear_coll(STAIR_COLL)
        _clear_coll(OPENING_PIECES_COLL)
        self.report({'INFO'}, "Interior Tool reset - set a new exterior shell to start")
        return {'FINISHED'}


# ---- mesh builders --------------------------------------------------------
def _make_face_object(coll, name, poly_xy, z):
    """Build the Floor Map as a single flat n-gon FACE (no triangulation) --
    this is the thing you slice directly to make rooms, not a wire outline."""
    bm = bmesh.new()
    vs = [bm.verts.new((p.x, p.y, z)) for p in poly_xy]
    try:
        bm.faces.new(vs)
    except ValueError:
        pass
    bm.normal_update()
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    ob.show_in_front = True
    return ob


def _face_centroid_xy(face, mat):
    pts = [mat @ v.co for v in face.verts]
    n = len(pts)
    return sum(p.x for p in pts) / n, sum(p.y for p in pts) / n


def _poly_centroid_xy(poly_xy):
    n = len(poly_xy)
    return sum(p[0] for p in poly_xy) / n, sum(p[1] for p in poly_xy) / n


def _replace_floor_map_face(floor, old_poly_xy, new_polys_xy):
    """On the Floor Map object for `floor`, find the face matching
    old_poly_xy (by centroid, the closest match) and replace it with one new
    face per polygon in new_polys_xy. Returns True on success."""
    fm_ob = _boundary_object_for_floor(floor)
    if fm_ob is None:
        return False
    bm = bmesh.new()
    bm.from_mesh(fm_ob.data)
    bm.faces.ensure_lookup_table()
    if not bm.faces:
        bm.free()
        return False
    mat = fm_ob.matrix_world
    inv = mat.inverted()
    tx, ty = _poly_centroid_xy(old_poly_xy)
    target, best_d = None, 1e18
    for f in bm.faces:
        fx, fy = _face_centroid_xy(f, mat)
        d = (fx - tx) ** 2 + (fy - ty) ** 2
        if d < best_d:
            best_d = d
            target = f
    if target is None:
        bm.free()
        return False
    face_z = (mat @ target.verts[0].co).z
    bmesh.ops.delete(bm, geom=[target], context='FACES')
    for poly in new_polys_xy:
        vs = [bm.verts.new(inv @ Vector((p.x, p.y, face_z))) for p in poly]
        try:
            bm.faces.new(vs)
        except ValueError:
            pass
    bm.normal_update()
    bm.to_mesh(fm_ob.data)
    fm_ob.data.update()
    bm.free()
    return True


def _reset_floor_map_face(floor, poly_xy):
    """Wipe every face on the Floor Map for `floor` and replace it with a
    single face matching poly_xy -- used by Reset Room Outline to drop any
    room-splitting cuts and go back to the whole-floor envelope."""
    fm_ob = _boundary_object_for_floor(floor)
    if fm_ob is None:
        return False
    bm = bmesh.new()
    bm.from_mesh(fm_ob.data)
    if bm.verts:
        bmesh.ops.delete(bm, geom=list(bm.verts), context='VERTS')
    vs = [bm.verts.new((p[0], p[1], floor.z)) for p in poly_xy]
    try:
        bm.faces.new(vs)
    except ValueError:
        pass
    bm.normal_update()
    bm.to_mesh(fm_ob.data)
    fm_ob.data.update()
    bm.free()
    return True


def _boundary_object_for_floor(floor, tol=0.05):
    """Find the GN_FloorMap_Floor* object for this floor, matched by Z (not
    name) so it's correct even if floors were added out of Z order. Checks
    the pre-rename 'GN_Boundaries' collection too -- if the migration timer
    (_deferred_restore) hasn't run yet on a scene from before the Floor Map
    rename, we must still find the object instead of concluding every
    floor's boundary was deleted and wiping the Floors list."""
    coll = bpy.data.collections.get(BOUND_COLL) or bpy.data.collections.get("GN_Boundaries")
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


def _sync_floors_with_boundaries(scene):
    """Drop any floor whose previously-generated boundary object was deleted
    by hand (e.g. in the Outliner) -- otherwise the Floors list keeps showing
    an entry with nothing left backing it. Floors that never had a boundary
    generated yet (no bound_json) are untouched. Takes a Scene (not a
    Context) so it's safe to call from a depsgraph handler, where context is
    not fully valid."""
    if not hasattr(scene, "gn_int"):
        return
    s = scene.gn_int
    stale = [i for i, f in enumerate(s.floors)
             if f.bound_json and _boundary_object_for_floor(f) is None]
    for i in reversed(stale):
        s.floors.remove(i)
    if stale and s.floor_index >= len(s.floors):
        s.floor_index = len(s.floors) - 1


@bpy.app.handlers.persistent
def _gn_depsgraph_sync(scene, depsgraph=None):
    # Panel.draw() is the wrong place to mutate scene data -- collection
    # edits made there can silently no-op depending on context, which is
    # exactly why the Floors list kept showing a deleted boundary's floor.
    # React to the real scene-graph change instead -- but depsgraph_update_post
    # fires during undo/redo too, and mutating bpy.data synchronously from
    # inside it (this handler used to call s.floors.remove() directly) can
    # crash Blender. Defer the actual mutation to the next event-loop tick,
    # same reentrancy-safe pattern as every other callback in this file.
    scene_name = scene.name

    def _do():
        sc = bpy.data.scenes.get(scene_name)
        if sc is None:
            return None
        try:
            _sync_floors_with_boundaries(sc)
        except Exception:
            pass
        try:
            _sync_lists_from_active(sc)      # msgbus misses some selections
        except Exception:
            pass
        return None

    try:
        bpy.app.timers.register(_do, first_interval=0.0)
    except Exception:
        pass


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
    locked_obs = {}
    if coll:
        for ob in coll.objects:
            u = ob.get("gn_room_uid")
            r = next((r for r in s.rooms if r.uid == u), None)
            if r is not None and r.lock:
                locked_obs[u] = ob
    for ob in list(coll.objects) if coll else []:
        u = ob.get("gn_room_uid")
        if u not in locked_obs:
            bpy.data.objects.remove(ob, do_unlink=True)
    coll = _get_coll(ROOM_COLL)
    for i, r in enumerate(s.rooms):
        if not r.uid:
            r.uid = _new_uid(s)
        if r.uid in locked_obs:
            continue    # keep the hand-edited mesh exactly as-is
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
    partition gap -- cutting the Floor Map's own face into two new faces.
    Returns count made (0 or 2)."""
    s = context.scene.gn_int
    if not (0 <= room_idx < len(s.rooms)):
        return 0
    rec = s.rooms[room_idx]
    fidx = rec.floor_index
    if not (0 <= fidx < len(s.floors)):
        return 0
    try:
        poly = [Vector(p) for p in json.loads(rec.poly_json)]
    except Exception:
        return 0
    a, b = split_polygon_path(poly, path, s.partition)
    if not a or not b:
        return 0
    if not _replace_floor_map_face(s.floors[fidx], poly, [a, b]):
        return 0
    s.rooms.remove(room_idx)
    for part in (a, b):
        r = s.rooms.add()
        r.floor_index = fidx
        r.uid = _new_uid(s)
        r.poly_json = json.dumps([[round(p.x, 4), round(p.y, 4)] for p in part])
    _dump_scene(context.scene)
    return 2


def seed_rooms_from_boundaries(context, floor_idxs=None):
    """Reset every floor in floor_idxs (all floors if None; a single int also
    accepted) back to ONE room = its Floor Map's whole envelope -- wiping any
    room-splitting cuts made on that floor's Floor Map face and starting
    over. Rooms on OTHER floors are left completely untouched. Does NOT
    build any 3D walls -- any walls already built for a replaced room are
    deleted so the floor drops back to a flat, outline-only state
    immediately; run Build Walls when ready to turn the outline into real
    geometry.

    Uses the pristine bound_json snapshot from when Generate Floor Map last
    ran (not the live Floor Map mesh, which may currently be split into
    several room faces). Returns the number of floors seeded.
    """
    s = context.scene.gn_int
    if floor_idxs is None:
        targets = set(range(len(s.floors)))
    elif isinstance(floor_idxs, int):
        targets = {floor_idxs}
    else:
        targets = set(floor_idxs)
    dropped_uids = {r.uid for r in s.rooms if r.floor_index in targets}
    room_coll = bpy.data.collections.get(ROOM_COLL)
    if room_coll:
        for ob in list(room_coll.objects):
            if ob.get("gn_room_uid") in dropped_uids:
                bpy.data.objects.remove(ob, do_unlink=True)
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
        if not f.bound_json:
            continue
        try:
            poly = json.loads(f.bound_json)
        except Exception:
            continue
        if len(poly) < 3 or not _reset_floor_map_face(f, poly):
            continue
        rec = s.rooms.add()
        rec.floor_index = i
        rec.uid = _new_uid(s)
        rec.poly_json = json.dumps(poly)
        made += 1
    _dump_scene(context.scene)
    return made


def split_room_record(context, room_idx, A, B):
    """Split room[room_idx] by line A-B with the partition gap -- cutting
    the Floor Map's own face into two new faces. Returns count made."""
    s = context.scene.gn_int
    if not (0 <= room_idx < len(s.rooms)):
        return 0
    rec = s.rooms[room_idx]
    fidx = rec.floor_index
    if not (0 <= fidx < len(s.floors)):
        return 0
    try:
        poly = [Vector(p) for p in json.loads(rec.poly_json)]
    except Exception:
        return 0
    pos, neg = split_polygon(poly, A, B, s.partition)
    if not pos or not neg:
        return 0
    if not _replace_floor_map_face(s.floors[fidx], poly, [pos, neg]):
        return 0
    s.rooms.remove(room_idx)
    for part in (pos, neg):
        r = s.rooms.add()
        r.floor_index = fidx
        r.uid = _new_uid(s)
        r.poly_json = json.dumps([[round(p.x, 4), round(p.y, 4)] for p in part])
    _dump_scene(context.scene)
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


def _room_for_opening(context, op):
    """(floor_idx, room_idx) this opening belongs to, or (floor_idx, None) /
    (None, None) if unmatched. An opening's (cx, cy) sits ON the wall plane,
    which a plain point-in-polygon test against the room's interior treats
    as ambiguous/outside -- so after trying the raw point, nudge it inward
    along the opening's own normal (trying both directions, since sign
    convention isn't guaranteed) by increasing amounts until it lands inside
    a room."""
    fi = _floor_idx_for_z(context, op.sill)
    if fi is None:
        return None, None
    base = Vector((op.cx, op.cy))
    ri = _room_at_point(context, fi, base)
    if ri is not None and ri >= 0:
        return fi, ri
    nrm = Vector((op.nx, op.ny))
    if nrm.length > 1e-6:
        nrm = nrm.normalized()
        for eps in (0.15, 0.3, 0.6, 1.0):
            for sign in (1, -1):
                ri = _room_at_point(context, fi, base + nrm * eps * sign)
                if ri is not None and ri >= 0:
                    return fi, ri
    return fi, None


def _group_openings_by_room(context):
    """Group opening indices by (floor_index, room_index) via _room_for_opening.
    Returns an ordered list of ((floor_idx_or_None, room_idx_or_None),
    [opening_index, ...]); openings that don't land in any known floor/room
    are grouped last under (None, None). room_idx matches GN_UL_rooms's own
    numbering (global position in s.rooms + 1), so the two panels
    cross-reference cleanly."""
    s = context.scene.gn_int
    groups = {}
    for i, op in enumerate(s.openings):
        key = _room_for_opening(context, op)
        groups.setdefault(key, []).append(i)

    def sort_key(k):
        fi, ri = k
        return (fi is None, fi if fi is not None else 0, ri is None, ri if ri is not None else 0)

    return [(k, groups[k]) for k in sorted(groups, key=sort_key)]


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
                      "rectangular room (clamped to the floor map)")

    def invoke(self, context, event):
        s = context.scene.gn_int
        fl = _floor_by_index(context, s.active_floor)
        if not fl or fl[2] is None:
            self.report({'ERROR'}, "Generate a floor map first (need a floor map)")
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
            self.report({'ERROR'}, "Generate a floor map first")
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
    bl_label = "Build Walls"
    bl_description = ("Turn the current room outline(s) into real 3D walls. "
                      "Safe to re-run any time to pick up outline edits -- "
                      "never touches the Floor Map or the outlines themselves")

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


def _mlo_purge_collection_objects(col):
    """Delete every object in col and its sub-collections (data-blocks too,
    via do_unlink), recursively. Used for Clean MLO's Collisions/Portals
    wipe -- those are pure Build MLO output with no prior state to restore
    to, so a full delete IS the correct undo. Returns count deleted."""
    n = 0
    for ob in list(col.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
        n += 1
    for child in list(col.children):
        n += _mlo_purge_collection_objects(child)
    return n


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
    bl_description = ("Undo the MLO build entirely, back to before Build MLO "
                      "ran: move each room shell mesh in Main back into "
                      "GN_Rooms under its original name, delete the shell "
                      "empty, delete every generated collision object and "
                      "portal (Collisions/Portals have no pre-MLO state to "
                      "restore to -- Build MLO creates them from scratch), "
                      "then remove the int_<name> collection scaffolding. Any "
                      "OTHER collection still holding real content (props/ "
                      "assets you placed by hand) is left in place")

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

        # Collisions and Portals are ENTIRELY generated by Build MLO (no
        # pre-existing content to restore elsewhere, unlike room shells) --
        # purge every object in them so the empty-tree cleanup below can
        # remove the collections themselves too
        purged = 0
        for child in main_col.children:
            if child.name in ("Collisions", "Portals"):
                purged += _mlo_purge_collection_objects(child)

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

        # empties Build MLO's own "Add Empties" step created (decals/details/
        # proxy/visuals/lights/custom, in each room's collection) are ALSO
        # pure Build MLO output with nothing to restore to -- delete them too,
        # but only EMPTY-type objects, so hand-placed prop/asset MESHES are
        # never touched
        for ob in list(main_col.all_objects):
            if ob.type == 'EMPTY':
                bpy.data.objects.remove(ob, do_unlink=True)
                purged += 1

        kept, col_count = _mlo_delete_empty_tree(main_col)
        msg = (f"Restored {restored} room(s) to GN_Rooms, deleted {purged} "
              f"collision/portal object(s), removed {col_count} empty collection(s)")
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
        col.prop(s, "build_empties")
        self.layout.separator()
        col2 = self.layout.column(align=True)
        col2.prop(s, "build_shell_collision")
        col2.prop(s, "build_portals")

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
            self.report({'ERROR'}, "No built rooms - run Build Walls first")
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
            _mlo_apply_empty_defaults(empty, mlo_name=name)

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
                        _mlo_make_collection(f"Assets_{room_token}", rcol).ragequit_type = 'assets'
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
            try:
                from Sollumz.sollumz_properties import SollumType
                ob.sollum_type = SollumType.DRAWABLE_MODEL
            except Exception:
                pass
            moved += 1

        _mlo_apply_collection_types(main_col)
        msg = f"Built int_{name}"
        if s.build_main:
            msg += f", {moved} room shell(s)"
            if replaced:
                msg += f" ({replaced} replaced)"

        if s.build_portals:
            try:
                result = bpy.ops.gn_int.create_portals()
                if 'FINISHED' in result:
                    msg += ", portals"
                else:
                    # create_portals() reports its own specific error (e.g. no
                    # openings), but returning {'CANCELLED'} raises no
                    # exception -- checking only for an exception here used to
                    # let this silently claim "portals" even when 0 were made
                    msg += ", portals SKIPPED (see previous error)"
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
        if s.build_shell_collision:
            # Same interactive material-mapping dialog as the standalone
            # "Create Shell Collision" tool (auto-guesses each material as a
            # starting suggestion, but always lets you review/override before
            # anything is actually built) -- chained right after this dialog
            # closes, rather than silently auto-assigning materials here.
            bpy.ops.gn_int.create_shell_collision('INVOKE_DEFAULT')
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
            try:
                _mlo_make_collection(f"Props_{rcol.name}", rcol).ragequit_type = 'room_props'
            except Exception:
                pass
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
                _mlo_make_collection(f"Assets_{rcol.name}", rcol).ragequit_type = 'assets'
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
def _mlo_apply_archetype_defaults(obj, set_static=False, mlo_name=None):
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


def _wall_match_for_opening(context, o, want_sign):
    """Full match details for one side of an opening, found by the SAME
    wall-segment alignment/proximity/Z-overlap test _opening_matches_any_wall
    uses to decide a wall actually cut this hole (rather than probing a
    point in space and checking polygon containment, which is fragile right
    next to a jog/notch in the room's boundary). Generalized to search every
    room (not just the opening's own nominal floor) so a manually z_offset
    half-floor/mezzanine room still matches.

    want_sign: +1 = the wall whose OWN inward normal points the same way as
    the opening's normal (so the opening's normal points INTO that room --
    it's the 'to' side); -1 = the opposite (the 'from' side, behind it).
    Picks the closest-aligned match if more than one room's wall qualifies.

    Returns (room, wn, wall_pt) or None. wall_pt is the opening's centre
    PROJECTED onto the wall's own line -- an opening's stored (cx, cy) can
    sit off that line by however thick the original window/door piece it
    was projected from was, so anything measuring outward FROM the wall
    (like a reveal depth) needs to start at wall_pt, not (o.cx, o.cy)."""
    s = context.scene.gn_int
    nrm = Vector((o.nx, o.ny))
    if nrm.length < 1e-6:
        return None
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
                best = (lat, r, wn, A + d * x)
    if best is None:
        return None
    _, r, wn, wall_pt = best
    return r, wn, wall_pt


def _wall_room_for_opening(context, o, want_sign):
    """Room token for one side of an opening (see _wall_match_for_opening),
    or 'limbo' if nothing matches on that side."""
    m = _wall_match_for_opening(context, o, want_sign)
    if m is None:
        return "limbo"
    token = _room_token_for_uid(context, m[0].uid)
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
            # extruded OUTWARD from the wall's own hole, away from that
            # room, by _build_wall -- so at a room/limbo boundary the portal
            # needs to sit at that jamb's outer rim, not the raw opening
            # centre. Two things the raw centre can't be trusted for here:
            # it can sit off the wall's actual line by however thick the
            # projected window/door piece was (_wall_match_for_opening's
            # wall_pt corrects that), and the reveal itself is measured
            # outward FROM that line, not from the (possibly offset) centre.
            # A room-to-room opening needs no shift: both sides' jambs
            # already meet exactly at the centre by construction.
            rd = 0.0
            if s.reveal:
                rd = (s.partition * 0.5) if o.is_door else s.wall_margin
            if rd > 1e-4 and (token_to == "limbo") != (token_from == "limbo"):
                m = _wall_match_for_opening(context, o, -1 if token_to == "limbo" else +1)
                if m:
                    _, wn, wall_pt = m
                    c = wall_pt + (-wn) * rd
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


def _mlo_apply_empty_defaults(empty, mlo_name=None, no_shadows=False):
    try:
        from Sollumz.sollumz_properties import SollumType
        empty.sollum_type = SollumType.DRAWABLE
    except Exception:
        pass
    _mlo_apply_archetype_defaults(empty, set_static=True, mlo_name=mlo_name)
    _mlo_apply_sollumz_lod_defaults(empty)
    if no_shadows:
        # decals/proxy/visuals empties shouldn't cast shadows in-game --
        # details/lights/custom keep Sollumz's own default (shadows on)
        try:
            ent = empty.ragequit_entity
            ent.flag_cast_static_shadows = False
            ent.flag_cast_dynamic_shadows = False
        except Exception:
            pass


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
    _mlo_apply_empty_defaults(empty, mlo_name=mlo_name,
                              no_shadows=empty_type_name in ('decals', 'proxy', 'visuals'))
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
def _gn_organise_renamed(context, obj, mlo_name, room, room_num, category):
    """Move a freshly-renamed <mlo>_<room_num>_<category> object into its
    room collection and parent it to the matching category empty (keeping
    world transform), same as scene_organizer.py's own Organise step that
    Smart Rename always runs afterward -- otherwise the renamed object is
    left wherever it happened to already be (often the scene root), not
    actually organised into the room hierarchy at all."""
    room_col = bpy.data.collections.get(room)
    if room_col is not None:
        for col in list(obj.users_collection):
            col.objects.unlink(obj)
        room_col.objects.link(obj)
    try:
        from Sollumz.sollumz_properties import SollumType
        obj.sollum_type = SollumType.DRAWABLE_MODEL
    except Exception:
        pass
    _mlo_apply_archetype_defaults(obj, set_static=True)
    _mlo_apply_sollumz_lod_defaults(obj)
    empty = bpy.data.objects.get(f"{mlo_name}_{room_num}_{category}")
    if empty is not None:
        world_mat = obj.matrix_world.copy()
        obj.parent = empty
        obj.matrix_parent_inverse = Matrix.Identity(4)
        obj.matrix_world = world_mat
    return room_col is not None, empty is not None


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

        # room_num (not the raw "r01" collection name) so this matches Add
        # Empties' own naming exactly (<mlo>_<room_num>_<category>) -- that's
        # what lets the object below find and parent to its category empty
        room_num = _gn_room_number_from_name(room)
        base = f"{name}_{room_num}_{category}"
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
                    moved, parented = _gn_organise_renamed(context, target, name, room, room_num, category)
                    msg = f"Renamed and merged {len(all_matching)} object(s) -> '{base}'"
                    if not moved:
                        msg += f" (room collection '{room}' not found)"
                    elif not parented:
                        msg += f" (no '{base}' empty to parent to)"
                    self.report({'INFO'}, msg)
                    return {'FINISHED'}
                except RuntimeError as e:
                    self.report({'WARNING'}, f"Join failed: {e}")

        moved_n = parented_n = 0
        for obj in objects:
            moved, parented = _gn_organise_renamed(context, obj, name, room, room_num, category)
            moved_n += moved
            parented_n += parented
        msg = f"Renamed {len(objects)} object(s) -> '{base}'"
        if moved_n < len(objects):
            msg += f" (room collection '{room}' not found)"
        elif parented_n < len(objects):
            msg += f" (no '{base}' empty to parent to)"
        self.report({'INFO'}, msg)
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
        _mlo_apply_archetype_defaults(root_obj, set_static=True, mlo_name=name)
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
        _mlo_apply_archetype_defaults(obj, set_static=True, mlo_name=name)
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
    bl_description = ("Reset every floor's Floor Map back to one whole-floor "
                      "room, dropping all room-splitting cuts, and delete any "
                      "built walls")

    def execute(self, context):
        seed_rooms_from_boundaries(context, None)
        return {'FINISHED'}


class GN_OT_seed_rooms(Operator):
    bl_idname = "gn_int.seed_rooms"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Reset Room Outline"
    bl_description = ("Undo all room-splitting cuts on every SELECTED floor "
                      "(its Floor Map object and/or rooms selected in the "
                      "viewport -- shift-click to pick several), going back "
                      "to one room = its whole Floor Map. NOT needed for a "
                      "new floor -- Generate Floor Map already leaves you "
                      "with one splittable room. Any walls already built for "
                      "that floor are removed. Falls back to the floor "
                      "list's active floor if nothing relevant is selected. "
                      "Other floors (and any splits already made there) are "
                      "left untouched")

    def execute(self, context):
        s = context.scene.gn_int
        if not s.floors:
            self.report({'ERROR'}, "Generate a floor map first")
            return {'CANCELLED'}
        idxs = {i for ob in context.selected_objects
               if (i := _floor_index_for_object(ob, s)) is not None}
        if not idxs:
            if not (0 <= s.floor_index < len(s.floors)):
                self.report({'ERROR'}, "Select a floor in the list")
                return {'CANCELLED'}
            idxs = {s.floor_index}
        made = seed_rooms_from_boundaries(context, idxs)
        if made == 0:
            self.report({'WARNING'}, "No floor map for the selected floor(s) - run Generate Floor Map")
            return {'CANCELLED'}
        names = ", ".join(str(i + 1) for i in sorted(idxs))
        self.report({'INFO'}, f"Floor{'s' if len(idxs) != 1 else ''} {names} room outline reset ({made} seeded) - Build Walls when ready")
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
            self.report({'ERROR'}, "No room outline on this floor - click 'Reset Room Outline' first")
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
        # prefer the floor of whatever room is actually selected/active --
        # s.floor_index (the Floors list's own selection) is a DIFFERENT,
        # easily out-of-sync property from the Rooms list's selection, so a
        # user who picked their room via the Rooms panel (or clicked it in
        # the viewport) could have it point at an entirely different floor,
        # silently searching the wrong floor's rooms for the path
        active = context.view_layer.objects.active
        room_fi = _room_index_for_object(active, s)
        self.floor_idx = s.rooms[room_fi].floor_index if room_fi is not None else s.floor_index
        fl = _floor_by_index(context, self.floor_idx)
        if not fl:
            self.report({'ERROR'}, "Select a floor in the Floors list")
            return {'CANCELLED'}
        if not any(r.floor_index == self.floor_idx for r in s.rooms):
            self.report({'ERROR'}, "No room outline on this floor - click 'Reset Room Outline' first")
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
        # Try the actual split against every room on this floor rather than
        # pre-checking whether some proxy point (the raw average of all
        # clicked points) sits inside one -- for a bent/L-shaped path near a
        # corner that average routinely lands outside the room even though
        # the path itself is a perfectly valid cut (split_polygon_path snaps
        # each endpoint to its nearest wall edge regardless of exactly where
        # it was clicked, so it doesn't need the points to be inside at all).
        s = context.scene.gn_int
        for i, r in enumerate(s.rooms):
            if r.floor_index != self.floor_idx:
                continue
            if split_room_record_path(context, i, self.pts):
                self.report({'INFO'}, "Room split")
                return
        self.report({'WARNING'},
                    "Split failed (path must cross a room on this floor, "
                    "ends on two different walls)")

    def _end(self, context, ok):
        bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
        context.area.header_text_set(None)
        context.area.tag_redraw()
        return {'FINISHED'} if ok else {'CANCELLED'}


class GN_OT_split_edges(Operator):
    bl_idname = "gn_int.split_edges"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Split at Selected Points"
    bl_description = ("Edit Mode: select two points (vertices) -- one on each "
                      "of two opposite walls -- and run this to cut exactly "
                      "between them. Subdivide a wall edge first (Blender's "
                      "own Subdivide/Loop Cut) to add a point wherever you "
                      "want the cut, then select it")

    @classmethod
    def poll(cls, context):
        ob = context.edit_object
        return ob is not None and ob.type == 'MESH'

    def execute(self, context):
        ob = context.edit_object
        bm = bmesh.from_edit_mesh(ob.data)
        mw = ob.matrix_world
        # cut line = the selected POINTS themselves, at their exact position
        # -- selecting an edge selects both its vertices too, so this covers
        # edge selection as well (using the edge's two endpoints), but the
        # intended way to work is Vertex select mode: pick precisely where
        # you want the cut to start and end
        pts = []
        for v in bm.verts:
            if v.select:
                w = mw @ v.co
                pts.append(Vector((w.x, w.y)))
        # collapse near-coincident points
        uniq = []
        for p in pts:
            if not any((p - q).length < 0.05 for q in uniq):
                uniq.append(p)
        if len(uniq) < 2:
            self.report({'ERROR'}, "Select two points on opposite walls")
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
        # which room? use the object's uid tag, else the room under the cut
        # midpoint -- ONLY among rooms on the same floor as the edited
        # object, otherwise a point that happens to fall inside some other
        # floor's room polygon (entirely possible -- floors routinely
        # overlap in X/Y, they're only separated in Z) would silently split
        # the wrong floor
        ruid = ob.get("gn_room_uid")
        mid = (A + B) * 0.5
        bpy.ops.object.mode_set(mode='OBJECT')
        s = context.scene.gn_int
        ridx = -1
        if ruid is not None:
            ridx = next((i for i, r in enumerate(s.rooms) if r.uid == ruid), -1)
        if ridx < 0:
            floor_idx = _floor_index_for_object(ob, s)
            for i, r in enumerate(s.rooms):
                if floor_idx is not None and r.floor_index != floor_idx:
                    continue
                try:
                    poly = [Vector(pt) for pt in json.loads(r.poly_json)]
                except Exception:
                    continue
                if _pt_in_poly(mid, poly):
                    ridx = i
                    break
        if ridx < 0:
            if not s.rooms:
                self.report({'WARNING'}, "No room outline yet - click 'Reset Room Outline' first")
            else:
                self.report({'WARNING'}, "Could not resolve which room these edges belong to")
            return {'CANCELLED'}
        if split_room_record(context, ridx, A, B):
            self.report({'INFO'}, "Room split between the selected points")
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


_STAIR_MATS = (("GN_StairTop", (0.55, 0.45, 0.35, 1.0)),   # 0 -- treads
              ("GN_StairSide", (0.55, 0.55, 0.57, 1.0)))   # 1 -- risers, side wedges, soffit
MAT_STAIR_TOP, MAT_STAIR_SIDE = 0, 1
_STAIR_NOSING_WIDTH_MULT = 1.8  # nosing overhang is wider than it is deep


def _build_stairs_between_edges(b0, b1, t0, t1, step_height, step_depth, nosing=0.0):
    """Solid staircase (treads, risers, closed sides, flat bottom, flat back)
    running from edge (b0,b1) up to edge (t0,t1). Each edge is assumed
    roughly level (flat at its own Z); the two edges need not be parallel or
    the same length -- the sides taper linearly between them. nosing > 0 cuts
    a small flat chamfer at each tread's top-front corner (depth = nosing on
    both the tread and the riser) -- NOT an overhang, NOT a curve: the tread
    and riser keep their normal positions, only that one small corner is cut
    away. The solid is a plain stepped block: flat at z_bot underneath and
    flat at the back (t=1), NOT a smooth diagonal soffit -- that read as a
    bizarre diagonal wedge cut through every step rather than a normal
    staircase silhouette. Returns (verts, faces, cats) in world space --
    cats parallels faces with a MAT_STAIR_* index per face -- or None if the
    edges are too close in height or in the travel direction to form a run."""
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

    def fwd(fx, fy, bx, by):
        v = Vector((bx - fx, by - fy))
        return v.normalized() if v.length > 1e-9 else Vector((0.0, 0.0))

    def fillet_arc(center, fdir, p_from, p_to, r, segs=4):
        # segs+1 points tracing a circular arc of radius r about `center`,
        # from p_from to p_to, confined to the vertical plane spanned by the
        # horizontal unit direction fdir and world +Z (true for every corner
        # here, since each fillet only ever moves along the tread-depth
        # direction and straight up/down). Endpoints must already sit
        # exactly on that circle -- this only fills in the curve between
        # them, it does not enforce the radius itself.
        def ang_of(p):
            a = (p[0] - center[0]) * fdir.x + (p[1] - center[1]) * fdir.y
            b = p[2] - center[2]
            return math.atan2(b, a)
        ang1, ang2 = ang_of(p_from), ang_of(p_to)
        while ang2 - ang1 > math.pi:
            ang2 -= 2 * math.pi
        while ang2 - ang1 < -math.pi:
            ang2 += 2 * math.pi
        pts = []
        for i in range(segs + 1):
            ang = ang1 + (ang2 - ang1) * (i / segs)
            ca, cb = math.cos(ang) * r, math.sin(ang) * r
            pts.append((center[0] + ca * fdir.x, center[1] + ca * fdir.y, center[2] + cb))
        return pts

    verts, faces, cats = [], [], []

    def quad(a, b, c, d_, cat):
        base = len(verts)
        verts.extend([a, b, c, d_])
        faces.append((base, base + 1, base + 2, base + 3))
        cats.append(cat)

    def tri(a, b, c, cat):
        base = len(verts)
        verts.extend([a, b, c])
        faces.append((base, base + 1, base + 2))
        cats.append(cat)

    def step_nosing_r(rise_):
        # R itself gets scaled up by _STAIR_NOSING_WIDTH_MULT for the
        # vertical drop face -- clamp so THAT scaled value still fits
        # within the riser, not R before scaling
        return min(nosing, rise_ * 0.9 / _STAIR_NOSING_WIDTH_MULT) if nosing > 1e-4 else 0.0

    # treads, risers, and the rounded nosing underside (per step, spanning
    # the full width from side0 to side1)
    for i in range(n):
        tf, tb = i / n, (i + 1) / n
        zl, zh = z_bot + i * rise, z_bot + (i + 1) * rise
        x0f, y0f = side_xy(b0, t0, tf); x1f, y1f = side_xy(b1, t1, tf)
        x0b, y0b = side_xy(b0, t0, tb); x1b, y1b = side_xy(b1, t1, tb)
        back0, back1 = (x0b, y0b, zh), (x1b, y1b, zh)
        bot0, bot1 = (x0f, y0f, zl), (x1f, y1f, zl)
        R = step_nosing_r(rise)

        if R > 1e-4:
            # square-edge nosing: right angles only, no curve, no diagonal.
            # The tread overhangs forward by R (tip), drops straight down by
            # Rd (the small vertical face), then tucks straight back to the
            # riser's own plane at xf (a small horizontal face) before the
            # riser continues straight down as normal. Rd > R so the
            # vertical face reads as a distinct lip rather than a square nub.
            Rd = R * _STAIR_NOSING_WIDTH_MULT
            # smooth rounded fillet at the two sharp right-angle corners of
            # the notch (tip<->drop and drop<->tuck) -- a small quarter-
            # circle arc, built as an explicit quad strip between the two
            # step sides (never a single n-gon), so it stays safe the same
            # way the square notch itself does. Bv < min(R, Rd) so each
            # fillet stays within its own two adjacent edges.
            Bv = min(R, Rd) * 0.35
            f0 = fwd(x0f, y0f, x0b, y0b)
            f1 = fwd(x1f, y1f, x1b, y1b)
            tip0 = (x0f - f0.x * R, y0f - f0.y * R, zh)
            tip1 = (x1f - f1.x * R, y1f - f1.y * R, zh)
            drop0 = (tip0[0], tip0[1], zh - Rd)
            drop1 = (tip1[0], tip1[1], zh - Rd)
            tuck0 = (x0f, y0f, zh - Rd)
            tuck1 = (x1f, y1f, zh - Rd)

            tipC0 = (tip0[0] + f0.x * Bv, tip0[1] + f0.y * Bv, zh - Bv)
            tipC1 = (tip1[0] + f1.x * Bv, tip1[1] + f1.y * Bv, zh - Bv)
            tipArc0 = fillet_arc(tipC0, f0, (tip0[0] + f0.x * Bv, tip0[1] + f0.y * Bv, zh), (tip0[0], tip0[1], zh - Bv), Bv)
            tipArc1 = fillet_arc(tipC1, f1, (tip1[0] + f1.x * Bv, tip1[1] + f1.y * Bv, zh), (tip1[0], tip1[1], zh - Bv), Bv)

            dropC0 = (drop0[0] + f0.x * Bv, drop0[1] + f0.y * Bv, drop0[2] + Bv)
            dropC1 = (drop1[0] + f1.x * Bv, drop1[1] + f1.y * Bv, drop1[2] + Bv)
            dropArc0 = fillet_arc(dropC0, f0, (drop0[0], drop0[1], drop0[2] + Bv), (drop0[0] + f0.x * Bv, drop0[1] + f0.y * Bv, drop0[2]), Bv)
            dropArc1 = fillet_arc(dropC1, f1, (drop1[0], drop1[1], drop1[2] + Bv), (drop1[0] + f1.x * Bv, drop1[1] + f1.y * Bv, drop1[2]), Bv)

            # the whole nosing lip -- tread out to the tip fillet, down the
            # vertical face, round the drop fillet, back to the riser plane
            # -- reads as one continuous decorative overhang, so it all
            # stays MAT_STAIR_TOP. Only the real riser below it (bot to
            # tuck) is MAT_STAIR_SIDE.
            quad(back0, back1, tipArc1[0], tipArc0[0], MAT_STAIR_TOP)
            for k in range(len(tipArc0) - 1):
                quad(tipArc0[k], tipArc1[k], tipArc1[k + 1], tipArc0[k + 1], MAT_STAIR_TOP)
            quad(tipArc0[-1], tipArc1[-1], dropArc1[0], dropArc0[0], MAT_STAIR_TOP)
            for k in range(len(dropArc0) - 1):
                quad(dropArc0[k], dropArc1[k], dropArc1[k + 1], dropArc0[k + 1], MAT_STAIR_TOP)
            quad(dropArc0[-1], dropArc1[-1], tuck1, tuck0, MAT_STAIR_TOP)
            quad(bot0, bot1, tuck1, tuck0, MAT_STAIR_SIDE)
        else:
            quad(back0, back1, (x1f, y1f, zh), (x0f, y0f, zh), MAT_STAIR_TOP)
            quad(bot0, bot1, (x1f, y1f, zh), (x0f, y0f, zh), MAT_STAIR_SIDE)

    # flat bottom (z_bot, full run) and flat back (t=1, full height) --
    # a plain block, not a diagonal soffit
    quad((b0.x, b0.y, z_bot), (b1.x, b1.y, z_bot), (t1.x, t1.y, z_bot), (t0.x, t0.y, z_bot), MAT_STAIR_SIDE)
    quad((t0.x, t0.y, z_bot), (t1.x, t1.y, z_bot), (t1.x, t1.y, z_top), (t0.x, t0.y, z_top), MAT_STAIR_SIDE)

    # sides: one flat face PER STEP, each reaching all the way down to
    # z_bot (not stacked on the previous step) -- consecutive step faces
    # sit side by side at different X ranges, so together they trace the
    # staircase silhouette with plain, planar quads. A single fan spanning
    # the whole run (the previous approach) is technically valid but its
    # internal diagonals, radiating from one far corner across every step,
    # show up as a visible mess of crossing lines -- this per-step version
    # is how room_tool.py builds its own stair stringers, for the same
    # reason ("avoid the concave N-gon fan triangulation that produced
    # diagonal artifacts").
    for b, t in ((b0, t0), (b1, t1)):
        for i in range(n):
            tf, tb = i / n, (i + 1) / n
            zh = z_bot + (i + 1) * rise
            xf, yf = side_xy(b, t, tf)
            xb, yb = side_xy(b, t, tb)
            R = step_nosing_r(rise)
            if R > 1e-4:
                # same square notch as the tread/riser above -- the full
                # per-step outline is concave at the notch (it juts out past
                # xf), so a single face there would leave the renderer to
                # auto-triangulate a concave shape, which is exactly what
                # produced the wrong (spiky) result before regardless of how
                # correct the boundary edges were. Split explicitly instead:
                # a plain, safely-convex quad for the step's main body (down
                # to the tuck corner, not z_bot to zh), plus 2 small
                # triangles -- anchored at the NEAR (xb, zh) corner, not a
                # far one -- closing just the small notch on its own.
                Rd = R * _STAIR_NOSING_WIDTH_MULT
                Bv = min(R, Rd) * 0.35
                f = fwd(xf, yf, xb, yb)
                tip = (xf - f.x * R, yf - f.y * R, zh)
                drop = (tip[0], tip[1], zh - Rd)
                tuck = (xf, yf, zh - Rd)
                tipC = (tip[0] + f.x * Bv, tip[1] + f.y * Bv, zh - Bv)
                tip_arc = fillet_arc(tipC, f, (tip[0] + f.x * Bv, tip[1] + f.y * Bv, zh), (tip[0], tip[1], zh - Bv), Bv)
                dropC = (drop[0] + f.x * Bv, drop[1] + f.y * Bv, drop[2] + Bv)
                drop_arc = fillet_arc(dropC, f, (drop[0], drop[1], drop[2] + Bv), (drop[0] + f.x * Bv, drop[1] + f.y * Bv, drop[2]), Bv)
                quad((xf, yf, z_bot), (xb, yb, z_bot), (xb, yb, zh), tuck, MAT_STAIR_SIDE)
                anchor = (xb, yb, zh)
                boundary = tip_arc + drop_arc[1:] + [tuck]
                for k in range(len(boundary) - 1):
                    tri(anchor, boundary[k], boundary[k + 1], MAT_STAIR_SIDE)
            else:
                quad((xf, yf, z_bot), (xb, yb, z_bot), (xb, yb, zh), (xf, yf, zh), MAT_STAIR_SIDE)

    return verts, faces, cats


def _apply_stair_mesh(ob, verts, faces, cats):
    """Write world-space verts/faces/cats (from _build_stairs_between_edges)
    into ob's existing mesh data, converting through the object's CURRENT
    matrix_world (so this still works if the stair object has been moved or
    rotated since it was created). cats assigns each face's material index
    (GN_StairTop for treads, GN_StairSide for everything else) -- material
    slots are (re)built fresh each time so a rebuild can't drift out of sync
    with face order."""
    mw_inv = ob.matrix_world.inverted()
    bm = bmesh.new()
    bverts = [bm.verts.new(mw_inv @ Vector(v)) for v in verts]
    cat_layer = bm.faces.layers.int.new("gn_stair_cat")
    for f, cat in zip(faces, cats):
        try:
            bf = bm.faces.new([bverts[i] for i in f])
            bf[cat_layer] = cat
        except ValueError:
            pass
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    ob.data.materials.clear()
    for name, color in _STAIR_MATS:
        ob.data.materials.append(_get_mat(name, color))
    cat_layer = bm.faces.layers.int["gn_stair_cat"]
    for bf in bm.faces:
        bf.material_index = bf[cat_layer]
    # smooth shading by angle: smooth across the rounded nosing arc (small
    # angle between its segments), crisp at real corners (tread/riser, ~90deg)
    ANGLE_THRESH = math.radians(35.0)
    for bf in bm.faces:
        bf.smooth = True
    for be in bm.edges:
        if len(be.link_faces) == 2:
            try:
                be.smooth = be.calc_face_angle() <= ANGLE_THRESH
            except Exception:
                be.smooth = False
        else:
            be.smooth = False
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()


def _rebuild_stair_mesh(ob, height, depth, nosing=0.0):
    """Re-run the stair build for an EXISTING stair object at new
    height/depth/nosing settings, using the original two edges stored on it
    at creation time. Returns True if ob was a stair and got rebuilt."""
    raw = ob.get("gn_stair_data")
    if not raw:
        return False
    try:
        d = json.loads(raw)
        b0, b1 = Vector(d["b0"]), Vector(d["b1"])
        t0, t1 = Vector(d["t0"]), Vector(d["t1"])
    except Exception:
        return False
    built = _build_stairs_between_edges(b0, b1, t0, t1, height, depth, nosing)
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
                      "floor maps) and build a solid flight of stairs "
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
                                            s.stair_step_height, s.stair_step_depth,
                                            s.stair_nosing)
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
        no_fit = []            # (width, height) of openings nothing fits
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
            if not op.is_door:
                # the piece's own bounds, BEFORE they're overwritten below with
                # the placed window's size -- a later swap still needs to know
                # how much room the exterior opening actually gives it
                op.avail_w, op.avail_h = op.hw * 2, op.top - op.sill
                fit = _best_fit_window_preset(context, op.avail_w, op.avail_h)
                if fit:
                    preset, w, h, rotated = fit
                    op.win_key = preset.library_key or preset.name
                    op.win_rotated = rotated
                    # hole is sized to the window actually being placed, not
                    # the exterior piece -- the piece was only a size reference
                    # for picking the fit. Re-center on the piece's own center,
                    # and cut it slightly SMALLER than the frame (FRAME_OVERLAP)
                    # so the frame overlaps the rough edge instead of leaving
                    # a visible gap between wall and window.
                    op.hw = max(w * 0.5 - FRAME_OVERLAP, 0.01)
                    op.sill = center.z - h * 0.5 + FRAME_OVERLAP
                    op.top = center.z + h * 0.5 - FRAME_OVERLAP
                    _place_frame_mesh(op, preset.mesh_object, w, h, 'WINDOW', rotated=rotated, context=context)
                    op.win_w, op.win_h = w, h
                    op.win_allow_curtain = preset.allow_curtain
                    op.win_allow_blinds = preset.allow_blinds
                else:
                    # nothing in the library fits -- leave the hole with no
                    # window rather than squeezing one in distorted
                    no_fit.append((op.avail_w, op.avail_h))
            made += 1
        rebuild_rooms(context)
        msg = f"Projected {made} selected opening(s)"
        if skipped:
            msg += f" ({skipped} already existed)"
        if no_fit:
            sizes = ", ".join(f"{w:.2f}x{h:.2f}" for w, h in no_fit[:3])
            if len(no_fit) > 3:
                sizes += f", +{len(no_fit) - 3} more"
            self.report({'WARNING'},
                        f"{msg} -- no library window fits {len(no_fit)} of them "
                        f"({sizes}); those got a hole but no window")
        else:
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
                      "or its floor map was regenerated). Also removes their "
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
        _clear_coll(CURTAIN_COLL)
        rebuild_rooms(context)
        return {'FINISHED'}


# ===========================================================================
# doors (Room-Tool-style edit mode)
# ===========================================================================
def _active_preset(s, kind):
    """Return (width, height, sill, mesh_object) for the active door/window
    preset. Windows use the scene's single default_window_sill for every
    manually-placed window -- not each preset's own registered sill --
    unless that preset is flagged starts_at_floor, in which case it starts
    at 0.0 (the room's floor) regardless of the default."""
    if kind == 'DOOR':
        if 0 <= s.active_door_preset < len(s.door_presets):
            p = s.door_presets[s.active_door_preset]
            return p.width, p.height, 0.0, p.mesh_object
        return 0.9, 2.0, 0.0, None
    if 0 <= s.active_window_preset < len(s.window_presets):
        p = s.window_presets[s.active_window_preset]
        sill = 0.0 if p.starts_at_floor else s.default_window_sill
        return p.width, p.height, sill, p.mesh_object
    return 1.0, 1.2, s.default_window_sill, None


def _window_bases(context):
    """Every window this project can place, as (source, width, height): its
    own presets (hand-made ones and previously pulled-in library windows)
    plus every not-yet-local entry in the shared library catalog. `source`
    is a GN_WindowPreset (already has a mesh_object) or a GN_WindowLibItem
    (needs _pull_library_entry_into_project first).

    For a preset that has a real mesh, width/height are MEASURED from that
    mesh, not read from the catalog -- the catalog's numbers can disagree
    with the geometry they point at (e.g. a sidecar edited without resyncing
    the .blend), and trusting them makes both fit-checking and placement
    stretch the window. Library-only entries have no local mesh to measure,
    so they fall back to the catalog until pulled in."""
    s = context.scene.gn_int
    local_keys = {p.library_key for p in s.window_presets if p.library_key}
    bases = []
    for p in s.window_presets:
        if p.mesh_object is not None:
            native = _native_frame_size(p.mesh_object)
            w, h = native if native else (p.width, p.height)
            bases.append((p, w, h))
    for it in s.winlib_items:
        if it.category == 'window' and it.key not in local_keys:
            bases.append((it, it.width, it.height))
    return bases


def _window_candidates(context, avail_w=0.0, avail_h=0.0):
    """_window_bases expanded into placement candidates
    (source, width, height, rotated), largest-area first. A can_rotate
    window also appears sideways, competing on equal footing. avail_w/
    avail_h > 0 keeps only what fits inside them; 0 means unconstrained
    (a manually placed window has no exterior opening bounding it)."""
    candidates = []
    for source, w, h in _window_bases(context):
        candidates.append((source, w, h, False))
        if source.can_rotate:
            candidates.append((source, h, w, True))
    if avail_w > 1e-4 and avail_h > 1e-4:
        # 2mm slack: sub-millimetre float noise (rounded catalog values, a
        # frozen baseline recomputed from a placed frame) must not be what
        # decides whether a window fits an opening
        candidates = [c for c in candidates
                      if c[1] <= avail_w + FIT_TOL and c[2] <= avail_h + FIT_TOL]
    return sorted(candidates, key=lambda c: c[1] * c[2], reverse=True)


def _candidate_key(source):
    """Stable id for a candidate, so an opening can record which window is
    placed there and later be matched back against the candidate list."""
    if isinstance(source, GN_WindowLibItem):
        return source.key
    return source.library_key or source.name


def _opening_avail(op):
    """Bounds a swapped-in window must fit inside for this opening. Falls
    back to the currently placed window's own size for projected openings
    recorded before avail_w/avail_h existed -- the best information left
    once hw/sill/top were resized to the placed window."""
    if op.avail_w > 1e-4 and op.avail_h > 1e-4:
        return op.avail_w, op.avail_h
    if op.projected:
        return (op.win_w or op.hw * 2), (op.win_h or (op.top - op.sill))
    return 0.0, 0.0


def _native_frame_size(mesh_src):
    """A mesh's own width/height in the axis convention _place_dressing_mesh
    uses (Z = height, the larger horizontal extent = width). Placing at these
    means the mesh isn't scaled at all -- no squashing when a catalog entry's
    recorded width/height disagrees with the geometry it points at."""
    if mesh_src is None or not mesh_src.data:
        return None
    bb = [Vector(c) for c in mesh_src.bound_box]
    xs = [c.x for c in bb]; ys = [c.y for c in bb]; zs = [c.z for c in bb]
    dx, dy, dz = max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)
    if dz < 1e-5 or max(dx, dy) < 1e-5:
        return None
    return max(dx, dy), dz


def _best_fit_window_preset(context, avail_w, avail_h):
    """Pick a window to fill a projected exterior opening of size avail_w x
    avail_h -- searching BOTH this project's already-local presets (hand-
    made ones like "Standard", and any previously pulled-in library
    windows) AND every window in the full shared library catalog, so
    auto-fit isn't limited to whatever's been manually added to the
    project first. Among candidates that fit within bounds, returns the
    one with the largest area (closest fit without exceeding) at its own
    native width/height -- it may leave a small reveal gap inside the
    rough opening, which is expected. A can_rotate candidate is also tried
    sideways (width/height swapped), competing on equal footing with its
    normal orientation. If NOTHING fits, returns None -- the opening still
    gets its hole, just no window, and the caller reports it. Squeezing the
    smallest window down to the opening (what this used to do) distorts it
    non-uniformly, which reads worse than an honest "nothing fits".
    If the winner came from the library rather than the project's own
    presets, it's pulled into the project on the spot (appended + a preset
    created) so it has a real mesh_object to place. A candidate that fails
    to pull in (e.g. a stale catalog entry whose object no longer exists in
    the library file) is skipped in favour of the next-best fit rather than
    failing the whole lookup -- one broken library entry shouldn't silently
    stop every projection.
    Returns (preset, width, height, rotated) or None if nothing usable."""
    bases = _window_bases(context)
    if not bases:
        return None
    fits = _window_candidates(context, avail_w, avail_h)

    def _resolve(source):
        if not isinstance(source, GN_WindowLibItem):
            return source
        resolved, err = _pull_library_entry_into_project(context, source)
        return resolved

    for source, w, h, rotated in fits:
        resolved = _resolve(source)
        if resolved is not None:
            return resolved, w, h, rotated
    return None


def _apply_window_to_opening(context, op, source, w, h, rotated):
    """Put `source` (at w x h) into an existing opening, re-cutting the hole
    to the new window and re-placing the frame. The opening stays where it
    is: a projected one keeps its CENTRE (that's how projection positioned
    it inside the exterior piece), a manually placed one keeps its SILL
    (that's how add_opening positioned it). Hole/frame keep the same
    FRAME_OVERLAP relationship as first placement.
    Returns None on success, an error message otherwise."""
    preset = source
    if isinstance(source, GN_WindowLibItem):
        preset, err = _pull_library_entry_into_project(context, source)
        if preset is None:
            return err or "Could not pull that window in from the library"
    if preset.mesh_object is None:
        return f"'{preset.name}' has no mesh"
    if op.projected and op.avail_w <= 1e-4:
        # freeze the available space BEFORE swapping. _opening_avail falls back
        # to the currently placed window for openings made before avail_* was
        # recorded -- without pinning it here, swapping to a smaller window
        # would shrink the budget and you could never swap back up again.
        op.avail_w, op.avail_h = _opening_avail(op)
    if not op.projected:
        # nothing is constraining this opening, so place the window at its
        # OWN size rather than scaling it to the catalog's numbers (which
        # squashes it whenever the two disagree). Projected openings keep
        # the fitted size -- there the exterior piece IS the constraint.
        native = _native_frame_size(preset.mesh_object)
        if native:
            w, h = (native[1], native[0]) if rotated else native
    if op.projected:
        center_z = (op.sill + op.top) * 0.5
        op.sill = center_z - h * 0.5 + FRAME_OVERLAP
        op.top = center_z + h * 0.5 - FRAME_OVERLAP
    else:
        op.top = op.sill + h - 2 * FRAME_OVERLAP
    op.hw = max(w * 0.5 - FRAME_OVERLAP, 0.01)
    op.win_w, op.win_h = w, h
    op.win_rotated = rotated
    op.win_key = _candidate_key(source)
    op.win_allow_curtain = preset.allow_curtain
    op.win_allow_blinds = preset.allow_blinds
    rebuild_rooms(context)
    _place_frame_mesh(op, preset.mesh_object, w, h, 'WINDOW',
                      rotated=rotated, context=context)
    return None


class GN_OT_swap_window(Operator):
    bl_idname = "gn_int.swap_window"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Swap Window"
    bl_description = ("Replace the window at this opening with a different one. "
                      "Only windows that still fit the original exterior opening "
                      "are offered")
    key: StringProperty()
    rotated: BoolProperty(default=False)
    index: IntProperty(default=-1)      # -1 = the selected opening

    def execute(self, context):
        s = context.scene.gn_int
        i = self.index if self.index >= 0 else s.opening_index
        if not (0 <= i < len(s.openings)):
            self.report({'WARNING'}, "No opening selected")
            return {'CANCELLED'}
        op = s.openings[i]
        if op.is_door:
            self.report({'ERROR'}, "That's a door, not a window")
            return {'CANCELLED'}
        avail_w, avail_h = _opening_avail(op)
        # candidates for this key that fit, in both orientations -- prefer the
        # requested one, else fall back to the other (a can_rotate window that
        # only fits sideways still fits)
        options = [c for c in _window_candidates(context, avail_w, avail_h)
                   if _candidate_key(c[0]) == self.key]
        match = next((c for c in options if c[3] == self.rotated), None) or \
            (options[0] if options else None)
        if match is None:
            self.report({'ERROR'},
                        f"That window doesn't fit this opening "
                        f"({avail_w:.2f} x {avail_h:.2f} available)")
            return {'CANCELLED'}
        source, w, h, rotated = match
        err = _apply_window_to_opening(context, op, source, w, h, rotated)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        self.report({'INFO'}, f"Swapped in '{source.name}'")
        return {'FINISHED'}


# ===========================================================================
# shared Window Library (cross-project, mirrors gn_mat_library's shared
# folder pattern -- but ONE library .blend instead of a folder of DDS files,
# with a JSON sidecar for the catalog since reading a library .blend's own
# object custom properties needs fully linking each one first)
# ===========================================================================
def _win_lib_prefs():
    try:
        return bpy.context.preferences.addons[__name__].preferences
    except Exception:
        return None


def _win_lib_sidecar_path(blend_path):
    base, _ext = os.path.splitext(blend_path)
    return base + ".windows.json"


def _bundled_win_lib_path():
    """The window library shipped INSIDE the add-on folder. Lets the tool
    work out of the box on a fresh install (any OS -- Windows included)
    without the user first pointing Preferences at a file."""
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(here, "window_library", "window_library.blend")
    return p if os.path.isfile(p) else ""


def _win_lib_path():
    """The active window library: the Preferences path if it's set and the
    file exists, else the bundled one. Everything that reads or writes the
    library goes through here, so the fallback is consistent and paths are
    always absolute (Blender's // relative form resolved)."""
    prefs = _win_lib_prefs()
    p = prefs.window_library_path if prefs else ""
    if p:
        ap = bpy.path.abspath(p)
        if os.path.isfile(ap):
            return ap
    return _bundled_win_lib_path()


class GN_IntPrefs(AddonPreferences):
    bl_idname = __name__
    window_library_path: StringProperty(
        name="Window Library", subtype='FILE_PATH',
        description="Optional: a shared .blend of library windows used by ALL "
        "projects. Leave EMPTY to use the library bundled with the add-on. "
        "Point this at your own file to share one library across a team/machine")

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "window_library_path")
        bundled = _bundled_win_lib_path()
        if not self.window_library_path:
            if bundled:
                layout.label(text="Using the bundled window library.", icon='CHECKMARK')
            else:
                layout.label(text="No bundled library found -- set a path above.",
                             icon='ERROR')
        layout.operator("gn_int.win_lib_refresh", icon='FILE_REFRESH')


def rescan_window_library(context):
    """Reload the shared Window Library's catalog into s.winlib_items from
    its JSON sidecar (name/width/height/sill/object_name per entry).
    Returns the number of entries found."""
    s = context.scene.gn_int
    s.winlib_items.clear()
    path = _win_lib_path()
    if not path or not os.path.isfile(path):
        return 0
    sidecar = _win_lib_sidecar_path(path)
    if not os.path.isfile(sidecar):
        return 0
    try:
        with open(sidecar, "r") as f:
            data = json.load(f)
    except Exception:
        return 0
    for entry in data.get("windows", []):
        key = entry.get("key")
        obj_name = entry.get("object_name")
        if not key or not obj_name:
            continue
        it = s.winlib_items.add()
        it.key = key
        it.object_name = obj_name
        it.name = entry.get("name", key)
        it.width = entry.get("width", 1.0)
        it.height = entry.get("height", 1.2)
        it.sill = entry.get("sill", 0.9)
        it.category = entry.get("category", "")
        it.can_rotate = entry.get("can_rotate", False)
        it.allow_curtain = entry.get("allow_curtain", True)
        it.allow_blinds = entry.get("allow_blinds", True)
        it.starts_at_floor = entry.get("starts_at_floor", False)
    return len(s.winlib_items)


class GN_OT_win_lib_refresh(Operator):
    bl_idname = "gn_int.win_lib_refresh"
    bl_options = {'REGISTER'}
    bl_label = "Refresh Window Library"
    bl_description = "Rescan the shared Window Library's catalog"

    def execute(self, context):
        n = rescan_window_library(context)
        path = _win_lib_path()
        if not path:
            self.report({'WARNING'},
                        "No window library found -- reinstall the add-on or set "
                        "one in Preferences")
        else:
            bundled = (path == _bundled_win_lib_path())
            self.report({'INFO'},
                        f"{n} window(s) in library" + (" (bundled)" if bundled else ""))
        return {'FINISHED'}


_CURTAIN_CATEGORIES = {"curtain", "blinds"}


def _pull_library_entry_into_project(context, entry):
    """Append (or, if editing the library file itself, just reference) the
    given catalog entry's object and add it as a project preset -- a
    window entry goes to window_presets, a curtain/blinds entry to
    curtain_presets. Shared by the explicit 'Add to Project' button AND
    the window auto-fit (_best_fit_window_preset), which pulls a library
    window in automatically the moment it's chosen -- no manual per-window
    curation required first.
    Returns (preset, None) on success, (None, error_message) on failure."""
    s = context.scene.gn_int
    is_curtain = entry.category in _CURTAIN_CATEGORIES
    target = s.curtain_presets if is_curtain else s.window_presets
    existing = next((p for p in target if p.library_key == entry.key), None)
    if existing is not None:
        return existing, None
    path = _win_lib_path()
    if not path or not os.path.isfile(path):
        return None, "Window Library file not set or missing"
    editing_library_itself = (bpy.data.filepath and
        os.path.abspath(bpy.data.filepath) == os.path.abspath(path))
    if editing_library_itself:
        # can't append a file into itself -- the object is already local,
        # just use it directly
        appended = bpy.data.objects.get(entry.object_name)
        if appended is None:
            return None, f"'{entry.object_name}' not found in this file"
    else:
        with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
            if entry.object_name not in data_from.objects:
                return None, f"'{entry.object_name}' not found in the library file"
            data_to.objects = [entry.object_name]
        appended = data_to.objects[0] if data_to.objects else None
        if appended is None:
            return None, "Append failed"
        if appended.name not in _get_coll(WINLIB_COLL).objects:
            _get_coll(WINLIB_COLL).objects.link(appended)
    p = target.add()
    p.name = entry.name
    p.mesh_object = appended
    p.library_key = entry.key
    if is_curtain:
        p.category = entry.category
        s.active_curtain_preset = len(s.curtain_presets) - 1
    else:
        p.width = entry.width
        p.height = entry.height
        p.sill = entry.sill
        p.can_rotate = entry.can_rotate
        p.allow_curtain = entry.allow_curtain
        p.allow_blinds = entry.allow_blinds
        p.starts_at_floor = entry.starts_at_floor
        s.active_window_preset = len(s.window_presets) - 1
    return p, None


def _resync_mesh_from_library(context, preset):
    """Re-append preset's source object from the shared library file and
    replace its already-local mesh's geometry IN PLACE (same Mesh ID, not
    a new datablock) -- every placed instance (GN_Frame_*/GN_Curtain_*)
    shares that one Mesh ID via _place_dressing_mesh, so this updates all
    of them at once without touching the scene. Also re-copies the
    catalog's width/height/sill/can_rotate/allow_curtain/allow_blinds onto
    the preset (window presets only -- curtain presets have no such
    fields), since those are cached separately from the mesh and drive
    wall-hole sizing for manual placement -- geometry alone getting out of
    sync with them would cut holes the wrong size. Fixes edits made
    directly in the library .blend (e.g. corrected normals, resized
    windows) not reaching windows already pulled into a project before the
    edit -- 'Refresh' only reloads the JSON sidecar's metadata display, it
    never touches an already-local preset's mesh or cached dimensions.
    Returns None on success, an error message on failure."""
    s = context.scene.gn_int
    if preset.mesh_object is None or preset.mesh_object.data is None:
        return "Preset has no mesh"
    path = _win_lib_path()
    if not path or not os.path.isfile(path):
        return "Window Library file not set or missing"
    if bpy.data.filepath and os.path.abspath(bpy.data.filepath) == os.path.abspath(path):
        return "Editing the library file itself -- nothing to resync"
    entry = next((it for it in s.winlib_items if it.key == preset.library_key), None)
    src_name = entry.object_name if entry else preset.mesh_object.name
    with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
        if src_name not in data_from.objects:
            return f"'{src_name}' not found in the library file"
        data_to.objects = [src_name]
    fresh = data_to.objects[0] if data_to.objects else None
    if fresh is None or fresh.data is None:
        return "Append failed"
    old_mesh = preset.mesh_object.data
    fresh_mesh = fresh.data
    bm = bmesh.new()
    bm.from_mesh(fresh_mesh)
    bm.to_mesh(old_mesh)
    bm.free()
    old_mesh.update()
    bpy.data.objects.remove(fresh, do_unlink=True)
    if fresh_mesh.users == 0:
        bpy.data.meshes.remove(fresh_mesh)
    if entry is not None and hasattr(preset, "width"):
        preset.width = entry.width
        preset.height = entry.height
        preset.sill = entry.sill
        preset.can_rotate = entry.can_rotate
        preset.allow_curtain = entry.allow_curtain
        preset.allow_blinds = entry.allow_blinds
        preset.starts_at_floor = entry.starts_at_floor
    return None


class GN_OT_win_lib_resync_mesh(Operator):
    bl_idname = "gn_int.win_lib_resync_mesh"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Resync Mesh Geometry from Library"
    bl_description = ("Pull updated geometry (e.g. fixed normals) from the "
                      "shared library file into every window/curtain mesh "
                      "already used in this project. Refresh above only "
                      "reloads names/dimensions, never geometry -- use this "
                      "after editing meshes directly in the library file")

    def execute(self, context):
        s = context.scene.gn_int
        presets = [p for p in list(s.window_presets) + list(s.curtain_presets)
                  if p.library_key]
        if not presets:
            self.report({'INFO'}, "No library-sourced presets in this project")
            return {'CANCELLED'}
        updated, errors = 0, []
        for p in presets:
            err = _resync_mesh_from_library(context, p)
            if err:
                errors.append(f"{p.name}: {err}")
            else:
                updated += 1
        if errors:
            self.report({'WARNING'},
                        f"Resynced {updated}, {len(errors)} failed -- " + "; ".join(errors[:3]))
        else:
            self.report({'INFO'}, f"Resynced {updated} mesh(es)")
        return {'FINISHED'}


class GN_OT_win_lib_add_to_project(Operator):
    bl_idname = "gn_int.win_lib_add_to_project"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Pick"
    bl_description = ("For windows: makes this the active window for Window "
                      "Edit Mode, pulling it into the project first if needed. "
                      "For curtains/blinds: adds it to the project so it can "
                      "be picked from the Curtains list")
    key: StringProperty()

    def execute(self, context):
        s = context.scene.gn_int
        entry = next((it for it in s.winlib_items if it.key == self.key), None)
        if entry is None:
            self.report({'ERROR'}, "Library entry not found - try refreshing the library")
            return {'CANCELLED'}
        is_curtain = entry.category in _CURTAIN_CATEGORIES
        target = s.curtain_presets if is_curtain else s.window_presets
        existing_idx = next((i for i, p in enumerate(target) if p.library_key == self.key), -1)
        if existing_idx >= 0:
            if not is_curtain:
                s.active_window_preset = existing_idx
                self.report({'INFO'}, f"'{entry.name}' is now the active window")
                return {'FINISHED'}
            self.report({'INFO'}, f"'{entry.name}' is already in this project")
            return {'CANCELLED'}
        p, err = _pull_library_entry_into_project(context, entry)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        self.report({'INFO'}, f"Added '{p.name}' to project")
        return {'FINISHED'}


class GN_OT_win_lib_remove_from_project(Operator):
    bl_idname = "gn_int.win_lib_remove_from_project"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Remove from Project"
    bl_description = ("Remove this preset from the project. Its appended "
                      "object is also deleted, unless another preset still uses it")
    index: IntProperty(default=-1)
    kind: EnumProperty(items=[('WINDOW', "Window", ""), ('CURTAIN', "Curtain", "")],
                       default='WINDOW', options={'HIDDEN'})

    def execute(self, context):
        s = context.scene.gn_int
        presets = s.curtain_presets if self.kind == 'CURTAIN' else s.window_presets
        active_attr = "active_curtain_preset" if self.kind == 'CURTAIN' else "active_window_preset"
        i = self.index if self.index >= 0 else getattr(s, active_attr)
        if not (0 <= i < len(presets)):
            self.report({'WARNING'}, "No preset selected")
            return {'CANCELLED'}
        ob = presets[i].mesh_object
        presets.remove(i)
        setattr(s, active_attr, max(0, min(i, len(presets) - 1)))
        if ob is not None and not any(p.mesh_object == ob for p in presets):
            me = ob.data
            bpy.data.objects.remove(ob, do_unlink=True)
            if me and me.users == 0:
                bpy.data.meshes.remove(me)
        return {'FINISHED'}


class GN_OT_lib_register_window(Operator):
    bl_idname = "gn_int.lib_register_window"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Register Selected"
    bl_description = ("Add the active mesh object to the shared Window "
                      "Library as a new catalog entry (re-registering the "
                      "same object updates its entry in place). Backs up "
                      "the library file first")

    reg_name: StringProperty(name="Name", default="")
    reg_category: EnumProperty(name="Category", default='window',
        items=[('window', "Window", ""), ('curtain', "Curtain", ""),
               ('blinds', "Blinds", "")])
    reg_can_rotate: BoolProperty(name="Can Rotate (works horizontal or vertical)", default=False)
    reg_allow_curtain: BoolProperty(name="Allows Curtains", default=True)
    reg_allow_blinds: BoolProperty(name="Allows Blinds", default=True)
    reg_starts_at_floor: BoolProperty(name="Starts at Floor", default=False,
        description="Starts at the room's floor instead of the scene's "
        "default window sill height (e.g. a French window/floor-length window)")

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.type == 'MESH'

    def invoke(self, context, event):
        if not self.reg_name:
            self.reg_name = context.active_object.name.replace("_", " ").title()
        return context.window_manager.invoke_props_dialog(self, width=380)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "reg_name")
        layout.prop(self, "reg_category")
        if self.reg_category == 'window':
            layout.prop(self, "reg_can_rotate")
            layout.prop(self, "reg_allow_curtain")
            layout.prop(self, "reg_allow_blinds")
            layout.prop(self, "reg_starts_at_floor")

    def execute(self, context):
        path = _win_lib_path()
        if not path:
            self.report({'ERROR'}, "Set a Window Library file in add-on Preferences first")
            return {'CANCELLED'}
        ob = context.active_object
        sidecar = _win_lib_sidecar_path(path)

        data = {"windows": []}
        if os.path.isfile(sidecar):
            try:
                with open(sidecar, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {"windows": []}
        data.setdefault("windows", [])

        # re-registering the exact same source object updates its own entry
        # in place; otherwise mint a fresh key that doesn't collide
        same_obj_entry = next((e for e in data["windows"] if e.get("object_name") == ob.name), None)
        if same_obj_entry:
            key = same_obj_entry["key"]
        else:
            base_key = re.sub(r"[^A-Za-z0-9_]+", "_", (self.reg_name or ob.name)).strip("_") or ob.name
            existing_keys = {e.get("key") for e in data["windows"] if e.get("key")}
            key = base_key
            i = 2
            while key in existing_keys:
                key = f"{base_key}_{i}"
                i += 1

        # back up the current library file + sidecar before touching either
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_blend = None
        if os.path.isfile(path):
            backup_blend = f"{path}.backup_{timestamp}.blend"
            shutil.copyfile(path, backup_blend)
        if os.path.isfile(sidecar):
            shutil.copyfile(sidecar, f"{sidecar}.backup_{timestamp}.json")

        # merge: pull in every OTHER existing library object from the
        # (just-made) backup copy, linked -- never link from and overwrite
        # the same live path in one operation
        existing_objs = []
        read_from = backup_blend or path
        rename_map = {}
        if os.path.isfile(read_from):
            # append (link=False), not link -- these need to become fully
            # local datablocks in this session for libraries.write() to
            # reliably re-serialize them into the (different) live path.
            # If an object of the same name already exists in THIS session
            # (e.g. it was previously pulled into a project), Blender
            # renames the freshly-appended copy (a ".001" suffix) -- track
            # that so the sidecar's object_name stays in sync with whatever
            # actually ends up written into the file
            with bpy.data.libraries.load(read_from, link=False) as (data_from, data_to):
                requested = [n for n in data_from.objects if n != ob.name]
                requested_names = list(requested)  # keep a copy: assigning
                data_to.objects = requested         # this mutates `requested`
                                                     # in place (strings -> objects)
            existing_objs = [o for o in data_to.objects if o is not None]
            rename_map = dict(zip(requested_names, [o.name for o in existing_objs]))

        bpy.data.libraries.write(path, set(existing_objs) | {ob}, fake_user=True)

        for o in existing_objs:
            bpy.data.objects.remove(o, do_unlink=True)

        mn = [min(v[i] for v in ob.bound_box) for i in range(3)]
        mx = [max(v[i] for v in ob.bound_box) for i in range(3)]
        entry = {
            "key": key, "object_name": ob.name, "name": self.reg_name or ob.name,
            "width": round(mx[0] - mn[0], 3), "height": round(mx[2] - mn[2], 3),
            "sill": 0.9, "category": self.reg_category,
        }
        if self.reg_category == 'window':
            entry["can_rotate"] = self.reg_can_rotate
            entry["allow_curtain"] = self.reg_allow_curtain
            entry["allow_blinds"] = self.reg_allow_blinds
            entry["starts_at_floor"] = self.reg_starts_at_floor

        data["windows"] = [e for e in data["windows"] if e.get("key") != key]
        for e in data["windows"]:
            old_name = e.get("object_name")
            if old_name in rename_map:
                e["object_name"] = rename_map[old_name]
        data["windows"].append(entry)
        with open(sidecar, "w") as f:
            json.dump(data, f, indent=2)

        rescan_window_library(context)
        self.report({'INFO'}, f"Registered '{entry['name']}' to the library")
        return {'FINISHED'}


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


def _place_dressing_mesh(coll, name, mesh_src, cx, cy, nx, ny, sill, width, height,
                         rotate90=False, reveal_depth=0.0):
    """Instance mesh_src into coll as `name`, oriented to the wall normal
    (nx, ny), scaled to width/height, centred at (cx, cy) with its bottom
    at `sill`. Detects the mesh's own axes from its bounding box (Z=height,
    the larger horizontal extent=width, the smaller=depth), so any panel-
    like mesh orients correctly regardless of how it was modelled. Origin-
    agnostic (uses bbox). Shared by door/window frames and curtain/blind
    dressing -- re-running with the same `name` replaces it in place.

    rotate90=True places the mesh on its side: its own height axis (Z) runs
    ALONG the wall (filling `width`) and its wide axis runs vertically
    (filling `height`) instead -- a real geometric rotation, not just a
    non-uniform squash of the normal placement (for a can_rotate window
    used sideways to fit a tall/narrow opening).

    reveal_depth=0.0 places the mesh's own depth-bbox CENTRE at (cx, cy) --
    (cx, cy) sits on the room's interior wall face, so with reveal_depth=0
    the mesh straddles that plane, filling only its own native depth
    (typically far shallower than the reveal jamb cut into the wall,
    leaving the jamb cavity visibly empty). reveal_depth > 0.0 instead
    places the mesh's OUTWARD-facing side flush with the plane that many
    metres out along (nx, ny) -- i.e. flush with the far/exterior end of
    the reveal recess, filling it from the outside in, matching how the
    reveal jambs built by _build_wall are actually cut (see wall_margin/
    win_reveal). 'Outward' is resolved from the mesh's own final placement
    matrix (which axis actually ends up pointing along +n), not assumed,
    since the right-handed-determinant fix below can flip it."""
    if mesh_src is None or not mesh_src.data:
        return None
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

    n = Vector((nx, ny, 0.0)).normalized()
    along = Vector((-n.y, n.x, 0.0))
    up = Vector((0.0, 0.0, 1.0))
    wide_is_x = dx >= dy
    wide = dx if wide_is_x else dy

    if not rotate90:
        sz = height / dz if dz > 1e-4 else 1.0
        sw = width / wide if wide > 1e-4 else 1.0
        colZ = up * sz
        if wide_is_x:
            colX = along * sw; colY = n
        else:
            colX = n; colY = along * sw
    else:
        sz = width / dz if dz > 1e-4 else 1.0        # mesh's Z -> along the wall
        sw = height / wide if wide > 1e-4 else 1.0   # mesh's wide axis -> world up
        colZ = along * sz
        if wide_is_x:
            colX = up * sw; colY = n
        else:
            colX = n; colY = up * sw

    def _mat(cX, cY, cZ):
        return Matrix(((cX.x, cY.x, cZ.x, 0.0),
                       (cX.y, cY.y, cZ.y, 0.0),
                       (cX.z, cY.z, cZ.z, 0.0),
                       (0.0, 0.0, 0.0, 1.0)))
    R = _mat(colX, colY, colZ)
    if R.to_3x3().determinant() < 0:                 # keep right-handed (no mirrored normals)
        if wide_is_x:
            colY = -colY
        else:
            colX = -colX
        R = _mat(colX, colY, colZ)
    # depth anchor: normally the bbox depth-centre, at world offset 0 along n.
    # With reveal_depth>0, anchor the bbox's OUTWARD face instead (whichever
    # local extreme actually maps to +n after the flip above), at world
    # offset reveal_depth along n.
    depth_min, depth_max = (min(ys), max(ys)) if wide_is_x else (min(xs), max(xs))
    depth_center = cyl if wide_is_x else cxl
    depth_col = colY if wide_is_x else colX          # exactly +-n, unit length
    if abs(reveal_depth) > 1e-6:
        # reveal_depth may be NEGATIVE: the offset runs along sign(reveal_depth)*n,
        # because an opening's normal points outward (away from its room) when it
        # came from an exterior piece, but INTO the room when it came from
        # _wall_under_cursor. Flush the mesh face that points that same way.
        rsign = 1.0 if reveal_depth > 0 else -1.0
        depth_anchor = depth_max if depth_col.dot(n) * rsign > 0 else depth_min
        target_offset = n * reveal_depth
    else:
        depth_anchor = depth_center
        target_offset = Vector((0.0, 0.0, 0.0))
    local_point = Vector((cxl, depth_anchor, zbot)) if wide_is_x else Vector((depth_anchor, cyl, zbot))
    R.translation = Vector((cx, cy, sill)) + target_offset - R.to_3x3() @ local_point
    inst.matrix_world = R
    return inst


def _place_frame_mesh(op, mesh_src, width, height, kind, rotated=False, context=None):
    """Instance a door/window frame mesh into the opening, oriented to the
    wall (thin wrapper around _place_dressing_mesh using the opening's own
    position/normal/sill). For windows, flushes the frame to the outward
    (exterior) end of the wall's reveal jamb rather than straddling the
    interior wall face -- matches how _build_wall actually cuts the reveal
    (wall_margin deep by default), so the frame fills the jamb cavity
    instead of leaving most of it visibly empty. Doors keep the old
    centred-on-face placement (their reveal is much shallower, partition*0.5).

    The reveal-depth offset is measured from the wall's own CURRENT line
    (via _wall_match_for_opening's wall_pt), not raw op.cx/cy -- op.cx/cy
    is the original projected piece's own centre, which can sit off the
    wall line by however thick that piece was (or drift if the room's
    wall was rebuilt since), and _wall_match_for_opening's docstring
    already flags this exact trap for anything measuring outward from the
    wall. Falls back to raw op.cx/cy if no current wall match is found
    (e.g. mid-edit, no room built yet) rather than failing to place.

    op.sill is the HOLE's bottom edge, already shrunk by FRAME_OVERLAP
    (both call sites -- GN_OT_project_openings and add_opening -- always
    apply this shrink before calling here). The frame itself must NOT be
    shrunk to match -- like width (which stays at its own centre via cx/cy,
    unaffected by op.hw), the frame's own vertical anchor is the hole's
    sill with that shrink undone, so the frame overlaps the hole by
    FRAME_OVERLAP at the bottom too, not just the sides/top."""
    coll = _get_coll(_frame_coll(kind))
    reveal_depth = 0.0
    cx, cy = op.cx, op.cy
    if context is not None and kind == 'WINDOW':
        s = context.scene.gn_int
        reveal_depth = s.wall_margin if s.reveal else 0.0
        if reveal_depth > 1e-6:
            # want_sign tells us which way THIS opening's normal points
            # relative to its owning room's wall -- projected-from-exterior
            # openings point outward (-1 matches), manually-placed ones (from
            # _wall_under_cursor) point INTO the room (+1 matches). That
            # direction decides which way the reveal offset must run: the
            # reveal is always cut from the interior face outward, so an
            # inward-pointing normal needs a NEGATIVE offset, otherwise the
            # frame gets pushed out of the wall and floats in front of it.
            m = _wall_match_for_opening(context, op, -1)
            if m is None:
                m = _wall_match_for_opening(context, op, 1)
                if m is not None:
                    reveal_depth = -reveal_depth
            if m:
                _, wn, wall_pt = m
                cx, cy = wall_pt.x, wall_pt.y
    frame_sill = op.sill - FRAME_OVERLAP
    _place_dressing_mesh(coll, f"GN_Frame_{op.uid}", mesh_src,
                         cx, cy, op.nx, op.ny, frame_sill, width, height,
                         rotate90=rotated, reveal_depth=reveal_depth)


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


def _place_curtain_mesh(context, op, mesh_src):
    """Place a curtain/blind overhanging op's actual placed window (op.win_w/
    win_h), falling back to the opening's own rough bounds if no window was
    ever placed there. Sized by the scene's curtain overhang tunables."""
    s = context.scene.gn_int
    win_w = op.win_w if op.win_w > 1e-4 else op.hw * 2
    win_h = op.win_h if op.win_h > 1e-4 else (op.top - op.sill)
    width = win_w + 2 * s.curtain_side_overhang
    height = win_h + s.curtain_top_overhang + s.curtain_bottom_drop
    sill = op.sill - s.curtain_bottom_drop
    coll = _get_coll(CURTAIN_COLL)
    _place_dressing_mesh(coll, f"GN_Curtain_{op.uid}", mesh_src,
                         op.cx, op.cy, op.nx, op.ny, sill, width, height)


def _remove_curtain_mesh(uid):
    ob = bpy.data.objects.get(f"GN_Curtain_{uid}")
    if ob:
        bpy.data.objects.remove(ob, do_unlink=True)


def _refresh_thresholds(context):
    """Rebuild threshold strips for all door openings (or clear them if disabled).
    The strip sits inside ONE room (the side the door normal points to, flippable),
    not bridging both. It spans from the gap centre into that room by threshold_depth."""
    s = context.scene.gn_int
    _clear_coll(THRESHOLD_COLL)
    if not s.add_threshold:
        _remove_coll_if_empty(THRESHOLD_COLL)
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
    _remove_coll_if_empty(THRESHOLD_COLL)   # e.g. enabled but no interior doors yet


def add_opening(context, xy, normal, base, kind):
    s = context.scene.gn_int
    w, h, sill, mesh = _active_preset(s, kind)
    if kind == 'WINDOW':
        # placement mode has no exterior opening to fit inside, so the window
        # goes in at its own size -- never scaled to the catalog's declared
        # width/height, which distorts it if the two disagree
        native = _native_frame_size(mesh)
        if native:
            w, h = native
    op = s.openings.add()
    op.uid = _new_uid(s)
    op.is_door = (kind == 'DOOR')
    op.cx, op.cy = xy.x, xy.y
    op.nx, op.ny = normal.x, normal.y
    # hole cut slightly smaller than the frame (FRAME_OVERLAP) so the frame
    # overlaps the rough edge instead of leaving a visible wall/frame gap
    op.hw = max(w * 0.5 - FRAME_OVERLAP, 0.01)
    op.sill = base + sill + FRAME_OVERLAP
    op.top = base + sill + h - FRAME_OVERLAP
    rebuild_rooms(context)                      # also refreshes thresholds
    _place_frame_mesh(op, mesh, w, h, kind, context=context)
    if kind == 'WINDOW':
        op.win_w, op.win_h = w, h
        if 0 <= s.active_window_preset < len(s.window_presets):
            active = s.window_presets[s.active_window_preset]
            op.win_allow_curtain = active.allow_curtain
            op.win_allow_blinds = active.allow_blinds
            op.win_key = active.library_key or active.name


def remove_opening(context, idx):
    s = context.scene.gn_int
    uid = s.openings[idx].uid
    _remove_frame_mesh(uid)
    _remove_curtain_mesh(uid)
    _remove_threshold(uid)
    s.openings.remove(idx)
    rebuild_rooms(context)
    for coll_name in (DOORFRAME_COLL, WINDOWFRAME_COLL, THRESHOLD_COLL, CURTAIN_COLL):
        _remove_coll_if_empty(coll_name)


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


def _wall_plane_z(context, event, cx, cy, nx, ny):
    """Intersect the mouse ray with the vertical plane through (cx, cy) whose
    normal is (nx, ny, 0) -- the wall's own plane -- returning the world Z
    where the ray crosses it, or None. Drives vertical (Shift-drag)
    repositioning of an existing opening the same way _wall_under_cursor
    drives horizontal placement/sliding."""
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return None
    co = (event.mouse_region_x, event.mouse_region_y)
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, co)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, co)
    plane_no = Vector((nx, ny, 0.0))
    if plane_no.length < 1e-6:
        return None
    p = intersect_line_plane(origin, origin + direction, Vector((cx, cy, 0.0)), plane_no)
    return p.z if p is not None else None


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
    bl_description = ("Hover a wall to preview; LMB = add. LMB-drag an existing "
                      "opening to slide it along the wall (hold Shift to move it "
                      "up/down instead); LMB click with no drag on one = remove. "
                      "Tab = next preset, Esc/RMB = exit (cancels an in-progress "
                      "drag back to its start instead, if one is active)")
    kind: EnumProperty(items=[('DOOR', "Door", ""), ('WINDOW', "Window", "")],
                       default='DOOR', options={'HIDDEN'})

    _DRAG_PX = 8

    def invoke(self, context, event):
        s = context.scene.gn_int
        presets, _ = _presets_for(s, self.kind)
        if not presets:
            p = presets.add()
            p.name = self.kind.title()
        self.hover = None
        self.remove_hover = False
        self._press = None       # set while LMB is down on an existing opening
        self._dragging = False   # True once the drag threshold is exceeded
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_opening_ghost, (self, context), 'WINDOW', 'POST_VIEW')
        context.window_manager.modal_handler_add(self)
        context.area.header_text_set(
            f"{self.kind.title()} Placement: LMB add · drag one to move "
            "(Shift = up/down) · click one to remove · Tab preset · Esc exit")
        return {'RUNNING_MODAL'}

    def _cleanup(self, context):
        try:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
        except Exception:
            pass
        if context.area:
            context.area.header_text_set(None)
            context.area.tag_redraw()

    def _frame_size_for(self, op):
        """(width, height) the frame was actually built at -- windows cache
        this on the opening itself (win_w/win_h); doors don't, so reconstruct
        it from the hole's own stored size, undoing the FRAME_OVERLAP shrink
        the same way _place_frame_mesh already does for the sill anchor."""
        if not op.is_door and op.win_w > 1e-4:
            return op.win_w, op.win_h
        return op.hw * 2 + 2 * FRAME_OVERLAP, (op.top - op.sill) + 2 * FRAME_OVERLAP

    def _reposition(self, context, op):
        frame = bpy.data.objects.get(f"GN_Frame_{op.uid}")
        if frame is None:
            return
        w, h = self._frame_size_for(op)
        _place_frame_mesh(op, frame, w, h, 'DOOR' if op.is_door else 'WINDOW',
                          rotated=op.win_rotated, context=context)

    def _cycle_opening_window(self, context, op, backwards=False):
        """Swap the hovered window for the next one that still fits its
        opening -- the same Tab that cycles the active preset when you're
        hovering bare wall, but aimed at an already-placed window."""
        cands = _window_candidates(context, *_opening_avail(op))
        if not cands:
            return
        cur = next((i for i, c in enumerate(cands)
                    if _candidate_key(c[0]) == op.win_key and c[3] == op.win_rotated), -1)
        source, w, h, rotated = cands[(cur + (-1 if backwards else 1)) % len(cands)]
        _apply_window_to_opening(context, op, source, w, h, rotated)

    def _end_drag(self, context, restore):
        # rebuild_rooms (bmesh wall/hole rebuild across every room) only
        # happens HERE, once, on release/cancel -- NOT on every MOUSEMOVE.
        # Doing full room rebuilds at mouse-move frequency during a drag
        # was heavy enough to be a real crash risk; live feedback during
        # the drag itself is just the frame object's matrix_world moving,
        # which is cheap, and the wall hole catches up in one shot at the end.
        p = self._press
        s = context.scene.gn_int
        idx = next((i for i, o in enumerate(s.openings) if o.uid == p["uid"]), -1)
        if idx >= 0:
            op = s.openings[idx]
            if restore:
                op.cx, op.cy = p["orig_cx"], p["orig_cy"]
                op.sill, op.top = p["orig_sill"], p["orig_top"]
            rebuild_rooms(context)
            self._reposition(context, op)
        self._press = None
        self._dragging = False

    def modal(self, context, event):
        try:
            if context.area:
                context.area.tag_redraw()
            s = context.scene.gn_int
            want_door = (self.kind == 'DOOR')
            if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
                return {'PASS_THROUGH'}

            if event.type == 'MOUSEMOVE':
                if self._press is not None:
                    dx = event.mouse_region_x - self._press["start_px"][0]
                    dy = event.mouse_region_y - self._press["start_px"][1]
                    if not self._dragging and (dx * dx + dy * dy) >= self._DRAG_PX ** 2:
                        self._dragging = True
                    if self._dragging:
                        p = self._press
                        idx = next((i for i, o in enumerate(s.openings) if o.uid == p["uid"]), -1)
                        if idx >= 0:
                            op = s.openings[idx]
                            if event.shift:
                                z = _wall_plane_z(context, event, op.cx, op.cy, op.nx, op.ny)
                                if z is not None and p["start_z"] is not None:
                                    delta = z - p["start_z"]
                                    op.sill = p["orig_sill"] + delta
                                    op.top = p["orig_top"] + delta
                            else:
                                w = _wall_under_cursor(context, event)
                                if w:
                                    xy, n, base = w
                                    # only accept a snap onto the SAME wall
                                    # direction the opening is already on --
                                    # don't let a drag jump it to a different
                                    # (e.g. perpendicular) wall
                                    if Vector((op.nx, op.ny)).dot(n) > 0.9:
                                        op.cx, op.cy = xy.x, xy.y
                            # live feedback only moves the frame object
                            # itself (cheap) -- the wall hole (rebuild_rooms,
                            # a full bmesh rebuild) only updates once, on
                            # release, see _end_drag
                            self._reposition(context, op)
                    return {'RUNNING_MODAL'}
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
                # hovering an already-placed window? Tab swaps THAT one for the
                # next library window that still fits its opening (Shift+Tab
                # goes back), instead of cycling the active preset
                if self.kind == 'WINDOW' and self.remove_hover and self.hover:
                    _, xy, _n, base = self.hover
                    idx = _opening_under(s, xy, base + 0.1, want_door)
                    if idx >= 0:
                        self._cycle_opening_window(context, s.openings[idx], event.shift)
                        return {'RUNNING_MODAL'}
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
                        op = s.openings[idx]
                        self._press = {
                            "uid": op.uid,
                            "start_px": (event.mouse_region_x, event.mouse_region_y),
                            "orig_cx": op.cx, "orig_cy": op.cy,
                            "orig_sill": op.sill, "orig_top": op.top,
                            "start_z": _wall_plane_z(context, event, op.cx, op.cy, op.nx, op.ny),
                        }
                        self._dragging = False
                    else:
                        add_opening(context, xy, n, base, self.kind)
                return {'RUNNING_MODAL'}

            if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
                if self._press is not None:
                    if self._dragging:
                        self._end_drag(context, restore=False)
                    else:
                        idx = next((i for i, o in enumerate(s.openings)
                                   if o.uid == self._press["uid"]), -1)
                        if idx >= 0:
                            remove_opening(context, idx)
                        self._press = None
                return {'RUNNING_MODAL'}

            if event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
                if self._dragging:
                    self._end_drag(context, restore=True)
                    return {'RUNNING_MODAL'}
                self._press = None
                self._cleanup(context)
                return {'FINISHED'}
            return {'RUNNING_MODAL'}
        except Exception as e:
            print("[UltimateMLO] opening_edit error:", e)
            self._cleanup(context)
            return {'CANCELLED'}


def _active_window_opening(context):
    """The GN_Opening whose frame is the active object -- or None if the
    active object isn't a window frame."""
    vl = context.view_layer
    ob = vl.objects.active if vl else None
    if ob is None:
        return None
    s = context.scene.gn_int
    idx = _opening_index_for_object(ob, s)
    if idx is None or not (0 <= idx < len(s.openings)):
        return None
    op = s.openings[idx]
    return None if op.is_door else op


def _reposition_frame_only(context, op):
    """Move/scale just the frame instance to match the opening's current
    numbers -- cheap (a matrix update), safe to call every mouse-move during
    a gizmo drag. The wall HOLE is not recut here; that's the expensive
    rebuild, debounced to drag-end by _schedule_win_edit_rebuild."""
    frame = bpy.data.objects.get(f"GN_Frame_{op.uid}")
    if frame is None:
        return
    w = op.win_w if op.win_w > 1e-4 else op.hw * 2
    h = op.win_h if op.win_h > 1e-4 else (op.top - op.sill)
    _place_frame_mesh(op, frame, w, h, 'WINDOW', rotated=op.win_rotated, context=context)


_GN_WIN_EDIT_REBUILD = {"pending": False}


def _schedule_win_edit_rebuild():
    """Recut the wall hole once a gizmo drag settles. rebuild_rooms is a full
    bmesh rebuild -- running it per mouse-move was a crash risk -- so debounce
    it: only the last drag event in a burst triggers one rebuild."""
    if _GN_WIN_EDIT_REBUILD["pending"]:
        return
    _GN_WIN_EDIT_REBUILD["pending"] = True

    def _do():
        _GN_WIN_EDIT_REBUILD["pending"] = False
        try:
            rebuild_rooms(bpy.context)
        except Exception as e:
            print("[UltimateMLO] window-edit rebuild:", e)
        return None
    bpy.app.timers.register(_do, first_interval=0.2)


def _win_gizmo_matrix(center, axis):
    """4x4 placing a gizmo at `center` (Vector) with its +Z along `axis`."""
    z = axis.normalized()
    up = Vector((0.0, 0.0, 1.0))
    x = (Vector((1.0, 0.0, 0.0)) if abs(z.dot(up)) > 0.99
         else up.cross(z).normalized())
    y = z.cross(x).normalized()
    return Matrix(((x.x, y.x, z.x, center.x),
                   (x.y, y.y, z.y, center.y),
                   (x.z, y.z, z.z, center.z),
                   (0.0, 0.0, 0.0, 1.0)))


class GN_GGT_window_edit(bpy.types.GizmoGroup):
    bl_idname = "GN_GGT_window_edit"
    bl_label = "Window Edit"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'3D', 'PERSISTENT'}

    @classmethod
    def poll(cls, context):
        s = getattr(context.scene, "gn_int", None)
        if not s or not s.win_edit_mode:
            return False
        return _active_window_opening(context) is not None

    def setup(self, context):
        a = self.gizmos.new("GIZMO_GT_arrow_3d")
        a.draw_style = 'NORMAL'; a.length = 0.8
        a.color = (0.2, 0.85, 0.2); a.alpha = 0.85
        a.color_highlight = (0.45, 1.0, 0.45); a.alpha_highlight = 1.0
        a.use_draw_modal = True
        a.target_set_handler("offset", get=self._get_along, set=self._set_along)
        self.g_along = a

        v = self.gizmos.new("GIZMO_GT_arrow_3d")
        v.draw_style = 'NORMAL'; v.length = 0.8
        v.color = (0.25, 0.5, 1.0); v.alpha = 0.85
        v.color_highlight = (0.45, 0.65, 1.0); v.alpha_highlight = 1.0
        v.use_draw_modal = True
        v.target_set_handler("offset", get=self._get_vert, set=self._set_vert)
        self.g_vert = v

        sc = self.gizmos.new("GIZMO_GT_arrow_3d")
        sc.draw_style = 'BOX'; sc.length = 0.6
        sc.color = (1.0, 0.85, 0.15); sc.alpha = 0.9
        sc.color_highlight = (1.0, 1.0, 0.4); sc.alpha_highlight = 1.0
        sc.use_draw_modal = True
        sc.target_set_handler("offset", get=self._get_scale, set=self._set_scale)
        self.g_scale = sc

    def refresh(self, context):
        op = _active_window_opening(context)
        if op is None:
            return
        center = Vector((op.cx, op.cy, (op.sill + op.top) * 0.5))
        n = Vector((op.nx, op.ny, 0.0))
        along = Vector((-n.y, n.x, 0.0))
        self.g_along.matrix_basis = _win_gizmo_matrix(center, along)
        self.g_vert.matrix_basis = _win_gizmo_matrix(center, Vector((0.0, 0.0, 1.0)))
        top_corner = Vector((op.cx, op.cy, op.top + 0.08))
        self.g_scale.matrix_basis = _win_gizmo_matrix(top_corner, along)

    # move along the wall -- capture the base position when the drag starts
    # (get is called once at drag start), apply the absolute offset in set
    def _get_along(self):
        op = _active_window_opening(bpy.context)
        self._base_cx = op.cx if op else 0.0
        self._base_cy = op.cy if op else 0.0
        return 0.0

    def _set_along(self, value):
        op = _active_window_opening(bpy.context)
        if op is None:
            return
        n = Vector((op.nx, op.ny))
        along = Vector((-n.y, n.x)).normalized()
        op.cx = self._base_cx + along.x * value
        op.cy = self._base_cy + along.y * value
        _reposition_frame_only(bpy.context, op)
        _schedule_win_edit_rebuild()

    # move up/down
    def _get_vert(self):
        op = _active_window_opening(bpy.context)
        self._base_sill = op.sill if op else 0.0
        self._base_top = op.top if op else 0.0
        return 0.0

    def _set_vert(self, value):
        op = _active_window_opening(bpy.context)
        if op is None:
            return
        op.sill = self._base_sill + value
        op.top = self._base_top + value
        _reposition_frame_only(bpy.context, op)
        _schedule_win_edit_rebuild()

    # uniform scale -- keeps aspect ratio; projected windows clamp to the
    # exterior opening they came from
    def _get_scale(self):
        op = _active_window_opening(bpy.context)
        if op:
            self._base_w = op.win_w if op.win_w > 1e-4 else op.hw * 2
            self._base_h = op.win_h if op.win_h > 1e-4 else (op.top - op.sill)
            self._base_cz = (op.sill + op.top) * 0.5
        else:
            self._base_w = self._base_h = 1.0
            self._base_cz = 0.0
        return 0.0

    def _set_scale(self, value):
        op = _active_window_opening(bpy.context)
        if op is None:
            return
        w0, h0 = self._base_w, self._base_h
        factor = max(0.1, 1.0 + value / max(w0, 0.3))
        if op.projected:
            aw, ah = _opening_avail(op)
            if aw > 1e-4 and ah > 1e-4:
                factor = min(factor, aw / w0, ah / h0)
        neww, newh = w0 * factor, h0 * factor
        cz = self._base_cz
        op.win_w, op.win_h = neww, newh
        op.hw = max(neww * 0.5 - FRAME_OVERLAP, 0.01)
        op.sill = cz - newh * 0.5 + FRAME_OVERLAP
        op.top = cz + newh * 0.5 - FRAME_OVERLAP
        _reposition_frame_only(bpy.context, op)
        _schedule_win_edit_rebuild()


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


def _room_locked_for_opening(context, op):
    """Index of the (locked) room this opening belongs to, or None -- same
    _room_for_opening lookup _group_openings_by_room uses, so the warning
    matches what the Openings panel shows as that opening's room."""
    s = context.scene.gn_int
    fi, ri = _room_for_opening(context, op)
    if ri is not None and ri >= 0 and s.rooms[ri].lock:
        return ri
    return None


class GN_OT_remove_opening(Operator):
    bl_idname = "gn_int.remove_opening"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Delete Opening"
    bl_description = "Delete the selected opening (its hole, frame mesh and threshold)"
    index: IntProperty(default=-1)

    def invoke(self, context, event):
        s = context.scene.gn_int
        i = self.index if self.index >= 0 else s.opening_index
        if 0 <= i < len(s.openings):
            ri = _room_locked_for_opening(context, s.openings[i])
            if ri is not None:
                self._locked_room = ri
                return context.window_manager.invoke_confirm(
                    self, event,
                    message=f"Room {ri + 1} is locked -- its wall geometry won't "
                            "change, only the opening record will be removed. Continue?")
        return self.execute(context)

    def execute(self, context):
        s = context.scene.gn_int
        i = self.index if self.index >= 0 else s.opening_index
        if 0 <= i < len(s.openings):
            remove_opening(context, i)
            s.opening_index = min(i, len(s.openings) - 1)
            return {'FINISHED'}
        self.report({'WARNING'}, "No opening selected")
        return {'CANCELLED'}


class GN_OT_add_curtain(Operator):
    bl_idname = "gn_int.add_curtain"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Add Curtain"
    bl_description = ("Dress the selected opening's window with the active "
                      "curtain/blind preset, overhanging it. Re-running this "
                      "replaces whatever curtain is already there")
    index: IntProperty(default=-1)

    def execute(self, context):
        s = context.scene.gn_int
        i = self.index if self.index >= 0 else s.opening_index
        if not (0 <= i < len(s.openings)):
            self.report({'WARNING'}, "No opening selected")
            return {'CANCELLED'}
        op = s.openings[i]
        if op.is_door:
            self.report({'ERROR'}, "Curtains/blinds are for windows, not doors")
            return {'CANCELLED'}
        if not (0 <= s.active_curtain_preset < len(s.curtain_presets)):
            self.report({'ERROR'}, "No curtain/blind preset selected - add one from the library first")
            return {'CANCELLED'}
        preset = s.curtain_presets[s.active_curtain_preset]
        if preset.mesh_object is None:
            self.report({'ERROR'}, "That preset has no mesh")
            return {'CANCELLED'}
        if preset.category == 'blinds' and not op.win_allow_blinds:
            self.report({'ERROR'}, "This window doesn't support blinds")
            return {'CANCELLED'}
        if preset.category == 'curtain' and not op.win_allow_curtain:
            self.report({'ERROR'}, "This window doesn't support curtains")
            return {'CANCELLED'}
        _place_curtain_mesh(context, op, preset.mesh_object)
        self.report({'INFO'}, f"Added '{preset.name}' to the selected opening")
        return {'FINISHED'}


class GN_OT_remove_curtain(Operator):
    bl_idname = "gn_int.remove_curtain"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Remove Curtain"
    bl_description = "Remove the curtain/blind from the selected opening (window stays)"
    index: IntProperty(default=-1)

    def execute(self, context):
        s = context.scene.gn_int
        i = self.index if self.index >= 0 else s.opening_index
        if not (0 <= i < len(s.openings)):
            self.report({'WARNING'}, "No opening selected")
            return {'CANCELLED'}
        _remove_curtain_mesh(s.openings[i].uid)
        _remove_coll_if_empty(CURTAIN_COLL)
        return {'FINISHED'}


class GN_OT_select_opening(Operator):
    bl_idname = "gn_int.select_opening"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Select Opening"
    bl_description = "Highlight this opening and select its frame/threshold object"
    index: IntProperty(default=-1)

    def execute(self, context):
        s = context.scene.gn_int
        if 0 <= self.index < len(s.openings):
            s.opening_index = self.index
            return {'FINISHED'}
        return {'CANCELLED'}


# ===========================================================================
# UI
# ===========================================================================
class GN_UL_door_presets(bpy.types.UIList):
    def draw_item(self, ctx, layout, data, item, icon, adata, aprop, index=0, flt=0):
        layout.prop(item, "name", text="", emboss=False, icon='MESH_DATA')
        layout.label(text=f"{item.width:.2f}x{item.height:.2f}")


class GN_UL_window_library(bpy.types.UIList):
    def draw_item(self, ctx, layout, data, item, icon, adata, aprop, index=0, flt=0):
        s = ctx.scene.gn_int
        row = layout.row(align=True)
        row.label(text=item.name, icon='MESH_PLANE')
        row.label(text=f"{item.width:.2f}x{item.height:.2f}  sill {item.sill:.2f}")
        active_idx = next((i for i, p in enumerate(s.window_presets)
                           if p.library_key == item.key), -1)
        is_active = active_idx >= 0 and active_idx == s.active_window_preset
        # clicking the row itself picks it (winlib_index's update callback);
        # this is just a status indicator, not a separate action
        if is_active:
            row.label(text="Active", icon='CHECKMARK')

    def filter_items(self, ctx, data, propname):
        items = getattr(data, propname)
        flt = [self.bitflag_filter_item] * len(items)
        for i, it in enumerate(items):
            if it.category in _CURTAIN_CATEGORIES:
                flt[i] &= ~self.bitflag_filter_item
        return flt, []


class GN_UL_curtain_library(bpy.types.UIList):
    def draw_item(self, ctx, layout, data, item, icon, adata, aprop, index=0, flt=0):
        row = layout.row(align=True)
        row.label(text=item.name, icon='MESH_PLANE')
        row.label(text=item.category.title() if item.category else "")
        # clicking the row picks it (winlib_index's update callback) -- this is
        # just a status indicator, like the window library list
        s = ctx.scene.gn_int
        idx = next((i for i, p in enumerate(s.curtain_presets)
                    if p.library_key == item.key), -1)
        if idx >= 0 and idx == s.active_curtain_preset:
            row.label(text="Active", icon='CHECKMARK')

    def filter_items(self, ctx, data, propname):
        items = getattr(data, propname)
        flt = [self.bitflag_filter_item] * len(items)
        for i, it in enumerate(items):
            if it.category not in _CURTAIN_CATEGORIES:
                flt[i] &= ~self.bitflag_filter_item
        return flt, []


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
        # "saved as final" indicator -- set automatically by Save Floor Map
        # as Final, not user-togglable (no room.prop here on purpose)
        if item.lock:
            row.label(text="", icon='CHECKMARK')
        op = row.operator("gn_int.remove_floor", text="", icon='TRASH')
        op.index = index


class GN_UL_rooms(bpy.types.UIList):
    def draw_item(self, ctx, layout, data, item, icon, adata, aprop, index=0, flt=0):
        row = layout.row(align=True)
        row.label(text=f"Floor {item.floor_index+1} · Room {index+1}", icon='MESH_CUBE')
        # lock: keep a hand-resized room's mesh exactly as-is (rebuild_rooms
        # skips it) -- its opening cuts also stop updating while locked
        row.prop(item, "lock", text="",
                 icon='LOCKED' if item.lock else 'UNLOCKED', emboss=False)
        op = row.operator("gn_int.remove_room", text="", icon='TRASH')
        op.index = index


class _PanelBase:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "UltimateMLO"


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
    bl_order = 0

    def draw(self, context):
        s = context.scene.gn_int
        col = self.layout.column(align=True)
        col.prop(s, "room_height")
        col.prop(s, "uv_scale")


class GN_PT_floors(_PanelBase, Panel):
    bl_parent_id = "GN_PT_interior"
    bl_label = "Floors & Floor Map"
    bl_order = 1

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
        layout.operator("gn_int.add_floor_sel", text="Create Floor Map from Edge", icon='EDGESEL')
        layout.operator("gn_int.pick_floor_z", text="Create Floor Map Slice",
                        icon='EMPTY_SINGLE_ARROW')

        box = layout.box()
        box.label(text="Floor Map cleanup", icon='MOD_BEVEL')
        col = box.column(align=True)
        col.prop(s, "detail_tol")
        col.prop(s, "bridge")
        col.prop(s, "wall_margin")
        col.prop(s, "sample_offset")
        box.operator("gn_int.save_floor_map_final", icon='CHECKMARK')
        box.label(text="Correct the Floor Map in Edit Mode, then save it as", icon='INFO')
        box.label(text="final -- Generate Floor Map will confirm before overwriting it")

        layout.operator("gn_int.seed_rooms", icon='MESH_PLANE')
        layout.operator("gn_int.clear", icon='TRASH')


class GN_PT_rooms(_PanelBase, Panel):
    bl_parent_id = "GN_PT_interior"
    bl_label = "Rooms"
    bl_order = 2

    def draw(self, context):
        s = context.scene.gn_int
        layout = self.layout
        layout.prop(s, "partition")
        layout.operator("gn_int.split_edges", icon='MOD_BEVEL')
        layout.label(text="Edit Mode: pick 2 points on the Floor Map, then Split", icon='INFO')
        layout.operator("gn_int.split_room_path", icon='GP_MULTIFRAME_EDITING')
        layout.label(text="Or click a bent path (L-shape) across a room", icon='INFO')
        layout.label(text=f"{len(s.rooms)} room(s)")
        if len(s.rooms):
            layout.template_list("GN_UL_rooms", "", s, "rooms", s, "room_index", rows=3)
            layout.label(text="Lock a room to keep a hand-edited resize", icon='INFO')
        layout.operator("gn_int.rebuild_rooms", icon='MOD_BUILD')
        row = layout.row(align=True)
        row.operator("gn_int.clear_rooms", icon='TRASH')
        layout.operator("gn_int.reunwrap", icon='UV')


class GN_PT_stairs(_PanelBase, Panel):
    bl_parent_id = "GN_PT_interior"
    bl_label = "Stairs"
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 6

    def draw(self, context):
        s = context.scene.gn_int
        layout = self.layout
        col = layout.column(align=True)
        col.prop(s, "stair_step_height")
        col.prop(s, "stair_step_depth")
        col.prop(s, "stair_nosing")
        layout.operator("gn_int.create_stairs", icon='MOD_ARRAY')
        layout.label(text="Edit Mode: pick 1 edge at the bottom, 1 at the", icon='INFO')
        layout.label(text="top (can be on 2 different objects), then run")


class GN_PT_openings(_PanelBase, Panel):
    bl_parent_id = "GN_PT_interior"
    bl_label = "Openings (project)"
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 3

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
            want = s.opening_filter
            for (fi, ri), idxs in _group_openings_by_room(context):
                idxs = [i for i in idxs if want == 'ALL'
                       or s.openings[i].is_door == (want == 'DOOR')]
                if not idxs:
                    continue
                if fi is None:
                    label, icon, room_rec = "Unmatched", 'QUESTION', None
                elif ri is None:
                    label, icon, room_rec = f"Floor {fi + 1} · unassigned", 'QUESTION', None
                else:
                    label, icon, room_rec = f"Floor {fi + 1} · Room {ri + 1}", 'MESH_CUBE', s.rooms[ri]
                box = layout.box()
                head = box.row(align=True)
                if room_rec is not None:
                    head.prop(room_rec, "openings_expanded", text="", emboss=False,
                             icon='TRIA_DOWN' if room_rec.openings_expanded else 'TRIA_RIGHT')
                head.label(text=f"{label}  ({len(idxs)})", icon=icon)
                if room_rec is not None and not room_rec.openings_expanded:
                    continue
                col = box.column(align=True)
                for i in idxs:
                    op = s.openings[i]
                    row = col.row(align=True)
                    row.active = (i == s.opening_index)
                    sel = row.operator("gn_int.select_opening", emboss=False,
                                       text=f"{'Door' if op.is_door else 'Window'}  "
                                            f"({op.cx:.1f}, {op.cy:.1f})  w{op.hw*2:.2f}",
                                       icon='MOD_BEVEL' if op.is_door else 'MOD_LATTICE')
                    sel.index = i
                    delop = row.operator("gn_int.remove_opening", text="", icon='X')
                    delop.index = i
        row = layout.row(align=True)
        row.operator("gn_int.remove_opening", text="Delete Selected", icon='X').index = -1
        row.operator("gn_int.clear_openings", text="Clear All", icon='TRASH')
        layout.operator("gn_int.clean_stale_openings", icon='ORPHAN_DATA')


class GN_PT_doors(_PanelBase, Panel):
    bl_parent_id = "GN_PT_interior"
    bl_label = "Doors"
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 4

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
    bl_order = 5

    def draw(self, context):
        s = context.scene.gn_int
        layout = self.layout

        box1 = layout.box()
        box1.label(text="1. Project from Exterior", icon='SELECT_DIFFERENCE')
        box1.label(text="Select exterior window pieces, then Project", icon='INFO')
        box1.label(text="Openings above -- hole is sized to the best-fitting")
        box1.label(text="library window, not the exterior piece itself")

        sel_op = (s.openings[s.opening_index]
                  if 0 <= s.opening_index < len(s.openings) else None)
        if sel_op is not None and not sel_op.is_door:
            box1.label(text=f"Selected window ({sel_op.cx:.1f}, {sel_op.cy:.1f}) --",
                       icon='RESTRICT_SELECT_OFF')
            box1.label(text="click a library window below to swap it")

        box2 = layout.box()
        box2.prop(s, "default_window_sill")
        if 0 <= s.active_window_preset < len(s.window_presets):
            active_name = s.window_presets[s.active_window_preset].name
        else:
            active_name = "None -- pick one below"
        box2.label(text=f"Active: {active_name}", icon='CHECKMARK')

        sub = box2.box()
        sub.label(text="Window Library", icon='ASSET_MANAGER')
        row = sub.row(align=True)
        row.operator("gn_int.win_lib_refresh", text="Refresh", icon='FILE_REFRESH')
        row.operator("gn_int.win_lib_resync_mesh", text="Resync Mesh", icon='IMPORT')
        if not s.winlib_items:
            sub.label(text="No entries -- set a Window Library file in", icon='ERROR')
            sub.label(text="add-on Preferences, then Refresh")
        else:
            sub.template_list("GN_UL_window_library", "winlib", s, "winlib_items",
                              s, "winlib_index", rows=4)
        sub.label(text="Pick a shape, then place it with Window Edit Mode", icon='INFO')
        sub.separator()
        sub.operator("gn_int.lib_register_window", icon='EXPORT')
        sub.label(text="Selects the active object -- writes to the shared", icon='INFO')
        sub.label(text="library file (backed up automatically)")

        box2.operator("gn_int.opening_edit", text="Window Placement Mode",
                      icon='GREASEPENCIL').kind = 'WINDOW'
        box2.prop(s, "win_edit_mode", toggle=True, icon='ORIENTATION_GIMBAL')
        if s.win_edit_mode:
            box2.label(text="Select a window -- drag the arrows to move,", icon='INFO')
            box2.label(text="the box handle to scale (uniform)")

        box3 = layout.box()
        box3.label(text="3. Curtains / Blinds", icon='MOD_CLOTH')
        if 0 <= s.opening_index < len(s.openings):
            sel = s.openings[s.opening_index]
            box3.label(text=f"Selected: {'Door' if sel.is_door else 'Window'} "
                            f"({sel.cx:.1f}, {sel.cy:.1f})",
                       icon='RESTRICT_SELECT_OFF')
        else:
            box3.label(text="No opening selected -- pick one in Openings above", icon='ERROR')
        if 0 <= s.active_curtain_preset < len(s.curtain_presets):
            box3.label(text=f"Active: {s.curtain_presets[s.active_curtain_preset].name}",
                       icon='CHECKMARK')
        if not s.winlib_items:
            box3.label(text="No entries -- set a Window Library file in", icon='ERROR')
            box3.label(text="add-on Preferences, then Refresh")
        else:
            box3.template_list("GN_UL_curtain_library", "curtainlib", s, "winlib_items",
                               s, "winlib_index", rows=3)
        col2 = box3.column(align=True)
        col2.prop(s, "curtain_side_overhang")
        col2.prop(s, "curtain_top_overhang")
        col2.prop(s, "curtain_bottom_drop")
        row2 = box3.row(align=True)
        row2.operator("gn_int.add_curtain", icon='ADD')
        row2.operator("gn_int.remove_curtain", icon='REMOVE')
        box3.label(text="Applies to the selected opening above", icon='INFO')


class GN_PT_mlo(_PanelBase, Panel):
    bl_parent_id = "GN_PT_interior"
    bl_label = "MLO Setup"
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 7

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
            s = context.scene.gn_int
            row = layout.row()
            row.prop(s, "show_portal_list",
                     icon='TRIA_DOWN' if s.show_portal_list else 'TRIA_RIGHT',
                     text=f"{len(portals_col.objects)} portal(s)", emboss=False)
            if s.show_portal_list:
                layout.template_list("GN_UL_portals", "", portals_col, "objects",
                                     s, "portal_index", rows=4)
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
# material tools -- cube-unwrap / rescale every face across the WHOLE scene
# that shares a material, not just this add-on's own generated objects
# ===========================================================================
class GN_OT_cube_unwrap_by_material(Operator):
    bl_idname = "gn_int.cube_unwrap_by_material"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Cube Unwrap by Material"
    bl_description = ("Box/cube-project UVs for every face across the whole "
                      "scene that uses the chosen material -- destructive, "
                      "replaces the active UV layer")

    material_name: StringProperty(name="Material", default="")
    tile_size: FloatProperty(name="Tile Size (m)",
        description="Metres per UV tile -- lower = more repetitions",
        default=1.0, min=0.001, soft_max=20.0)
    projection: EnumProperty(name="Projection", items=[
        ('LOCAL', "Local", "Box-project using object-local coordinates"),
        ('WORLD', "World", "Box-project using world-space coordinates -- "
         "identical tiling on every object")], default='LOCAL')
    rotate_uvs: EnumProperty(name="Rotate UVs", items=[
        ('0', "0", "No rotation"), ('90', "90", "Rotate 90 degrees CCW"),
        ('180', "180", "Rotate 180 degrees"),
        ('270', "270", "Rotate 270 degrees CCW")], default='0')

    def invoke(self, context, event):
        ob = context.active_object
        if ob and ob.active_material:
            self.material_name = ob.active_material.name
        return context.window_manager.invoke_props_dialog(self, width=380)

    def draw(self, context):
        layout = self.layout
        layout.prop_search(self, "material_name", bpy.data, "materials",
                           text="Material", icon='MATERIAL')
        layout.separator(factor=0.3)
        layout.prop(self, "projection", expand=True)
        layout.separator(factor=0.3)
        layout.prop(self, "tile_size", slider=False)
        layout.separator(factor=0.3)
        layout.prop(self, "rotate_uvs", expand=True)
        layout.separator(factor=0.3)
        layout.label(text="Destructive -- replaces the active UV layer", icon='INFO')

    def _apply(self, context):
        mat = bpy.data.materials.get(self.material_name)
        if mat is None:
            return 0
        sc = 1.0 / max(self.tile_size, 1e-4)
        world = (self.projection == 'WORLD')
        angle = math.radians(float(self.rotate_uvs))
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        patched = 0
        for ob in context.scene.objects:
            if ob.type != 'MESH':
                continue
            mat_indices = {i for i, slot in enumerate(ob.material_slots) if slot.material is mat}
            if not mat_indices:
                continue
            me = ob.data
            if not me.uv_layers:
                me.uv_layers.new(name="UVMap")
            uv = me.uv_layers.active
            if uv is None or not uv.data:
                continue
            mw = ob.matrix_world
            mw3 = mw.to_3x3()
            for poly in me.polygons:
                if poly.material_index not in mat_indices:
                    continue
                n = (mw3 @ poly.normal).normalized() if world else poly.normal
                ax, ay, az = abs(n.x), abs(n.y), abs(n.z)
                for li in range(poly.loop_start, poly.loop_start + poly.loop_total):
                    vi = me.loops[li].vertex_index
                    co = (mw @ me.vertices[vi].co) if world else me.vertices[vi].co
                    if az >= ax and az >= ay:
                        u, v = co.x * sc, co.y * sc
                    elif ax >= ay:
                        u, v = co.y * sc, co.z * sc
                    else:
                        u, v = co.x * sc, co.z * sc
                    uv.data[li].uv = (u * cos_a - v * sin_a, u * sin_a + v * cos_a)
            me.update()
            patched += 1
        return patched

    def check(self, context):
        self._apply(context)
        return True

    def execute(self, context):
        mat = bpy.data.materials.get(self.material_name)
        if mat is None:
            self.report({'ERROR'}, f"Material '{self.material_name}' not found")
            return {'CANCELLED'}
        patched = self._apply(context)
        if patched == 0:
            self.report({'WARNING'}, f"No objects use '{self.material_name}'")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Cube-unwrapped {patched} object(s) using '{mat.name}'")
        return {'FINISHED'}


class GN_OT_scale_uv_by_material(Operator):
    bl_idname = "gn_int.scale_uv_by_material"
    bl_options = {'REGISTER', 'UNDO'}
    bl_label = "Scale UVs by Material"
    bl_description = ("Scale the UVs of every face across the whole scene "
                      "that uses the chosen material -- unifies tiling on "
                      "everything sharing it, live-previewed as you drag")

    material_name: StringProperty(name="Material", default="")
    factor: FloatProperty(name="Scale Factor",
        description="> 1 tiles more, < 1 tiles less",
        default=2.0, min=0.0001, soft_max=100.0)

    def _snapshot(self, mat):
        self._data = []
        for ob in bpy.data.objects:
            if ob.type != 'MESH':
                continue
            me = ob.data
            uv = me.uv_layers.active
            if not uv:
                continue
            mat_slots = {i for i, slot in enumerate(ob.material_slots) if slot.material is mat}
            if not mat_slots:
                continue
            loop_indices = set()
            for poly in me.polygons:
                if poly.material_index in mat_slots:
                    loop_indices.update(range(poly.loop_start, poly.loop_start + poly.loop_total))
            if not loop_indices:
                continue
            orig = [l.uv.copy() for l in uv.data]
            self._data.append((ob, uv, orig, loop_indices))

    def _apply_factor(self):
        f = self.factor
        for ob, uv, orig, loop_indices in self._data:
            for li, loop in enumerate(uv.data):
                loop.uv = orig[li]
            for li in loop_indices:
                uv.data[li].uv = orig[li] * f
            ob.data.update()

    def invoke(self, context, event):
        ob = context.active_object
        if ob and ob.active_material:
            self.material_name = ob.active_material.name
        self._data = []
        mat = bpy.data.materials.get(self.material_name)
        if mat:
            self._snapshot(mat)
        return context.window_manager.invoke_props_dialog(self, width=380)

    def draw(self, context):
        layout = self.layout
        layout.prop_search(self, "material_name", bpy.data, "materials",
                           text="Material", icon='MATERIAL')
        layout.separator(factor=0.4)
        layout.prop(self, "factor", slider=True)
        layout.separator(factor=0.3)
        layout.label(text=f"{len(getattr(self, '_data', []))} object(s) with "
                     "this material will be affected", icon='INFO')

    def check(self, context):
        mat = bpy.data.materials.get(self.material_name)
        if mat:
            cur = {id(ob.data) for ob, *_ in getattr(self, '_data', [])}
            match = {id(ob.data) for ob in bpy.data.objects
                    if ob.type == 'MESH' and any(sl.material is mat for sl in ob.material_slots)}
            if match != cur:
                self._snapshot(mat)
        else:
            self._data = []
        if self._data:
            self._apply_factor()
        return True

    def execute(self, context):
        mat = bpy.data.materials.get(self.material_name)
        if mat is None:
            self.report({'ERROR'}, f"Material '{self.material_name}' not found")
            return {'CANCELLED'}
        if not getattr(self, '_data', None):
            self._snapshot(mat)
        if not self._data:
            self.report({'WARNING'}, f"No objects with a UV map use '{self.material_name}'")
            return {'CANCELLED'}
        self._apply_factor()
        self.report({'INFO'}, f"UV scale x{self.factor:.3f} applied to "
                    f"{len(self._data)} object(s) using '{mat.name}'")
        return {'FINISHED'}


class GN_PT_material_tools(_PanelBase, Panel):
    bl_parent_id = "GN_PT_interior"
    bl_label = "Material Tools"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.label(text="Affects every object in the scene sharing the "
                     "chosen material, not just this add-on's own", icon='INFO')
        layout.operator("gn_int.cube_unwrap_by_material", icon='MOD_UVPROJECT')
        layout.operator("gn_int.scale_uv_by_material", icon='FULLSCREEN_ENTER')


# ===========================================================================
_classes = (
    GN_FloorLevel, GN_Room, GN_Opening, GN_DoorPreset, GN_WindowPreset,
    GN_WindowLibItem, GN_CurtainPreset, GN_IntPrefs,
    GN_CustomEmptyItem, GN_IntProps,
    GN_CollMatSearchItem, GN_ShellCollMappingItem,
    GN_OT_set_exterior, GN_OT_add_floor_sel,
    GN_OT_add_floor_z, GN_OT_pick_floor_z,
    GN_OT_remove_floor, GN_OT_gen_boundaries, GN_OT_save_floor_map_final,
    GN_OT_clear, GN_OT_clean_interior,
    GN_OT_draw_room, GN_OT_add_room, GN_OT_remove_room, GN_OT_rebuild_rooms,
    GN_OT_clear_rooms, GN_OT_seed_rooms, GN_OT_split_room, GN_OT_split_room_path,
    GN_OT_split_edges, GN_OT_create_stairs,
    GN_OT_reunwrap, GN_OT_build_mlo, GN_OT_clean_mlo,
    GN_OT_add_room_collections, GN_OT_add_prop_collections, GN_OT_add_asset_collections,
    GN_OT_create_shell_collision,
    GN_OT_create_portals, GN_OT_remove_portal, GN_OT_flip_portal, GN_OT_clear_portals,
    GN_OT_add_empties, GN_OT_add_custom_empty, GN_OT_remove_custom_empty,
    GN_OT_smart_rename, GN_OT_create_asset,
    GN_OT_cube_unwrap_by_material, GN_OT_scale_uv_by_material,
    GN_OT_split_opening_pieces, GN_OT_project_openings, GN_OT_clear_openings,
    GN_OT_opening_edit, GN_GGT_window_edit,
    GN_OT_preset_add, GN_OT_preset_remove, GN_OT_remove_opening,
    GN_OT_select_opening, GN_OT_clean_stale_openings,
    GN_OT_win_lib_refresh, GN_OT_win_lib_add_to_project, GN_OT_win_lib_remove_from_project,
    GN_OT_win_lib_resync_mesh, GN_OT_lib_register_window, GN_OT_swap_window,
    GN_OT_add_curtain, GN_OT_remove_curtain,
    GN_UL_door_presets, GN_UL_window_library, GN_UL_curtain_library,
    GN_UL_openings, GN_UL_floors, GN_UL_rooms,
    GN_UL_shell_coll_mappings, GN_UL_portals,
    GN_PT_interior, GN_PT_setup, GN_PT_floors, GN_PT_rooms,
    GN_PT_openings, GN_PT_doors, GN_PT_windows, GN_PT_stairs,
    GN_PT_mlo, GN_PT_manual_setup,
    GN_PT_add_empties, GN_PT_smart_rename, GN_PT_create_asset,
    GN_PT_material_tools,
)


_SETTINGS_KEYS = ("wall_margin", "room_height", "sample_offset",
                  "cleanup", "detail_tol", "bridge", "square", "ang_tol",
                  "allow45", "partition", "reveal", "uid_counter", "uv_scale",
                  "active_floor", "snap", "active_door_preset",
                  "active_window_preset", "add_threshold", "threshold_height",
                  "threshold_depth", "threshold_flip", "threshold_offset",
                  "active_curtain_preset", "curtain_side_overhang",
                  "curtain_top_overhang", "curtain_bottom_drop",
                  "mlo_name", "timecycle_name", "timecycle_auto",
                  "build_main", "build_room_colls", "build_prop_colls",
                  "build_asset_colls", "build_shell_collision", "build_portals",
                  "build_empties", "stair_step_height", "stair_step_depth",
                  "stair_nosing", "default_window_sill")


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
                   "poly_json": r.poly_json, "z_offset": r.z_offset} for r in s.rooms],
        "openings": [{k: getattr(o, k) for k in
                      ("cx", "cy", "nx", "ny", "hw", "sill", "top", "uid",
                       "is_door", "projected", "win_w", "win_h",
                       "win_allow_curtain", "win_allow_blinds",
                       "avail_w", "avail_h", "win_key", "win_rotated")}
                     for o in s.openings],
        "door_presets": [{"name": p.name, "width": p.width, "height": p.height,
                          "mesh": p.mesh_object.name if p.mesh_object else ""}
                         for p in s.door_presets],
        "window_presets": [{"name": p.name, "width": p.width, "height": p.height,
                            "sill": p.sill, "library_key": p.library_key,
                            "can_rotate": p.can_rotate, "allow_curtain": p.allow_curtain,
                            "allow_blinds": p.allow_blinds, "starts_at_floor": p.starts_at_floor,
                            "mesh": p.mesh_object.name if p.mesh_object else ""}
                           for p in s.window_presets],
        "curtain_presets": [{"name": p.name, "library_key": p.library_key,
                             "category": p.category,
                             "mesh": p.mesh_object.name if p.mesh_object else ""}
                            for p in s.curtain_presets],
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
        it.z_offset = r.get("z_offset", 0.0)
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
        it.library_key = p.get("library_key", "")
        it.can_rotate = p.get("can_rotate", False)
        it.allow_curtain = p.get("allow_curtain", True)
        it.allow_blinds = p.get("allow_blinds", True)
        it.starts_at_floor = p.get("starts_at_floor", False)
        mn = p.get("mesh", "")
        if mn and mn in bpy.data.objects:
            it.mesh_object = bpy.data.objects[mn]
    s.curtain_presets.clear()
    for p in data.get("curtain_presets", []):
        it = s.curtain_presets.add()
        it.name = p.get("name", "Curtain")
        it.library_key = p.get("library_key", "")
        it.category = p.get("category", "")
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


def _migrate_floor_map_naming():
    """One-time rename of the old 'GN_Boundaries' collection (from before the
    Floor Map rename) to the new name -- in place, so every boundary object
    and its bound_json data survives untouched, just under the new label."""
    old = bpy.data.collections.get("GN_Boundaries")
    if old is not None and bpy.data.collections.get(BOUND_COLL) is None:
        old.name = BOUND_COLL


def _deferred_restore():
    # runs after enable completes (bpy.data is accessible here, unlike register())
    _migrate_floor_map_naming()
    for scene in bpy.data.scenes:
        try:
            s = scene.gn_int
            if len(s.rooms) == 0 and len(s.floors) == 0 and scene.get("gn_int_backup"):
                _restore_scene(scene)           # only when a reload wiped the data
        except Exception as e:
            print("[UltimateMLO] restore failed:", e)
    return None                                 # don't repeat


def _setup_sync_handlers():
    """(Re-)establish the depsgraph handler and msgbus subscription that
    drive list<->viewport selection sync. Both need this re-run on every
    file load, not just on addon enable: a plain .append() on a handlers
    list is dropped on file load unless the function itself is decorated
    @persistent (added below), and empirically the msgbus subscription's
    own 'PERSISTENT' option was NOT enough to survive switching between
    .blend files in testing -- so this gets called from a load_post
    handler too, not just register(), to be robust either way."""
    if _gn_depsgraph_sync in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_gn_depsgraph_sync)
    bpy.app.handlers.depsgraph_update_post.append(_gn_depsgraph_sync)
    bpy.msgbus.clear_by_owner(_GN_MSGBUS_OWNER)
    bpy.msgbus.subscribe_rna(
        key=(bpy.types.LayerObjects, "active"),
        owner=_GN_MSGBUS_OWNER, args=(), notify=_gn_active_object_changed,
        options={'PERSISTENT'})


def _deferred_rescan_lib():
    """Populate the window-library browse list from disk once, after register
    or a file load, so the bundled library shows up without a manual Refresh."""
    try:
        if hasattr(bpy.context.scene, "gn_int"):
            rescan_window_library(bpy.context)
    except Exception:
        pass
    return None


@bpy.app.handlers.persistent
def _gn_on_load_post(dummy):
    _setup_sync_handlers()
    bpy.app.timers.register(_deferred_rescan_lib, first_interval=0.1)


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
    bpy.app.timers.register(_deferred_rescan_lib, first_interval=0.1)
    if _HL_HANDLE is None:
        _HL_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
            _draw_opening_highlight, (), 'WINDOW', 'POST_VIEW')
    _setup_sync_handlers()
    if _gn_on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_gn_on_load_post)
    bpy.app.handlers.load_post.append(_gn_on_load_post)


def unregister():
    global _HL_HANDLE
    if _gn_on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_gn_on_load_post)
    bpy.msgbus.clear_by_owner(_GN_MSGBUS_OWNER)
    if _gn_depsgraph_sync in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_gn_depsgraph_sync)
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
