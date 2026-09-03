# GN Interior Tool (MLO)

Exterior-aware interior generator for GTA MLO work in Blender (4.2+, tested on 5.0).
N-panel: **GN Interior**.

## Pipeline
1. **Set exterior** shell + **floor levels** (pick a facade bottom edge → base Z, ceiling auto +2.95 m, or pick two edges for a custom height).
2. **Generate Boundaries** — per floor, cross-section the wall outline, clean it up (Douglas–Peucker), inset the wall margin (20 cm).
3. **Rooms = Envelope** — each floor's boundary becomes an editable room.
4. **Split at Selected Edges** — Edit Mode: select two edges on opposite walls → cut the room in two with a partition-wall gap. Recursive, L-shape safe.
5. **Project Openings** — select separated window/door pieces (planes or meshes) → parametric holes cut into the matching room walls, with optional reveal jambs. Survives splits.
6. **Doors** (Room-Tool style) — door presets (width/height + optional frame mesh); **Door Edit** snaps a door to the nearest wall and cuts both adjacent rooms (doorway between rooms).

Room data (floors / rooms / openings / presets) is backed up to the scene and restored across add-on reloads.

## Install
Blender → Preferences → Add-ons → Install → pick the folder zipped, enable **GN Interior Tool (MLO)**.
