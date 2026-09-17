# Ultimate MLO Tool

Exterior-aware interior generator for GTA V MLO work in Blender (4.2+, tested on 5.0).
Sidebar (press **N**) → **UltimateMLO** tab.

Works on Windows, macOS and Linux. A **window library is bundled inside the add-on**,
so windows work out of the box — nothing to configure.

## Install

Use the ready-to-install ZIP:

1. Blender → **Edit → Preferences → Add-ons**
2. Drop-down (⌄) → **Install from Disk…** → pick `gn_interior_tool.zip`
3. Enable **UltimateMLO**

> Install the ZIP, not a loose folder. A GitHub "Download ZIP" unpacks to
> `gn_interior_tool-main`; rename it to exactly `gn_interior_tool` before zipping,
> or use the provided `gn_interior_tool.zip`.

**Full usage guide: see [`HOW_TO_USE.txt`](HOW_TO_USE.txt).**

## Pipeline

1. **Set exterior** shell + **floor levels**.
2. **Generate Floor Map** — cross-section the shell into a clean per-floor outline; hand-tidy, then **Save as Final**.
3. **Rooms** — each floor's outline becomes an editable room; **Split at Selected Points** carves partitions.
4. **Build Walls** — real 3D room shells.

## Windows

- **Project from Exterior** — select exterior window pieces → best-fitting library window dropped in at its true size (never stretched); a message if nothing fits.
- **Manual Placement** — pick a window from the library, click a wall to place it. Default sill height, plus "Starts at Floor" windows.
- **Swap** — select a placed window, click a different library window to swap it (errors if it won't fit a projected opening).
- **Window Edit Mode** — gizmo on the selected window: green = slide along wall, blue = up/down, yellow box = uniform scale.
- **Curtains / Blinds** — added per window, gated by each window's own allow-flags.
- **Register** your own window meshes into the library (auto-backed-up).

## Window library

Bundled in `window_library/`. To share one library across machines, set a path in
**Preferences → Add-ons → UltimateMLO → Window Library**; leave empty to use the bundled one.

Interior data (floors / rooms / openings / presets) is stored in the `.blend` and survives
saving and add-on reloads.
