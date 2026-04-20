"""Interactive GUI editor for Steiner Tree Problem instances.

Run from the project root:

    python SteinerTreeProblemQUBO/instance_editor.py

Builds weighted graphs (nodes, weighted edges, terminal set) by pointing and
clicking on a canvas, and saves them as JSON files under
``SteinerTreeProblemQUBO/STPInstances/``.

JSON format written:

    {
        "nodes":     ["v0", "v1", ...],
        "edges":     [["v0", "v1", 5], ...],      # [u, v, weight]
        "terminals": ["v0", "v3"],
        "positions": {"v0": [x, y], ...}           # optional, for round-tripping
    }
"""

import json
import math
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INSTANCES_DIR = SCRIPT_DIR / "STPInstances"

NODE_RADIUS = 22
NODE_COLOR = "#4a90e2"
TERMINAL_COLOR = "#e74c3c"
NODE_OUTLINE = "#1f3a5f"
SELECTED_OUTLINE = "#f1c40f"
EDGE_COLOR = "#34495e"
SOLUTION_EDGE_COLOR = "#27ae60"
FLOW_LABEL_FG = "#145a32"
FLOW_LABEL_BG = "#e6f8ea"
ROOT_RING_COLOR = "#f39c12"
ROOT_RING_PADDING = 6


def _point_segment_distance(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    cx = x1 + t * dx
    cy = y1 + t * dy
    return math.hypot(px - cx, py - cy)


def _fmt_flow(value):
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.2f}"


def _ensure_project_on_path():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


class InstanceEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Steiner Tree Instance Editor")
        self.root.geometry("1150x720")

        # graph data
        self.nodes = {}           # name -> (x, y)
        self.edges = {}           # frozenset({u, v}) -> int weight
        self.terminals = set()
        self.node_counter = 0

        # canvas bookkeeping
        self.node_items = {}       # name -> (circle_id, text_id)
        self.edge_items = {}       # frozenset({u, v}) -> (line_id, bg_id, text_id)
        self.edge_extra_items = {} # frozenset({u, v}) -> list of extra canvas ids

        # interaction state
        self.mode = tk.StringVar(value="add_node")
        self.edge_start = None
        self.drag_node = None
        self.drag_offset = (0, 0)

        # solver state
        self.solver_choice = tk.StringVar(value="li_et_al")
        self.solution_edges = set()    # frozenset({u, v}) keys in the solution
        self.solution_flows = {}       # {(u, v): flow_value} (directional)
        self.root_node = None          # name of the node picked as root
        self.root_ring = None          # canvas id for the root highlight ring

        self._build_ui()
        self._set_status_for_mode()

    # ----- UI -----

    def _build_ui(self):
        # Status bar packed first so it spans the full width at the bottom.
        self.status_var = tk.StringVar()
        status = ttk.Label(
            self.root, textvariable=self.status_var, relief="sunken", anchor="w"
        )
        status.pack(side="bottom", fill="x")

        left = ttk.Frame(self.root, padding=10)
        left.pack(side="left", fill="y")

        ttk.Label(left, text="Mode", font=("", 11, "bold")).pack(anchor="w")
        for value, label in [
            ("add_node", "Add node"),
            ("add_edge", "Add edge"),
            ("move", "Move / rename"),
            ("toggle_terminal", "Toggle terminal"),
            ("delete", "Delete"),
        ]:
            ttk.Radiobutton(
                left, text=label, variable=self.mode, value=value,
                command=self._on_mode_change,
            ).pack(anchor="w")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        ttk.Button(left, text="New / Clear", command=self.clear_all).pack(fill="x", pady=2)
        ttk.Button(left, text="Save\u2026", command=self.save_instance).pack(fill="x", pady=2)
        ttk.Button(left, text="Load\u2026", command=self.load_instance).pack(fill="x", pady=2)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        summary_frame = ttk.LabelFrame(left, text="Summary", padding=8)
        summary_frame.pack(fill="x")
        self.summary_var = tk.StringVar()
        ttk.Label(summary_frame, textvariable=self.summary_var).pack(anchor="w")
        self._refresh_summary()

        solve_frame = ttk.LabelFrame(left, text="Solve", padding=8)
        solve_frame.pack(fill="x", pady=(10, 0))
        ttk.Radiobutton(
            solve_frame, text="Li et al.",
            variable=self.solver_choice, value="li_et_al",
        ).pack(anchor="w")
        ttk.Radiobutton(
            solve_frame, text="My formulation (flow)",
            variable=self.solver_choice, value="my_formulation",
        ).pack(anchor="w")
        ttk.Button(
            solve_frame, text="Solve current graph", command=self.solve_current,
        ).pack(fill="x", pady=(6, 2))
        ttk.Button(
            solve_frame, text="Clear solution", command=self.clear_solution,
        ).pack(fill="x")

        ttk.Label(left, text="Tips", font=("", 10, "bold")).pack(anchor="w", pady=(12, 2))
        tips = (
            "\u2022 Add node: click empty space\n"
            "\u2022 Add edge: click two nodes;\n"
            "  enter a non-negative weight\n"
            "\u2022 Move: drag nodes around\n"
            "\u2022 Double-click a node to rename\n"
            "\u2022 Terminals are drawn in red"
        )
        tk.Label(left, text=tips, justify="left").pack(anchor="w")

        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(side="right", fill="both", expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)

    def _on_mode_change(self):
        self._clear_edge_start()
        self._set_status_for_mode()

    def _set_status_for_mode(self):
        msgs = {
            "add_node": "Add node: click anywhere on the canvas.",
            "add_edge": "Add edge: click the first node, then the second.",
            "move": "Move: drag nodes. Double-click to rename.",
            "toggle_terminal": "Toggle terminal: click a node.",
            "delete": "Delete: click a node or an edge to remove it.",
        }
        self.status_var.set(msgs.get(self.mode.get(), ""))

    # ----- canvas event routing -----

    def _on_click(self, event):
        mode = self.mode.get()
        hit_node = self._node_at(event.x, event.y)
        hit_edge = None if hit_node else self._edge_at(event.x, event.y)

        if mode == "add_node":
            if hit_node is None:
                self._add_node(event.x, event.y)
        elif mode == "add_edge":
            if hit_node is not None:
                self._handle_edge_click(hit_node)
            else:
                self._clear_edge_start()
        elif mode == "move":
            if hit_node is not None:
                self.drag_node = hit_node
                nx_, ny_ = self.nodes[hit_node]
                self.drag_offset = (nx_ - event.x, ny_ - event.y)
        elif mode == "toggle_terminal":
            if hit_node is not None:
                self._toggle_terminal(hit_node)
        elif mode == "delete":
            if hit_node is not None:
                self._delete_node(hit_node)
            elif hit_edge is not None:
                self._delete_edge(hit_edge)

    def _on_drag(self, event):
        if self.mode.get() == "move" and self.drag_node is not None:
            x = event.x + self.drag_offset[0]
            y = event.y + self.drag_offset[1]
            self._move_node(self.drag_node, x, y)

    def _on_release(self, _event):
        self.drag_node = None

    def _on_double_click(self, event):
        hit_node = self._node_at(event.x, event.y)
        if hit_node is not None:
            self._rename_node(hit_node)

    # ----- hit testing -----

    def _node_at(self, x, y):
        for name, (nx_, ny_) in self.nodes.items():
            if (nx_ - x) ** 2 + (ny_ - y) ** 2 <= NODE_RADIUS ** 2:
                return name
        return None

    def _edge_at(self, x, y, tolerance=8):
        best_key = None
        best_d = tolerance
        for key in self.edges:
            u, v = tuple(key)
            x1, y1 = self.nodes[u]
            x2, y2 = self.nodes[v]
            d = _point_segment_distance(x, y, x1, y1, x2, y2)
            if d <= best_d:
                best_d = d
                best_key = key
        return best_key

    # ----- node operations -----

    def _next_node_name(self):
        while True:
            candidate = f"v{self.node_counter}"
            self.node_counter += 1
            if candidate not in self.nodes:
                return candidate

    def _add_node(self, x, y, name=None):
        if name is None:
            name = self._next_node_name()
        if name in self.nodes:
            messagebox.showerror("Duplicate node", f"A node named '{name}' already exists.")
            return
        self.nodes[name] = (x, y)
        self._draw_node(name)
        self._refresh_summary()

    def _draw_node(self, name):
        x, y = self.nodes[name]
        fill = TERMINAL_COLOR if name in self.terminals else NODE_COLOR
        circle = self.canvas.create_oval(
            x - NODE_RADIUS, y - NODE_RADIUS,
            x + NODE_RADIUS, y + NODE_RADIUS,
            fill=fill, outline=NODE_OUTLINE, width=2, tags=("node",),
        )
        text = self.canvas.create_text(
            x, y, text=name, fill="white", font=("", 10, "bold"), tags=("node",),
        )
        self.node_items[name] = (circle, text)
        self.canvas.tag_raise("node")

    def _move_node(self, name, x, y):
        self.nodes[name] = (x, y)
        circle, text = self.node_items[name]
        self.canvas.coords(
            circle, x - NODE_RADIUS, y - NODE_RADIUS,
            x + NODE_RADIUS, y + NODE_RADIUS,
        )
        self.canvas.coords(text, x, y)
        for key in list(self.edges):
            if name in key:
                self._redraw_edge(key)
        if name == self.root_node and self.root_ring is not None:
            pad = NODE_RADIUS + ROOT_RING_PADDING
            self.canvas.coords(
                self.root_ring,
                x - pad, y - pad, x + pad, y + pad,
            )
        self.canvas.tag_raise("node")

    def _rename_node(self, old):
        new = simpledialog.askstring(
            "Rename node", f"New name for '{old}':",
            initialvalue=old, parent=self.root,
        )
        if new is None:
            return
        new = new.strip()
        if not new or new == old:
            return
        if new in self.nodes:
            messagebox.showerror("Duplicate", f"Node '{new}' already exists.")
            return

        self.nodes[new] = self.nodes.pop(old)
        self.node_items[new] = self.node_items.pop(old)

        new_edges = {}
        new_edge_items = {}
        for key, w in self.edges.items():
            if old in key:
                other = next(iter(key - {old}))
                new_key = frozenset({new, other})
                new_edges[new_key] = w
                new_edge_items[new_key] = self.edge_items[key]
            else:
                new_edges[key] = w
                new_edge_items[key] = self.edge_items[key]
        self.edges = new_edges
        self.edge_items = new_edge_items

        if old in self.terminals:
            self.terminals.remove(old)
            self.terminals.add(new)

        # redraw the renamed node's label
        circle, text = self.node_items[new]
        self.canvas.itemconfigure(text, text=new)
        self._invalidate_solution()
        self._refresh_summary()

    def _delete_node(self, name):
        for cid in self.node_items.pop(name, ()):
            self.canvas.delete(cid)
        for key in [k for k in self.edges if name in k]:
            self._delete_edge(key, refresh=False)
        self.nodes.pop(name, None)
        self.terminals.discard(name)
        if self.edge_start == name:
            self.edge_start = None
            self._set_status_for_mode()
        self._invalidate_solution()
        self._refresh_summary()

    def _toggle_terminal(self, name):
        if name in self.terminals:
            self.terminals.remove(name)
        else:
            self.terminals.add(name)
        circle, _ = self.node_items[name]
        fill = TERMINAL_COLOR if name in self.terminals else NODE_COLOR
        self.canvas.itemconfigure(circle, fill=fill)
        self._invalidate_solution()
        self._refresh_summary()

    # ----- edge operations -----

    def _handle_edge_click(self, name):
        if self.edge_start is None:
            self.edge_start = name
            circle, _ = self.node_items[name]
            self.canvas.itemconfigure(circle, outline=SELECTED_OUTLINE, width=3)
            self.status_var.set(f"Selected '{name}'. Click another node to connect.")
            return
        if self.edge_start == name:
            self._clear_edge_start()
            return
        key = frozenset({self.edge_start, name})
        if key in self.edges:
            messagebox.showinfo(
                "Edge exists",
                f"An edge between '{self.edge_start}' and '{name}' already exists.",
            )
            self._clear_edge_start()
            return
        weight = simpledialog.askinteger(
            "Edge weight",
            f"Weight for edge {self.edge_start} \u2014 {name}:",
            parent=self.root, minvalue=0, initialvalue=1,
        )
        self._clear_edge_start()
        if weight is None:
            return
        self.edges[key] = int(weight)
        self._draw_edge(key)
        self._refresh_summary()

    def _clear_edge_start(self):
        if self.edge_start is not None and self.edge_start in self.node_items:
            circle, _ = self.node_items[self.edge_start]
            self.canvas.itemconfigure(circle, outline=NODE_OUTLINE, width=2)
        self.edge_start = None
        self._set_status_for_mode()

    def _draw_edge(self, key):
        u, v = tuple(key)
        x1, y1 = self.nodes[u]
        x2, y2 = self.nodes[v]
        weight = self.edges[key]

        selected = key in self.solution_edges

        # decide arrow direction from flow (only shown for selected edges)
        arrow_opt = "none"
        flow_value = None
        if selected and self.solution_flows:
            f_uv = float(self.solution_flows.get((u, v), 0.0))
            f_vu = float(self.solution_flows.get((v, u), 0.0))
            if f_uv > f_vu + 1e-6:
                arrow_opt = "last"
                flow_value = f_uv
            elif f_vu > f_uv + 1e-6:
                arrow_opt = "first"
                flow_value = f_vu
            elif f_uv > 1e-6:
                flow_value = f_uv

        line_color = SOLUTION_EDGE_COLOR if selected else EDGE_COLOR
        line_width = 4 if selected else 2

        # shrink endpoints to the node-circle boundary so arrowheads are
        # visible instead of being hidden underneath the target node
        dx_total, dy_total = x2 - x1, y2 - y1
        seg_len = math.hypot(dx_total, dy_total)
        if seg_len > 2 * NODE_RADIUS + 4:
            ux, uy = dx_total / seg_len, dy_total / seg_len
            lx1 = x1 + ux * NODE_RADIUS
            ly1 = y1 + uy * NODE_RADIUS
            lx2 = x2 - ux * NODE_RADIUS
            ly2 = y2 - uy * NODE_RADIUS
        else:
            lx1, ly1, lx2, ly2 = x1, y1, x2, y2

        line = self.canvas.create_line(
            lx1, ly1, lx2, ly2,
            fill=line_color, width=line_width,
            arrow=arrow_opt, arrowshape=(14, 18, 7),
            tags=("edgeline",),
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        label_w = 7 * max(2, len(str(weight))) + 6
        bg = self.canvas.create_rectangle(
            mx - label_w / 2, my - 10, mx + label_w / 2, my + 10,
            fill="white", outline="", tags=("edgelabel",),
        )
        txt = self.canvas.create_text(
            mx, my, text=str(weight), font=("", 9, "bold"), tags=("edgelabel",),
        )
        self.edge_items[key] = (line, bg, txt)

        extras = []
        if flow_value is not None:
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy) or 1.0
            px, py = -dy / length, dx / length
            offset = 20
            fx, fy = mx + px * offset, my + py * offset
            flow_text = f"f={_fmt_flow(flow_value)}"
            fw = 7 * max(3, len(flow_text)) + 8
            fbg = self.canvas.create_rectangle(
                fx - fw / 2, fy - 10, fx + fw / 2, fy + 10,
                fill=FLOW_LABEL_BG, outline=SOLUTION_EDGE_COLOR,
                tags=("edgelabel",),
            )
            ftxt = self.canvas.create_text(
                fx, fy, text=flow_text, fill=FLOW_LABEL_FG,
                font=("", 9, "bold"), tags=("edgelabel",),
            )
            extras = [fbg, ftxt]
        self.edge_extra_items[key] = extras

        self.canvas.tag_lower("edgeline")
        self.canvas.tag_raise("edgelabel", "edgeline")
        self.canvas.tag_raise("node")

    def _redraw_edge(self, key):
        for cid in self.edge_items.pop(key, ()):
            self.canvas.delete(cid)
        for cid in self.edge_extra_items.pop(key, ()):
            self.canvas.delete(cid)
        self._draw_edge(key)

    def _delete_edge(self, key, refresh=True):
        for cid in self.edge_items.pop(key, ()):
            self.canvas.delete(cid)
        for cid in self.edge_extra_items.pop(key, ()):
            self.canvas.delete(cid)
        self.edges.pop(key, None)
        self._invalidate_solution()
        if refresh:
            self._refresh_summary()

    # ----- summary / clear -----

    def _refresh_summary(self):
        self.summary_var.set(
            f"{len(self.nodes)} nodes, {len(self.edges)} edges, "
            f"{len(self.terminals)} terminals"
        )

    def clear_all(self):
        if self.nodes and not messagebox.askyesno(
            "Clear all",
            "Remove all nodes, edges, and terminals from the canvas?",
        ):
            return
        self.canvas.delete("all")
        self.nodes.clear()
        self.edges.clear()
        self.terminals.clear()
        self.node_items.clear()
        self.edge_items.clear()
        self.edge_extra_items.clear()
        self.solution_edges.clear()
        self.solution_flows.clear()
        self._clear_root_ring()
        self.node_counter = 0
        self.edge_start = None
        self._refresh_summary()
        self._set_status_for_mode()

    # ----- persistence -----

    def _canonical_edge_pairs(self):
        pairs = []
        for key in self.edges:
            u, v = sorted(key)
            pairs.append((u, v))
        pairs.sort()
        return pairs

    def save_instance(self):
        if not self.nodes:
            messagebox.showerror("Nothing to save", "Add at least one node first.")
            return
        if not self.edges:
            messagebox.showerror("Nothing to save", "Add at least one edge first.")
            return
        if not self.terminals:
            messagebox.showerror("Nothing to save", "Mark at least one node as a terminal.")
            return

        INSTANCES_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="Save Steiner Tree instance",
            initialdir=str(INSTANCES_DIR),
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        payload = {
            "nodes": list(self.nodes.keys()),
            "edges": [
                [u, v, int(self.edges[frozenset({u, v})])]
                for u, v in self._canonical_edge_pairs()
            ],
            "terminals": [n for n in self.nodes if n in self.terminals],
            "positions": {n: [float(x), float(y)] for n, (x, y) in self.nodes.items()},
        }
        try:
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
        except OSError as e:
            messagebox.showerror("Save failed", str(e))
            return
        self.status_var.set(f"Saved {Path(path).name}")

    def load_instance(self):
        INSTANCES_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.askopenfilename(
            title="Load Steiner Tree instance",
            initialdir=str(INSTANCES_DIR),
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            messagebox.showerror("Failed to load", str(e))
            return

        try:
            nodes = list(data["nodes"])
            raw_edges = [(str(u), str(v), int(w)) for u, v, w in data["edges"]]
            terminals = [str(t) for t in data["terminals"]]
        except (KeyError, TypeError, ValueError) as e:
            messagebox.showerror("Invalid file", f"Bad JSON structure: {e}")
            return

        positions = data.get("positions", {}) or {}

        self.clear_all_silent()
        self.terminals = {n for n in terminals if n in set(nodes)}

        # lay nodes out on a circle if positions are missing
        self.canvas.update_idletasks()
        width = self.canvas.winfo_width() or 800
        height = self.canvas.winfo_height() or 600
        cx, cy = width / 2, height / 2
        r = max(60.0, min(width, height) / 2 - 80)

        for i, name in enumerate(nodes):
            pos = positions.get(name)
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                x, y = float(pos[0]), float(pos[1])
            else:
                angle = 2 * math.pi * i / max(1, len(nodes))
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)
            self.nodes[name] = (x, y)
            self._draw_node(name)

        # bump node_counter so auto-generated names don't collide
        max_i = -1
        for name in nodes:
            if name.startswith("v") and name[1:].isdigit():
                max_i = max(max_i, int(name[1:]))
        self.node_counter = max_i + 1

        for u, v, w in raw_edges:
            if u in self.nodes and v in self.nodes and u != v:
                key = frozenset({u, v})
                if key not in self.edges:
                    self.edges[key] = int(w)
                    self._draw_edge(key)

        self._refresh_summary()
        self.status_var.set(f"Loaded {Path(path).name}")

    def clear_all_silent(self):
        self.canvas.delete("all")
        self.nodes.clear()
        self.edges.clear()
        self.terminals.clear()
        self.node_items.clear()
        self.edge_items.clear()
        self.edge_extra_items.clear()
        self.solution_edges.clear()
        self.solution_flows.clear()
        self._clear_root_ring()
        self.node_counter = 0
        self.edge_start = None

    # ----- solver integration -----

    def _invalidate_solution(self):
        if self.solution_edges or self.solution_flows or self.root_node is not None:
            self.clear_solution()

    def _set_root_node(self, name):
        self._clear_root_ring()
        if name is None or name not in self.nodes:
            self.root_node = None
            return
        self.root_node = name
        x, y = self.nodes[name]
        pad = NODE_RADIUS + ROOT_RING_PADDING
        self.root_ring = self.canvas.create_oval(
            x - pad, y - pad, x + pad, y + pad,
            outline=ROOT_RING_COLOR, width=3, dash=(4, 2),
            tags=("rootring",),
        )
        self.canvas.tag_raise("node")

    def _clear_root_ring(self):
        if self.root_ring is not None:
            self.canvas.delete(self.root_ring)
            self.root_ring = None
        self.root_node = None

    def clear_solution(self):
        had_state = (
            self.solution_edges or self.solution_flows or self.root_node is not None
        )
        if not had_state:
            return
        self.solution_edges = set()
        self.solution_flows = {}
        self._clear_root_ring()
        for key in list(self.edges):
            self._redraw_edge(key)
        self._set_status_for_mode()

    def solve_current(self):
        if not self.nodes:
            messagebox.showerror("Cannot solve", "Add at least one node first.")
            return
        if not self.edges:
            messagebox.showerror("Cannot solve", "Add at least one edge first.")
            return
        if not self.terminals:
            messagebox.showerror("Cannot solve", "Mark at least one node as a terminal.")
            return

        _ensure_project_on_path()
        try:
            from SteinerTreeProblemQUBO.SteinerTree import SteinerTree
        except ImportError as e:
            messagebox.showerror("Import error", f"Could not import SteinerTree:\n{e}")
            return

        try:
            nodes = list(self.nodes.keys())
            edges = [
                (u, v, int(self.edges[frozenset({u, v})]))
                for u, v in self._canonical_edge_pairs()
            ]
            terminals = [n for n in nodes if n in self.terminals]
            problem = SteinerTree(nodes, edges, terminals)
        except Exception as e:
            messagebox.showerror("Invalid instance", str(e))
            return

        choice = self.solver_choice.get()
        label = "Li et al." if choice == "li_et_al" else "my formulation"
        self.status_var.set(f"Solving with {label}\u2026")
        self.root.update_idletasks()

        try:
            if choice == "li_et_al":
                from SteinerTreeProblemQUBO.Li_et_al.gurobi_solver import (
                    solve_ilp_li_et_al,
                )
                result = solve_ilp_li_et_al(problem)
            else:
                from SteinerTreeProblemQUBO.MyFormulization.gurobi_solver import (
                    solve_ilp,
                )
                result = solve_ilp(problem)
        except ImportError as e:
            messagebox.showerror(
                "Solver unavailable",
                f"Could not import the solver. Is gurobipy installed and licensed?\n\n{e}",
            )
            self._set_status_for_mode()
            return
        except Exception as e:
            messagebox.showerror("Solver error", f"{type(e).__name__}: {e}")
            self._set_status_for_mode()
            return

        if result.get("status") != "OPTIMAL":
            messagebox.showinfo(
                "No optimal solution",
                f"Solver returned status: {result.get('status')}",
            )
            self._set_status_for_mode()
            return

        solution_edges = set()
        for u, v, _ in result.get("edges", []):
            key = frozenset({u, v})
            if key in self.edges:
                solution_edges.add(key)

        flows = result.get("flows") or {}

        self.solution_edges = solution_edges
        self.solution_flows = dict(flows)
        self._set_root_node(result.get("root"))
        for key in list(self.edges):
            self._redraw_edge(key)

        cost = result.get("cost")
        if cost is None:
            cost = result.get("tree_cost")
        if cost is None:
            cost = result.get("objective")
        cost_str = f"{cost:.4g}" if cost is not None else "n/a"
        root_str = f", root = {self.root_node}" if self.root_node else ""
        self.status_var.set(
            f"{label}: cost = {cost_str}, edges = {len(solution_edges)}{root_str}"
        )


def main():
    root = tk.Tk()
    InstanceEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
