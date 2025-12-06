"""
Lightweight HTTP server for graph visualization.

Serves an interactive D3.js force-directed graph with:
- Real-time parameter adjustment via control panel
- Focus query cycling for exploring rank topology shifts
- Visual encoding of bridges, API surface, rank concentration

Architecture:
- Pure stdlib http.server (no Flask dependency)
- Static HTML/JS/CSS embedded for zero-config deployment
- REST API for graph data updates and parameter changes
"""

import json
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Optional, Callable, Dict, Any
from urllib.parse import parse_qs, urlparse

from .exporter import ExportedGraph
from .tuner import HeuristicRegistry, TunerState


class VisualizerHandler(SimpleHTTPRequestHandler):
    """HTTP handler for visualization server.

    Routes:
    - GET / → Serve main HTML page
    - GET /api/graph → Current graph data as JSON
    - GET /api/graph?focus=X&preset=Y → Graph with focus and preset
    - GET /api/schema → Heuristic schema for UI rendering
    - GET /api/presets → Available presets
    - POST /api/tuner → Update tuner state (enabled heuristics + params)
    - GET /api/health → Health check
    """

    # Class-level state shared across requests
    graph_provider: Optional[Callable[..., ExportedGraph]] = None
    current_params: Dict[str, Any] = {}
    tuner_state: TunerState = TunerState()
    registry: HeuristicRegistry = HeuristicRegistry()

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/api/graph":
            self._serve_graph(parsed.query)
        elif path == "/api/health":
            self._json_response({"status": "ok"})
        elif path == "/api/params":
            self._json_response(self.current_params)
        elif path == "/api/schema":
            self._json_response(self.registry.get_api_schema())
        elif path == "/api/presets":
            self._json_response({"presets": self.registry.get_all_presets()})
        elif path == "/api/tuner":
            self._json_response(self.tuner_state.to_dict())
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/params":
            self._update_params()
        elif parsed.path == "/api/focus":
            self._update_focus()
        elif parsed.path == "/api/tuner":
            self._update_tuner()
        elif parsed.path == "/api/preset":
            self._apply_preset()
        else:
            self.send_error(404)

    def _serve_html(self):
        """Serve the main visualization page."""
        html = get_visualization_html()
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_graph(self, query_string: str):
        """Serve graph data with current tuner state."""
        params = parse_qs(query_string)
        focus = params.get("focus", [None])[0]

        # Update tuner state with focus from query
        if focus:
            VisualizerHandler.tuner_state.focus = focus

        # Access graph_provider through class to avoid descriptor binding
        # (self.graph_provider would pass self as first arg since it's a function)
        provider = VisualizerHandler.graph_provider
        if provider:
            try:
                # Pass tuner state to provider so it can apply heuristics
                graph = provider(
                    focus_query=focus,
                    tuner_state=VisualizerHandler.tuner_state
                )
                self._json_response(graph.to_dict())
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._json_response({"error": str(e)}, status=500)
        else:
            self._json_response({"error": "No graph provider configured"}, status=503)

    def _update_params(self):
        """Update hyperparameters from POST body."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            new_params = json.loads(body)
            self.current_params.update(new_params)
            self._json_response({"status": "ok", "params": self.current_params})
        except Exception as e:
            self._json_response({"error": str(e)}, status=400)

    def _update_focus(self):
        """Update focus query from POST body."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            focus = data.get("focus", "")
            self.current_params["focus"] = focus
            self.tuner_state.focus = focus
            self._json_response({"status": "ok", "focus": focus})
        except Exception as e:
            self._json_response({"error": str(e)}, status=400)

    def _update_tuner(self):
        """Update tuner state (heuristics + params)."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            self.tuner_state = TunerState.from_dict(data)
            warnings = self.registry.validate_state(self.tuner_state)
            self._json_response({
                "status": "ok",
                "state": self.tuner_state.to_dict(),
                "warnings": warnings
            })
        except Exception as e:
            self._json_response({"error": str(e)}, status=400)

    def _apply_preset(self):
        """Apply a named preset."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            preset_id = data.get("preset", "default")
            preset_state = self.registry.get_preset(preset_id)
            if preset_state:
                self.tuner_state = TunerState(
                    enabled=preset_state.enabled.copy(),
                    focus=self.tuner_state.focus,  # Preserve current focus
                    preset=preset_id
                )
                self._json_response({
                    "status": "ok",
                    "preset": preset_id,
                    "state": self.tuner_state.to_dict()
                })
            else:
                self._json_response({"error": f"Unknown preset: {preset_id}"}, status=400)
        except Exception as e:
            self._json_response({"error": str(e)}, status=400)

    def _json_response(self, data: dict, status: int = 200):
        """Send JSON response."""
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


class VisualizerServer:
    """Web server for graph visualization.

    Usage:
        def get_graph(focus_query=None):
            # Run grepmap with focus_query, return ExportedGraph
            ...

        server = VisualizerServer(graph_provider=get_graph, port=8765)
        server.start()  # Starts in background thread
        # ... graph updates will be visible on refresh
        server.stop()
    """

    def __init__(
        self,
        graph_provider: Callable[..., ExportedGraph],
        port: int = 8765,
        host: str = "localhost",
        auto_open: bool = True
    ):
        """Initialize visualization server.

        Args:
            graph_provider: Callable that returns ExportedGraph,
                           accepts optional focus_query kwarg
            port: HTTP port to listen on
            host: Host to bind to
            auto_open: Open browser automatically on start
        """
        self.graph_provider = graph_provider
        self.port = port
        self.host = host
        self.auto_open = auto_open
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self):
        """Start the server in a background thread."""
        VisualizerHandler.graph_provider = self.graph_provider
        VisualizerHandler.current_params = {}

        self.server = HTTPServer((self.host, self.port), VisualizerHandler)

        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        url = f"http://{self.host}:{self.port}"
        print(f"🌐 Graph visualizer running at {url}")

        if self.auto_open:
            webbrowser.open(url)

    def stop(self):
        """Stop the server."""
        if self.server:
            self.server.shutdown()
            self.server = None
        if self.thread:
            self.thread.join(timeout=1)
            self.thread = None


def get_visualization_html() -> str:
    """Return the embedded visualization HTML/JS/CSS.

    D3.js force-directed graph with:
    - Node size ~ rank (log scale)
    - Node color ~ cluster (directory)
    - Special styling for bridges (diamond), API (ring)
    - Edge thickness ~ weight
    - Control panel for focus cycling and params
    """
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>grepmap - Graph Visualizer</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            overflow: hidden;
        }
        #container { display: flex; height: 100vh; }
        #graph-container {
            flex: 1;
            position: relative;
        }
        #controls {
            width: 320px;
            background: #161b22;
            padding: 16px;
            border-left: 1px solid #30363d;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        h1 { font-size: 18px; color: #58a6ff; margin-bottom: 8px; }
        h2 { font-size: 14px; color: #8b949e; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
        .section { background: #0d1117; padding: 12px; border-radius: 6px; border: 1px solid #30363d; }

        /* Focus input */
        #focus-input {
            width: 100%;
            padding: 8px 12px;
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            font-size: 14px;
        }
        #focus-input:focus { outline: none; border-color: #58a6ff; }

        /* Buttons */
        .btn {
            padding: 8px 16px;
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }
        .btn:hover { background: #30363d; border-color: #58a6ff; }
        .btn.active { background: #388bfd; color: white; border-color: #388bfd; }
        .btn-row { display: flex; gap: 8px; flex-wrap: wrap; }

        /* Presets */
        .preset-btn {
            padding: 4px 10px;
            font-size: 12px;
        }

        /* Sliders */
        .slider-group { margin-bottom: 12px; }
        .slider-label {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            margin-bottom: 4px;
        }
        .slider-label .value { color: #58a6ff; font-family: monospace; }
        input[type="range"] {
            width: 100%;
            height: 4px;
            background: #30363d;
            border-radius: 2px;
            appearance: none;
        }
        input[type="range"]::-webkit-slider-thumb {
            appearance: none;
            width: 14px;
            height: 14px;
            background: #58a6ff;
            border-radius: 50%;
            cursor: pointer;
        }

        /* Stats */
        .stats { font-size: 13px; font-family: monospace; }
        .stats .row { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #21262d; }
        .stats .label { color: #8b949e; }
        .stats .value { color: #58a6ff; }
        .stats .value.high { color: #3fb950; }
        .stats .value.medium { color: #d29922; }
        .stats .value.low { color: #f85149; }

        /* Legend */
        .legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: 12px; }
        .legend-item { display: flex; align-items: center; gap: 6px; }
        .legend-dot { width: 10px; height: 10px; border-radius: 50%; }
        .legend-diamond { width: 10px; height: 10px; background: #f0883e; transform: rotate(45deg); }
        .legend-ring { width: 10px; height: 10px; border: 2px solid #a371f7; border-radius: 50%; }

        /* Animation controls */
        .cycle-controls { display: flex; gap: 8px; align-items: center; }
        #cycle-list {
            width: 100%;
            padding: 8px;
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #8b949e;
            font-size: 12px;
            font-family: monospace;
            resize: vertical;
            min-height: 80px;
        }

        /* Toast notifications */
        #toast {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: #21262d;
            color: #c9d1d9;
            padding: 12px 24px;
            border-radius: 6px;
            border: 1px solid #30363d;
            font-size: 14px;
            opacity: 0;
            transition: opacity 0.3s;
            pointer-events: none;
            z-index: 1000;
        }
        #toast.show { opacity: 1; }

        /* Node tooltip */
        .tooltip {
            position: absolute;
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 10px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            max-width: 280px;
            z-index: 100;
        }
        .tooltip.show { opacity: 1; }
        .tooltip .file { color: #8b949e; margin-bottom: 4px; font-family: monospace; }
        .tooltip .symbol { color: #58a6ff; font-weight: bold; font-size: 14px; }
        .tooltip .stats { margin-top: 8px; }
        .tooltip .badge {
            display: inline-block;
            padding: 2px 6px;
            background: #30363d;
            border-radius: 4px;
            font-size: 10px;
            margin-right: 4px;
        }
        .tooltip .badge.recent { background: #238636; }
        .tooltip .badge.churn { background: #9e6a03; }
        .tooltip .badge.emergent { background: #8957e5; }
        .tooltip .badge.bridge { background: #f0883e; }
        .tooltip .badge.api { background: #a371f7; }
    </style>
</head>
<body>
    <div id="container">
        <div id="graph-container">
            <svg id="graph"></svg>
            <div class="tooltip" id="tooltip"></div>
        </div>
        <div id="controls">
            <div>
                <h1>grepmap visualizer</h1>
                <p style="font-size: 12px; color: #8b949e;">Explore ranking topology and rhizome patterns</p>
            </div>

            <div class="section">
                <h2>Focus Query</h2>
                <input type="text" id="focus-input" placeholder="e.g., ranking, facade, parser...">
                <div class="btn-row" style="margin-top: 8px;">
                    <button class="btn preset-btn" data-focus="">∅ clear</button>
                    <button class="btn preset-btn" data-focus="ranking">ranking</button>
                    <button class="btn preset-btn" data-focus="render">render</button>
                    <button class="btn preset-btn" data-focus="parse">parse</button>
                </div>
            </div>

            <div class="section">
                <h2>Focus Cycling</h2>
                <textarea id="cycle-list" placeholder="One focus per line...">ranking
render
bridge
api
parser
config
cache</textarea>
                <div class="cycle-controls" style="margin-top: 8px;">
                    <button class="btn" id="cycle-btn">▶ Cycle</button>
                    <span style="font-size: 12px; color: #8b949e;" id="cycle-status">stopped</span>
                </div>
            </div>

            <div class="section">
                <h2>Display</h2>
                <div class="slider-group">
                    <div class="slider-label">
                        <span>Link distance</span>
                        <span class="value" id="link-dist-val">100</span>
                    </div>
                    <input type="range" id="link-distance" min="30" max="300" value="100">
                </div>
                <div class="slider-group">
                    <div class="slider-label">
                        <span>Charge strength</span>
                        <span class="value" id="charge-val">-150</span>
                    </div>
                    <input type="range" id="charge" min="-500" max="-30" value="-150">
                </div>
                <div class="slider-group">
                    <div class="slider-label">
                        <span>Node scale</span>
                        <span class="value" id="node-scale-val">1.0</span>
                    </div>
                    <input type="range" id="node-scale" min="0.3" max="3" value="1" step="0.1">
                </div>
                <div class="btn-row">
                    <button class="btn" id="toggle-labels">Labels: ON</button>
                    <button class="btn" id="toggle-orphans">Orphans: ON</button>
                </div>
            </div>

            <div class="section">
                <h2>Graph Stats</h2>
                <div class="stats" id="stats">
                    <div class="row"><span class="label">Nodes</span><span class="value">-</span></div>
                    <div class="row"><span class="label">Edges</span><span class="value">-</span></div>
                    <div class="row"><span class="label">Density</span><span class="value">-</span></div>
                    <div class="row"><span class="label">Gini</span><span class="value">-</span></div>
                    <div class="row"><span class="label">Confidence</span><span class="value">-</span></div>
                    <div class="row"><span class="label">Orphans</span><span class="value">-</span></div>
                    <div class="row"><span class="label">Intent</span><span class="value">-</span></div>
                </div>
            </div>

            <div class="section">
                <h2>Legend</h2>
                <div class="legend">
                    <div class="legend-item"><div class="legend-diamond"></div> Bridge</div>
                    <div class="legend-item"><div class="legend-ring"></div> API</div>
                    <div class="legend-item"><div class="legend-dot" style="background: #8b949e;"></div> Orphan</div>
                    <div class="legend-item"><div class="legend-dot" style="background: #58a6ff;"></div> Normal</div>
                </div>
            </div>

            <div class="section">
                <h2>Top Hubs</h2>
                <div id="hubs" style="font-size: 12px; font-family: monospace;"></div>
            </div>
        </div>
    </div>
    <div id="toast"></div>

<script>
// State
let graphData = null;
let simulation = null;
let showLabels = true;
let showOrphans = true;
let cycleInterval = null;
let cycleIndex = 0;

// Color scale for clusters
const clusterColors = d3.scaleOrdinal(d3.schemeTableau10);

// SVG setup
const svg = d3.select("#graph");
const container = document.getElementById("graph-container");
let width = container.clientWidth;
let height = container.clientHeight;
svg.attr("width", width).attr("height", height);

// Groups for layering
const g = svg.append("g");
const linkGroup = g.append("g").attr("class", "links");
const nodeGroup = g.append("g").attr("class", "nodes");
const labelGroup = g.append("g").attr("class", "labels");

// Zoom
const zoom = d3.zoom()
    .scaleExtent([0.1, 10])
    .on("zoom", (e) => g.attr("transform", e.transform));
svg.call(zoom);

// Tooltip
const tooltip = document.getElementById("tooltip");

// Load initial graph
fetchGraph();

// Focus input
const focusInput = document.getElementById("focus-input");
focusInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        fetchGraph(focusInput.value);
    }
});

// Preset buttons
document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        const focus = btn.dataset.focus;
        focusInput.value = focus;
        fetchGraph(focus);
    });
});

// Cycle controls
const cycleBtn = document.getElementById("cycle-btn");
const cycleStatus = document.getElementById("cycle-status");
const cycleList = document.getElementById("cycle-list");

cycleBtn.addEventListener("click", () => {
    if (cycleInterval) {
        stopCycle();
    } else {
        startCycle();
    }
});

function getCycleQueries() {
    return cycleList.value.split("\\n").map(s => s.trim()).filter(s => s);
}

function startCycle() {
    const queries = getCycleQueries();
    if (queries.length === 0) return;

    cycleIndex = 0;
    cycleBtn.textContent = "⏹ Stop";
    cycleBtn.classList.add("active");

    function tick() {
        const query = queries[cycleIndex % queries.length];
        focusInput.value = query;
        fetchGraph(query);
        cycleStatus.textContent = `${cycleIndex + 1}/${queries.length}: ${query}`;
        cycleIndex++;
    }

    tick();
    cycleInterval = setInterval(tick, 2000);
}

function stopCycle() {
    clearInterval(cycleInterval);
    cycleInterval = null;
    cycleBtn.textContent = "▶ Cycle";
    cycleBtn.classList.remove("active");
    cycleStatus.textContent = "stopped";
}

// Display controls
const linkDistSlider = document.getElementById("link-distance");
const chargeSlider = document.getElementById("charge");
const nodeScaleSlider = document.getElementById("node-scale");

linkDistSlider.addEventListener("input", updateSimulation);
chargeSlider.addEventListener("input", updateSimulation);
nodeScaleSlider.addEventListener("input", updateNodeScale);

document.getElementById("toggle-labels").addEventListener("click", (e) => {
    showLabels = !showLabels;
    e.target.textContent = `Labels: ${showLabels ? "ON" : "OFF"}`;
    labelGroup.style("display", showLabels ? null : "none");
});

document.getElementById("toggle-orphans").addEventListener("click", (e) => {
    showOrphans = !showOrphans;
    e.target.textContent = `Orphans: ${showOrphans ? "ON" : "OFF"}`;
    renderGraph();
});

function updateSimulation() {
    if (!simulation) return;
    const linkDist = +linkDistSlider.value;
    const charge = +chargeSlider.value;
    document.getElementById("link-dist-val").textContent = linkDist;
    document.getElementById("charge-val").textContent = charge;

    simulation
        .force("link").distance(linkDist);
    simulation
        .force("charge").strength(charge);
    simulation.alpha(0.3).restart();
}

function updateNodeScale() {
    const scale = +nodeScaleSlider.value;
    document.getElementById("node-scale-val").textContent = scale.toFixed(1);
    nodeGroup.selectAll("circle, path")
        .attr("transform", d => `scale(${scale})`);
}

// Fetch graph from server
async function fetchGraph(focus = null) {
    try {
        const url = focus ? `/api/graph?focus=${encodeURIComponent(focus)}` : "/api/graph";
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        graphData = await resp.json();
        renderGraph();
        updateStats();
        toast(`Loaded: ${graphData.metadata.num_nodes} nodes, ${graphData.metadata.num_edges} edges`);
    } catch (err) {
        toast(`Error: ${err.message}`, true);
    }
}

// Render the graph
function renderGraph() {
    if (!graphData) return;

    let nodes = graphData.nodes;
    let edges = graphData.edges;

    // Filter orphans if hidden
    if (!showOrphans) {
        const connectedIds = new Set();
        edges.forEach(e => {
            connectedIds.add(e.source);
            connectedIds.add(e.target);
        });
        nodes = nodes.filter(n => connectedIds.has(n.id));
    }

    // Create node map for edge resolution
    const nodeById = new Map(nodes.map(n => [n.id, n]));

    // Filter edges to only include existing nodes
    edges = edges.filter(e => nodeById.has(e.source) && nodeById.has(e.target));

    // Size scale (log of rank)
    const rankExtent = d3.extent(nodes, d => d.rank);
    const sizeScale = d3.scaleLog()
        .domain([Math.max(rankExtent[0], 0.0001), Math.max(rankExtent[1], 0.001)])
        .range([4, 20])
        .clamp(true);

    // Links
    const linkData = linkGroup.selectAll("line").data(edges, d => `${d.source}-${d.target}`);
    linkData.exit().remove();
    const linkEnter = linkData.enter().append("line")
        .attr("stroke", "#30363d")
        .attr("stroke-opacity", 0.6)
        .attr("stroke-width", d => Math.sqrt(d.weight));
    const links = linkEnter.merge(linkData);

    // Nodes
    const nodeData = nodeGroup.selectAll("g").data(nodes, d => d.id);
    nodeData.exit().remove();

    const nodeEnter = nodeData.enter().append("g")
        .attr("class", "node")
        .call(d3.drag()
            .on("start", dragStarted)
            .on("drag", dragged)
            .on("end", dragEnded))
        .on("mouseover", showTooltip)
        .on("mouseout", hideTooltip);

    // Different shapes for special nodes
    nodeEnter.each(function(d) {
        const g = d3.select(this);
        const size = sizeScale(d.rank);

        if (d.is_bridge) {
            // Diamond for bridges
            g.append("path")
                .attr("d", d3.symbol().type(d3.symbolDiamond).size(size * size * 2))
                .attr("fill", "#f0883e")
                .attr("stroke", "#fff")
                .attr("stroke-width", 1);
        } else {
            // Circle for regular nodes
            g.append("circle")
                .attr("r", size)
                .attr("fill", d => d.is_orphan ? "#8b949e" : clusterColors(d.cluster))
                .attr("stroke", d => d.is_api ? "#a371f7" : "#fff")
                .attr("stroke-width", d => d.is_api ? 3 : 1);
        }
    });

    const nodeGs = nodeEnter.merge(nodeData);

    // Labels
    const labelData = labelGroup.selectAll("text").data(nodes, d => d.id);
    labelData.exit().remove();
    const labelEnter = labelData.enter().append("text")
        .attr("font-size", 10)
        .attr("fill", "#8b949e")
        .attr("dx", 12)
        .attr("dy", 4)
        .text(d => d.label);
    const labels = labelEnter.merge(labelData);
    labelGroup.style("display", showLabels ? null : "none");

    // Simulation
    simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(edges).id(d => d.id).distance(+linkDistSlider.value))
        .force("charge", d3.forceManyBody().strength(+chargeSlider.value))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide().radius(d => sizeScale(d.rank) + 5));

    simulation.on("tick", () => {
        links
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        nodeGs.attr("transform", d => `translate(${d.x},${d.y})`);
        labels.attr("x", d => d.x).attr("y", d => d.y);
    });
}

// Drag handlers
function dragStarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
}

function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
}

function dragEnded(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}

// Tooltip
function showTooltip(event, d) {
    const badges = [];
    if (d.is_bridge) badges.push('<span class="badge bridge">bridge</span>');
    if (d.is_api) badges.push('<span class="badge api">api</span>');
    if (d.badges.includes("recent")) badges.push('<span class="badge recent">recent</span>');
    if (d.badges.includes("high-churn")) badges.push('<span class="badge churn">high-churn</span>');
    if (d.badges.includes("emergent")) badges.push('<span class="badge emergent">emergent</span>');

    tooltip.innerHTML = `
        <div class="file">${d.file}</div>
        <div class="symbol">${d.symbol}</div>
        ${badges.length ? '<div style="margin-top: 6px;">' + badges.join("") + '</div>' : ''}
        <div class="stats">
            <div class="row"><span class="label">Rank</span><span class="value">${d.rank.toFixed(4)}</span></div>
            <div class="row"><span class="label">In-degree</span><span class="value">${d.in_degree}</span></div>
            <div class="row"><span class="label">Out-degree</span><span class="value">${d.out_degree}</span></div>
            <div class="row"><span class="label">Cluster</span><span class="value">${d.cluster}</span></div>
        </div>
    `;
    tooltip.style.left = (event.pageX + 15) + "px";
    tooltip.style.top = (event.pageY - 10) + "px";
    tooltip.classList.add("show");
}

function hideTooltip() {
    tooltip.classList.remove("show");
}

// Update stats panel
function updateStats() {
    if (!graphData) return;
    const m = graphData.metadata;

    const confidenceClass = m.confidence === 'high' ? 'high' : m.confidence === 'medium' ? 'medium' : 'low';

    document.getElementById("stats").innerHTML = `
        <div class="row"><span class="label">Nodes</span><span class="value">${m.num_nodes}</span></div>
        <div class="row"><span class="label">Edges</span><span class="value">${m.num_edges}</span></div>
        <div class="row"><span class="label">Density</span><span class="value">${(m.density * 100).toFixed(2)}%</span></div>
        <div class="row"><span class="label">Gini</span><span class="value">${m.gini.toFixed(2)}</span></div>
        <div class="row"><span class="label">Confidence</span><span class="value ${confidenceClass}">${m.confidence}</span></div>
        <div class="row"><span class="label">Orphans</span><span class="value">${m.orphan_count}</span></div>
        <div class="row"><span class="label">Intent</span><span class="value">${m.intent}</span></div>
        ${m.cliff_percentile ? `<div class="row"><span class="label">Cliff @</span><span class="value">p${m.cliff_percentile}</span></div>` : ''}
    `;

    // Hubs
    const hubsHtml = m.hub_symbols.slice(0, 8).map(([name, deg]) =>
        `<div style="display: flex; justify-content: space-between; padding: 2px 0;">
            <span style="color: #c9d1d9;">${name}</span>
            <span style="color: #58a6ff;">${deg}↓</span>
        </div>`
    ).join("");
    document.getElementById("hubs").innerHTML = hubsHtml;
}

// Toast notification
function toast(msg, isError = false) {
    const el = document.getElementById("toast");
    el.textContent = msg;
    el.style.borderColor = isError ? "#f85149" : "#30363d";
    el.classList.add("show");
    setTimeout(() => el.classList.remove("show"), 2500);
}

// Resize handler
window.addEventListener("resize", () => {
    width = container.clientWidth;
    height = container.clientHeight;
    svg.attr("width", width).attr("height", height);
    if (simulation) {
        simulation.force("center", d3.forceCenter(width / 2, height / 2));
        simulation.alpha(0.3).restart();
    }
});
</script>
</body>
</html>'''
