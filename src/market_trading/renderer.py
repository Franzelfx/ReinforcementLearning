"""Interactive Flask/pyecharts chart server for market-trading evaluation episodes.

Fixes two issues with gym-trading-env's built-in Renderer:
- Port 5000 is blocked by macOS AirPlay Receiver — we auto-select a free port.
- The HTML template hardcodes http://127.0.0.1:5000 — replaced with relative URLs.
"""

from __future__ import annotations

import glob
import socket
import threading
import time
import webbrowser
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template_string

from gym_trading_env.renderer import charts as _charts


# ---------------------------------------------------------------------------
# HTML template — identical to gym_trading_env's index.html except the two
# AJAX URLs are relative (no hardcoded host:port).
# ---------------------------------------------------------------------------

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MarketTrading Viewer</title>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.4/jquery.min.js"></script>
    <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@100;200;300;400;500;600;700;800;900&display=swap" rel="stylesheet">
</head>
<style>
body { font-family: 'Poppins', sans-serif; font-size:14px; color:#333; }
#main { display:flex; justify-content:center; text-align:center; }
#info { position:-webkit-sticky; position:sticky; top:30px; max-width:300px; }
select { padding:10px; border:1px solid #ededed; border-radius:5%; max-width:100%; white-space:nowrap; text-overflow:ellipsis; }
#metrics { display:flex; flex-wrap:wrap; justify-content:center; gap:15px; }
.metric { padding:10px; border:1px solid #ededed; max-width:110px; border-radius:5%; }
#author { color:lightgrey; margin-top:20px; }
</style>
<body>
<div id="main">
    <div>
        <div id="chart" style="width:950px; height:900px;"></div>
    </div>
    <div>
        <div id="info">
            <h2>Metrics</h2>
            <div id="metrics"></div>
            <h2 style="margin-top:40px;">Parameters</h2>
            <p>Select episode</p>
            <select name="select_dataset" id="select_dataset">
                {% for name in render_names|sort %}
                    <option value="{{ name }}">{{ name }}</option>
                {% endfor %}
            </select>
            <br>
            <button id="btn_refresh" style="margin-top:10px; padding:6px 14px; border:1px solid #ededed; border-radius:5%; cursor:pointer; background:#f5f5f5;">&#8635; Refresh episodes</button>
            <div id="author">MarketTrading RL Viewer</div>
        </div>
    </div>
</div>
<script>
    var chart = echarts.init(document.getElementById('chart'), 'white', {renderer: 'canvas'});

    $(document).ready(function () {
        fetchDataChart($('#select_dataset option:selected').val());
    });
    $('#select_dataset').on('change', function () {
        fetchDataChart(this.value);
    });
    $('#btn_refresh').on('click', function () {
        $.ajax({
            type: "GET",
            url: "/episodes",
            dataType: 'json',
            success: function (names) {
                var sel = $('#select_dataset');
                var current = sel.val();
                sel.empty();
                $.each(names, function (_, name) {
                    sel.append($('<option>', {value: name, text: name}));
                });
                if (names.indexOf(current) >= 0) {
                    sel.val(current);
                } else if (names.length > 0) {
                    sel.val(names[names.length - 1]);
                    fetchDataChart(names[names.length - 1]);
                }
            }
        });
    });
    function fetchDataChart(name) {
        $.ajax({
            type: "GET",
            url: "/update_data/" + name,
            dataType: 'json',
            success: function (results) {
                chart.setOption(results);
                $.ajax({
                    type: "GET",
                    url: "/metrics",
                    dataType: 'json',
                    success: function (metrics) {
                        var html = $("#metrics");
                        html.empty();
                        for (var m of metrics) {
                            html.append(
                                $("<div>").addClass('metric')
                                    .append($("<p style='font-weight:700'>").text(m.name))
                                    .append($("<p>").text(m.value))
                            );
                        }
                    }
                });
            }
        });
    }
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Metrics computed over the episode DataFrame
# ---------------------------------------------------------------------------

_METRICS: list[dict] = [
    {
        "name": "Market Return",
        "fn":   lambda df: f"{(df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100:.2f}%",
    },
    {
        "name": "Portfolio Return",
        "fn":   lambda df: f"{(df['portfolio_valuation'].iloc[-1] / df['portfolio_valuation'].iloc[0] - 1) * 100:.2f}%",
    },
    {
        "name": "# Trades",
        "fn":   lambda df: str(int((df["position"].diff() != 0).sum())),
    },
    {
        "name": "Steps",
        "fn":   lambda df: str(len(df)),
    },
]


# ---------------------------------------------------------------------------
# Free-port detection
# ---------------------------------------------------------------------------

def _free_port(start: int = 5001) -> int:
    """Returns the first available TCP port >= start."""
    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port found in range [5001, 5200]")


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------

def _build_app(logs_dir: str) -> Flask:
    app  = Flask(__name__)
    _state: dict = {"df": None}

    @app.route("/")
    def index():
        names = sorted(Path(p).name for p in glob.glob(f"{logs_dir}/*.pkl"))
        return render_template_string(_TEMPLATE, render_names=names)

    @app.route("/episodes")
    def episodes():
        names = sorted(Path(p).name for p in glob.glob(f"{logs_dir}/*.pkl"))
        return jsonify(names)

    @app.route("/update_data/<name>")
    def update_data(name: str):
        _state["df"] = pd.read_pickle(f"{logs_dir}/{name}")
        return _charts(_state["df"], []).dump_options_with_quotes()

    @app.route("/metrics")
    def get_metrics():
        df = _state["df"]
        if df is None:
            return jsonify([])
        return jsonify([
            {"name": m["name"], "value": m["fn"](df)}
            for m in _METRICS
        ])

    return app


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def serve(logs_dir: str) -> None:
    """Finds a free port, starts Flask, and opens the browser automatically."""

    logs_dir = str(Path(logs_dir).resolve())
    logs     = glob.glob(f"{logs_dir}/*.pkl")
    if not logs:
        print(
            f"No render logs found in '{logs_dir}'.\n"
            "Run evaluation first:\n"
            "  python -m market_trading.train --mode evaluate"
        )
        return

    port = _free_port()
    url  = f"http://127.0.0.1:{port}"
    print(
        f"Found {len(logs)} render log(s).\n"
        f"Starting chart server at {url}  (Ctrl+C to stop)\n"
    )

    app = _build_app(logs_dir)

    def _open():
        time.sleep(1.2)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()
    app.run(host="127.0.0.1", port=port)
