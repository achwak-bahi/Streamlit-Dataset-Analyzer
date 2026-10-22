import math
from pathlib import Path
import altair as alt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import streamlit as st


# ==============================================================================
# 1. Clean SVG Icons (Zero Emojis, Pure Modern UI Components)
# ==============================================================================
def svg_icon(name, size=18, color="currentColor"):
    icons = {
        "sparkles": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/><path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/></svg>""",
        "database": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/></svg>""",
        "activity": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>""",
        "gauge": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/></svg>""",
        "layers": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/></svg>""",
        "table": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/><path d="M9 3v18"/><path d="M15 3v18"/></svg>""",
        "shield_check": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/></svg>""",
        "alert_triangle": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>""",
        "correlation": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>""",
        "chart_wave": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>""",
        "check": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>""",
        "file": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/></svg>""",
        "download": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>""",
        "scale": f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/></svg>""",
    }
    return icons.get(name, "")


# ==============================================================================
# 2. Strict Light Mode & Modern Blue Custom CSS
# ==============================================================================
def _inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        /* Force Pure Light Canvas - Overrides any Dark Mode extensions */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .stApp, section.main {
            background-color: #ffffff !important;
            color: #0f172a !important;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        }

        /* Top Brand Navigation */
        .brand-nav {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 0 18px 0;
            border-bottom: 1px solid #e2e8f0;
            margin-bottom: 24px;
        }
        .brand-logo-group {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .brand-icon {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 50%, #06b6d4 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.28);
        }
        .brand-title {
            font-weight: 800;
            font-size: 1.25rem;
            color: #0f172a !important;
            letter-spacing: -0.02em;
        }
        .brand-tag {
            font-size: 0.7rem;
            font-weight: 700;
            color: #1d4ed8 !important;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            padding: 2px 8px;
            border-radius: 9999px;
            margin-left: 6px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .nav-status-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 600;
            color: #334155 !important;
        }
        .pulse-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #10b981;
            box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2);
        }

        /* Hero Typography */
        .hero-section {
            padding: 6px 0 24px 0;
            max-width: 900px;
        }
        .hero-badge-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #1d4ed8 !important;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 5px 14px;
            border-radius: 9999px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 12px;
        }
        .hero-title {
            font-family: 'Newsreader', Georgia, serif;
            font-size: 3.1rem;
            font-weight: 500;
            line-height: 1.12;
            letter-spacing: -0.03em;
            color: #0f172a !important;
            margin: 0 0 12px 0;
        }
        .gradient-blue-text {
            background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 45%, #0ea5e9 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 600;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            line-height: 1.6;
            color: #64748b !important;
            font-weight: 400;
            margin-bottom: 8px;
        }

        /* Top Bento Grid Cards */
        .bento-card {
            background: #ffffff !important;
            border-radius: 20px;
            border: 1px solid #dbeafe;
            padding: 22px;
            box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04), 0 2px 6px -1px rgba(15, 23, 42, 0.02);
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative;
            overflow: hidden;
            height: 100%;
        }
        .bento-card:hover {
            box-shadow: 0 16px 32px -8px rgba(37, 99, 235, 0.12);
            border-color: #93c5fd;
            transform: translateY(-2px);
        }
        .bento-gradient-top {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #1d4ed8 0%, #3b82f6 50%, #06b6d4 100%);
        }
        .card-tag {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.72rem;
            font-weight: 700;
            color: #1d4ed8 !important;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            padding: 3px 10px;
            border-radius: 9999px;
            margin-bottom: 12px;
        }
        .card-heading {
            font-family: 'Newsreader', Georgia, serif;
            font-size: 1.45rem;
            font-weight: 600;
            color: #0f172a !important;
            letter-spacing: -0.02em;
            margin: 0 0 6px 0;
        }
        .card-desc {
            font-size: 0.85rem;
            color: #64748b !important;
            line-height: 1.5;
            margin-bottom: 14px;
        }

        /* Metric Tiles */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 10px;
        }
        .metric-tile {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 12px 14px;
            transition: all 0.2s ease;
        }
        .metric-tile:hover {
            background: #f0f7ff;
            border-color: #bfdbfe;
        }
        .metric-tile-lbl {
            font-size: 0.7rem;
            font-weight: 700;
            color: #64748b !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }
        .metric-tile-val {
            font-size: 1.55rem;
            font-weight: 800;
            color: #0f172a !important;
            line-height: 1.1;
        }

        /* Speedometer Gauge Styling */
        .gauge-wrapper {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2px 0 0 0;
        }
        .gauge-status-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.78rem;
            font-weight: 700;
            padding: 4px 12px;
            border-radius: 9999px;
            margin-top: -6px;
        }
        .gauge-status-good {
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
            color: #065f46 !important;
        }
        .gauge-status-fair {
            background: #fffbeb;
            border: 1px solid #fde68a;
            color: #92400e !important;
        }
        .gauge-status-poor {
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #991b1b !important;
        }

        /* Section Segments with Collapsible Headers */
        .segment-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 28px 0 10px 0;
        }
        .segment-badge-icon {
            width: 32px;
            height: 32px;
            border-radius: 9px;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #1d4ed8;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .segment-title {
            font-family: 'Newsreader', Georgia, serif;
            font-size: 1.55rem;
            font-weight: 600;
            color: #0f172a !important;
            letter-spacing: -0.02em;
            margin: 0;
        }
        .segment-desc {
            font-size: 0.85rem;
            color: #64748b !important;
            margin: 0 0 12px 44px;
        }

        /* Modern Expander Card (White & Blue) */
        div[data-testid="stExpander"] {
            background: #ffffff !important;
            border: 1px solid #dbeafe !important;
            border-radius: 18px !important;
            box-shadow: 0 4px 16px -2px rgba(37, 99, 235, 0.05), 0 2px 4px -1px rgba(15, 23, 42, 0.02) !important;
            overflow: hidden !important;
            margin-bottom: 24px !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stExpander"]:hover {
            border-color: #93c5fd !important;
            box-shadow: 0 8px 24px -4px rgba(37, 99, 235, 0.1) !important;
        }
        div[data-testid="stExpander"] > details > summary {
            background-color: #f8fbff !important;
            color: #0f172a !important;
            font-weight: 700 !important;
            font-size: 1.02rem !important;
            padding: 14px 20px !important;
            border-radius: 18px 18px 0 0 !important;
            border-bottom: 1px solid #e0effe !important;
        }
        div[data-testid="stExpander"] > details > summary:hover {
            color: #1d4ed8 !important;
            background-color: #f0f7ff !important;
        }
        div[data-testid="stExpander"] > details[open] > summary {
            border-bottom: 1px solid #dbeafe !important;
        }
        div[data-testid="stExpander"] > details > div[data-testid="stExpanderDetails"] {
            background-color: #ffffff !important;
            color: #0f172a !important;
            padding: 22px !important;
        }
        div[data-testid="stExpander"] svg {
            color: #1d4ed8 !important;
        }

        /* Two-Blue Alternating Row Table */
        .table-scroll-container {
            width: 100%;
            overflow-x: auto;
            border-radius: 14px;
            border: 1px solid #bfdbfe;
            background: #ffffff !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.04);
            margin: 8px 0 16px 0;
        }
        .two-blue-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.86rem;
            color: #0f172a !important;
            text-align: left;
            font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
        }
        .two-blue-table th {
            background-color: #dbeafe !important;
            color: #1e3a8a !important;
            font-weight: 700;
            padding: 12px 16px;
            border-bottom: 2px solid #bfdbfe;
            white-space: nowrap;
        }
        .two-blue-table tr.row-even {
            background-color: #f7faff !important;
        }
        .two-blue-table tr.row-odd {
            background-color: #ebf4ff !important;
        }
        .two-blue-table tr:hover {
            background-color: #dbeafe !important;
        }
        .two-blue-table td {
            padding: 10px 16px;
            border-bottom: 1px solid #e0effe;
            color: #1e293b !important;
            white-space: nowrap;
        }

        /* Top Correlated Pair Card */
        .pair-card {
            background: #ffffff !important;
            border: 1px solid #dbeafe;
            border-radius: 14px;
            padding: 14px;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 6px rgba(0,0,0,0.02);
            transition: all 0.15s ease;
        }
        .pair-card:hover {
            border-color: #93c5fd;
            background: #f8fbff !important;
        }
        .pair-badge {
            background: linear-gradient(135deg, #1d4ed8, #0ea5e9);
            color: white !important;
            font-weight: 700;
            font-size: 0.8rem;
            padding: 4px 10px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25);
        }

        /* Insight Chips */
        .insight-chip {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            background: #f8fbff;
            border: 1px solid #dbeafe;
            border-radius: 14px;
            padding: 12px 16px;
            margin-bottom: 10px;
            font-size: 0.86rem;
            color: #1e293b !important;
            line-height: 1.45;
        }
        .insight-chip-good {
            background: #f0fdf4 !important;
            border-color: #bbf7d0 !important;
            color: #166534 !important;
        }
        .insight-chip-warning {
            background: #fffbeb !important;
            border-color: #fde68a !important;
            color: #92400e !important;
        }

        /* Streamlit Buttons & Inputs */
        div.stButton > button {
            background: linear-gradient(135deg, #2563eb 0%, #0284c7 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 10px 22px !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
            transition: all 0.2s ease !important;
        }
        div.stButton > button:hover {
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
            transform: translateY(-1px) !important;
        }
        div[data-testid="stFileUploader"] {
            background: #f8fafc !important;
            border: 1px dashed #93c5fd !important;
            border-radius: 16px !important;
            padding: 8px !important;
        }
        div[data-testid="stSelectbox"] > div {
            border-radius: 12px !important;
            background: #ffffff !important;
            border-color: #cbd5e1 !important;
        }

        /* Overall Section & Subsets Card */
        .overall-section-card {
            background: #ffffff !important;
            border: 1px solid #dbeafe !important;
            border-radius: 18px !important;
            padding: 20px 24px 18px 24px !important;
            position: relative !important;
            margin-top: 20px !important;
            margin-bottom: 24px !important;
            box-shadow: 0 4px 16px -2px rgba(37, 99, 235, 0.05), 0 2px 4px -1px rgba(15, 23, 42, 0.02) !important;
        }
        .overall-top-gradient {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #1d4ed8 0%, #3b82f6 50%, #06b6d4 100%);
            border-radius: 18px 18px 0 0;
        }
        .overall-header-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 12px;
        }
        .overall-title-group {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .overall-icon-badge {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #1d4ed8;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
        .overall-main-title {
            font-family: 'Newsreader', Georgia, serif;
            font-size: 1.4rem;
            font-weight: 600;
            color: #0f172a !important;
            letter-spacing: -0.01em;
            margin: 0;
        }
        .overall-desc-row {
            font-size: 0.85rem;
            color: #64748b !important;
            margin: 2px 0 0 0;
            line-height: 1.45;
        }
        .overall-meta-chips {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }
        .overall-chip-count {
            font-size: 0.72rem;
            font-weight: 700;
            color: #1d4ed8 !important;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            padding: 4px 12px;
            border-radius: 9999px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .overall-chip-joinable {
            font-size: 0.72rem;
            font-weight: 700;
            color: #15803d !important;
            background: #dcfce7;
            border: 1px solid #bbf7d0;
            padding: 4px 12px;
            border-radius: 9999px;
            letter-spacing: 0.04em;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .overall-chip-active {
            font-size: 0.72rem;
            font-weight: 700;
            color: #1d4ed8 !important;
            background: #f0f7ff;
            border: 1px solid #93c5fd;
            padding: 4px 12px;
            border-radius: 9999px;
            letter-spacing: 0.04em;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .overall-banner {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 12px;
            padding: 10px 16px;
            margin-bottom: 14px;
            font-size: 0.85rem;
            color: #166534 !important;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Modern Blue & White Pills */
        div[data-testid="stPills"], div[data-testid="stButtonGroup"] {
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 8px !important;
            margin-bottom: 8px !important;
        }
        div[data-testid="stPills"] button, div[data-testid="stButtonGroup"] button {
            background-color: #f8fbff !important;
            border: 1px solid #bfdbfe !important;
            color: #1e3a8a !important;
            border-radius: 9999px !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            padding: 6px 14px !important;
            transition: all 0.15s ease !important;
        }
        div[data-testid="stPills"] button:hover, div[data-testid="stButtonGroup"] button:hover {
            border-color: #3b82f6 !important;
            background-color: #eff6ff !important;
            color: #1d4ed8 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 2px 6px rgba(37, 99, 235, 0.1) !important;
        }
        div[data-testid="stPills"] button[aria-selected="true"],
        div[data-testid="stButtonGroup"] button[aria-selected="true"],
        div[data-testid="stPills"] button[aria-checked="true"],
        div[data-testid="stButtonGroup"] button[aria-checked="true"] {
            background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 50%, #0284c7 100%) !important;
            color: #ffffff !important;
            border-color: #1d4ed8 !important;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.28) !important;
            font-weight: 700 !important;
        }
        div[data-testid="stPills"] button[aria-selected="true"] *,
        div[data-testid="stButtonGroup"] button[aria-selected="true"] *,
        div[data-testid="stPills"] button[aria-checked="true"] *,
        div[data-testid="stButtonGroup"] button[aria-checked="true"] * {
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 3. Two-Blue Alternating Row Table Helper
# ==============================================================================
def _clean_table_val(val):
    if val is None or pd.isna(val):
        return "-"
    if isinstance(val, float):
        if abs(val) < 1e-4 and val != 0:
            return f"{val:.2e}"
        return f"{val:.3f}".rstrip("0").rstrip(".") if abs(val) < 1000 else f"{val:,.2f}"
    if isinstance(val, int):
        return f"{val:,}"
    return str(val)


def _render_two_blue_table(df, max_rows=None):
    """Renders any DataFrame in a modern, light two-blue alternating row style."""
    if df is None or df.empty:
        st.info("No records to display.")
        return

    view_df = df.head(max_rows) if (max_rows and len(df) > max_rows) else df

    headers = "".join(f"<th>{col}</th>" for col in view_df.columns)

    rows = []
    for idx, (_, row) in enumerate(view_df.iterrows()):
        row_cls = "row-even" if idx % 2 == 0 else "row-odd"
        cells = "".join(f"<td>{_clean_table_val(val)}</td>" for val in row.values)
        rows.append(f"<tr class='{row_cls}'>{cells}</tr>")

    table_html = f"""
    <div class="table-scroll-container">
        <table class="two-blue-table">
            <thead><tr>{headers}</tr></thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


# ==============================================================================
# 4. Quality Speedometer SVG Component
# ==============================================================================
def _calculate_quality_score(result):
    score = 100.0
    meta = result.get("meta", {})
    n_rows = max(1, meta.get("n_rows", 1))
    n_cols = max(1, meta.get("n_columns", 1))

    # Missing values penalty
    missing = result.get("missing_data", {})
    total_missing = missing.get("total_missing_cells", 0)
    missing_pct = (total_missing / (n_rows * n_cols)) * 100.0
    score -= min(35.0, missing_pct * 1.8)

    # Duplicate penalty
    noisy = result.get("noisy_data", {})
    duplicates = noisy.get("duplicate_rows", 0)
    dup_pct = (duplicates / n_rows) * 100.0
    score -= min(20.0, dup_pct * 2.0)

    # Constant columns penalty
    constant_cols = noisy.get("constant_columns", [])
    score -= len(constant_cols) * 5.0

    # Outliers penalty
    outliers = result.get("outliers", {}).get("per_column", {})
    if outliers:
        total_outlier_pct = sum(
            col_data.get("iqr", {}).get("pct", 0.0) for col_data in outliers.values()
        )
        avg_outlier_pct = total_outlier_pct / len(outliers)
        score -= min(15.0, avg_outlier_pct * 0.8)

    return max(0.0, min(100.0, round(score, 1)))


def _render_speedometer_svg(score):
    radius = 72
    center_x = 100
    center_y = 90
    circumference = math.pi * radius
    dashoffset = circumference * (1.0 - (score / 100.0))

    angle = 180.0 - (score / 100.0) * 180.0
    rad = math.radians(angle)
    needle_len = 54
    nx = center_x + needle_len * math.cos(rad)
    ny = center_y - needle_len * math.sin(rad)

    svg = (
        f'<div class="gauge-wrapper">'
        f'<svg viewBox="0 0 200 115" width="100%" height="135" style="overflow: visible;">'
        f'<defs>'
        f'<linearGradient id="gaugeBlueGrad" x1="0%" y1="0%" x2="100%" y2="0%">'
        f'<stop offset="0%" stop-color="#1d4ed8"/>'
        f'<stop offset="45%" stop-color="#3b82f6"/>'
        f'<stop offset="100%" stop-color="#06b6d4"/>'
        f'</linearGradient>'
        f'<filter id="gaugeShadow" x="-20%" y="-20%" width="140%" height="140%">'
        f'<feDropShadow dx="0" dy="4" stdDeviation="4" flood-color="#3b82f6" flood-opacity="0.32"/>'
        f'</filter>'
        f'</defs>'
        f'<path d="M {center_x - radius} {center_y} A {radius} {radius} 0 0 1 {center_x + radius} {center_y}" fill="none" stroke="#f1f5f9" stroke-width="13" stroke-linecap="round"/>'
        f'<path d="M {center_x - radius} {center_y} A {radius} {radius} 0 0 1 {center_x + radius} {center_y}" fill="none" stroke="url(#gaugeBlueGrad)" stroke-width="13" stroke-linecap="round" stroke-dasharray="{circumference:.1f}" stroke-dashoffset="{dashoffset:.1f}" filter="url(#gaugeShadow)"/>'
        f'<line x1="{center_x}" y1="{center_y}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#0f172a" stroke-width="3.5" stroke-linecap="round"/>'
        f'<circle cx="{center_x}" cy="{center_y}" r="6" fill="#0f172a"/>'
        f'<circle cx="{center_x}" cy="{center_y}" r="2" fill="#38bdf8"/>'
        f'<text x="{center_x}" y="74" font-family="\'Plus Jakarta Sans\', system-ui, sans-serif" font-weight="800" font-size="24" fill="#0f172a" text-anchor="middle">{score:.1f}%</text>'
        f'<text x="{center_x}" y="108" font-family="\'Plus Jakarta Sans\', system-ui, sans-serif" font-weight="700" font-size="9" fill="#94a3b8" text-anchor="middle" letter-spacing="1.2">QUALITY INDEX</text>'
        f'<text x="{center_x - radius + 2}" y="{center_y + 16}" font-size="9" font-weight="600" fill="#94a3b8" text-anchor="middle">0%</text>'
        f'<text x="{center_x + radius - 2}" y="{center_y + 16}" font-size="9" font-weight="600" fill="#94a3b8" text-anchor="middle">100%</text>'
        f'</svg>'
        f'</div>'
    )
    return svg


# ==============================================================================
# 5. Brand Header & Navbar
# ==============================================================================
def _render_brand_navbar(dataset_name):
    sparkle_svg = svg_icon("sparkles", 18, "#ffffff")
    file_svg = svg_icon("file", 14, "#1d4ed8")

    st.markdown(
        f"""
        <div class="brand-nav">
            <div class="brand-logo-group">
                <div class="brand-icon">{sparkle_svg}</div>
                <div class="brand-title">DataPulse <span class="brand-tag">v2.0 PRO</span></div>
            </div>
            <div class="nav-status-badge">
                <span class="pulse-dot"></span>
                <span>Active Dataset: <strong>{dataset_name}</strong></span>
            </div>
        </div>
        <div class="hero-section">
            <div class="hero-badge-pill">{file_svg} Automated EDA Engine</div>
            <h1 class="hero-title">
                Everything Your Data Needs.<br>
                <span class="gradient-blue-text">Nothing It Doesn't.</span>
            </h1>
            <p class="hero-subtitle">
                Comprehensive automated exploratory analysis, quality health score, anomaly detection,
                and correlation discovery with zero manual setup.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 6. Top Bento Grid Cards
# ==============================================================================
def _render_bento_cards(result, controller, group_names, current_group, active_label):
    col_source, col_gauge, col_stats = st.columns([1.1, 1.0, 1.1])

    meta = result.get("meta", {})
    n_rows = meta.get("n_rows", 0)
    n_cols = meta.get("n_columns", 0)
    overview = result.get("data_description", {}).get("overview", {})
    num_cols = len(overview.get("numeric_columns", []))

    sparkle_svg = svg_icon("sparkles", 13, "#1d4ed8")
    gauge_svg = svg_icon("gauge", 13, "#1d4ed8")
    layers_svg = svg_icon("layers", 13, "#1d4ed8")
    file_svg = svg_icon("file", 13, "#1d4ed8")
    activity_svg = svg_icon("activity", 13, "#1d4ed8")

    # 1. Source & Ingestion Card
    with col_source:
        st.markdown(
            f"""
            <div class="bento-card">
                <div class="bento-gradient-top"></div>
                <div class="card-tag">{sparkle_svg} Smart Ingestion</div>
                <div class="card-heading">Dataset Source</div>
                <div class="card-desc">Select a preloaded benchmark or drop your own dataset.</div>
            """,
            unsafe_allow_html=True,
        )

        selected_group = st.selectbox(
            "Benchmark dataset",
            group_names,
            index=group_names.index(current_group) if current_group in group_names else 0,
            label_visibility="collapsed",
        )

        uploaded = st.file_uploader(
            "Upload dataset",
            type=["csv", "xlsx", "xls", "json"],
            help="Supported: CSV, Excel (.xlsx, .xls), JSON",
            label_visibility="collapsed",
        )

        st.markdown(
            f"""
                <div style="margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap;">
                    <span class="nav-status-badge">{file_svg} {active_label}</span>
                    <span class="nav-status-badge">{activity_svg} {n_rows:,} rows</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Quality Speedometer Card
    quality_score = _calculate_quality_score(result)
    with col_gauge:
        check_svg = svg_icon("shield_check", 14, "#065f46" if quality_score >= 90 else "#92400e")
        if quality_score >= 90:
            status_text = "Pristine Health"
            status_cls = "gauge-status-good"
        elif quality_score >= 70:
            status_text = "Good Quality"
            status_cls = "gauge-status-fair"
        else:
            status_text = "Attention Needed"
            status_cls = "gauge-status-poor"

        gauge_html = _render_speedometer_svg(quality_score)
        st.markdown(
            f"""
            <div class="bento-card" style="text-align: center;">
                <div class="bento-gradient-top"></div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="card-tag">{gauge_svg} Performance Engine</div>
                    <div class="gauge-status-chip {status_cls}">{check_svg} {status_text}</div>
                </div>
                <div class="card-heading" style="text-align: left; margin-bottom: 2px;">Data Quality Index</div>
                {gauge_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Architecture Card
    with col_stats:
        missing_count = result.get("missing_data", {}).get("total_missing_cells", 0)

        st.markdown(
            f"""
            <div class="bento-card">
                <div class="bento-gradient-top"></div>
                <div class="card-tag">{layers_svg} Architecture</div>
                <div class="card-heading">Dataset Dimensions</div>
                <div class="card-desc">Structural composition and health breakdown.</div>
                <div class="metric-grid">
                    <div class="metric-tile">
                        <div class="metric-tile-lbl">Total Rows</div>
                        <div class="metric-tile-val">{n_rows:,}</div>
                    </div>
                    <div class="metric-tile">
                        <div class="metric-tile-lbl">Features</div>
                        <div class="metric-tile-val">{n_cols}</div>
                    </div>
                    <div class="metric-tile">
                        <div class="metric-tile-lbl">Numerical</div>
                        <div class="metric-tile-val">{num_cols}</div>
                    </div>
                    <div class="metric-tile">
                        <div class="metric-tile-lbl">Missing Cells</div>
                        <div class="metric-tile-val" style="color: {'#dc2626' if missing_count > 0 else '#10b981'};">{missing_count:,}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_group, uploaded


def _render_overall_and_subsets_section(current_group, group_info, controller, selected_subset):
    """Renders the Overall section first, with a small summary table of all subsets and view switcher."""
    files = group_info.get("files", [])
    labels = group_info.get("labels", {})
    joinable = group_info.get("joinable", False)
    group_desc = group_info.get("description", "")
    n_files = len(files)

    layers_svg = svg_icon("layers", 18, "#1d4ed8")
    table_svg = svg_icon("table", 13, "#047857")
    check_svg = svg_icon("check", 13, "#15803d")
    sparkle_svg = svg_icon("sparkles", 13, "#1d4ed8")

    active_is_overall = (selected_subset == "__overall__")
    if active_is_overall:
        active_chip_label = "Overall View (Active)"
    else:
        active_chip_label = f"Subset: {labels.get(selected_subset, selected_subset)}"

    joinable_chip = (
        f'<span class="overall-chip-joinable">{check_svg} Joinable (Concatenated View Available)</span>'
        if joinable else
        f'<span class="overall-chip-count">{n_files} Independent Subsets</span>'
    )

    st.markdown(
        f"""
        <div class="overall-section-card">
            <div class="overall-top-gradient"></div>
            <div class="overall-header-bar">
                <div class="overall-title-group">
                    <div class="overall-icon-badge">{layers_svg}</div>
                    <div>
                        <div class="overall-main-title">{current_group} · Architecture & Subsets</div>
                        <div class="overall-desc-row">{group_desc}</div>
                    </div>
                </div>
                <div class="overall-meta-chips">
                    <span class="overall-chip-count">{n_files} Subsets</span>
                    {joinable_chip}
                    <span class="overall-chip-active">{table_svg} {active_chip_label}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Summary Table of All Subsets (row counts, basic stats)
    summary_df = controller.get_subsets_summary(current_group)
    if summary_df is not None and not summary_df.empty:
        with st.expander(f"All Subsets Summary Table ({n_files} Subsets)", expanded=True):
            if joinable:
                st.markdown(
                    f"""
                    <div class="overall-banner">
                        {sparkle_svg} <strong>Overall View (Joined):</strong>
                        All {n_files} subsets have been concatenated into the master DataFrame with an added identifier column.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            _render_two_blue_table(summary_df)

    # 2. Scope Switcher (Overall view is default option, with individual subsets available)
    scope_options = ["__overall__"] + files

    def _format_scope(opt):
        if opt == "__overall__":
            return f"Overall View (All {n_files} Subsets Concatenated)" if joinable else f"Overall View (All {n_files} Subsets Aggregated)"
        sub_name = labels.get(opt, Path(opt).stem)
        return f"{sub_name} ({opt})"

    current_idx = scope_options.index(selected_subset) if selected_subset in scope_options else 0

    col_select, col_helper = st.columns([1.6, 1.0])
    with col_select:
        new_choice = st.selectbox(
            "Select View Scope / Individual Subset",
            scope_options,
            index=current_idx,
            format_func=_format_scope,
            key=f"scope_select_{current_group}",
        )
    with col_helper:
        if new_choice == "__overall__":
            if joinable:
                st.caption("Active: Concatenated Overall Dataset across all subsets.")
            else:
                st.caption("Active: Overall Multi-Subset View (Summary + Primary Subset).")
        else:
            sub_lbl = labels.get(new_choice, new_choice)
            st.caption(f"Active: Individual Subset '{sub_lbl}'.")

    return new_choice


# ==============================================================================
# 7. Helper for Segment Header
# ==============================================================================
def _render_segment_header(icon_name, title, subtitle):
    icon_html = svg_icon(icon_name, 18, "#1d4ed8")
    st.markdown(
        f"""
        <div class="segment-header">
            <div class="segment-badge-icon">{icon_html}</div>
            <h2 class="segment-title">{title}</h2>
        </div>
        <p class="segment-desc">{subtitle}</p>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 8. Segments (One Page Scroll with Accordion Drop-Down/Drop-Up)
# ==============================================================================
def _render_segment_statistics(result):
    """Segment 1: Feature Statistics & Catalog."""
    _render_segment_header(
        "table",
        "Feature Statistics & Data Catalog",
        "Distribution parameters, data types, and scale metrics across all features.",
    )

    with st.expander("Feature Statistics & Data Catalog (Click to Collapse/Expand)", expanded=True):
        dd = result.get("data_description", {})
        statistics = dd.get("statistics", {})

        if statistics:
            stats_rows = []
            for col_name, st_data in statistics.items():
                stats_rows.append(
                    {
                        "Feature": col_name,
                        "Type": st_data.get("type", "unknown").capitalize(),
                        "Count": st_data.get("count"),
                        "Unique": st_data.get("n_unique"),
                        "Mean": st_data.get("avg"),
                        "Std Dev": st_data.get("std"),
                        "Min": st_data.get("min"),
                        "Q1 (25%)": st_data.get("q1"),
                        "Median (50%)": st_data.get("median"),
                        "Q3 (75%)": st_data.get("q3"),
                        "Max": st_data.get("max"),
                        "Skewness": st_data.get("skewness"),
                    }
                )
            df_stats = pd.DataFrame(stats_rows)
            _render_two_blue_table(df_stats)

        # Covariance Matrix
        covariance = dd.get("covariance", {})
        if covariance and covariance.get("matrix"):
            st.markdown("<h4 style='font-size: 1.1rem; color: #0f172a; margin-top: 14px;'>Covariance Matrix</h4>", unsafe_allow_html=True)
            cov_labels = covariance.get("labels", [])
            cov_matrix = covariance.get("matrix", [])
            df_cov = pd.DataFrame(cov_matrix, index=cov_labels, columns=cov_labels).reset_index()
            df_cov.rename(columns={"index": "Feature"}, inplace=True)
            _render_two_blue_table(df_cov)


def _render_segment_quality(result, df=None):
    """Segment 2: Data Quality & Noise Diagnostics."""
    _render_segment_header(
        "shield_check",
        "Quality & Noise Diagnostics",
        "Automated audit of missing values, duplicate records, zero concentrations, and rolling noise.",
    )

    with st.expander("Quality & Noise Diagnostics (Click to Collapse/Expand)", expanded=True):
        col1, col2 = st.columns(2)

        missing = result.get("missing_data", {})
        total_missing = missing.get("total_missing_cells", 0)
        per_col_missing = missing.get("per_column", {})

        check_svg = svg_icon("check", 16, "#166534")
        alert_svg = svg_icon("alert_triangle", 16, "#92400e")

        with col1:
            st.markdown(
                f"""
                <div class="bento-card" style="margin-bottom: 14px;">
                    <div class="card-tag">{svg_icon('activity', 13, '#1d4ed8')} Missing Data Audit</div>
                    <div class="card-heading">{total_missing:,} Missing Cells</div>
                    <div class="card-desc">Total null or unobserved values across the entire dataset.</div>
                """,
                unsafe_allow_html=True,
            )

            missing_rows = [
                {"Feature": c, "Missing Count": d.get("missing", 0), "Percentage (%)": d.get("pct", 0.0)}
                for c, d in per_col_missing.items()
                if d.get("missing", 0) > 0
            ]
            if missing_rows:
                df_miss = pd.DataFrame(missing_rows)
                _render_two_blue_table(df_miss)
            else:
                st.markdown(
                    f"""
                    <div class="insight-chip insight-chip-good">
                        <div>{check_svg}</div>
                        <div><strong>100% Complete:</strong> No missing values detected in any feature.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        noisy = result.get("noisy_data", {})
        duplicates = noisy.get("duplicate_rows", 0)

        with col2:
            st.markdown(
                f"""
                <div class="bento-card" style="margin-bottom: 14px;">
                    <div class="card-tag">{svg_icon('layers', 13, '#1d4ed8')} Duplication Check</div>
                    <div class="card-heading">{duplicates:,} Duplicate Rows</div>
                    <div class="card-desc">Identical records identified across all columns.</div>
                """,
                unsafe_allow_html=True,
            )
            if duplicates == 0:
                st.markdown(
                    f"""
                    <div class="insight-chip insight-chip-good">
                        <div>{check_svg}</div>
                        <div><strong>Unique Records:</strong> Zero duplicate rows discovered.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="insight-chip insight-chip-warning">
                        <div>{alert_svg}</div>
                        <div><strong>Caution:</strong> {duplicates:,} identical rows found. Consider deduplicating prior to downstream modeling.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # Missing values bar plot (Seaborn) - compact column layout matching Feature Importance
        if df is not None and not df.empty:
            missing_series = df.isna().sum()
            col_chart, col_side = st.columns([1.6, 1.0])
            with col_chart:
                cols_with_miss = missing_series[missing_series > 0]
                if len(cols_with_miss) > 0 and len(missing_series) > 15:
                    plot_series = cols_with_miss.head(15)
                    chart_sub = " (Features with missing values)"
                elif len(missing_series) > 15:
                    plot_series = missing_series.head(15)
                    chart_sub = " (First 15 features)"
                else:
                    plot_series = missing_series
                    chart_sub = ""

                fig_miss, ax_miss = plt.subplots(figsize=(6.5, 3.5))
                sns.barplot(x=plot_series.index, y=plot_series.values, ax=ax_miss, color="#2563eb")
                ax_miss.set_xticks(range(len(plot_series)))
                ax_miss.set_xticklabels(plot_series.index, rotation=45, ha="right", fontsize=8)
                ax_miss.set_title(f"Missing values per column{chart_sub}", fontsize=11, fontweight="bold", pad=8)
                ax_miss.set_ylabel("Missing Count", fontsize=9)
                ax_miss.grid(axis="y", linestyle="--", alpha=0.3)
                fig_miss.tight_layout()
                st.pyplot(fig_miss)
                plt.close(fig_miss)

            with col_side:
                n_miss_cols = int((missing_series > 0).sum())
                pct_complete = 100.0 * (1.0 - (missing_series.sum() / max(1, df.size)))
                st.markdown(
                    f"""
                    <div class="bento-card">
                        <div class="bento-gradient-top"></div>
                        <div class="card-tag">{svg_icon('shield_check', 13, '#1d4ed8')} Completeness Audit</div>
                        <div class="card-heading" style="font-size: 1.2rem;">{pct_complete:.1f}% Integrity</div>
                        <div class="card-desc"><strong>{n_miss_cols}</strong> of <strong>{len(missing_series)}</strong> features contain unobserved values.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Suspicious Zeros
        suspicious_zeros = noisy.get("suspicious_zeros", [])
        if suspicious_zeros:
            st.markdown("<h4 style='font-size: 1.1rem; color: #0f172a; margin-top: 16px;'>Suspicious Zero Concentrations</h4>", unsafe_allow_html=True)
            df_zeros = pd.DataFrame(suspicious_zeros)
            _render_two_blue_table(df_zeros)

        # Rolling Noise
        rolling_noise = noisy.get("rolling_noise", {})
        if rolling_noise and rolling_noise.get("columns"):
            st.markdown("<h4 style='font-size: 1.1rem; color: #0f172a; margin-top: 16px;'>Rolling Noise Diagnostics</h4>", unsafe_allow_html=True)
            st.caption(rolling_noise.get("note", "Computed using rolling standard deviation."))
            df_noise = pd.DataFrame(rolling_noise.get("columns"))
            _render_two_blue_table(df_noise)


def _render_segment_outliers(result, df=None):
    """Segment 3: Outlier & Anomaly Analysis."""
    _render_segment_header(
        "alert_triangle",
        "Outlier & Anomaly Analysis",
        "Robust outlier detection using Interquartile Range (IQR 1.5x), Z-Score (3.0σ) thresholds, and distribution boxplots.",
    )

    with st.expander("Outlier & Anomaly Analysis (Click to Collapse/Expand)", expanded=True):
        outliers = result.get("outliers", {})
        per_col = outliers.get("per_column", {})

        if df is not None and not df.empty:
            num_cols = list(df.select_dtypes(include=np.number).columns)
            if num_cols:
                col_sel, col_info = st.columns([1.6, 1.0])
                with col_sel:
                    options = ["All Features (3x3 Grid Overview)"] + num_cols
                    selected_col = st.selectbox(
                        "Select Feature for Outlier Boxplot Inspection",
                        options=options,
                        key="outlier_feature_selector",
                    )
                with col_info:
                    st.caption("Inspect individual feature distributions or switch to a multi-feature 3x3 grid overview.")

                col_chart, col_side = st.columns([1.6, 1.0])

                with col_chart:
                    if selected_col == "All Features (3x3 Grid Overview)":
                        plot_cols = num_cols[:9]
                        fig, axes = plt.subplots(3, 3, figsize=(7.5, 4.5))
                        axes_flat = axes.flatten()
                        for ax, col in zip(axes_flat, plot_cols):
                            sns.boxplot(y=df[col], ax=ax, color="#93c5fd")
                            ax.set_title(col, fontsize=8, fontweight="bold")
                            ax.tick_params(labelsize=7)
                            ax.grid(axis="y", linestyle="--", alpha=0.3)
                        for ax in axes_flat[len(plot_cols):]:
                            ax.set_visible(False)
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        fig, ax = plt.subplots(figsize=(6.5, 3.5))
                        sns.boxplot(y=df[selected_col], ax=ax, color="#60a5fa", width=0.35)
                        ax.set_title(f"Outlier Boxplot — {selected_col}", fontsize=11, fontweight="bold", pad=8)
                        ax.tick_params(labelsize=8)
                        ax.grid(axis="y", linestyle="--", alpha=0.3)
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                with col_side:
                    if selected_col != "All Features (3x3 Grid Overview)" and selected_col in per_col:
                        cdata = per_col[selected_col]
                        iqr_info = cdata.get("iqr", {})
                        z_info = cdata.get("zscore", {})
                        cnt = iqr_info.get("count", 0)
                        pct = iqr_info.get("pct", 0.0)
                        st.markdown(
                            f"""
                            <div class="bento-card">
                                <div class="bento-gradient-top"></div>
                                <div class="card-tag">{svg_icon('alert_triangle', 13, '#1d4ed8')} Feature Outliers</div>
                                <div class="card-heading" style="font-size: 1.2rem;">{cnt:,} Outliers ({pct:.1f}%)</div>
                                <div class="card-desc">IQR Lower: <strong>{iqr_info.get('lower', 'N/A')}</strong> | Upper: <strong>{iqr_info.get('upper', 'N/A')}</strong></div>
                                <div style="margin-top: 10px; font-size: 0.8rem; color: #64748b;">Z-Score > 3.0σ count: <strong>{z_info.get('count', 0):,}</strong> ({z_info.get('pct', 0.0):.1f}%)</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        total_outlier_cols = sum(1 for c, d in per_col.items() if d.get("iqr", {}).get("count", 0) > 0)
                        st.markdown(
                            f"""
                            <div class="bento-card">
                                <div class="bento-gradient-top"></div>
                                <div class="card-tag">{svg_icon('layers', 13, '#1d4ed8')} Overview Metrics</div>
                                <div class="card-heading" style="font-size: 1.2rem;">{total_outlier_cols} Affected Columns</div>
                                <div class="card-desc">Numerical features with values outside the 1.5× IQR boundary thresholds.</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        outliers = result.get("outliers", {})
        per_col = outliers.get("per_column", {})

        if not per_col:
            st.info("No numerical features available for outlier detection.")
            return

        outlier_list = []
        for col, data in per_col.items():
            iqr = data.get("iqr", {})
            zsc = data.get("zscore", {})
            outlier_list.append(
                {
                    "Feature": col,
                    "IQR Count": iqr.get("count", 0),
                    "IQR (%)": iqr.get("pct", 0.0),
                    "Lower Bound": iqr.get("lower"),
                    "Upper Bound": iqr.get("upper"),
                    "Z-Score Count": zsc.get("count", 0),
                    "Z-Score (%)": zsc.get("pct", 0.0),
                }
            )
        df_outliers = pd.DataFrame(outlier_list)

        # Chart of outlier rates
        df_chart = df_outliers[df_outliers["IQR Count"] > 0]
        if not df_chart.empty:
            chart = (
                alt.Chart(df_chart)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#2563eb")
                .encode(
                    x=alt.X("Feature:N", sort="-y", title="Feature"),
                    y=alt.Y("IQR (%):Q", title="Outlier Rate (%)"),
                    tooltip=["Feature", "IQR Count", "IQR (%)", "Lower Bound", "Upper Bound"],
                )
                .properties(height=260)
                .configure_axis(grid=False, labelFont="Plus Jakarta Sans", titleFont="Plus Jakarta Sans")
            )
            st.altair_chart(chart, width="stretch")
        else:
            check_svg = svg_icon("check", 16, "#166534")
            st.markdown(
                f"""
                <div class="insight-chip insight-chip-good" style="margin-bottom: 14px;">
                    <div>{check_svg}</div>
                    <div><strong>Zero Extreme Outliers:</strong> All values lie within normal 1.5× IQR boundaries.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        _render_two_blue_table(df_outliers)


def _render_segment_balance(df):
    """Segment: Categorical Class Balance."""
    _render_segment_header(
        "scale",
        "Categorical Class Balance",
        "Frequency distribution and class balance inspection for categorical and discrete features.",
    )

    with st.expander("Categorical Class Balance (Click to Collapse/Expand)", expanded=True):
        if df is None or df.empty:
            st.info("No data available for class balance inspection.")
            return

        cat_cols = [c for c in df.columns if df[c].dtype == "object" or df[c].nunique() <= 30]
        if not cat_cols:
            st.info("No categorical or discrete features found for balance inspection.")
            return

        default_cols = [c for c in ["Precip Type", "Summary"] if c in df.columns]
        if not default_cols:
            default_cols = cat_cols[:2]

        selected_cols = st.multiselect(
            "Select Categorical Feature(s) to Inspect Balance",
            options=cat_cols,
            default=default_cols,
            key="balance_multiselect",
        )

        if not selected_cols:
            st.info("Please select at least one categorical feature to display class balance.")
            return

        cols_ui = st.columns(min(len(selected_cols), 2))
        for idx, col in enumerate(selected_cols):
            with cols_ui[idx % len(cols_ui)]:
                fig, ax = plt.subplots(figsize=(7, 4))
                df[col].value_counts().head(10).plot(kind="bar", ax=ax, color="#2563eb")
                ax.set_title(f"Balance — {col}", fontsize=11, fontweight="bold", pad=10)
                ax.set_ylabel("Frequency", fontsize=9)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
                ax.grid(axis="y", linestyle="--", alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)


def _render_segment_correlation(result):
    """Segment 4: Correlation Matrix & Feature Coupling."""
    _render_segment_header(
        "correlation",
        "Correlation Matrix & Feature Coupling",
        "Pairwise linear correlation coefficients (Pearson) with strongest coupled relationships.",
    )

    with st.expander("Correlation Matrix & Feature Coupling (Click to Collapse/Expand)", expanded=True):
        vis = result.get("visualizations", {})
        corr_data = vis.get("correlation", {})

        if not corr_data or not corr_data.get("matrix"):
            st.info("Correlation matrix is not available for this dataset.")
            return

        labels = corr_data.get("labels", [])
        matrix = corr_data.get("matrix", [])
        top_pairs = corr_data.get("top_pairs", [])

        col_chart, col_top = st.columns([1.6, 1.0])

        with col_chart:
            records = []
            for i in range(len(labels)):
                for j in range(len(labels)):
                    val = matrix[i][j]
                    corr_val = round(float(val), 2) if val is not None else None
                    records.append({
                        "Feature 1": labels[i],
                        "Feature 2": labels[j],
                        "Correlation": corr_val,
                    })
            df_corr = pd.DataFrame(records)

            base = alt.Chart(df_corr).encode(
                x=alt.X(
                    "Feature 1:O",
                    title=None,
                    axis=alt.Axis(labelAngle=-45, labelFont="Plus Jakarta Sans", labelFontSize=11, labelColor="#1e293b"),
                ),
                y=alt.Y(
                    "Feature 2:O",
                    title=None,
                    axis=alt.Axis(labelFont="Plus Jakarta Sans", labelFontSize=11, labelColor="#1e293b"),
                ),
            )

            heatmap = base.mark_rect().encode(
                color=alt.Color(
                    "Correlation:Q",
                    scale=alt.Scale(scheme="blues", domain=[-1, 1]),
                    title="Correlation (r)",
                ),
                tooltip=["Feature 1", "Feature 2", "Correlation"],
            )

            font_size = 11 if len(labels) <= 6 else (9 if len(labels) <= 12 else 8)

            text = base.mark_text(baseline="middle", fontSize=font_size, fontWeight=600).encode(
                text=alt.Text("Correlation:Q", format=".2f"),
                color=alt.condition(
                    alt.datum.Correlation > 0.45,
                    alt.value("#ffffff"),
                    alt.value("#0f172a"),
                ),
            )

            corr_chart = (heatmap + text).properties(
                height=max(380, len(labels) * 36)
            )
            st.altair_chart(corr_chart, width="stretch")

        with col_top:
            sparkle_svg = svg_icon("sparkles", 13, "#1d4ed8")
            st.markdown(
                f"""
                <div class="bento-card">
                    <div class="bento-gradient-top"></div>
                    <div class="card-tag">{sparkle_svg} Top Associations</div>
                    <div class="card-heading" style="font-size: 1.25rem;">Leading Correlated Pairs</div>
                    <div class="card-desc">Strongest absolute correlation coefficients discovered.</div>
                """,
                unsafe_allow_html=True,
            )
            if top_pairs:
                for pair in top_pairs[:6]:
                    v1 = pair.get("var1")
                    v2 = pair.get("var2")
                    r_val = pair.get("r")
                    r_str = f"{r_val:+.3f}" if r_val is not None else "N/A"
                    st.markdown(
                        f"""
                        <div class="pair-card">
                            <div>
                                <strong style="color: #0f172a;">{v1}</strong>
                                <span style="color: #94a3b8; margin: 0 4px;">↔</span>
                                <strong style="color: #0f172a;">{v2}</strong>
                            </div>
                            <span class="pair-badge">r = {r_str}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.write("No distinct pairs found.")
            st.markdown("</div>", unsafe_allow_html=True)


def _render_segment_feature_importance(df, result=None):
    """Segment: Feature Importance via RandomForest."""
    _render_segment_header(
        "sparkles",
        "Feature Importance (RandomForest)",
        "Automated Gini / variance reduction feature importances trained using an ensemble of decision trees.",
    )

    with st.expander("Feature Importance (RandomForest) (Click to Collapse/Expand)", expanded=True):
        if df is None or df.empty:
            st.info("No data available for feature importance analysis.")
            return

        num_cols = list(df.select_dtypes(include=np.number).columns)
        all_cols = list(df.columns)
        if len(num_cols) < 2 and len(all_cols) < 2:
            st.info("At least two columns are required to train a RandomForest model.")
            return

        meta_target = result.get("meta", {}).get("target") if result else None

        if "Temperature (C)" in df.columns:
            default_target = "Temperature (C)"
        elif meta_target and meta_target in df.columns:
            default_target = meta_target
        elif num_cols:
            default_target = num_cols[-1]
        else:
            default_target = all_cols[-1]

        target_idx = all_cols.index(default_target) if default_target in all_cols else 0

        col_tgt, col_info = st.columns([1.6, 1.0])
        with col_tgt:
            target_col = st.selectbox(
                "Select Target Column (Dependent Variable y)",
                options=all_cols,
                index=target_idx,
                key="rf_target_selector",
            )
        with col_info:
            st.caption("Predictor features (X) will automatically exclude the target variable.")

        X_cols = [c for c in num_cols if c != target_col]
        if not X_cols:
            st.warning(f"No numeric predictor features available for X when target is '{target_col}'.")
            return

        target_series = df[target_col]
        is_continuous = pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 20

        data = df[X_cols + [target_col]].dropna(subset=[target_col])
        if len(data) > 5000:
            data = data.sample(5000, random_state=42)

        X = data[X_cols].fillna(data[X_cols].median())
        y = data[target_col]

        with st.spinner("Training RandomForest model..."):
            if is_continuous:
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X, y)
                model_type_lbl = "RandomForestRegressor"
            else:
                y_enc = pd.factorize(y)[0]
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X, y_enc)
                model_type_lbl = "RandomForestClassifier"

            imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

        col_chart, col_tbl = st.columns([1.6, 1.0])

        with col_chart:
            plot_imp = imp.tail(20) if len(imp) > 20 else imp
            fig, ax = plt.subplots(figsize=(8, max(4.5, len(plot_imp) * 0.3)))
            plot_imp.plot(kind="barh", ax=ax, color="#2563eb")
            title_suffix = " (Top 20)" if len(imp) > 20 else ""
            ax.set_title(f"Feature importance ({model_type_lbl}){title_suffix}", fontsize=11, fontweight="bold", pad=10)
            ax.set_xlabel("Importance Score", fontsize=9)
            ax.grid(axis="x", linestyle="--", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_tbl:
            df_imp = (
                pd.DataFrame({"Feature": imp.index, "Importance": imp.values})
                .sort_values(by="Importance", ascending=False)
                .reset_index(drop=True)
            )
            df_imp["Share (%)"] = (df_imp["Importance"] * 100).round(2)
            df_imp["Importance"] = df_imp["Importance"].round(4)
            st.markdown(
                f"""
                <div class="bento-card">
                    <div class="bento-gradient-top"></div>
                    <div class="card-tag">{svg_icon('sparkles', 13, '#1d4ed8')} Model Info</div>
                    <div class="card-heading" style="font-size: 1.2rem;">{model_type_lbl}</div>
                    <div class="card-desc">Target: <strong>{target_col}</strong> ({len(X):,} samples analyzed).</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            _render_two_blue_table(df_imp)


def _render_segment_kde(result):
    """Segment 5: Continuous Distribution Curves (KDE)."""
    _render_segment_header(
        "chart_wave",
        "Continuous Distribution & Density (KDE)",
        "Non-parametric probability density estimates computed using Gaussian kernels.",
    )

    with st.expander("Continuous Distribution & Density (KDE) (Click to Collapse/Expand)", expanded=True):
        vis = result.get("visualizations", {})
        kde_data = vis.get("kde", {})
        columns_kde = kde_data.get("columns", {})

        if not columns_kde:
            st.info("KDE distributions are not available for this dataset.")
            return

        available_cols = list(columns_kde.keys())
        selected_col = st.selectbox("Select feature to inspect density curve", available_cols)

        if selected_col in columns_kde:
            cdata = columns_kde[selected_col]
            x_vals = cdata.get("x", [])
            d_vals = cdata.get("density", [])

            df_curve = pd.DataFrame({"Value": x_vals, "Density": d_vals})

            col_plot, col_info = st.columns([1.7, 1.0])

            with col_plot:
                area_chart = (
                    alt.Chart(df_curve)
                    .mark_area(
                        opacity=0.35,
                        color="#38bdf8",
                        line={"color": "#1d4ed8", "width": 2.5},
                    )
                    .encode(
                        x=alt.X("Value:Q", title=f"{selected_col} Scale"),
                        y=alt.Y("Density:Q", title="Probability Density"),
                        tooltip=["Value", "Density"],
                    )
                    .properties(height=320)
                    .configure_axis(grid=True, gridColor="#f1f5f9", labelFont="Plus Jakarta Sans", titleFont="Plus Jakarta Sans")
                )
                st.altair_chart(area_chart, width="stretch")

            with col_info:
                stats = result.get("data_description", {}).get("statistics", {}).get(selected_col, {})
                st.markdown(
                    f"""
                    <div class="bento-card">
                        <div class="bento-gradient-top"></div>
                        <div class="card-tag">{svg_icon('activity', 13, '#1d4ed8')} Feature Dynamics</div>
                        <div class="card-heading" style="font-size: 1.25rem;">{selected_col}</div>
                        <div class="card-desc">Statistical parameters & shape metrics.</div>
                        <div class="metric-grid">
                            <div class="metric-tile">
                                <div class="metric-tile-lbl">Mean</div>
                                <div class="metric-tile-val" style="font-size: 1.25rem;">{_clean_table_val(stats.get('avg'))}</div>
                            </div>
                            <div class="metric-tile">
                                <div class="metric-tile-lbl">Median</div>
                                <div class="metric-tile-val" style="font-size: 1.25rem;">{_clean_table_val(stats.get('median'))}</div>
                            </div>
                            <div class="metric-tile">
                                <div class="metric-tile-lbl">Std Dev</div>
                                <div class="metric-tile-val" style="font-size: 1.25rem;">{_clean_table_val(stats.get('std'))}</div>
                            </div>
                            <div class="metric-tile">
                                <div class="metric-tile-lbl">Range</div>
                                <div class="metric-tile-val" style="font-size: 1.25rem;">{_clean_table_val(stats.get('range'))}</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def _render_segment_raw_data(df):
    """Segment 6: Raw Dataset Explorer & Export."""
    _render_segment_header(
        "database",
        "Raw Dataset Explorer & Export",
        "Interactive record viewer with two-blue zebra striping and CSV export.",
    )

    with st.expander("Raw Dataset Explorer & Export (Click to Collapse/Expand)", expanded=True):
        if df is None or df.empty:
            st.info("Raw dataset preview is not available.")
            return

        col_rows, col_download = st.columns([1.5, 1.0])
        with col_rows:
            row_limit = st.selectbox("Display rows", [25, 50, 100, 200, "All"], index=0)

        with col_download:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Export CSV Dataset",
                data=csv_bytes,
                file_name="dataset_export.csv",
                mime="text/csv",
            )

        limit_num = None if row_limit == "All" else int(row_limit)
        _render_two_blue_table(df, max_rows=limit_num)


# ==============================================================================
# 9. Main Controller & Page Flow
# ==============================================================================
@st.cache_data(show_spinner=False)
def _cached_group_analysis(group_name, subset_file=None):
    import controller

    return controller.analyze_group(group_name, subset_file=subset_file)


@st.cache_data(show_spinner=False)
def _cached_group_dataframe(group_name, subset_file=None):
    import controller

    return controller.get_group_dataframe(group_name, subset_file=subset_file)


@st.cache_data(show_spinner=False)
def _cached_upload_analysis(filename, content):
    import controller

    return controller.analyze_uploaded_file(filename, content)


@st.cache_data(show_spinner=False)
def _cached_dataframe(filename=None, content=None):
    import controller

    return controller.get_dataframe(filename=filename, content=content)


def render(controller):
    st.set_page_config(
        page_title="DataPulse | Automated EDA",
        page_icon="https://cdn-icons-png.flaticon.com/512/2103/2103633.png",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    _inject_custom_css()

    groups = controller.get_dataset_groups()
    group_names = list(groups.keys())
    if not group_names:
        st.error("No benchmark datasets available.")
        return

    # 1. Initialize logical group state
    if "selected_group" not in st.session_state or st.session_state["selected_group"] not in groups:
        st.session_state["selected_group"] = group_names[0]

    current_group = st.session_state["selected_group"]
    group_info = groups.get(current_group, {})
    files = group_info.get("files", [])
    labels = group_info.get("labels", {})

    # 2. Scope / Subset state (default is "__overall__")
    scope_key = f"scope_select_{current_group}"
    if scope_key in st.session_state and (st.session_state[scope_key] == "__overall__" or st.session_state[scope_key] in files):
        st.session_state["selected_subset"] = st.session_state[scope_key]
    elif "selected_subset" not in st.session_state or (st.session_state["selected_subset"] != "__overall__" and st.session_state["selected_subset"] not in files):
        st.session_state["selected_subset"] = "__overall__"

    selected_subset = st.session_state["selected_subset"]
    uploaded_file = st.session_state.get("uploaded_file", None)

    try:
        if uploaded_file is not None:
            filename = uploaded_file.name
            content = uploaded_file.getvalue()
            result = _cached_upload_analysis(filename, content)
            df = _cached_dataframe(filename=filename, content=content)
            display_name = f"Custom Upload · {filename}"
            badge_label = filename
        else:
            result = _cached_group_analysis(current_group, selected_subset)
            df = _cached_group_dataframe(current_group, selected_subset)
            if selected_subset == "__overall__":
                display_name = f"{current_group} · Overall View"
                badge_label = f"{current_group} (Overall)"
            else:
                subset_display = labels.get(selected_subset, Path(selected_subset).stem)
                display_name = f"{current_group} · {subset_display}"
                badge_label = subset_display
    except Exception as exc:
        st.error(f"Error during dataset analysis: {exc}")
        return

    # 3. Header & Brand Navigation
    _render_brand_navbar(display_name)

    # 4. Top Bento Grid (Dropdown displays ONLY original logical dataset names)
    new_selected_group, new_upload = _render_bento_cards(
        result=result,
        controller=controller,
        group_names=group_names,
        current_group=current_group,
        active_label=badge_label,
    )

    # State update on user selection in Bento cards
    if new_upload is not None and new_upload != uploaded_file:
        st.session_state["uploaded_file"] = new_upload
        st.rerun()
    elif new_selected_group != current_group and new_upload is None:
        st.session_state["selected_group"] = new_selected_group
        st.session_state["selected_subset"] = "__overall__"  # Default Overall view on group switch!
        st.session_state["uploaded_file"] = None
        st.rerun()

    # 5. Clear Overall section first for multi-file datasets
    if uploaded_file is None and len(files) > 1:
        new_choice = _render_overall_and_subsets_section(
            current_group=current_group,
            group_info=group_info,
            controller=controller,
            selected_subset=selected_subset,
        )
        if new_choice != selected_subset:
            st.session_state["selected_subset"] = new_choice
            st.rerun()

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 6. All Segments on Single Page (Default Dropped Down / Collapsible)
    _render_segment_statistics(result)
    _render_segment_quality(result, df)
    _render_segment_outliers(result, df)
    _render_segment_balance(df)
    _render_segment_correlation(result)
    _render_segment_feature_importance(df, result)
    _render_segment_kde(result)
    _render_segment_raw_data(df)
