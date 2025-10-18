#!/usr/bin/env python3
"""
Lookup JSON produced (mangalore-wards-map.json)

Features
- Single-point or CSV batch lookup (lat,lon in WGS84).
- Uses EPSG:3857 Web-Mercator math (no external libs).
- On-edge is treated as inside.
- Optional nearest-ward fallback with distance in meters.
- Emits winner info (2019 MCC) when present.

Usage
  Single point:
    python ward_lookup.py --data mangalore-wards-slim.json --lat 12.884965 --lon 74.834136
"""

from __future__ import annotations
import argparse, csv, json, math, sys
from typing import Any, Dict, List, Tuple

# --- Geometry helpers ---------------------------------------------------------

def lonlat_to_webmercator(lon: float, lat: float) -> Tuple[float, float]:
    lat = max(min(lat, 85.05112878), -85.05112878)  # projection limits
    R = 6378137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.pi/4 + math.radians(lat)/2))
    return x, y

def signed_area(ring: List[List[float]]) -> float:
    s = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s  # >0 CCW, <0 CW (ArcGIS shells are typically CW)

def point_in_ring(x: float, y: float, ring: List[List[float]]) -> bool:
    # Ray casting with explicit on-edge check
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]

        # On-edge?
        dx, dy = x2 - x1, y2 - y1
        if (min(x1, x2) <= x <= max(x1, x2)) and (min(y1, y2) <= y <= max(y1, y2)):
            if abs((x - x1) * dy - (y - y1) * dx) <= 1e-9 * max(1.0, abs(dx) + abs(dy)):
                return True

        # Crossing?
        if (y1 > y) != (y2 > y):
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 0.0) + x1
            if xinters == x:
                return True
            if xinters > x:
                inside = not inside
    return inside

def point_in_polygon(x: float, y: float, rings: List[List[List[float]]]) -> bool:
    shells, holes = [], []
    for r in rings:
        (holes if signed_area(r) > 0 else shells).append(r)
    # If orientation isn’t reliable, treat all as shells (no holes).
    if not shells and rings:
        shells = rings
        holes = []
    for shell in shells:
        if point_in_ring(x, y, shell):
            for hole in holes:
                if point_in_ring(x, y, hole):
                    return False
            return True
    return False

def dist_point_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    # Squared distance from point to segment
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    denom = vx * vx + vy * vy
    t = 0.0 if denom == 0 else max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    cx, cy = x1 + t * vx, y1 + t * vy
    dx, dy = px - cx, py - cy
    return math.hypot(dx, dy)

def distance_to_polygon(x: float, y: float, rings: List[List[List[float]]]) -> float:
    dmin = float("inf")
    for ring in rings:
        n = len(ring)
        for i in range(n):
            x1, y1 = ring[i]
            x2, y2 = ring[(i + 1) % n]
            dmin = min(dmin, dist_point_segment(x, y, x1, y1, x2, y2))
    return dmin

# --- Data loading & lookup ----------------------------------------------------

def load_slim(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "wards" not in data:
        raise SystemExit("This script expects the slim format with a top-level 'wards' array.")
    return data

def lookup(wards: List[Dict[str, Any]], lat: float, lon: float, nearest: bool = False) -> Dict[str, Any] | None:
    x, y = lonlat_to_webmercator(lon, lat)

    # Quick bbox filter
    cands = []
    for w in wards:
        minx, miny, maxx, maxy = w["bbox"]
        if (minx <= x <= maxx) and (miny <= y <= maxy):
            cands.append(w)

    for w in cands:
        if point_in_polygon(x, y, w["rings"]):
            return {"match": "inside", "ward": w}

    if nearest:
        best, bestd = None, float("inf")
        for w in wards:
            d = distance_to_polygon(x, y, w["rings"])
            if d < bestd:
                best, bestd = w, d
        return {"match": "nearest", "ward": best, "distance_m": round(bestd, 2)} if best else None

    return None

# --- CLI ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Ward lookup using mangalore-wards-slim.json")
    ap.add_argument("--data", required=True, help="Path to mangalore-wards-slim.json")
    ap.add_argument("--lat", type=float, help="Latitude (WGS84)")
    ap.add_argument("--lon", type=float, help="Longitude (WGS84)")
    ap.add_argument("--csv", help="CSV input with 'lat,lon' columns")
    ap.add_argument("--out", help="CSV output (for --csv mode)")
    ap.add_argument("--nearest", action="store_true", help="Return nearest ward if point is outside")
    args = ap.parse_args()

    data = load_slim(args.data)
    wards = data["wards"]

    if args.csv:
        if not args.out:
            sys.exit("Provide --out for CSV output.")
        with open(args.csv, newline="", encoding="utf-8") as fin, open(args.out, "w", newline="", encoding="utf-8") as fout:
            r = csv.DictReader(fin)
            headers = r.fieldnames + ["ward_no","ward_name","winner_name","winner_party","winner_margin","match","distance_m"]
            w = csv.DictWriter(fout, fieldnames=headers)
            w.writeheader()
            for row in r:
                lat = float(row["lat"]); lon = float(row["lon"])
                res = lookup(wards, lat, lon, nearest=args.nearest)
                out = {k: row.get(k, "") for k in r.fieldnames}
                if res:
                    wrow = res["ward"]
                    out["ward_no"] = wrow["ward_no"]
                    out["ward_name"] = wrow["ward_name"]
                    win = wrow.get("winner") or {}
                    out["winner_name"] = win.get("name", "")
                    out["winner_party"] = win.get("party", "")
                    out["winner_margin"] = win.get("margin", "")
                    out["match"] = res["match"]
                    out["distance_m"] = res.get("distance_m", "")
                w.writerow(out)
        print(f"Wrote {args.out}")
        return

    if args.lat is None or args.lon is None:
        ap.error("Provide --lat and --lon, or use --csv/--out for batch mode.")

    res = lookup(wards, args.lat, args.lon, nearest=args.nearest)
    if not res:
        print("No ward found and nearest was not requested.")
        sys.exit(1)

    w = res["ward"]
    print(f"Ward No.: {w['ward_no']}\nWard Name: {w['ward_name']}")
    if "winner" in w and w["winner"]:
        print(f"Winner (2019): {w['winner']['name']} ({w['winner'].get('party','')})",
              end="")
        if w["winner"].get("margin") is not None:
            print(f" — margin: {w['winner']['margin']} votes")
        else:
            print()
    if res["match"] == "nearest":
        print(f"(Point is outside; nearest ward distance ≈ {res['distance_m']} m)")

if __name__ == "__main__":
    main()
