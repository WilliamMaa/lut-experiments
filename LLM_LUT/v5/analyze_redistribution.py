"""
Analyze how to redistribute Phase 4 groups if gate_proj scale is reduced.
Reads sensitivity_scan.json and current Phase 4 configs from run_phase4_down_o_gate_5pct.sh.
"""

import json
import re
from pathlib import Path


def parse_config_str(s):
    """Parse '15:9;1;9;28,...' into list of (layer, group_ids)."""
    if not s:
        return []
    entries = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        layer_str, rest = part.split(":")
        layer = int(layer_str)
        items = rest.split(";")
        count = int(items[0])
        group_ids = [int(x) for x in items[1:]]
        assert len(group_ids) == count, f"layer {layer}: count {count} != {len(group_ids)} ids"
        entries.append((layer, group_ids))
    return entries


def load_current_configs():
    script = Path(__file__).parent / "run_phase4_down_o_gate_5pct.sh"
    text = script.read_text(encoding="utf-8")
    down = re.search(r'DOWN_CONFIGS="([^"]+)"', text).group(1)
    o = re.search(r'O_CONFIGS="([^"]+)"', text).group(1)
    gate = re.search(r'GATE_CONFIGS="([^"]+)"', text).group(1)
    return {
        "down_proj": parse_config_str(down),
        "o_proj": parse_config_str(o),
        "gate_proj": parse_config_str(gate),
    }


def selected_set(configs):
    sel = set()
    for module, layers in configs.items():
        for layer, gids in layers:
            for gid in gids:
                sel.add((module, layer, gid))
    return sel


def build_candidates(scan, module, selected):
    cands = []
    for item in scan["scans"][module]:
        key = (module, item["layer"], item["group_id"])
        if key in selected:
            continue
        mac = item["mac_saved_per_token"]
        delta_kl = item["delta_kl"]
        cost_per_mac = delta_kl / mac if mac > 0 else float("inf")
        cands.append({
            "module": module,
            "layer": item["layer"],
            "group_id": item["group_id"],
            "mac": mac,
            "delta_kl": delta_kl,
            "cost_per_mac": cost_per_mac,
        })
    return cands


def format_config(module_items):
    by_layer = {}
    for layer, gid in module_items:
        by_layer.setdefault(layer, []).append(gid)
    parts = []
    for layer in sorted(by_layer):
        gids = sorted(by_layer[layer])
        parts.append(f"{layer}:{len(gids)};" + ";".join(str(g) for g in gids))
    return ",".join(parts)


def make_scenario(scan, selected, gate_max, target_mac, per_layer_caps=None):
    per_layer_caps = per_layer_caps or {"down_proj": 15, "o_proj": 16, "gate_proj": 60}

    def count_per_layer(configs, module, layer):
        return sum(1 for l, _ in configs[module] if l == layer)

    # Keep top gate_max gate groups from current selection (lowest cost per mac).
    gate_items = []
    for item in scan["scans"]["gate_proj"]:
        key = ("gate_proj", item["layer"], item["group_id"])
        if key in selected:
            gate_items.append((item["layer"], item["group_id"], item["mac_saved_per_token"], item["delta_kl"]))
    gate_items.sort(key=lambda x: x[3] / x[2])
    kept_gate = []
    layer_counts = {}
    for layer, gid, mac, dkl in gate_items:
        if len(kept_gate) >= gate_max:
            break
        if layer_counts.get(("gate_proj", layer), 0) >= per_layer_caps.get("gate_proj", 999):
            continue
        kept_gate.append((layer, gid, mac, dkl))
        layer_counts[("gate_proj", layer)] = layer_counts.get(("gate_proj", layer), 0) + 1

    new_sel = set()
    for layer, gid, mac, dkl in kept_gate:
        new_sel.add(("gate_proj", layer, gid))

    total_mac = sum(mac for _, _, mac, _ in kept_gate)
    total_kl = sum(dkl for _, _, _, dkl in kept_gate)

    cands = []
    for module in ["down_proj", "o_proj"]:
        cands.extend(build_candidates(scan, module, selected | new_sel))
    cands.sort(key=lambda x: x["cost_per_mac"])

    for cand in cands:
        if total_mac >= target_mac:
            break
        key = (cand["module"], cand["layer"], cand["group_id"])
        if key in new_sel:
            continue
        if layer_counts.get((cand["module"], cand["layer"]), 0) >= per_layer_caps.get(cand["module"], 999):
            continue
        new_sel.add(key)
        layer_counts[(cand["module"], cand["layer"])] = layer_counts.get((cand["module"], cand["layer"]), 0) + 1
        total_mac += cand["mac"]
        total_kl += cand["delta_kl"]

    configs = {m: [] for m in ["down_proj", "o_proj", "gate_proj"]}
    for module, layer, gid in new_sel:
        configs[module].append((layer, gid))

    return configs, total_mac, total_kl


def main():
    scan_path = Path(__file__).parent / "results" / "sensitivity_scan.json"
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    current = load_current_configs()
    selected = selected_set(current)

    total_full_model_mac = 7_142_496_256
    target_mac = int(total_full_model_mac * 0.05)

    print(f"Target 5% MAC: {target_mac:,}")
    print(f"Current selected: down={sum(len(g) for _, g in current['down_proj'])}, "
          f"o={sum(len(g) for _, g in current['o_proj'])}, "
          f"gate={sum(len(g) for _, g in current['gate_proj'])}")
    print()

    for gate_max, name in [(0, "no gate, down+o only"),
                           (200, "gate=200, fill with down+o"),
                           (400, "gate=400, fill with down+o"),
                           (600, "gate=600, fill with down+o")]:
        configs, total_mac, total_kl = make_scenario(scan, selected, gate_max, target_mac)
        ratio = total_mac / total_full_model_mac * 100
        down_n = len(configs["down_proj"])
        o_n = len(configs["o_proj"])
        gate_n = len(configs["gate_proj"])
        print(f"--- {name} ---")
        print(f"  down={down_n}, o={o_n}, gate={gate_n}")
        print(f"  total MAC saved: {total_mac:,} ({ratio:.2f}%)")
        print(f"  total delta KL: {total_kl:.4f}")
        print(f"  DOWN_CONFIGS=\"{format_config(configs['down_proj'])}\"")
        print(f"  O_CONFIGS=\"{format_config(configs['o_proj'])}\"")
        print(f"  GATE_CONFIGS=\"{format_config(configs['gate_proj'])}\"")
        print()


if __name__ == "__main__":
    main()
