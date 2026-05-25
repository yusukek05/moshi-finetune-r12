#!/usr/bin/env python3
"""Static validator for .drawio files. Catches common render-time errors
like 'd.setId is not a function' that drawio emits when the XML has
structural issues.

Run:
    python3 validate_drawio.py path/to/file.drawio

Exit code 0 if clean, 1 if issues found.
"""

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def validate(path: Path) -> int:
    errors: list[str] = []
    warnings: list[str] = []

    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        print(f"[FAIL] XML parse error: {e}")
        return 1

    root = tree.getroot()
    cells = root.findall(".//mxCell")

    # 1. Collect IDs and basic structure
    ids: list[str] = []
    by_id: dict[str, ET.Element] = {}
    for c in cells:
        cid = c.get("id")
        if cid is None or cid == "":
            errors.append(f"cell has empty/missing id: {ET.tostring(c)[:120]!r}")
            continue
        ids.append(cid)
        by_id[cid] = c

    # 2. Duplicate IDs (causes many render failures)
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        errors.append(f"duplicate cell IDs: {sorted(dup)}")

    # 3. Invalid ID characters (drawio requires DOM-safe IDs)
    id_pattern = re.compile(r"^[A-Za-z0-9_\-]+$")
    for cid in ids:
        if not id_pattern.match(cid):
            warnings.append(f"id with special chars (may break): {cid!r}")

    # 3b. IDs that collide with JS Object/Array/String prototype names.
    # drawio stores cells in a plain JS object {}; an id like 'concat' makes
    # `cellMap['concat']` resolve to Array.prototype.concat instead of
    # undefined, and the decoder later fails with "d.setId is not a function".
    # This was the actual cause of an extended debugging session — keep this
    # list comprehensive and flag as ERROR (not warning).
    js_proto_names = {
        # Object.prototype
        "constructor", "hasOwnProperty", "isPrototypeOf",
        "propertyIsEnumerable", "toLocaleString", "toString", "valueOf",
        "__proto__", "__defineGetter__", "__defineSetter__",
        "__lookupGetter__", "__lookupSetter__",
        # Array.prototype (and a few that overlap with String)
        "length", "at", "concat", "copyWithin", "entries", "every",
        "fill", "filter", "find", "findIndex", "findLast", "findLastIndex",
        "flat", "flatMap", "forEach", "includes", "indexOf", "join",
        "keys", "lastIndexOf", "map", "pop", "push", "reduce",
        "reduceRight", "reverse", "shift", "slice", "some", "sort",
        "splice", "unshift", "values", "with", "group", "groupBy",
        # String.prototype extras
        "charAt", "charCodeAt", "codePointAt", "endsWith", "match",
        "matchAll", "normalize", "padEnd", "padStart", "repeat",
        "replace", "replaceAll", "search", "split", "startsWith",
        "substring", "substr", "toLowerCase", "toUpperCase", "trim",
        "trimEnd", "trimStart", "trimLeft", "trimRight",
        # Function.prototype
        "apply", "bind", "call", "name",
    }
    for cid in ids:
        if cid in js_proto_names:
            errors.append(
                f"id {cid!r} collides with JS prototype method/property — "
                f"will cause 'setId is not a function' in drawio"
            )

    # 4. source/target/parent refs
    for c in cells:
        cid = c.get("id", "?")
        for attr in ("source", "target", "parent"):
            ref = c.get(attr)
            if ref and ref not in ids:
                errors.append(f"cell {cid!r}: {attr}={ref!r} not defined")

    # 5. Edges must have source AND target (else 'setId' on undefined)
    for c in cells:
        if c.get("edge") == "1":
            cid = c.get("id", "?")
            src = c.get("source")
            tgt = c.get("target")
            # Edge can have explicit sourcePoint/targetPoint instead of source/target;
            # here we warn if both ref and point are missing
            geom = c.find("mxGeometry")
            has_src_point = (
                geom is not None
                and geom.find("mxPoint[@as='sourcePoint']") is not None
            )
            has_tgt_point = (
                geom is not None
                and geom.find("mxPoint[@as='targetPoint']") is not None
            )
            if not src and not has_src_point:
                errors.append(f"edge {cid!r}: no source and no sourcePoint")
            if not tgt and not has_tgt_point:
                errors.append(f"edge {cid!r}: no target and no targetPoint")

    # 6. vertex and edge are mutually exclusive per mxCell
    for c in cells:
        cid = c.get("id", "?")
        if c.get("vertex") == "1" and c.get("edge") == "1":
            errors.append(f"cell {cid!r}: both vertex=1 and edge=1 set")

    # 7. Non-root mxCell should have parent
    for c in cells:
        cid = c.get("id", "?")
        if cid in ("0", "1"):
            continue
        if c.get("parent") is None:
            errors.append(f"cell {cid!r}: missing parent attribute")

    # 8. Edge label pattern: if parent is an edge, child should be vertex=1 connectable=0
    for c in cells:
        cid = c.get("id", "?")
        par_id = c.get("parent")
        if par_id and par_id in by_id:
            par = by_id[par_id]
            if par.get("edge") == "1":
                if c.get("vertex") != "1":
                    warnings.append(
                        f"cell {cid!r} is child of edge {par_id!r} but missing vertex=1 (edge label convention)"
                    )
                style = c.get("style", "")
                if "edgeLabel" not in style:
                    warnings.append(
                        f"cell {cid!r} is child of edge {par_id!r} but style lacks 'edgeLabel' (may still work but non-standard)"
                    )

    # 9. Styles: malformed key=value; pairs
    for c in cells:
        cid = c.get("id", "?")
        style = c.get("style")
        if style:
            # drawio styles are 'key=value;key=value;...' optionally starting with 'shape=...'
            # empty segments are OK (trailing ';') but 'key=' or '=value' alone is suspicious
            for seg in style.split(";"):
                seg = seg.strip()
                if not seg:
                    continue
                if "=" in seg:
                    k, v = seg.split("=", 1)
                    if not k.strip():
                        warnings.append(f"cell {cid!r} style has empty key in '{seg}'")
                    # We won't flag empty values (some styles have 'value=')
                # seg without '=' is allowed (e.g., 'rounded' alone in some older styles)

    # 10. Circular parent
    for start_id in ids:
        seen: set[str] = set()
        cur = start_id
        while cur and cur not in ("0", "1"):
            if cur in seen:
                errors.append(f"circular parent chain starting from {start_id!r}: {seen}")
                break
            seen.add(cur)
            nxt = by_id.get(cur)
            if nxt is None:
                break
            cur = nxt.get("parent") or ""

    # Report
    print(f"File: {path}")
    print(f"  cells={len(cells)} | unique_ids={len(set(ids))}")
    print(f"  vertices={sum(1 for c in cells if c.get('vertex') == '1')}")
    print(f"  edges={sum(1 for c in cells if c.get('edge') == '1')}")
    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  [warn] {w}")
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  [err]  {e}")
        return 1
    print("\nOK — no structural errors detected.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: validate_drawio.py path/to/file.drawio")
        sys.exit(2)
    sys.exit(validate(Path(sys.argv[1])))
