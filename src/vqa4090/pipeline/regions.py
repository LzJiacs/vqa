from __future__ import annotations

from vqa4090.data.schemas import Region


def _zone_from_bbox(bbox: list[int], page_w: int, page_h: int) -> str:
    if len(bbox) != 4 or page_w <= 0 or page_h <= 0:
        return "unknown"
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    horiz = "left" if cx < page_w * 0.5 else "right"
    if cy < page_h * 0.2:
        vert = "header"
    elif cy > page_h * 0.85:
        vert = "footer"
    else:
        vert = "body"
    return f"{vert}_{horiz}"


def build_regions_from_ocr(
    doc_id: str,
    ocr_items: list[dict],
    merge_lines: bool = True,
    y_merge_thresh: int = 20,
    page_size: tuple[int, int] | None = None,
) -> list[Region]:
    if not ocr_items:
        return []

    # Sort top-to-bottom then left-to-right.
    sorted_items = sorted(ocr_items, key=lambda x: (x.get("bbox", [0, 0, 0, 0])[1], x.get("bbox", [0, 0, 0, 0])[0]))
    page_w, page_h = page_size if page_size else (0, 0)

    regions: list[Region] = []
    if not merge_lines:
        for i, x in enumerate(sorted_items):
            bbox = x.get("bbox", [])
            zone = _zone_from_bbox(bbox, page_w, page_h)
            text = x.get("text", "").strip()
            if zone != "unknown":
                text = f"[{zone}] {text}"
            regions.append(
                Region(
                    doc_id=doc_id,
                    region_id=f"{doc_id}_r{i}",
                    text=text,
                    bbox=bbox,
                    image_path=x.get("image_path"),
                )
            )
        return regions

    # Merge neighboring OCR lines into layout-aware text blocks.
    blocks: list[list[dict]] = []
    current: list[dict] = []
    last_y = None
    for item in sorted_items:
        y0 = item.get("bbox", [0, 0, 0, 0])[1]
        if not current:
            current = [item]
            last_y = y0
            continue

        if last_y is not None and abs(y0 - last_y) <= y_merge_thresh:
            current.append(item)
            last_y = y0
        else:
            blocks.append(current)
            current = [item]
            last_y = y0

    if current:
        blocks.append(current)

    for i, blk in enumerate(blocks):
        txts = [x.get("text", "").strip() for x in blk if x.get("text", "").strip()]
        if not txts:
            continue
        xs0 = [x.get("bbox", [0, 0, 0, 0])[0] for x in blk]
        ys0 = [x.get("bbox", [0, 0, 0, 0])[1] for x in blk]
        xs1 = [x.get("bbox", [0, 0, 0, 0])[2] for x in blk]
        ys1 = [x.get("bbox", [0, 0, 0, 0])[3] for x in blk]
        bbox = [min(xs0), min(ys0), max(xs1), max(ys1)]
        zone = _zone_from_bbox(bbox, page_w, page_h)

        text = " ".join(txts)
        if zone != "unknown":
            text = f"[{zone}] {text}"

        regions.append(
            Region(
                doc_id=doc_id,
                region_id=f"{doc_id}_r{i}",
                text=text,
                bbox=bbox,
                image_path=blk[0].get("image_path"),
            )
        )

    return regions
