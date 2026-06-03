import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Any, Dict, List, Optional

SIDEBAR_WIDTH = 300
BG_COLOR = (30, 30, 46)
TEXT_COLOR = (205, 214, 244)
ACCENT_COLOR = (137, 180, 250)
GREEN_COLOR = (166, 227, 161)
RED_COLOR = (243, 139, 168)
DIM_COLOR = (108, 112, 134)


def _get_font(size=14, bold=False):
    try:
        if bold:
            return ImageFont.truetype("arialbd.ttf", size)
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _truncate(s: str, max_len: int = 35) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len - 3] + "..."


class ToolCallVisualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._call_counter = 0

    def save(self, image: np.ndarray, tool_name: str, args: dict,
             result: Any, call_index: Optional[int] = None,
             iteration: int = 0, is_error: bool = False) -> int:
        if call_index is None:
            call_index = self._call_counter
            self._call_counter += 1

        h, w = image.shape[:2]
        canvas = Image.new('RGB', (w + SIDEBAR_WIDTH, h), BG_COLOR)
        img_bgr = image.copy()

        if not is_error:
            self._draw_overlay(img_bgr, tool_name, result, args)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        main_img = Image.fromarray(img_rgb)
        canvas.paste(main_img, (SIDEBAR_WIDTH, 0))

        self._draw_sidebar(canvas, tool_name, args, result,
                           call_index, iteration, is_error)

        filename = f"{call_index:02d}.png"
        canvas.save(os.path.join(self.output_dir, filename))
        return call_index

    def _draw_overlay(self, img: np.ndarray, tool_name: str,
                      result: Any, args: dict):
        dispatch = {
            "ocr_text_locator": self._overlay_ocr,
            "axis_line_locator": self._overlay_axis,
            "scatter_point_extractor_v1": self._overlay_scatter,
            "sam3_box_extractor": self._overlay_sam3,
            "pie_slice_extractor": self._overlay_pie,
            "heatmap_digitizer": self._overlay_heatmap,
            "color_cluster_point_sampler": self._overlay_sampler,
            "color_cluster_pixel_counter": self._overlay_counter,
        }
        fn = dispatch.get(tool_name)
        if fn:
            fn(img, result, args)

    def _overlay_ocr(self, img: np.ndarray, result: Any, args: dict):
        if not isinstance(result, list):
            return
        for item in result:
            box = item.get("box")
            cx, cy = int(item.get("cx", 0)), int(item.get("cy", 0))
            text = item.get("text", "")
            if box:
                pts = np.array(box, dtype=np.int32)
                cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
            if text:
                cv2.putText(img, text, (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    def _overlay_axis(self, img: np.ndarray, result: Any, args: dict):
        if not isinstance(result, list):
            return
        for item in result:
            p1 = tuple(item["p1"])
            p2 = tuple(item["p2"])
            line_type = item.get("type", "")
            lid = item.get("id", 0)
            color = (255, 255, 0) if line_type == "horizontal" else (255, 0, 255)
            cv2.line(img, p1, p2, color, 3)
            label = f"#{lid} {line_type[0].upper()}"
            cv2.putText(img, label, (p1[0] + 5, p1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _overlay_scatter(self, img: np.ndarray, result: Any, args: dict):
        if not isinstance(result, dict):
            return
        tr = args.get("target_rect")
        if tr and len(tr) == 4:
            cv2.rectangle(img, (tr[0], tr[1]), (tr[2], tr[3]),
                          (128, 128, 128), 2)
        for pt in result.get("points", []):
            x, y = int(pt["x"]), int(pt["y"])
            rgb = pt.get("rgb", [255, 0, 0])
            color = tuple(rgb[::-1])
            cv2.circle(img, (x, y), 5, color, -1)
            cv2.circle(img, (x, y), 5, (255, 255, 255), 1)

    def _overlay_sam3(self, img: np.ndarray, result: Any, args: dict):
        if not isinstance(result, list):
            return
        for idx, box in enumerate(result):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            c = ((idx * 47 + 30) % 200 + 55,
                 (idx * 89 + 60) % 200 + 55,
                 (idx * 137 + 90) % 200 + 55)
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
            cv2.putText(img, str(idx), (x1 + 3, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

    def _overlay_pie(self, img: np.ndarray, result: Any, args: dict):
        if not isinstance(result, list):
            return
        for idx, sl in enumerate(result):
            poly = sl.get("polygon", [])
            centroid = sl.get("centroid", [0, 0])
            hex_c = sl.get("color", "#ff0000")
            try:
                bgr = (int(hex_c[5:7], 16), int(hex_c[3:5], 16), int(hex_c[1:3], 16))
            except Exception:
                bgr = (0, 0, 255)
            if poly:
                pts = np.array(poly, dtype=np.int32)
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], bgr)
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
                cv2.polylines(img, [pts], True, bgr, 2)
            cx, cy = int(centroid[0]), int(centroid[1])
            cv2.drawMarker(img, (cx, cy), (255, 255, 255),
                           cv2.MARKER_CROSS, 12, 2)

    def _overlay_heatmap(self, img: np.ndarray, result: Any, args: dict):
        if not isinstance(result, dict):
            return
        hr = result.get("heatmap_rect")
        if hr and len(hr) == 4:
            cv2.rectangle(img, (hr[0], hr[1]), (hr[2], hr[3]), (0, 255, 0), 3)
            cv2.putText(img, "Heatmap", (hr[0] + 3, hr[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        lr = result.get("legend_rect")
        if lr and len(lr) == 4:
            cv2.rectangle(img, (lr[0], lr[1]), (lr[2], lr[3]), (255, 100, 0), 3)
            cv2.putText(img, "Legend", (lr[0] + 3, lr[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

    def _overlay_sampler(self, img: np.ndarray, result: Any, args: dict):
        if not isinstance(result, dict):
            return
        tr = args.get("target_rect")
        if tr and len(tr) == 4:
            cv2.rectangle(img, (tr[0], tr[1]), (tr[2], tr[3]),
                          (128, 128, 128), 2)
        for cl in result.get("clusters", []):
            rgb = cl.get("mean_color", [255, 0, 0])
            color = tuple(rgb[::-1])
            for pt in cl.get("sampled_points", []):
                x, y = int(pt[0]), int(pt[1])
                cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, 6, 1)

    def _overlay_counter(self, img: np.ndarray, result: Any, args: dict):
        if not isinstance(result, dict):
            return
        tr = args.get("target_rect")
        if tr and len(tr) == 4:
            cv2.rectangle(img, (tr[0], tr[1]), (tr[2], tr[3]),
                          (128, 128, 128), 2)

    def _draw_sidebar(self, canvas: Image.Image, tool_name: str,
                      args: dict, result: Any, call_index: int,
                      iteration: int, is_error: bool):
        draw = ImageDraw.Draw(canvas)
        font_title = _get_font(18, bold=True)
        font_body = _get_font(13)
        font_small = _get_font(11)

        x = 10
        y = 10

        draw.text((x, y), tool_name, fill=ACCENT_COLOR, font=font_title)
        y += 28
        draw.text((x, y), f"Call #{call_index}  |  Iter {iteration + 1}",
                  fill=DIM_COLOR, font=font_small)
        y += 22
        draw.line([(x, y), (x + SIDEBAR_WIDTH - 20, y)], fill=DIM_COLOR)
        y += 12

        draw.text((x, y), "Parameters", fill=TEXT_COLOR, font=font_body)
        y += 18
        if args:
            for k, v in args.items():
                vs = _truncate(str(v), 38)
                draw.text((x, y), f"{k}:", fill=DIM_COLOR, font=font_small)
                draw.text((x + 8, y + 14), vs, fill=TEXT_COLOR, font=font_small)
                y += 30
        else:
            draw.text((x + 8, y), "(none)", fill=DIM_COLOR, font=font_small)
            y += 18

        y += 10
        draw.line([(x, y), (x + SIDEBAR_WIDTH - 20, y)], fill=DIM_COLOR)
        y += 12

        draw.text((x, y), "Result", fill=TEXT_COLOR, font=font_body)
        y += 18

        if is_error:
            err_msg = result if isinstance(result, str) else str(result)
            draw.text((x + 8, y), _truncate(err_msg, 42),
                      fill=RED_COLOR, font=font_small)
            return

        summary, colors = self._get_summary(result, tool_name)
        for line in summary:
            if y > canvas.height - 20:
                break
            draw.text((x + 8, y), _truncate(line, 42),
                      fill=GREEN_COLOR, font=font_small)
            y += 16

        y += 8
        for color_rgb in colors:
            if y > canvas.height - 20:
                break
            c_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            hex_str = "#{:02X}{:02X}{:02X}".format(*color_rgb)
            draw.rectangle([x + 8, y, x + 22, y + 14],
                           fill=tuple(c_bgr), outline=(255, 255, 255))
            draw.text((x + 28, y), hex_str, fill=TEXT_COLOR, font=font_small)
            y += 20

    def _get_summary(self, result: Any, tool_name: str):
        summary = []
        colors = []

        if isinstance(result, str):
            return [result], []

        if isinstance(result, list):
            summary.append(f"{len(result)} items detected")
            if tool_name == "pie_slice_extractor":
                for i, sl in enumerate(result):
                    centroid = sl.get("centroid", [0, 0])
                    cx_str = f"{centroid[0]:.0f}"
                    cy_str = f"{centroid[1]:.0f}"
                    summary.append(f"  Slice {i}: ({cx_str},{cy_str})")
                    hex_c = sl.get("color", "#000000")
                    try:
                        colors.append((int(hex_c[1:3], 16),
                                       int(hex_c[3:5], 16),
                                       int(hex_c[5:7], 16)))
                    except Exception:
                        pass
            elif tool_name == "sam3_box_extractor":
                for i, box in enumerate(result):
                    summary.append(
                        f"  Box {i}: [{int(box[0])},{int(box[1])},"
                        f"{int(box[2])},{int(box[3])}]")

        elif isinstance(result, dict):
            if "error" in result:
                summary.append(f"Error: {result['error']}")
            elif tool_name == "ocr_text_locator":
                summary.append(f"{len(result)} texts detected")
            elif tool_name == "scatter_point_extractor_v1":
                summary.append(
                    f"Points: {result.get('point_count', 0)}")
                summary.append(
                    f"Clusters: {result.get('cluster_count', 0)}")
                for cl in result.get("clusters", []):
                    summary.append(
                        f"  C{cl['id']}: {cl.get('mean_hex','')} "
                        f"({cl['count']}pts)")
                    colors.append(tuple(cl.get("mean_rgb", [0, 0, 0])))
            elif tool_name == "heatmap_digitizer":
                lo = result.get("legend_orientation", "unknown")
                summary.append(f"Legend: {lo}")
                hr = result.get("heatmap_rect")
                if hr:
                    summary.append(f"Heatmap: {hr}")
                lr = result.get("legend_rect")
                if lr:
                    summary.append(f"Legend rect: {lr}")
            elif tool_name == "color_cluster_point_sampler":
                for cl in result.get("clusters", []):
                    n = len(cl.get("sampled_points", []))
                    mc = cl.get("mean_color", [0, 0, 0])
                    summary.append(
                        f"  C{cl['cluster_idx']}: {n} pts "
                        f"RGB{tuple(mc)}")
                    colors.append(tuple(mc))
            elif tool_name == "color_cluster_pixel_counter":
                for cl in result.get("clusters", []):
                    mc = cl.get("mean_color", [0, 0, 0])
                    pc = cl.get("pixel_count", 0)
                    summary.append(
                        f"  C{cl['cluster_idx']}: {pc}px "
                        f"RGB{tuple(mc)}")
                    colors.append(tuple(mc))
            else:
                for k, v in result.items():
                    summary.append(f"{k}: {_truncate(str(v), 35)}")

        return summary, colors
