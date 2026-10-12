"""
Webcam inference for YOLOv8n-OSRA (YOLOv8-seg, full-integer-quant TFLite).

Model I/O:
  input : int8 [1,640,640,3]   real = 0.00392157 * (q + 128)
  out 0 : int8 [1,39,8500]     real = 0.0194168  * (q + 5)     (4 box + 3 cls + 32 mask coeffs)
  out 1 : int8 [1,160,160,32]  real = 0.0242659  * (q + 117)   (mask prototypes)

Install:  pip install opencv-python numpy ai-edge-litert
          (or tflite-runtime, or full tensorflow)

Run:      python webcam_infer.py --model best_saved_model/best_full_integer_quant.tflite
NPU:      python webcam_infer.py --npu                       # uses /usr/lib/libvx_delegate.so
          python webcam_infer.py --delegate /path/to/other_delegate.so
Keys:     q / ESC = quit,  m = toggle masks
"""
import argparse
import time

import cv2
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter, load_delegate
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter, load_delegate

DEFAULT_NPU_DELEGATE = "/usr/lib/libvx_delegate.so"

# Change the order if it does not match your training data.yaml
CLASS_NAMES = ["corrosion", "crack", "Coating spalling"]
COLORS = [(0, 0, 255), (0, 140, 255), (255, 200, 0)]  # BGR

IMG_SIZE = 640
NUM_MASK = 32


def dequant(arr, detail):
    scale, zp = detail["quantization"]
    return (arr.astype(np.float32) - zp) * scale


def letterbox(img, size=IMG_SIZE, color=114):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_l = (size - nw) // 2
    pad_t = (size - nh) // 2
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    canvas[pad_t:pad_t + nh, pad_l:pad_l + nw] = resized
    return canvas, r, pad_l, pad_t, nw, nh


class Detector:
    def __init__(self, model_path, threads=4, delegate_path=None):
        delegates = []
        if delegate_path:
            try:
                delegates.append(load_delegate(delegate_path))
                print(f"[info] NPU delegate loaded: {delegate_path}")
            except Exception as e:
                print(f"[warn] Failed to load delegate '{delegate_path}': {e}")
                print("[warn] Falling back to CPU.")
        self.using_npu = bool(delegates)
        self.interp = Interpreter(
            model_path=model_path,
            num_threads=threads,
            experimental_delegates=delegates or None,
        )
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        outs = self.interp.get_output_details()
        # Identify outputs by rank: 3-D = detections, 4-D = prototypes
        self.out_det = next(o for o in outs if len(o["shape"]) == 3)
        self.out_proto = next(o for o in outs if len(o["shape"]) == 4)

    def __call__(self, frame_bgr, conf_thr=0.25, iou_thr=0.45, with_masks=True):
        fh, fw = frame_bgr.shape[:2]
        img, r, pad_l, pad_t, nw, nh = letterbox(frame_bgr)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # real = pixel/255 -> q = real/scale + zp  (== pixel - 128 for this model)
        scale, zp = self.inp["quantization"]
        x = rgb.astype(np.float32) / 255.0
        x = np.clip(np.round(x / scale + zp), -128, 127).astype(self.inp["dtype"])
        self.interp.set_tensor(self.inp["index"], x[None])
        self.interp.invoke()

        det = dequant(self.interp.get_tensor(self.out_det["index"]), self.out_det)[0].T  # (8500, 39)
        nc = det.shape[1] - 4 - NUM_MASK

        boxes = det[:, :4].copy()
        scores = det[:, 4:4 + nc]
        coefs = det[:, 4 + nc:]

        # TFLite exports usually give normalised xywh; handle pixel units too
        if boxes.max() <= 2.0:
            boxes *= IMG_SIZE

        cls_ids = scores.argmax(1)
        confs = scores.max(1)
        keep = confs > conf_thr
        if not keep.any():
            return []
        boxes, confs, cls_ids, coefs = boxes[keep], confs[keep], cls_ids[keep], coefs[keep]

        # xywh (centre) -> xyxy in letterboxed space
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        nms_boxes = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in xyxy]
        idx = cv2.dnn.NMSBoxes(nms_boxes, confs.tolist(), conf_thr, iou_thr)
        if len(idx) == 0:
            return []
        idx = np.array(idx).flatten()
        xyxy, confs, cls_ids, coefs = xyxy[idx], confs[idx], cls_ids[idx], coefs[idx]

        masks = None
        if with_masks:
            proto = dequant(self.interp.get_tensor(self.out_proto["index"]), self.out_proto)[0]  # (160,160,32)
            ph, pw = proto.shape[:2]
            m = 1.0 / (1.0 + np.exp(-(coefs @ proto.reshape(-1, NUM_MASK).T)))  # (N, 160*160)
            masks = m.reshape(-1, ph, pw)

        results = []
        for i in range(len(idx)):
            x1, y1, x2, y2 = xyxy[i]
            # back to original frame coordinates
            fx1 = np.clip((x1 - pad_l) / r, 0, fw - 1)
            fy1 = np.clip((y1 - pad_t) / r, 0, fh - 1)
            fx2 = np.clip((x2 - pad_l) / r, 0, fw - 1)
            fy2 = np.clip((y2 - pad_t) / r, 0, fh - 1)

            full_mask = None
            if masks is not None:
                m640 = cv2.resize(masks[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
                box_mask = np.zeros_like(m640)
                bx1, by1 = max(int(x1), 0), max(int(y1), 0)
                bx2, by2 = min(int(x2), IMG_SIZE), min(int(y2), IMG_SIZE)
                box_mask[by1:by2, bx1:bx2] = m640[by1:by2, bx1:bx2]
                crop = box_mask[pad_t:pad_t + nh, pad_l:pad_l + nw]
                full_mask = cv2.resize(crop, (fw, fh), interpolation=cv2.INTER_LINEAR) > 0.5

            results.append({
                "box": (int(fx1), int(fy1), int(fx2), int(fy2)),
                "conf": float(confs[i]),
                "cls": int(cls_ids[i]),
                "mask": full_mask,
            })
        return results


def draw(frame, results):
    overlay = frame.copy()
    for r in results:
        if r["mask"] is not None:
            overlay[r["mask"]] = COLORS[r["cls"] % len(COLORS)]
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    for r in results:
        x1, y1, x2, y2 = r["box"]
        color = COLORS[r["cls"] % len(COLORS)]
        name = CLASS_NAMES[r["cls"]] if r["cls"] < len(CLASS_NAMES) else str(r["cls"])
        label = f"{name} {r['conf']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, max(y1 - th - 8, 0)), (x1 + tw + 4, max(y1, th + 8)), color, -1)
        cv2.putText(frame, label, (x1 + 2, max(y1 - 4, th + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="best_saved_model/best_full_integer_quant.tflite")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-mask", action="store_true", help="boxes only (faster)")
    ap.add_argument("--npu", action="store_true",
                    help=f"use NPU delegate ({DEFAULT_NPU_DELEGATE} unless --delegate is given)")
    ap.add_argument("--delegate", default=None,
                    help="path to a delegate .so (implies --npu)")
    args = ap.parse_args()

    delegate_path = args.delegate or (DEFAULT_NPU_DELEGATE if args.npu else None)
    det = Detector(args.model, args.threads, delegate_path)

    # First invoke on the NPU compiles the graph and can take several seconds
    if det.using_npu:
        print("[info] Warming up NPU (first run can be slow)...")
        det(np.zeros((480, 640, 3), np.uint8), with_masks=False)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    with_masks = not args.no_mask
    fps, t_prev = 0.0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = det(frame, args.conf, args.iou, with_masks)
        frame = draw(frame, results)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now
        cv2.putText(frame, f"FPS: {fps:.1f} ({'NPU' if det.using_npu else 'CPU'})", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Surface defect detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("m"):
            with_masks = not with_masks

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
