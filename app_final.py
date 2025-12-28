import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO


MAX_INPUT_WIDTH = 800
MAX_PANO_WIDTH = 2500


def resize_to_max_width(img, max_width=MAX_INPUT_WIDTH):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / float(w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def to_gray(img):
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def crop_black_borders(image, threshold=10):
    if image is None or image.size == 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray > threshold).astype(np.uint8) * 255
    if cv2.countNonZero(mask) == 0:
        return image

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    margin = 3
    x = max(0, x + margin)
    y = max(0, y + margin)
    w = max(1, w - 2 * margin)
    h = max(1, h - 2 * margin)

    return image[y:y+h, x:x+w]

def feather_blend(base, warped, feather_iters=2):
    base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    base_mask = (base_gray > 0).astype(np.uint8)
    warped_mask = (warped_gray > 0).astype(np.uint8)

    overlap = (base_mask & warped_mask).astype(np.uint8)

    if cv2.countNonZero(overlap) == 0:
        out = base.copy()
        out[warped_mask == 1] = warped[warped_mask == 1]
        return out

    warped_mask_255 = (warped_mask * 255).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    for _ in range(max(1, feather_iters)):
        warped_mask_255 = cv2.erode(warped_mask_255, kernel, iterations=1)

    if cv2.countNonZero(warped_mask_255) == 0:
        warped_mask_255 = (warped_mask * 255).astype(np.uint8)

    dist = cv2.distanceTransform(warped_mask_255, cv2.DIST_L2, 3)
    alpha = dist / (dist.max() + 1e-6) if dist.max() > 0 else dist
    alpha *= warped_mask

    alpha_3 = np.dstack([alpha, alpha, alpha]).astype(np.float16)
    base_f = base.astype(np.float16)
    warped_f = warped.astype(np.float16)
    out = (warped_f * alpha_3 + base_f * (1.0 - alpha_3)).astype(np.uint8)

    out[(base_mask == 0) & (warped_mask == 1)] = warped[(base_mask == 0) & (warped_mask == 1)]
    out[(base_mask == 1) & (warped_mask == 0)] = base[(base_mask == 1) & (warped_mask == 0)]
    return out

# ==========================
# Cylindrical warp (ADDED)
# ==========================
def cylindrical_warp(img, f):
    """
    Projects image onto a cylinder. Helps real panoramas (reduces wedge/curvature).
    f: focal length in pixels (slider).
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    y_i, x_i = np.indices((h, w))
    x = (x_i - cx) / f
    y = (y_i - cy) / f

    x_c = np.tan(x)
    y_c = y / np.cos(x)

    map_x = (f * x_c + cx).astype(np.float32)
    map_y = (f * y_c + cy).astype(np.float32)

    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return warped

# ==========================
# Features + Matching
# ==========================
def detect_and_describe(img, method="SIFT"):
    gray = to_gray(img)
    if method == "SIFT":
        detector = cv2.SIFT_create()
    else:
        detector = cv2.ORB_create(5000)
    kps, des = detector.detectAndCompute(gray, None)
    if kps is None or len(kps) == 0 or des is None:
        return None, None
    pts = np.float32([kp.pt for kp in kps])
    return pts, des

def _bfmatcher(method):
    if method == "SIFT":
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

def mutual_ratio_matches(des1, des2, method="SIFT", ratio=0.75):
    matcher = _bfmatcher(method)

    m12 = matcher.knnMatch(des1, des2, k=2)
    good12 = {}
    for pair in m12:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good12[(m.queryIdx, m.trainIdx)] = m.distance

    m21 = matcher.knnMatch(des2, des1, k=2)
    good21 = {}
    for pair in m21:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good21[(m.trainIdx, m.queryIdx)] = m.distance

    mutual = []
    for (q, t), d in good12.items():
        if (q, t) in good21:
            mutual.append((q, t, d))

    mutual.sort(key=lambda x: x[2])
    return mutual

def draw_match_viz(img1, img2, pts1, pts2, mutual_matches, inlier_mask=None, max_draw=120):
    kps1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in pts1]
    kps2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in pts2]

    matches = []
    for (q, t, _) in mutual_matches[:max_draw]:
        matches.append(cv2.DMatch(_queryIdx=int(q), _trainIdx=int(t), _imgIdx=0, _distance=0))

    if inlier_mask is not None and len(inlier_mask) == len(mutual_matches):
        inliers = [i for i, v in enumerate(inlier_mask[:len(mutual_matches)]) if v]
        if len(inliers) > 0:
            matches = [cv2.DMatch(_queryIdx=int(mutual_matches[i][0]),
                                  _trainIdx=int(mutual_matches[i][1]),
                                  _imgIdx=0, _distance=0)
                       for i in inliers[:max_draw]]

    vis = cv2.drawMatches(img1, kps1, img2, kps2, matches, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

# ==========================
# Homography / Warp
# ==========================
def find_homography(src_pts, dst_pts, reproj=4.0):
    if hasattr(cv2, "USAC_MAGSAC"):
        H, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.USAC_MAGSAC, ransacReprojThreshold=reproj)
    else:
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj)
    return H, mask

# ---- Homography sanity check (ADDED) ----
def homography_is_sane(H, img2_shape, max_expand=3.0):
    """
    Reject homographies that create the huge wedge/triangle explosion.
    max_expand: how much larger the projected bbox may be compared to input.
    """
    if H is None:
        return False

    # Too much perspective often means wrong matches
    if abs(H[2, 0]) > 0.01 or abs(H[2, 1]) > 0.01:
        return False

    h2, w2 = img2_shape[:2]
    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    wc = cv2.perspectiveTransform(corners2, H)
    xs = wc[:, 0, 0]
    ys = wc[:, 0, 1]
    w_proj = (xs.max() - xs.min())
    h_proj = (ys.max() - ys.min())

    if w_proj <= 0 or h_proj <= 0:
        return False

    # If projection bbox is insanely larger than original → reject
    if w_proj > max_expand * w2 or h_proj > max_expand * h2:
        return False

    return True

def warp_and_compose(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners2 = cv2.perspectiveTransform(corners2, H)
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    all_corners = np.concatenate((corners1, warped_corners2), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = [-xmin, -ymin]
    T = np.array([[1, 0, translation[0]],
                  [0, 1, translation[1]],
                  [0, 0, 1]], dtype=np.float64)

    output_size = (xmax - xmin, ymax - ymin)
    ow, oh = output_size

    # canvas cap (prevents multi-GB)
    max_canvas_w = MAX_PANO_WIDTH * 2
    max_canvas_h = 4000
    if ow > max_canvas_w or oh > max_canvas_h:
        scale = min(max_canvas_w / float(ow), max_canvas_h / float(oh))
        output_size = (max(1, int(ow * scale)), max(1, int(oh * scale)))
        S = np.array([[scale, 0, 0],
                      [0, scale, 0],
                      [0, 0, 1]], dtype=np.float64)
        H = S @ H
        T = S @ T
        translation = [int(translation[0] * scale), int(translation[1] * scale)]
        img1 = cv2.resize(img1, (int(w1 * scale), int(h1 * scale)), interpolation=cv2.INTER_AREA)
        h1, w1 = img1.shape[:2]

    warped = cv2.warpPerspective(img2, T @ H, output_size)

    canvas = np.zeros_like(warped)
    y0, x0 = translation[1], translation[0]
    y1, x1 = y0 + h1, x0 + w1

    y0c, x0c = max(0, y0), max(0, x0)
    y1c, x1c = min(canvas.shape[0], y1), min(canvas.shape[1], x1)
    if y1c > y0c and x1c > x0c:
        canvas[y0c:y1c, x0c:x1c] = img1[0:(y1c - y0c), 0:(x1c - x0c)]

    result = feather_blend(canvas, warped, feather_iters=2)
    return result

def stitch_pair(img1, img2, method="SIFT", diagnostics=False):
    diag = {}

    pts1, des1 = detect_and_describe(img1, method=method)
    pts2, des2 = detect_and_describe(img2, method=method)
    if des1 is None or des2 is None or pts1 is None or pts2 is None:
        raise ValueError("Not enough features in one of the images.")

    ratios = [0.75, 0.85] if method == "SIFT" else [0.70, 0.80]

    for r in ratios:
        mutual = mutual_ratio_matches(des1, des2, method=method, ratio=r)
        if len(mutual) < (25 if method == "ORB" else 12):
            continue

        mutual_use = mutual[:800]
        src = np.float32([pts2[t] for (q, t, _) in mutual_use])  # img2
        dst = np.float32([pts1[q] for (q, t, _) in mutual_use])  # img1

        H, mask = find_homography(src, dst, reproj=4.0 if method == "SIFT" else 6.0)
        if H is None or mask is None:
            continue

        inliers = int(mask.sum())
        inlier_ratio = inliers / float(len(mutual_use))

        # ---- Sanity gate (ADDED) ----
        if not homography_is_sane(H, img2.shape, max_expand=3.0):
            continue

        # quality gate
        if inliers < 18 or inlier_ratio < 0.25:
            continue

        if diagnostics:
            diag["method"] = method
            diag["ratio_used"] = r
            diag["kps_img1"] = len(pts1)
            diag["kps_img2"] = len(pts2)
            diag["mutual_matches"] = len(mutual_use)
            diag["inliers"] = inliers
            diag["inlier_ratio"] = round(inlier_ratio, 3)
            diag["match_viz_rgb"] = draw_match_viz(
                img1, img2, pts1, pts2, mutual_use,
                inlier_mask=mask.ravel().astype(bool)
            )

        return warp_and_compose(img1, img2, H), diag

    # ---- Affine fallback (keeps it stable, no wedge explosion) ----
    mutual = mutual_ratio_matches(des1, des2, method=method, ratio=0.9)
    if len(mutual) >= 20:
        mutual_use = mutual[:600]
        src = np.float32([pts2[t] for (q, t, _) in mutual_use])
        dst = np.float32([pts1[q] for (q, t, _) in mutual_use])

        A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=6.0)
        if A is not None:
            H_aff = np.array([[A[0, 0], A[0, 1], A[0, 2]],
                              [A[1, 0], A[1, 1], A[1, 2]],
                              [0, 0, 1]], dtype=np.float64)

            if diagnostics:
                diag["used_affine_fallback"] = True
                diag["kps_img1"] = len(pts1)
                diag["kps_img2"] = len(pts2)
                diag["mutual_matches"] = len(mutual_use)
                diag["inliers"] = int(inl.sum()) if inl is not None else 0
                diag["match_viz_rgb"] = draw_match_viz(img1, img2, pts1, pts2, mutual_use, inlier_mask=None)

            return warp_and_compose(img1, img2, H_aff), diag

    raise ValueError("Homography estimation failed (matches unreliable / overlap weak).")

# ==========================
# Multi-image stitching
# ==========================
def stitch_images_sequential(images, method="SIFT", diagnostics=False, progress_callback=None):
    pano = images[0].copy()
    all_diags = []
    total_steps = len(images) - 1

    for i in range(1, len(images)):
        if progress_callback:
            pct = int((i / total_steps) * 100)
            progress_callback(pct, f"Stitching image {i+1}/{len(images)} (Sequential)...")

        pano, diag = stitch_pair(pano, images[i], method=method, diagnostics=diagnostics)
        all_diags.append(diag)

    pano = crop_black_borders(pano, threshold=10)
    return pano, all_diags

def stitch_images_center(images, method="SIFT", diagnostics=False, progress_callback=None):
    center_idx = len(images) // 2
    pano = images[center_idx].copy()
    all_diags = []

    total_ops = len(images) - 1
    done = 0

    for i in range(center_idx - 1, -1, -1):
        done += 1
        if progress_callback:
            pct = int((done / total_ops) * 100)
            progress_callback(pct, f"Stitching image {i+1} (Left of center)...")

        pano, diag = stitch_pair(images[i], pano, method=method, diagnostics=diagnostics)
        all_diags.append(diag)

    for i in range(center_idx + 1, len(images)):
        done += 1
        if progress_callback:
            pct = int((done / total_ops) * 100)
            progress_callback(pct, f"Stitching image {i+1} (Right of center)...")

        pano, diag = stitch_pair(pano, images[i], method=method, diagnostics=diagnostics)
        all_diags.append(diag)

    pano = crop_black_borders(pano, threshold=10)
    return pano, all_diags

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="Panorama Stitcher", layout="wide")
st.title("🌄 Panoramic Image Stitching")
st.write("Features → Mutual Match → Homography (USAC/MAGSAC) → Warp → Blend → Crop")

with st.sidebar:
    st.header("⚙️ Settings")

    method = st.selectbox("Feature Method", ["SIFT", "ORB"])
    stitch_mode = st.selectbox("Stitching Mode", ["Sequential", "Center Reference"])
    diagnostics = st.checkbox("Show diagnostics", value=True)

    resize_enabled = st.checkbox("Resize images (recommended)", value=True)
    custom_width = st.slider("Max input width", 400, 1500, 800, 100, disabled=not resize_enabled)
    MAX_INPUT_WIDTH = custom_width if resize_enabled else 5000

    # ---- Cylindrical option (ADDED) ----
    use_cyl = st.checkbox("Use cylindrical projection (recommended)", value=True)
    f_slider = st.slider("Cylindrical focal length (px)", 400, 2000, 900, 50,
                         help="Try 700-1200. Higher = less distortion, lower = stronger cylinder.")

uploaded_files = st.file_uploader(
    "Choose images (2 or more)",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"],
    accept_multiple_files=True,
)

if uploaded_files:
    if len(uploaded_files) < 2:
        st.warning("⚠️ Upload at least 2 images.")
    else:
        st.success(f"✅ {len(uploaded_files)} images uploaded")

        with st.expander("Preview", expanded=False):
            cols = st.columns(min(4, len(uploaded_files)))
            for idx, f in enumerate(uploaded_files):
                cols[idx % 4].image(Image.open(f).convert("RGB"), caption=f"Image {idx+1}", use_container_width=True)

        if st.button("🚀 Stitch Panorama", type="primary"):
            cv_images = []
            with st.spinner("Loading images..."):
                for f in uploaded_files:
                    pil = Image.open(f).convert("RGB")
                    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                    img = resize_to_max_width(img, MAX_INPUT_WIDTH)

                    if use_cyl:
                        img = cylindrical_warp(img, f=f_slider)

                    cv_images.append(img)

            progress = st.progress(0)
            status = st.empty()

            def cb(p, msg):
                progress.progress(int(np.clip(p, 0, 100)))
                status.text(msg)

            try:
                with st.spinner("Stitching..."):
                    if stitch_mode == "Sequential":
                        pano, diags = stitch_images_sequential(cv_images, method=method, diagnostics=diagnostics, progress_callback=cb)
                    else:
                        pano, diags = stitch_images_center(cv_images, method=method, diagnostics=diagnostics, progress_callback=cb)

                progress.progress(100)
                status.text("✅ Done!")

                pano_rgb = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
                st.image(pano_rgb, caption="Panorama", use_container_width=True)

                buf = BytesIO()
                Image.fromarray(pano_rgb).save(buf, format="PNG")
                st.download_button("⬇️ Download Panorama", buf.getvalue(), "panorama.png", "image/png")

                if diagnostics:
                    st.subheader("Diagnostics")
                    for i, d in enumerate(diags, 1):
                        st.markdown(f"**Step {i}**")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("KP img1", d.get("kps_img1", "-"))
                        c2.metric("KP img2", d.get("kps_img2", "-"))
                        c3.metric("Mutual matches", d.get("mutual_matches", "-"))
                        c4.metric("Inliers", d.get("inliers", "-"))

                        if "inlier_ratio" in d:
                            st.caption(f"Inlier ratio: {d['inlier_ratio']} | ratio_used: {d.get('ratio_used')}")

                        if d.get("used_affine_fallback"):
                            st.warning("Affine fallback used (homography unstable).")

                        if "match_viz_rgb" in d:
                            st.image(d["match_viz_rgb"], use_container_width=True)

                        st.divider()

            except Exception as e:
                st.error(f"❌ Stitching failed: {e}")
                st.info(
                    "If you still see a huge black wedge / stretched triangle:\n"
                    "- Turn ON cylindrical projection\n"
                    "- Use SIFT\n"
                    "- Reduce width to 600–800\n"
                    "- Ensure real overlap (30–50%) and correct order\n"
                    "- Try focal length 700–1200\n"
                )
else:
    st.info("👆 Upload images to start.")
