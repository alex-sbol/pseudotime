import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from matplotlib.patches import Rectangle

def plot_pseudotime_for_cells(df, num_cells=10):
    num_cells = min(num_cells, len(df))

    fig, axes = plt.subplots(
        max(num_cells, 2), 2,
        figsize=(12, 3 * num_cells),
        sharex=False
    )

    dfs = df.sample(n=num_cells, random_state=42)

    if num_cells == 1:
        axes = np.atleast_1d(axes)

    # global max width for symmetric y-limits
    all_widths = np.concatenate(
        dfs.iloc[:num_cells]["pseudotime_widths"]
        .apply(lambda x: np.array(x, dtype=int))
        .values
    )
    max_width = all_widths.max()

    for i in range(num_cells):
        row = dfs.iloc[i]

        ids = row["pseudotime"]
        widths = np.array(row["pseudotime_widths"], dtype=int)
        x = np.arange(len(widths))

        # center bars at y = 0
        bottoms = -widths / 2

        axes[i, 0].bar(x, widths, bottom=bottoms)
        axes[i, 0].axhline(0, linewidth=1)

        axes[i, 0].set_ylabel("Width")
        axes[i, 0].set_title(f"Cell {i}")
        axes[i, 0].set_ylim(-max_width, max_width)

        # label pseudotime IDs at bar centers (y = 0)
        for xi, label in zip(x, ids):
            axes[i, 0].text(
                xi,
                0,
                str(label),
                ha="center",
                va="center",
                fontsize=8
            )

        img = imread(row["path_dapi"]).astype(float)
        print(row["path_dapi"])
        bg  = imread(row["path_background"]).astype(float)

        H, W = img.shape[:2]
        pad = 25

        cx = row["centered_center_x"] + img.shape[1] // 2
        cy = row["centered_center_y"] + img.shape[0] // 2
        bbox_min_x = row["centered_bbox_min_x"] + img.shape[1] // 2
        bbox_max_x = row["centered_bbox_max_x"] + img.shape[1] // 2
        bbox_min_y = row["centered_bbox_min_y"] + img.shape[0] // 2
        bbox_max_y = row["centered_bbox_max_y"] + img.shape[0] // 2

        x0 = max(int(bbox_min_x) - pad, 0)
        y0 = max(int(bbox_min_y) - pad, 0)

        x1 = min(int(bbox_max_x) + pad, W)
        y1 = min(int(bbox_max_y) + pad, H)

        img = img[y0:y1, x0:x1]
        bg  = bg[y0:y1, x0:x1]



        #Plot center dot and bbox
        axes[i, 1].plot(cx - x0, cy - y0, marker='o', markersize=1, color='red')
        rect = Rectangle(
            (bbox_min_x - x0, bbox_min_y - y0
             ),
            bbox_max_x - bbox_min_x,
            bbox_max_y - bbox_min_y,
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        axes[i, 1].add_patch(rect)

            # normalize if needed

        
        img /= img.max()
        bg  /= bg.max()

        alpha = 0.5
        blended = (1 - alpha) * img + alpha * bg

        axes[i, 1].imshow(blended, cmap="gray")
        axes[i, 1].axis("off")


    axes[-1, 0].set_xlabel("Index along pseudotime")
    fig.suptitle("Pseudotime widths centered on origin", y=1.02)

    plt.tight_layout()
    plt.show()

def rotation_viz_with_dapi(
    BG, rotated, SD, blend_mode="add", alpha=0.5,
    max_rows=30, dot_size=3
):
    import matplotlib.cm as cm
    DF = SD.dataframe

    BG = BG.astype(np.float32)
    RW, RH = rotated.shape[:2]
    cry = (RW - 1) / 2
    crx = (RH - 1) / 2

    # --- 1. Build a composite DAPI image ---
    dapi_stack = []
    for idx, row in DF.iloc[:max_rows].iterrows():
        try:
            path = row["path_dapi"]
            dapi_img = imread(path)
        except Exception as e:
            print(f"Could not read DAPI image for row {idx}: {e}")
            continue

        H, W = BG.shape[:2]
        h2, w2 = dapi_img.shape[:2]

        cy_big = (H - 1) / 2
        cx_big = (W - 1) / 2

        cy_small = (h2 - 1) / 2
        cx_small = (w2 - 1) / 2

        x0 = int(round(cx_big - cx_small))
        y0 = int(round(cy_big - cy_small))

        canvas = np.zeros((H, W), dtype=float)
        #canvas = np.zeros((H, W, 2), dtype=float)

        # Destination slice (big image)
        y1 = max(0, y0)
        x1 = max(0, x0)
        y2 = min(H, y0 + h2)
        x2 = min(W, x0 + w2)

        # Corresponding source slice (small image)
        sy1 = y1 - y0
        sx1 = x1 - x0
        sy2 = sy1 + (y2 - y1)
        sx2 = sx1 + (x2 - x1)

        # Write intensity
        #canvas[y1:y2, x1:x2, 0] = dapi_img[sy1:sy2, sx1:sx2]

        # Write alpha (fully opaque where image exists)
        #canvas[y1:y2, x1:x2, 1] = 1.0

        canvas[y1:y2, x1:x2] = dapi_img[sy1:sy2, sx1:sx2]


        if canvas.shape[:2] != BG.shape[:2]:
            raise ValueError(f"DAPI shape {canvas.shape} != BG shape {BG.shape}")

        dapi_stack.append(canvas.astype(np.float32))

    if len(dapi_stack) == 0:
        raise RuntimeError("No DAPI images could be loaded.")

    dapi_composite = np.sum(dapi_stack, axis=0)

    # --- 2. Blend BG + DAPI ---
    if blend_mode == "add":
        combined = BG.astype(np.float32) + dapi_composite
    elif blend_mode == "alpha":
        combined = (1 - alpha) * BG.astype(np.float32) + alpha * dapi_composite
    else:
        raise ValueError("blend_mode must be 'add' or 'alpha'")

    combined = np.clip(combined, 0, 255).astype(np.uint8)

    # --- 3. Plot setup ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(combined, cmap="gray")
    axes[0].set_title("Combined BG + DAPI with Annotated Objects")
    axes[0].axis("off")

    axes[1].imshow(rotated, cmap="gray")
    axes[1].set_title("Rotated Image with Annotated Objects")
    axes[1].axis("off")

    # --- 5. Colormap for unique tint per object ---
    colormap = cm.get_cmap("tab20", max_rows)

    # --- 6. Plot each object ---
    for row_index, (idx, row) in enumerate(DF.iloc[:max_rows].iterrows()):

        color = colormap(row_index)  # unique RGBA color

        # Original coords
        x_c = (float(row["centered_center_x"])) + cx_big
        y_c = (float(row["centered_center_y"])) + cy_big

        x_min = float(row["centered_bbox_min_x"]) + cx_big
        y_min = float(row["centered_bbox_min_y"]) + cy_big
        x_max = float(row["centered_bbox_max_x"]) + cx_big
        y_max = float(row["centered_bbox_max_y"]) + cy_big

        # Draw original
        axes[0].plot(x_c, y_c, marker='o', markersize=dot_size, color=color)

        axes[0].text(
        x_c, y_c - 5,            
        f"{idx}",                
        color=color,           
        fontsize=6,
        ha="center", va="bottom"
)
        rect = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=1,
            edgecolor=color,
            facecolor='none'
        )
        axes[0].add_patch(rect)

        # Rotated coords
        x_rot_c = row["center_rot_x"] + crx
        y_rot_c = row["center_rot_y"] + cry

        x_min_rot = row["bbox_rot_min_x"] + crx
        y_min_rot = row["bbox_rot_min_y"] + cry
        x_max_rot = row["bbox_rot_max_x"] + crx
        y_max_rot = row["bbox_rot_max_y"] + cry

        # Draw rotated
        axes[1].plot(x_rot_c, y_rot_c, marker='o', markersize=dot_size, color=color)

        axes[1].text(
        x_rot_c, y_rot_c - 5,             # position (slightly above the point)
        f"{idx}",                 # text label – here using the row index
        color=color,              # same tint as the object
        fontsize=6,
        ha="center", va="bottom"
)
        
        rect = Rectangle(
            (x_min_rot, y_min_rot),
            x_max_rot - x_min_rot,
            y_max_rot - y_min_rot,
            linewidth=1,
            edgecolor=color,
            facecolor='none'
        )
        axes[1].add_patch(rect)

    axes[0].plot(cx_big, cy_big, markersize=5, color='red',  marker='o')
    axes[1].plot(crx, cry, markersize=5, color='red', marker='o')
    plt.tight_layout()
    plt.show()

def line_id_color(line_id, T, cm):
    if line_id < 0:
        return 'red' 
    return cm(line_id / T)

def visualize_confidence(results):
    if results is None:
        print("No corridor results to visualize.")
        return
    
    max_len = max(len(r["confidence"]) for r in results)
    C = np.full((len(results), max_len), np.nan)

    for i, r in enumerate(results):
        C[i, :len(r["confidence"])] = r["confidence"]

    plt.figure(figsize=(16, 16))
    plt.imshow(C, aspect="auto", cmap="inferno")
    plt.colorbar(label="confidence")
    plt.xlabel("pseudotime")
    plt.ylabel("corridor")
    plt.title("Confidence heatmap")
    plt.show()

def visualize_template(template):
    if template is None:
        print("No template to visualize.")
        return
    plt.figure(figsize=(8, 3))
    plt.plot(template, marker="o")
    plt.title("Canonical period template")
    plt.xlabel("Line ID")
    plt.ylabel("Signal value")
    plt.grid(True)
    plt.show()

def visualize_overlay(
    img,
    corridor_bbox,
    results,
    T,
    m,
    alpha_scale=0.8,
    show_ids=False,
    figsize=(16, 16),
    DF=None,
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, cmap="gray")
    ax.set_title("Corridor line IDs and periods")
    ax.axis("off")
    cmap = plt.get_cmap("turbo", T)
    colors = ["red", "blue"]
    period_idx = 0

    

    for res in results:
        bbox = res["bbox"]
        x0, x1 = bbox["x0"], bbox["x1"]
        y0, y1 = bbox["y0"], bbox["y1"]

        signal = res["signal"]
        line_id = bbox["line_id"]
        conf = res["confidence"]

        xs = list(range(x0, x1 + 1, m))
        period_start_x = None
        period_top = None
        period_bottom = None


        for i, x in enumerate(xs):
            
            if i >= len(line_id):
                break

            lid = line_id[i] 
            c = conf[i]

            
            color = line_id_color(lid, T, cmap)
            alpha = alpha_scale * 1

            col = img[y0:y1+1, x]
            if np.all(col == 0):
                continue
            ys = np.flatnonzero(col)
            y_top = y0 + int(ys[0])
            y_bottom = y0 + int(ys[-1])


            if lid == 0:
                period_start_x = x
                period_top = y_top
                period_bottom = y_bottom

            elif lid == T - 1 and period_start_x is not None:
                width = x - period_start_x
                height = period_bottom - period_top

                col = colors[period_idx % 2]

                rect = Rectangle(
                    (period_start_x, period_top),  # bottom-left corner
                    width,
                    height,
                    linewidth=1,
                    edgecolor=col,
                    facecolor=col,
                    alpha=0.5
                )

                plt.gca().add_patch(rect)
                period_idx += 1

            ax.plot([x, x], [y_top, y_bottom],
                    color=color, alpha=alpha, linewidth=1)

        # draw bbox
        plt.plot([x0, x1], [y0, y0], color='red', linewidth=0.5)
        plt.plot([x0, x1], [y1, y1], color='red', linewidth=0.5)
        plt.plot([x0, x0], [y0, y1], color='red', linewidth=0.5)
        plt.plot([x1, x1], [y0, y1], color='red', linewidth=0.5)
    
    plt.show()