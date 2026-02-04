# Our Project Explained — From Zero (Simple Version)

This document explains **what this project is**, **what we did so far**, and **what you have in hand right now** — in the simplest words possible.

---

## 1. What Problem Are We Solving?

Imagine a **factory** that makes bottles, leather bags, or electronic parts. Before shipping, someone has to **check each item** to see if it has a **defect** (crack, scratch, wrong color, missing part, etc.).

- **Humans doing this** = slow, expensive, and they get tired and miss defects.
- **Teaching a computer with lots of photos of defects** = in real factories you **don’t have many defect photos** (defects are rare), so that’s hard.

So the question is:

> **Can we teach the computer using ONLY photos of GOOD (normal) products, and still detect when something is BAD (defective)?**

That’s exactly what this project does. We never show the computer “this is a defect.” We only show “this is normal.” The computer **learns what “normal” looks like** and then **flags anything that doesn’t look normal** as a possible defect. That’s called **unsupervised anomaly detection**.

---

## 2. What Is “Anomaly” and “Anomaly Detection”?

- **Normal** = the usual, good product (e.g. a perfect bottle).
- **Anomaly** = something unusual, not normal (e.g. a broken or dirty bottle).

**Anomaly detection** = the system looks at a new image and says: “This looks normal” or “This looks like an anomaly (maybe a defect).”

We do this **without** ever training on defect images — only on normal ones. So it’s **unsupervised** in the sense that we don’t need defect labels.

---

## 3. What Is This Project About? (One Paragraph)

This project is about building an **AI system** that:

1. **Learns** what “normal” looks like using **only images of good products** (no defect images in training).
2. **Detects anomalies** by checking if a new image is “far from normal” or “close to normal.”
3. Uses a special method called **I-JEPA** (a type of self‑supervised learning) to learn good **semantic** representations (meaning the AI understands “what the object is,” not just pixels). We will compare this with a simpler **baseline** method (autoencoder) to show that I-JEPA is better.

So in short: **Unsupervised anomaly detection using deep learning — train on normal only, detect defects without defect labels.**

---

## 4. What Dataset Are We Using?

We use **MVTec Anomaly Detection** — a standard dataset used in industry and research.

- It has **15 types of objects**: bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper.
- For each type:
  - **Training set**: only **good (normal)** images. We are **not allowed** to use any defect image here. That’s the main rule.
  - **Test set**: mix of **good** and **defective** images. We use this only to **evaluate** how well our system detects anomalies (we don’t train on these).

So: **train = only normal**, **test = normal + defects** (only for measuring performance).

---

## 5. What Did We Do So Far? (Phases 0, 1, 2)

### Phase 0 — Setup (Environment)

- Installed **Python**, created a **virtual environment** (`ijepa_env`), and installed libraries: **PyTorch** (deep learning), **NumPy**, **Pandas**, **Matplotlib**, **scikit-learn**, **OpenCV**, etc.
- Checked that **GPU (CUDA)** works so training can run faster.
- **Deliverable**: A working environment where we can run all the code.

---

### Phase 1 — Understand the Data

- **Downloaded and extracted** the MVTec dataset into the `data/` folder.
- **Built data loaders** so the program can:
  - Load images,
  - Resize them to 224×224,
  - Normalize them (same way as ImageNet, so we can use standard models later).
- Added a **patchify** step: we cut each image into small patches (like a grid). This will be used later for I-JEPA (which works on patches).
- **Explored the data**: looked at how many images per category, what kind of defects exist (e.g. for leather: color, cut, fold, glue, poke).
- **Deliverable**: Dataset ready to use, code to load it, and a clear rule: **train only on normal (train/good)**.

---

### Phase 2 — Baseline (Autoencoder)

Before building the fancy I-JEPA method, we built a **simple baseline** so we can later say “I-JEPA is better than this.”

- **What is the baseline?**  
  An **autoencoder**: a network that tries to **reconstruct** the image (input → compress → reconstruct). It is trained **only on normal images**.  
  Idea: **Normal images** → low reconstruction error. **Defect images** → higher error (because the network has never “seen” defects, so it reconstructs them poorly). So **reconstruction error = anomaly score**.

- **What we implemented:**
  1. **Autoencoder model** (`model_autoencoder.py`): takes 224×224 image → encodes to a small “latent” vector → decodes back to 224×224.
  2. **Training script** (`train_baseline.py`): trains this autoencoder **only on normal images** (e.g. leather, bottle).
  3. **Evaluation script** (`anomaly_eval.py`): for each test image, we compute **reconstruction error** as the anomaly score, then we compare with the true labels (normal vs defect) and compute **ROC-AUC** (and other metrics). Higher ROC-AUC = better at separating normal from defect.

- **Deliverable**:  
  - A **trained baseline model** (e.g. `checkpoints/autoencoder_leather.pth`).  
  - **Evaluation results** for that category: ROC-AUC, precision, recall, F1, confusion matrix, and saved scores/labels in `results/baseline/<category>/`.  
  - A **comparison baseline** for later: when we add I-JEPA, we will show that I-JEPA gets **better ROC-AUC** than this autoencoder.

---

## 6. What Do You Have *Right Now* After Two Phases? (Deliverables)

After Phase 0, 1, and 2 you have:

| What | Where / What it is |
|------|--------------------|
| **Working environment** | Python + PyTorch + GPU, virtual env `ijepa_env` |
| **Dataset** | MVTec AD in `data/` — 15 categories, train = normal only, test = normal + defects |
| **Data loading & preprocessing** | `src/datasets.py` — load images, resize, normalize, patchify |
| **Baseline model** | Autoencoder in `src/model_autoencoder.py` |
| **Training (normal only)** | `src/train_baseline.py` — train autoencoder on one category |
| **Evaluation** | `src/anomaly_eval.py` — anomaly score = reconstruction error, then ROC-AUC etc. |
| **One-command run** | `run_baseline.py` — train + evaluate for a category (e.g. leather) |
| **Saved baseline results** | e.g. `results/baseline/leather/metrics_baseline.json`, scores, labels |
| **Saved checkpoint** | e.g. `checkpoints/autoencoder_leather.pth` (and history JSON) |

So the **deliverable so far** is: **a complete pipeline that trains a reconstruction-based anomaly detector on normal images only and evaluates it on MVTec (with ROC-AUC and related metrics), plus a saved baseline to compare I-JEPA against later.**

---

## 7. Simple Flow: From Image to “Normal or Defect?”

**Training (what we did in Phase 2):**

1. Take **only normal** images (e.g. good leather).
2. Train the autoencoder: input image → compress → reconstruct. Goal: reconstruction as close to input as possible.
3. Save the trained model.

**Testing (evaluation):**

1. Take a **test image** (can be normal or defective).
2. Pass it through the autoencoder and get the **reconstructed** image.
3. Compute **reconstruction error** (e.g. mean squared error between input and reconstruction). This is the **anomaly score**.
4. **High error** → likely defect. **Low error** → likely normal.
5. We compare these scores with the true labels and compute **ROC-AUC** (and other metrics) to see how well we separate normal from defect.

---

## 8. What Comes Next? (Phase 3 and Beyond)

- **Phase 3**: Build **I-JEPA** — a more advanced method that learns in **latent space** (semantic features) instead of raw pixels. Train it again **only on normal** images.
- **Phase 4–5**: Use I-JEPA’s **embeddings** (feature vectors) to model “normal” (e.g. with Mahalanobis distance or k‑NN) and compute anomaly scores from **distance to normal**.
- **Phase 6**: **Compare** I-JEPA’s ROC-AUC (and other metrics) with the **autoencoder baseline** we built in Phase 2. The goal is to show that I-JEPA is better.
- Later phases: more evaluation, visualizations, ablation studies, and a **research report / paper**.

---

## 9. One-Sentence Summary of the Whole Project

**We are building an AI that learns “normal” from only good product images, then flags defects by spotting what doesn’t look normal; we first built a simple baseline (autoencoder) and will next build I-JEPA and show it performs better.**

---

## 10. Glossary (Very Short)

- **Anomaly** = something that is not normal (e.g. a defect).
- **Unsupervised** = we don’t use defect labels during training; we only use normal images.
- **MVTec AD** = standard dataset of industrial objects with normal and defect images.
- **Autoencoder** = network that compresses then reconstructs the input; we use **reconstruction error** as anomaly score.
- **Baseline** = simple method we built first so we can compare and prove that I-JEPA is better.
- **ROC-AUC** = a number between 0 and 1 that says how well we separate “normal” from “defect”; higher is better (1 = perfect).
- **I-JEPA** = the advanced method we will use next; it learns semantic representations and predicts in “latent space” instead of pixels.
- **Train on normal only** = we never use defect images for training; that’s the core rule of this project.

If you want, we can next add a **one-page diagram** (flowchart) of “image → model → anomaly score → ROC-AUC” to this document.
