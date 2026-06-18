"""
ConvNeXt-V2 classifier for 6 bus-component classes (Colab).

Method: frozen ConvNeXt-V2 embeddings -> logistic-regression head. Sweeps weight
variants x seeds, keeps the BEST (weights + seed), persists it to Drive, reuses
it on later runs. Corruption-safe image loading. Two parts below: CELL 1
(train/sweep) and CELL 2 (predict). Paste each part into its own Colab cell.

Classes (folder names in /content/new_dataset):
  bus_front_side, bus_back_side, Front_windshield,
  back_windshield, luggage_compartment, Jack and Spare_tyre
"""

# ============================================================
#  CELL 1 — train / sweep / save best
# ============================================================
import os, time, pickle
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import timm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_DIR      = "/content/new_dataset"
DRIVE_CACHE   = "/content/drive/MyDrive/bus_images/cache6_cnx"
CLASS_FOLDERS = ["bus_front_side", "bus_back_side", "Front_windshield",
                 "back_windshield", "luggage_compartment", "Jack and Spare_tyre"]
VARIANTS      = ["convnextv2_tiny.fcmae_ft_in22k_in1k",
                 "convnextv2_base.fcmae_ft_in22k_in1k"]
SEEDS         = [42, 123, 2024]
IMG_SIZE      = 224
VAL_FRAC, TEST_FRAC = 0.15, 0.15
SELECT_BY     = "accuracy"
FORCE_RESWEEP = False

os.makedirs(DRIVE_CACHE, exist_ok=True)
BEST_OUT = f"{DRIVE_CACHE}/convnextv2_BEST_6cls.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device)

if os.path.exists(BEST_OUT) and not FORCE_RESWEEP:
    obj = pickle.load(open(BEST_OUT, "rb"))
    print("\n========== REUSING SAVED BEST (ConvNeXt-V2, 6 classes) ==========")
    print(f"classes      : {obj['classes']}")
    print(f"BEST weights : {obj['model_name']}")
    print(f"BEST seed    : {obj['seed']}")
    print(f"BEST accuracy: {obj['accuracy']:.4f}  (macro_f1 {obj.get('macro_f1', float('nan')):.4f})")
    print("Set FORCE_RESWEEP=True to run the sweep again.")
else:
    IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(path, size):
        try:
            img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
        except Exception as e:
            print(f"  [skip unreadable] {path} ({e})")
            img = Image.new("RGB", (size, size))
        arr = (np.asarray(img, np.uint8).astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))

    root = Path(DATA_DIR)
    classes = CLASS_FOLDERS[:]
    samples = []
    for lbl, c in enumerate(classes):
        folder = root / c
        assert folder.is_dir(), f"missing folder: {folder}"
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in IMG_EXT and p.is_file():
                samples.append((str(p), lbl))
    y_all = np.array([l for _, l in samples])
    print(f"classes={len(classes)}  total images={len(samples)}")
    for i, c in enumerate(classes):
        print(f"  {c:<25s} {int((y_all==i).sum())}")

    def split_indices(labels, val_frac, test_frac, seed):
        rng = np.random.default_rng(seed); tr, va, te = [], [], []
        for c in np.unique(labels):
            ids = np.where(labels == c)[0]; rng.shuffle(ids)
            n = len(ids); n_te = int(n*test_frac); n_va = int(n*val_frac)
            te += ids[:n_te].tolist(); va += ids[n_te:n_te+n_va].tolist(); tr += ids[n_te+n_va:].tolist()
        return np.array(tr), np.array(va), np.array(te)

    @torch.no_grad()
    def embed_all(variant, batch=32):
        cache_f = Path(f"{DRIVE_CACHE}/emb6_{variant.split('.')[0]}_{IMG_SIZE}.npy")
        if cache_f.exists():
            print(f"  [cache] {cache_f.name}"); return np.load(cache_f)
        model = timm.create_model(variant, pretrained=True, num_classes=0).eval().to(device)
        X = []
        for i in range(0, len(samples), batch):
            chunk = samples[i:i+batch]
            t = torch.stack([preprocess(p, IMG_SIZE) for p, _ in chunk]).to(device)
            v = model(t).cpu().numpy().astype(np.float32)
            v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
            X.append(v)
            if i % (batch*10) == 0:
                print(f"  embedding {variant.split('.')[0]} {min(i+batch,len(samples))}/{len(samples)}", flush=True)
        X = np.concatenate(X); np.save(cache_f, X)
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
        return X

    results, best = [], None
    for variant in VARIANTS:
        print(f"\n===== variant: {variant} =====")
        t0 = time.perf_counter(); X = embed_all(variant)
        print(f"  ready in {(time.perf_counter()-t0)/60:.1f} min")
        for seed in SEEDS:
            tr, va, te = split_indices(y_all, VAL_FRAC, TEST_FRAC, seed)
            clf = LogisticRegression(max_iter=3000, C=1.0)
            clf.fit(X[tr], y_all[tr])
            preds = clf.predict(X[te])
            acc = accuracy_score(y_all[te], preds)
            mf1 = f1_score(y_all[te], preds, average="macro")
            score = acc if SELECT_BY == "accuracy" else mf1
            results.append((variant, seed, acc, mf1))
            print(f"  seed={seed}:  acc={acc:.4f}  macro_f1={mf1:.4f}")
            if best is None or score > best["score"]:
                best = dict(score=score, variant=variant, seed=seed, acc=acc, mf1=mf1, clf=clf, te=te)

    print("\n==================== BEST (ConvNeXt-V2, 6 classes) ====================")
    print(f"BEST weights : {best['variant']}   seed: {best['seed']}")
    print(f"BEST accuracy: {best['acc']:.4f}   macro_f1: {best['mf1']:.4f}")
    Xb = embed_all(best["variant"])
    bp = best["clf"].predict(Xb[best["te"]]); byt = y_all[best["te"]]
    print("\nPer-class report (BEST):")
    print(classification_report(byt, bp, target_names=classes, digits=3, zero_division=0))

    with open(BEST_OUT, "wb") as f:
        pickle.dump({"clf": best["clf"], "classes": classes, "model_name": best["variant"],
                     "seed": best["seed"], "img_size": IMG_SIZE,
                     "accuracy": best["acc"], "macro_f1": best["mf1"]}, f)
    print(f"\nsaved BEST model -> {BEST_OUT}")


# ============================================================
#  CELL 2 — predict one image using the saved best
# ============================================================
def predict(IMAGE_PATH, MODEL_PATH="/content/drive/MyDrive/bus_images/cache6_cnx/convnextv2_BEST_6cls.pkl"):
    import pickle, numpy as np, torch
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    import timm
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj = pickle.load(open(MODEL_PATH, "rb"))
    clf, classes, model_name, img_size = obj["clf"], obj["classes"], obj["model_name"], obj["img_size"]
    print(f"using BEST weights={model_name}  seed={obj['seed']}  (acc {obj['accuracy']:.3f})")
    backbone = timm.create_model(model_name, pretrained=True, num_classes=0).eval().to(device)
    img = Image.open(IMAGE_PATH).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = (np.asarray(img, np.uint8).astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).unsqueeze(0).to(device)
    with torch.no_grad():
        v = backbone(x).cpu().numpy().astype(np.float32)
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    probs = clf.predict_proba(v)[0]
    best = int(probs.argmax())
    print(f"\nImage: {IMAGE_PATH}")
    print(f"PREDICTION: {classes[best]}   (confidence {probs[best]*100:.1f}%)\n")
    for i in probs.argsort()[::-1]:
        print(f"  {classes[i]:<25s} {probs[i]*100:6.2f}%")


if __name__ == "__main__":
    # In Colab: run CELL 1, then:  predict("/content/new_dataset/Front_windshield/IMG.jpeg")
    pass
