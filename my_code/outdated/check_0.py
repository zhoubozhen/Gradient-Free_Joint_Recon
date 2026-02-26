#!/usr/bin/env python3
import numpy as np
import h5py
import scipy.io as sio

# ==========================================================
# 路径（与你 gfjr_stable.py 一致）
# ==========================================================
nhp_directory = (
    "/shared/anastasio-s3/CommonData/bozhen/gfjr_data/"
    "kevin_rsaas/KH250319_headphantom/data/"
)
directory = (
    "/shared/anastasio-s3/CommonData/bozhen/"
    "gfjr_data/kevin_rsaas/KH250727_JRwithAubry/"
)

pressure_name = "nhp_3_nsp"   # 按你的 .sh
DTYPE = np.float32

# ==========================================================
# 统计参数（可按需改）
# ==========================================================
SAMPLE_MAX = 2_000_000        # 超大数组抽样上限（元素个数）
SAMPLE_SEED = 0
Q = [0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0]  # 分位点
MASK_UNIQUE_MAX = 16          # unique 值超过这个就不全打（防止爆输出）

rng = np.random.default_rng(SAMPLE_SEED)

# ==========================================================
# 工具函数
# ==========================================================
def _bytes_gb(nbytes: int) -> str:
    return f"{nbytes/1e9:.3f} GB"

def info(name, arr):
    print(f"{name:45s} | dtype={arr.dtype} shape={arr.shape} bytes={_bytes_gb(arr.nbytes)}")

def _is_mask_like(arr: np.ndarray) -> bool:
    # bool 一定是 mask；整数且 unique 少也基本是 mask；float 也可能是 0/1 mask
    if arr.dtype == np.bool_:
        return True
    if arr.dtype.kind in ("i", "u"):
        return True
    # float 情况：如果 unique 很少且都在 {0,1}
    if arr.dtype.kind == "f":
        # 只抽样判断
        s = sample_flat(arr)
        u = np.unique(s)
        if len(u) <= 4 and np.all(np.isin(u, [0.0, 1.0])):
            return True
    return False

def sample_flat(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    n = x.size
    if n <= SAMPLE_MAX:
        return x.reshape(-1)
    idx = rng.integers(0, n, size=SAMPLE_MAX, endpoint=False)
    flat = x.reshape(-1)
    return flat[idx]

def stat_numeric(name, arr, force_sample=False):
    x = np.asarray(arr)
    print(f"\n--- {name} ---")
    print(f"dtype={x.dtype} shape={x.shape} bytes={_bytes_gb(x.nbytes)}")

    # 只对数值数组做
    if x.dtype.kind not in ("b", "i", "u", "f"):
        print("non-numeric dtype, skip numeric stats.")
        return

    # 采样 or 全量
    use_sample = force_sample or (x.size > SAMPLE_MAX)
    s = sample_flat(x) if use_sample else x.reshape(-1)

    # nan/inf（整数/bool 没这个概念，float 才需要）
    if s.dtype.kind == "f":
        n_nan = int(np.count_nonzero(np.isnan(s)))
        n_inf = int(np.count_nonzero(np.isinf(s)))
        print(f"sample={use_sample}  sample_n={s.size}  nan={n_nan}  inf={n_inf}")

        ss = s[np.isfinite(s)]
        if ss.size == 0:
            print("all sample values are non-finite.")
            return
    else:
        print(f"sample={use_sample}  sample_n={s.size}")
        ss = s

    # 基本统计
    mn = float(np.min(ss))
    mx = float(np.max(ss))
    mean = float(np.mean(ss))
    std = float(np.std(ss))
    print(f"min={mn:.6g}  max={mx:.6g}  mean={mean:.6g}  std={std:.6g}")

    # 分位数
    qs = np.quantile(ss, Q)
    q_str = "  ".join([f"q{int(q*1000)/10:g}%={v:.6g}" for q, v in zip(Q, qs)])
    print("quantiles:", q_str)

def stat_mask(name, arr):
    x = np.asarray(arr)
    print(f"\n--- {name} (mask stats) ---")
    print(f"dtype={x.dtype} shape={x.shape} bytes={_bytes_gb(x.nbytes)}")

    # mask 统一转 uint8 方便 unique
    if x.dtype == np.bool_:
        y = x.astype(np.uint8, copy=False)
    else:
        # float mask (0/1) 也转
        y = x.astype(np.float32, copy=False)

    # unique + counts（全量做，mask 一般 unique 少）
    uniq, cnt = np.unique(y, return_counts=True)
    print(f"unique values ({len(uniq)}):")
    if len(uniq) <= MASK_UNIQUE_MAX:
        for u, c in zip(uniq, cnt):
            print(f"  value={u!s:>8}  count={int(c):>12}  ratio={c/y.size:.6e}")
    else:
        print("  (too many unique values; showing first 16)")
        for u, c in list(zip(uniq, cnt))[:MASK_UNIQUE_MAX]:
            print(f"  value={u!s:>8}  count={int(c):>12}")

    nnz = int(np.count_nonzero(y))
    print(f"nonzero voxels = {nnz} / {y.size} ({nnz/y.size:.6e})")

# ==========================================================
print("\n================= CHECK INPUTS (recon_opt = 0) =================\n")

# 1) rec_pos
rec_pos = np.fromfile(nhp_directory + "mp_raw.DAT", dtype=DTYPE).reshape(3, -1).T
info("rec_pos (mp_raw.DAT)", rec_pos)
Nrec = rec_pos.shape[0]
print(f"{'Nrec':45s} = {Nrec}")
stat_numeric("rec_pos (mp_raw.DAT)", rec_pos, force_sample=False)

# 2) forward pressure
with h5py.File(directory + f"vit_data/{pressure_name}.hdf5", "r") as f:
    p0_forward = f["forward"][:].astype(DTYPE)

info("p0_forward raw", p0_forward)

if p0_forward.ndim == 1:
    Nt = p0_forward.size // Nrec
    assert Nt * Nrec == p0_forward.size, "pressure size not divisible by Nrec"
    p0_forward = p0_forward.reshape(Nt, Nrec)
elif p0_forward.ndim == 2:
    Nt, _ = p0_forward.shape
else:
    raise ValueError(f"Unexpected pressure ndim={p0_forward.ndim}")

info("p0_forward reshaped", p0_forward)
print(f"{'Nt':45s} = {Nt}")

# pressure 内容统计：默认抽样（太大）
stat_numeric("p0_forward (pressure)", p0_forward, force_sample=False)

# 3) acoustic phantom (1p1l)
mat = sio.loadmat(nhp_directory + "acoustic/nhp_1p1l_vifov.mat")

print("\n================= ACOUSTIC PHANTOM CONTENT =================\n")
print("Loaded keys:", [k for k in mat.keys() if not k.startswith("__")])

# 对每个 3D 场做内容统计（全量统计可能慢，默认抽样）
for k, v in mat.items():
    if k.startswith("__"):
        continue
    if isinstance(v, np.ndarray) and v.ndim >= 2:
        info(f"acnhp['{k}']", v)
        stat_numeric(f"acnhp['{k}']", v, force_sample=False)

# 4) fn (1p1l)
fn = np.fromfile(directory + "param/fn_1p1l_nhp_3.DAT", dtype=DTYPE).reshape(-1, 4)
info("fn_1p1l_nhp_3.DAT", fn)
stat_numeric("fn_1p1l_nhp_3.DAT", fn, force_sample=False)

# 5) ROI files（内容必须全量统计）
print("\n================= ROI CONTENT =================\n")
opt_roi = np.fromfile(nhp_directory + "param/roi_intran.DAT", dtype=DTYPE).reshape(702, 702, 350)
info("opt_roi (roi_intran)", opt_roi)
stat_mask("opt_roi (roi_intran)", opt_roi)

cor_roi = np.load(nhp_directory + "param/roi_cor.npy")
info("cor_roi", cor_roi)
stat_mask("cor_roi", cor_roi)

skull_roi = np.load(nhp_directory + "param/roi_diluted_skull.npy")
info("skull_roi", skull_roi)
stat_mask("skull_roi", skull_roi)

# ROI 关系
print("\n================= ROI RELATION =================\n")
opt_nz = opt_roi != 0
cor_nz = cor_roi != 0
skull_nz = skull_roi != 0
print(f"opt ∩ cor   = {int(np.count_nonzero(opt_nz & cor_nz))}")
print(f"opt ∩ skull = {int(np.count_nonzero(opt_nz & skull_nz))}")
print(f"cor ∩ skull = {int(np.count_nonzero(cor_nz & skull_nz))}")
print(f"opt only    = {int(np.count_nonzero(opt_nz & ~cor_nz & ~skull_nz))}")

# 6) noise（也很大，默认抽样统计）
print("\n================= NOISE CONTENT =================\n")
with h5py.File(nhp_directory + "param/noise_filtered_unit_std.hdf5", "r") as f:
    p0_noise = f["forward"][:].astype(DTYPE)

info("p0_noise raw", p0_noise)

if p0_noise.ndim == 1:
    Nt_noise = p0_noise.size // Nrec
    assert Nt_noise * Nrec == p0_noise.size, "noise size not divisible by Nrec"
    p0_noise = p0_noise.reshape(Nt_noise, Nrec)

info("p0_noise reshaped", p0_noise)
stat_numeric("p0_noise", p0_noise, force_sample=False)

print("\n================= ALL CHECKS DONE =================\n")
