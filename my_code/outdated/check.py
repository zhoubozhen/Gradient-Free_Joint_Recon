#!/usr/bin/env python3
import h5py
import numpy as np
from pathlib import Path

BASE = Path(
    "/shared/anastasio-s3/CommonData/bozhen/"
    "gfjr_data/kevin_rsaas/KH250727_JRwithAubry/vit_data"
)

FILES = {
    "full": BASE / "nhp_3_nsp.hdf5",
    "3ppw": BASE / "nhp_3_nsp_3ppw.hdf5",
}

NT = 4800   # <<< 已确认

def analyze_per_receiver(fname):
    with h5py.File(fname, "r") as f:
        p = f["forward"][:].astype(np.float32)
        attrs = dict(f["forward"].attrs)

    ntot = p.size
    assert ntot % NT == 0, "total size not divisible by Nt"
    nrec = ntot // NT

    # === 关键：reshape 为 (Nt, Nrec) ===
    p = p.reshape(NT, nrec)

    # FFT along TIME axis (axis=0)
    fft = np.fft.rfft(p, axis=0)
    psd = np.abs(fft) ** 2

    dt = attrs.get("dt", None)
    if dt is None:
        freqs = np.fft.rfftfreq(NT)
        dt_info = "N/A"
    else:
        freqs = np.fft.rfftfreq(NT, d=dt)
        dt_info = dt

    nyq = freqs[-1]
    hf_mask = freqs > 0.6 * nyq

    psd_sum = psd.sum(axis=0)
    centroid = (psd * freqs[:, None]).sum(axis=0) / psd_sum
    hf_ratio = psd[hf_mask, :].sum(axis=0) / psd_sum

    return {
        "nrec": nrec,
        "dt": dt_info,
        "centroid": centroid,
        "hf_ratio": hf_ratio,
    }

print("\n=========== PER-RECEIVER (Nt, Nrec) CHECK ===========\n")

results = {}
for tag, fn in FILES.items():
    print(f"[{tag}] {fn}")
    r = analyze_per_receiver(fn)
    results[tag] = r

    print(f"  Nt            : {NT}")
    print(f"  Nrec          : {r['nrec']}")
    print(f"  dt            : {r['dt']}")
    print(f"  centroid mean : {r['centroid'].mean():.5e}")
    print(f"  centroid med  : {np.median(r['centroid']):.5e}")
    print(f"  centroid max  : {r['centroid'].max():.5e}")
    print(f"  hf_ratio mean : {r['hf_ratio'].mean():.5e}")
    print(f"  hf_ratio med  : {np.median(r['hf_ratio']):.5e}")
    print(f"  hf_ratio max  : {r['hf_ratio'].max():.5e}")

    idx = np.argsort(r["hf_ratio"])[-5:][::-1]
    print("  Top-5 HF-heavy receivers:")
    for i in idx:
        print(f"    rec {i:5d}: hf_ratio={r['hf_ratio'][i]:.5e}, "
              f"centroid={r['centroid'][i]:.5e}")
    print()

print("================ FINAL JUDGEMENT ================\n")

full = results["full"]
ppw3 = results["3ppw"]

print(f"Mean centroid ratio : {ppw3['centroid'].mean() / full['centroid'].mean():.3f}")
print(f"Mean hf_ratio ratio : {ppw3['hf_ratio'].mean() / full['hf_ratio'].mean():.3f}")

print("\n=================================================\n")
