import os
import json
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.linalg import qr, lstsq
from scipy.ndimage import affine_transform

from tranPACT import TranPACTModel, TranPACTWaveSolver


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rotation_about_axis(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    c_vec: np.ndarray,
    R_vec: np.ndarray,
    angle: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    r_in = np.array([x, y, z])
    c_vec = c_vec.reshape(3, 1)
    r_in_shifted = r_in - c_vec

    qr_matrix = np.array([
        [R_vec[0], 1, 1],
        [R_vec[1], 0, 0],
        [R_vec[2], 0, 0],
    ])
    Q, _ = qr(qr_matrix)

    vx = Q[:, 1]
    vy = Q[:, 2]
    vz = R_vec.flatten() / np.linalg.norm(R_vec)

    basis_matrix = np.stack([vx, vy, vz], axis=1)
    if np.linalg.det(basis_matrix) < 0:
        vx = -vx
        basis_matrix = np.stack([vx, vy, vz], axis=1)

    Rz = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1],
    ])
    P = basis_matrix
    M = P @ Rz @ P.T

    r_out_shifted = M @ r_in_shifted
    r_out = r_out_shifted + c_vec

    xr = r_out[0, :].reshape(-1, 1)
    yr = r_out[1, :].reshape(-1, 1)
    zr = r_out[2, :].reshape(-1, 1)
    return xr, yr, zr


def load_h5_variable(filepath: str, var_name: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with h5py.File(filepath, "r") as f:
        if var_name not in f:
            raise KeyError(f"Variable '{var_name}' not found in {filepath}")
        data = f[var_name][()]
        data = data.transpose(2, 1, 0)
        return data


def generate_nhp_data_and_pos(file_prefix: str, directory: str) -> Tuple[np.ndarray, np.ndarray]:
    rec_map_file = os.path.join(directory, "info/rec_map.mat")
    rec_map_contents = loadmat(rec_map_file)

    id_rearrange = rec_map_contents["id_rearrange"].flatten()
    trans_select = rec_map_contents["trans_select"].flatten().astype(bool)
    Ax = rec_map_contents["Ax"].flatten().astype(float)
    PointOnAx = rec_map_contents["PointOnAx"].flatten().astype(float)
    transducer_x_ms = rec_map_contents["transducer_x_ms"].flatten().astype(float)
    transducer_y_ms = rec_map_contents["transducer_y_ms"].flatten().astype(float)
    transducer_z_ms = rec_map_contents["transducer_z_ms"].flatten().astype(float)

    angle_file = os.path.join(directory, file_prefix + ".csv")
    the_angle_df = pd.read_csv(angle_file, header=None)
    the_angle = the_angle_df.values.flatten()
    theta = the_angle[10:110] / 180.0 * np.pi
    steps = theta.size

    n_rot = transducer_x_ms.size
    total_size = steps * n_rot

    transducer_x_r = np.zeros((total_size, 1), dtype=np.float64)
    transducer_y_r = np.zeros((total_size, 1), dtype=np.float64)
    transducer_z_r = np.zeros((total_size, 1), dtype=np.float64)

    for k in range(steps):
        xr, yr, zr = rotation_about_axis(
            transducer_x_ms,
            transducer_y_ms,
            transducer_z_ms,
            PointOnAx,
            Ax,
            theta[k],
        )
        s = k * n_rot
        e = (k + 1) * n_rot
        transducer_x_r[s:e] = xr
        transducer_y_r[s:e] = yr
        transducer_z_r[s:e] = zr

    rec_pos = np.concatenate((transducer_x_r, transducer_y_r, transducer_z_r), axis=1)
    rec_pos *= 1000.0
    rec_pos -= np.min(rec_pos, axis=0) - np.array([1.5, 1.5, 1.5])

    data_file = os.path.join(directory, file_prefix + ".mat")
    voltage_raw = load_h5_variable(data_file, "VOLTAGE")

    sinogram_slice = voltage_raw[:, :, 10:110]
    reorder_indices = id_rearrange.astype(int) - 1
    sinogram_reorder = sinogram_slice[reorder_indices, :, :]
    sinogram_select = sinogram_reorder[trans_select, :, :]
    sinogram_permute = np.transpose(sinogram_select, (1, 2, 0))
    dim1_size = sinogram_permute.shape[0]
    forward = sinogram_permute.reshape(dim1_size, -1)

    print(f"Final rec_pos shape: {rec_pos.shape}", flush=True)
    print(f"Final forward shape: {forward.shape}", flush=True)
    return rec_pos, forward


def load_exp_data_from_config(cfg: dict) -> dict:
    exp_cfg = dict(cfg.get("exp", {}) or {})

    prefix = exp_cfg["prefix"]
    folder = exp_cfg["folder"]
    model_mode = int(exp_cfg.get("model_mode", 0))
    ct_path = exp_cfg["ct_path"]
    model_path = exp_cfg["model_path"]
    sample_delay = int(exp_cfg.get("sample_delay", 16))
    fs = float(exp_cfg.get("fs", 20.0))
    gridsize = float(exp_cfg.get("gridsize", 0.5))

    rec_pos, forward_raw = generate_nhp_data_and_pos(file_prefix=prefix, directory=folder)

    if model_mode == 0:
        skull_model = np.zeros((556, 556, 300), dtype=np.float32)
    else:
        with h5py.File(model_path, "r") as f:
            skull_model = f["mask"][()]

    ct_skull = None
    if model_mode == 2:
        with h5py.File(ct_path, "r") as f:
            ct_skull = f["ct"][()]

    shape = tuple(skull_model.shape)
    forward = forward_raw[sample_delay:, :]
    dt = 1.0 / fs
    Nt = forward.shape[0]
    opt_roi = np.ones(shape, dtype=np.float32)

    if model_mode == 2:
        fn = np.array(
            [[1e3, 2146.225, 0, 0],
             [1.8e3, 10434.0, 3528.0, 0.5]],
            dtype=np.float32,
        )
    else:
        fn = np.array(
            [[1e3, 2265.025, 0, 0],
             [1.8e3, 9738.0, 3280.5, 0.5]],
            dtype=np.float32,
        )

    return {
        "prefix": prefix,
        "folder": folder,
        "model_mode": model_mode,
        "skull_model": skull_model,
        "ct_skull": ct_skull,
        "rec_pos": rec_pos.astype(np.float32),
        "forward": forward.astype(np.float32),
        "dt": dt,
        "Nt": Nt,
        "gridsize": gridsize,
        "shape": shape,
        "opt_roi": opt_roi,
        "fn": fn,
    }


def setup_wave_solver_from_config(cfg: dict, data_dict: dict):
    exp_cfg = dict(cfg.get("exp", {}) or {})
    nbl = int(exp_cfg.get("nbl", 10))
    to = float(exp_cfg.get("to", 2.0))

    DTYPE = np.float32
    origin = (0, 0, 0)
    spacing = (data_dict["gridsize"], data_dict["gridsize"], data_dict["gridsize"])
    fn = data_dict["fn"]
    model_mode = int(data_dict["model_mode"])

    cl_start = np.sqrt((fn[:, 1] + 4 * fn[:, 2] / 3) / fn[:, 0])
    cs_start = np.sqrt(fn[:, 2] / fn[:, 0])

    if model_mode == 2:
        model = TranPACTModel(
            space_order=10,
            medium_param=fn,
            water_index=0,
            ct_data=data_dict["ct_skull"].astype(DTYPE),
            cor_length=1,
            medium_mode="aubry",
            origin=origin,
            opt_roi=data_dict["opt_roi"],
            shape=data_dict["shape"],
            spacing=spacing,
            nbl=nbl,
            dtype=DTYPE, use_static=False
        )
        skull_type = "aubry"
    else:
        model = TranPACTModel(
            space_order=10,
            medium_geometry=data_dict["skull_model"].astype(DTYPE),
            medium_param=fn,
            origin=origin,
            opt_roi=data_dict["opt_roi"],
            water_index=0,
            shape=data_dict["shape"],
            spacing=spacing,
            nbl=nbl,
            dtype=DTYPE,
        )
        skull_type = "1p1l"

    solver = TranPACTWaveSolver(
        model,
        rec_pos=data_dict["rec_pos"],
        Nt=data_dict["Nt"],
        dt=data_dict["dt"],
        to=to,
    )
    return solver, model, cl_start, cs_start, fn, skull_type


def build_saving_dir(cfg: dict, skull_type: str) -> str:
    paths = dict(cfg.get("paths", {}) or {})
    exp_cfg = dict(cfg.get("exp", {}) or {})
    fista_cfg = dict(cfg.get("fista", {}) or {})

    saving_root = paths["saving_root"]
    prefix = exp_cfg["prefix"]
    reg_param = float(fista_cfg.get("reg", 0.0))

    saving_dir = os.path.join(saving_root, prefix)
    if skull_type == "aubry":
        saving_dir += "_aubry"
    else:
        saving_dir += "_1p1l"

    if reg_param != 0:
        tmp_name = "r%.0e" % reg_param
        reg_name = tmp_name[:-2] + tmp_name[-1]
    else:
        reg_name = "nr"

    saving_dir += "_" + reg_name
    saving_dir += "/"
    os.makedirs(saving_dir, exist_ok=True)
    return saving_dir


def find_affine_transformation_scaled(
    r_source: np.ndarray,
    r_target: np.ndarray,
    source_dx: float,
    target_dx: float,
) -> np.ndarray:
    N = r_source.shape[0]
    A = np.hstack((r_source, np.ones((N, 1))))
    B = r_target
    M_T, _, _, _ = lstsq(A, B)
    M = np.vstack((M_T.T, np.array([0, 0, 0, 1])))
    return M


def transform_coordinates(coords_n3: np.ndarray, M_index_to_index: np.ndarray) -> np.ndarray:
    N = coords_n3.shape[0]
    coords_h = np.hstack((coords_n3, np.ones((N, 1))))
    coords_prime_h_T = M_index_to_index @ coords_h.T
    coords_prime = coords_prime_h_T[:3, :].T / coords_prime_h_T[3, :].T[:, None]
    return coords_prime


def transform_volume_with_scaling(volume_3d: np.ndarray, M_index_to_index: np.ndarray):
    old_shape = np.array(volume_3d.shape)

    corners_old = np.array([
        [0, 0, 0], [old_shape[0], 0, 0], [0, old_shape[1], 0], [0, 0, old_shape[2]],
        [old_shape[0], old_shape[1], 0], [old_shape[0], 0, old_shape[2]],
        [0, old_shape[1], old_shape[2]], [old_shape[0], old_shape[1], old_shape[2]],
    ])

    M_inverse = np.linalg.inv(M_index_to_index)
    M_resample = M_inverse[:3, :3]
    T_resample = M_inverse[:3, 3]

    corners_new = transform_coordinates(corners_old, M_index_to_index)
    min_coords = np.floor(np.min(corners_new, axis=0)).astype(int)
    max_coords = np.ceil(np.max(corners_new, axis=0)).astype(int)
    new_shape = (max_coords - min_coords).astype(int)

    T_resample_adjusted = T_resample + M_resample @ min_coords

    resampled_volume = affine_transform(
        volume_3d,
        matrix=M_resample,
        offset=T_resample_adjusted,
        output_shape=new_shape,
        order=1,
        mode="constant",
        cval=0.0,
    )

    return resampled_volume, min_coords
