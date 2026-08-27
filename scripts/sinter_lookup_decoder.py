#!/usr/bin/env python3
"""sinter_lookup_decoder.py —— LookupDecoder 的 sinter.Decoder 适配

把几何论自研查表解码器（10.30/10.83）接入 sinter/stim 生态：
- 实现 sinter.Decoder.decode_via_files（文件式 b8 解码）
- dets（bit-packed）→ 我们的 LookupDecoder 查表恢复 → 预测 observable flips
- 可用于 sinter collect（大规模采样 + 自定义解码器统计 p_L），
  以及任何调用 sinter.Decoder 的生态（含 tqec 若其支持 sinter 接口）

b8 格式（官方 result_formats.md）：LSB-first bit-packing，
dets: num_shots × ceil(num_dets/8) 字节；obs: num_shots × ceil(num_obs/8) 字节。

使用（与 sinter collect 集成）：
    import sinter
    decoder = sinter.Decoder(
        lookup_decoder_factory=my_factory)   # sinter 1.16 支持传入工厂？
    实际 sinter 用法：sinter.collect(..., custom_decoders={'lookup': LookupSinterDecoder()})
    其中 LookupSinterDecoder 是 sinter.Decoder 子类（本文件提供）。

运行: .venv311/bin/python scripts/sinter_lookup_decoder.py（自测）
"""
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sinter
import stim

from qecgeo import LookupDecoder
from qecgeo.pauli import Pauli


def _b8_unpack(data: np.ndarray, num_dets: int, num_shots: int) -> np.ndarray:
    """b8 → (num_shots, num_dets) uint8 探测器矩阵。"""
    dets = np.zeros((num_shots, num_dets), dtype=np.uint8)
    n_bytes = (num_dets + 7) // 8
    for s in range(num_shots):
        row = data[s * n_bytes:(s + 1) * n_bytes]
        for b in range(n_bytes):
            byte = int(row[b])
            for bit in range(8):
                idx = b * 8 + bit
                if idx < num_dets:
                    dets[s, idx] = (byte >> bit) & 1
    return dets


def _b8_pack(obs: np.ndarray, num_obs: int, num_shots: int) -> np.ndarray:
    """(num_shots, num_obs) uint8 → b8。"""
    n_bytes = (num_obs + 7) // 8
    out = np.zeros(num_shots * n_bytes, dtype=np.uint8)
    for s in range(num_shots):
        for o in range(num_obs):
            if obs[s, o]:
                out[s * n_bytes + o // 8] |= 1 << (o % 8)
    return out


class LookupSinterDecoder(sinter.Decoder):
    """LookupDecoder 的 sinter 适配：dets → 查表恢复 → observable flip 预测。

    用法：传给 sinter.collect(..., custom_decoders={'lookup': LookupSinterDecoder(...)})。
    构造参数：
      dec_z / dec_x: LookupDecoder 实例（X 错误由 Z 稳定子检测 → dec_z，
                      Z 错误由 X 稳定子检测 → dec_x）
      z_det_indices / x_det_indices: Z/X 稳定子对应的探测器索引
      obs_map_z: 列表，第 k 项 = 逻辑 observable k 的支撑（X 恢复翻转的 obs 集）
      obs_map_x: 列表，第 k 项 = 逻辑 observable k 的支撑（Z 恢复翻转的 obs 集）
    """

    def __init__(self, dec_z=None, dec_x=None,
                 z_det_indices=None, x_det_indices=None,
                 obs_map_z=None, obs_map_x=None):
        self.dec_z = dec_z
        self.dec_x = dec_x
        self.z_det_indices = z_det_indices if z_det_indices is not None else []
        self.x_det_indices = x_det_indices if x_det_indices is not None else []
        self.obs_map_z = obs_map_z if obs_map_z is not None else []
        self.obs_map_x = obs_map_x if obs_map_x is not None else []

    def decode_via_files(self, *, num_shots, num_dets, num_obs,
                         dem_path, dets_b8_in_path, obs_predictions_b8_out_path,
                         tmp_dir):
        """sinter 文件式解码：读 dets，写 obs 预测（b8 格式）。"""
        # 读 dets
        dets_b8 = np.fromfile(dets_b8_in_path, dtype=np.uint8)
        dets = _b8_unpack(dets_b8, num_dets, num_shots)
        # 逐 shot 解码
        nz, nx = len(self.z_det_indices), len(self.x_det_indices)
        obs_pred = np.zeros((num_shots, num_obs), dtype=np.uint8)
        for s in range(num_shots):
            # X 错误 syndrome（Z 稳定子探测器翻转）
            dz = tuple(int(dets[s, self.z_det_indices[k]]) for k in range(nz))
            ez = self.dec_z.decode(dz) if self.dec_z else None
            if ez is not None:
                for o, supp in enumerate(self.obs_map_z):
                    flip = 0
                    for j in supp:
                        if ez.t[j]:
                            flip ^= 1
                    obs_pred[s, o] ^= flip
            # Z 错误 syndrome（X 稳定子探测器翻转）
            dx = tuple(int(dets[s, self.x_det_indices[k]]) for k in range(nx))
            ex = self.dec_x.decode(dx) if self.dec_x else None
            if ex is not None:
                for o, supp in enumerate(self.obs_map_x):
                    flip = 0
                    for j in supp:
                        if ex.t[j]:
                            flip ^= 1
                    obs_pred[s, o] ^= flip
        # 写 obs 预测（b8）
        packed = _b8_pack(obs_pred, num_obs, num_shots)
        packed.tofile(obs_predictions_b8_out_path)


# ---------- 自测 ----------

def _self_test():
    """验证 b8 编解码 + 解码链路（用 AG(4,1) 数据，见 ag_stim_memory）。"""
    import tempfile
    from scripts.ag_stim_memory import build_rounds2, extract_stabilizers
    from scripts.ag_stim_sim import rm_single_monomials

    m, r = 4, 1
    n = 1 << m
    dec_z, dec_x, z_dets, x_dets = extract_stabilizers(m, r)
    # obs 支撑（逻辑 Z = x_1x_2，X 恢复翻 obs）
    I = tuple(range(r + 1))
    lz = [a for a in range(n) if all((a >> (m - 1 - i)) & 1 for i in I)]
    obs_map_z = [lz]  # 逻辑 observable 0 的 X 恢复支撑

    dec = LookupSinterDecoder(dec_z=dec_z, dec_x=dec_x,
                              z_det_indices=list(z_dets), x_det_indices=list(x_dets),
                              obs_map_z=obs_map_z, obs_map_x=[[]])

    # 生成测试数据：stim 采样 + 注入已知错误
    c, _ = build_rounds2(m, r, 0.02, 0.01)
    sampler = c.compile_detector_sampler(seed=0)
    dets, obs = sampler.sample(2000, separate_observables=True)
    num_dets, num_obs = dets.shape[1], 1

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        dets_path = td / "dets.b8"
        out_path = td / "obs.b8"
        dem_path = td / "dem"
        c.detector_error_model(decompose_errors=True).to_file(dem_path)
        # 写 b8
        n_bytes = (num_dets + 7) // 8
        packed = np.zeros(2000 * n_bytes, dtype=np.uint8)
        for s in range(2000):
            for dd in range(num_dets):
                if dets[s, dd]:
                    packed[s * n_bytes + dd // 8] |= 1 << (dd % 8)
        packed.tofile(dets_path)
        # 调用 sinter 接口
        dec.decode_via_files(num_shots=2000, num_dets=num_dets, num_obs=num_obs,
                             dem_path=dem_path, dets_b8_in_path=dets_path,
                             obs_predictions_b8_out_path=out_path, tmp_dir=td)
        pred = np.fromfile(out_path, dtype=np.uint8)
        pred_bits = np.zeros((2000, 1), dtype=np.uint8)
        for s in range(2000):
            pred_bits[s, 0] = pred[s] & 1  # obs 0 在最低位
        # 与手工解码对比
        nz, nx = len(z_dets), len(x_dets)
        manual = np.zeros(2000, dtype=np.uint8)
        for s in range(2000):
            dz = tuple(int(dets[s, z_dets[k]]) for k in range(nz))
            ez = dec_z.decode(dz)
            flip = 0
            if ez is not None:
                for j in lz:
                    if ez.t[j]:
                        flip ^= 1
            manual[s] = flip
        match = np.mean(pred_bits[:, 0] == manual)
        print(f"[自测] sinter 接口 vs 手工解码一致率: {match:.2%}")
        # p_L 对比
        pL_sinter = np.mean((obs[:, 0] ^ pred_bits[:, 0]) != 0)
        print(f"[自测] sinter 接口 p_L = {pL_sinter:.5f}")


if __name__ == "__main__":
    print("LookupSinterDecoder —— LookupDecoder 的 sinter.Decoder 适配（10.83）")
    print("=" * 72)
    _self_test()
