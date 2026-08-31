#!/usr/bin/env python3
"""Q-006 实测定标(spec §5.2/§14.1):邻居列表重建间隔 / 每 step 越界概率 / 每 ns 重建次数。

**仪器选择(E0a 纪律)**:轨迹用 OpenMM(CPU 平台)积分,A1 已证 opus 参考
实现与其能量/力逐项一致(rel<1e-10);Q-006 是物理定标(稠密水扩散位移统计),
不是引擎正确性比较,故用外部仪器。opus 引擎 direct_space 为逐对 Python 循环,
N=582 单次力计算 23.7 s ⇒ 100 ps 需 ~30 天,不可用(实测,已登记)。

方法:
  - 体系:TIP3P-FB 刚性水(HBonds+rigidWater),PME α=0.35,rc=1.0(生产同参)
  - 积分:LangevinMiddleIntegrator(BAOAB 族,与生产 §6.1 同族),γ=1/ps,T=300
  - 臂:dt=4 fs+HMR×3(生产,Q-011);dt=2 fs 无 HMR(步长对照)
  - box ∈ {1.8, 2.4, 3.0}³ nm(ln N 极值标度)
  - episode:每 2 ps 重置原点;per-step 记录 max |x−x_ref|(MIC)与平均 MSD
  - 事后:任意 d 的首穿 renewal 间隔 / 每 ns 重建次数 / MSD→D(原位测定)
  - K 窗口佐证:max_disp(100 fs / 200 fs)分布 → flag 触发率 vs s_window=0.12

输出:results/<tag>.npz + <tag>.json 摘要。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import openmm
import openmm.app
from openmm import unit as u

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)
BOX_V = None  # per-run orthogonal box (nm)


def build_water(box_nm: float):
    import io
    pdb = openmm.app.PDBFile(io.StringIO(
        "HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O\n"
        "HETATM    2  H1  HOH A   1       0.096   0.000   0.000  1.00  0.00           H\n"
        "HETATM    3  H2  HOH A   1      -0.024   0.093   0.000  1.00  0.00           H\n"
        "END\n"))
    ff = openmm.app.ForceField("amber14/tip3pfb.xml")
    m = openmm.app.Modeller(pdb.topology, pdb.positions)
    m.addSolvent(ff, model="tip3p", boxSize=openmm.Vec3(box_nm, box_nm, box_nm) * u.nanometer)
    sys_ = ff.createSystem(m.topology, nonbondedMethod=openmm.app.PME,
                           nonbondedCutoff=1.0 * u.nanometer,
                           constraints=openmm.app.HBonds, rigidWater=True)
    nb = [f for f in sys_.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    g = max(16, int(2 ** np.ceil(np.log2(box_nm / 0.1))))
    # α = 0.35 Å⁻¹ = 3.5 nm⁻¹(AMBER 标准;erfc(3.5×1.0)≈7e-6,截断误差可略)。
    # 注意:M0.5 fixture 写 0.35/nm 是单位错位(10×),erfc(0.35)=0.62,直空间静电
    # 在 rc 处被截 62% ⇒ 水锁死不扩散(2026-08-28 实测登记,见 docs/m0/q006_rebuild.md)
    nb.setPMEParameters(3.5 / u.nanometer, g, g, g)
    pos = np.array(m.positions.value_in_unit(u.nanometer))
    return sys_, pos, (g, g, g)


def hmr(sys_, factor=3.0):
    n = 0
    for i in range(sys_.getNumParticles()):
        m = sys_.getParticleMass(i).value_in_unit(u.dalton)
        if 0.5 < m < 2.0:
            sys_.setParticleMass(i, m * factor * u.dalton)
            n += 1
    return n


def run(tag, box, dt_fs, use_hmr, equil_ps, meas_ps, episode_ps=5.0, seed=7):
    t0 = time.time()
    sys_, pos, grid = build_water(box)
    n_h = hmr(sys_) if use_hmr else 0
    integ = openmm.LangevinMiddleIntegrator(300 * u.kelvin, 1.0 / u.picosecond,
                                            dt_fs * u.femtosecond)
    integ.setRandomNumberSeed(seed)
    dt = dt_fs * 1e-3  # ps
    plat = openmm.Platform.getPlatformByName("CPU")
    ctx = openmm.Context(sys_, integ, plat)
    ctx.setPositions(pos * u.nanometer)
    # G3b 纪律:未弛豫初始构象 max|F| 超门 ⇒ 先最小化(spec §12 G3b)
    openmm.LocalEnergyMinimizer.minimize(ctx, 1.0, 200)
    ctx.setVelocitiesToTemperature(300 * u.kelvin)
    # 熔化预备:10 ps @2fs(最小化后淬火态的结极弛豫),再切目标步长
    if dt_fs != 2.0:
        integ.setStepSize(2.0 * u.femtosecond)
        integ.step(int(10.0 / 2e-3))
        integ.setStepSize(dt_fs * u.femtosecond)
    # 液体自检门:末 16 ps 平衡段的 MSD_O@16ps ≥ 0.10 nm²(玻璃/固态拒绝入库)
    masses_chk = np.array([sys_.getParticleMass(i).value_in_unit(u.dalton)
                           for i in range(sys_.getNumParticles())])
    heavy_chk = masses_chk > 3.5 if use_hmr else masses_chk > 2.5
    xs = []
    for _ in range(8):
        integ.step(int(2.0 / dt))
        xs.append(ctx.getState(getPositions=True).getPositions(asNumpy=True)
                  .value_in_unit(u.nanometer)[heavy_chk])
    m16 = ((np.array(xs)[7] - np.array(xs)[0]) ** 2).sum(-1).mean()
    assert m16 > 0.10, f"liquid sanity gate failed: MSD_O@16ps={m16:.4f} (glass?)"
    rest = int(max(0.0, equil_ps - 16.0 - (10.0 if dt_fs != 2.0 else 0.0)) / dt)
    integ.step(rest)

    L = box
    inv_L = 1.0 / L

    n_steps = int(round(meas_ps / dt))
    ep_len = int(round(episode_ps / dt))       # post-hoc d-grid episodes
    w_len = int(round(0.1 / dt))               # K=25 @4fs 窗口 = 100 fs
    max_d = np.zeros(n_steps)                  # max disp since episode origin
    msd = np.zeros(n_steps)
    mw = np.zeros(n_steps)                     # max disp since window origin
    mw_heavy = np.zeros(n_steps)               # 同上，仅重原子
    mde_heavy = np.zeros(n_steps)              # episode 原点，仅重原子(干净，无中途回置)
    msd_heavy = np.zeros(n_steps)              # O 扩散(episode 原点，仅重原子)
    # 内联三套统计:
    #  (a) 纯 renewal(越过 d_primary 立即重置)⇒ Q-006 重建间隔
    #  (b) 窗口模拟(每 100 fs 重置;窗口内越界 = flag)⇒ Q-006b/c 生产语义
    #  (c) MSD 滚动原点(独立,每 episode 重置)⇒ D 原位测定
    d_primary = 0.05
    masses = np.array([sys_.getParticleMass(i).value_in_unit(u.dalton)
                       for i in range(sys_.getNumParticles())])
    heavy = masses > 2.5  # HMR×3 后水 H = 3.02，仍 < 2.5? 否：HMR 后 H=3.02>2.5
    # 注意：HMR×3 时 H 质量变为 ~3.02，重原子阔值改为 3.5
    heavy = masses > 3.5 if use_hmr else heavy
    x0 = ctx.getState(getPositions=True).getPositions(asNumpy=True) \
        .value_in_unit(u.nanometer).copy()
    x_ref = x0.copy()                          # (a)
    x_ref_h = x0.copy()                        # (a') heavy-only 独立原点
    x_win = x0.copy()                          # (b)
    renew_iv = []                              # (a) intervals in steps
    renew_heavy_iv = []                       # (a') heavy-atom-only renewal
    n_flags = 0                                # (b)
    for s in range(n_steps):
        integ.step(1)
        x = ctx.getState(getPositions=True).getPositions(asNumpy=True) \
            .value_in_unit(u.nanometer)
        d = x - x0
        d -= np.round(d * inv_L) * L
        max_d[s] = np.sqrt((d * d).sum(-1)).max()
        msd[s] = (d * d).sum() / len(d)
        dh = d[heavy]
        mde_heavy[s] = np.sqrt((dh * dh).sum(-1)).max()
        msd_heavy[s] = (dh * dh).sum() / len(dh)
        # (a) pure renewal at d_primary
        dr = x - x_ref
        dr -= np.round(dr * inv_L) * L
        dr_norm = np.sqrt((dr * dr).sum(-1))
        if dr_norm.max() > d_primary:
            renew_iv.append(s)              # absolute step of rebuild event
            x_ref = x.copy()
        # (a') heavy-atom renewal:独立参考点 x_ref_h
        drh = x - x_ref_h
        drh -= np.round(drh * inv_L) * L
        dr_heavy_max = np.sqrt((drh[heavy] * drh[heavy]).sum(-1)).max() \
            if heavy.any() else 0.0
        if dr_heavy_max > d_primary:
            renew_heavy_iv.append(s)
            x_ref_h = x.copy()
        # (b) window simulation: resync at 100 fs boundaries; mid-window
        #     crossing = C1 flag event (window would abort+rollback)
        dw = x - x_win
        dw -= np.round(dw * inv_L) * L
        mw[s] = np.sqrt((dw * dw).sum(-1)).max()
        mw_heavy[s] = np.sqrt((dw[heavy] * dw[heavy]).sum(-1)).max() \
            if heavy.any() else 0.0
        w_crossed = mw[s] > d_primary
        at_boundary = (s + 1) % w_len == 0
        if w_crossed:
            n_flags += 1
        if w_crossed or at_boundary:
            x_win = x.copy()
        if (s + 1) % ep_len == 0:
            x0 = x.copy()
    # renewal 间隔 = 相邻重建事件的绝对步号差(首个事件从 0 计)
    renew_steps = np.diff([0] + renew_iv) if renew_iv else np.array([])
    renew_heavy_steps = (np.diff([0] + renew_heavy_iv)
                         if renew_heavy_iv else np.array([]))
    wall = time.time() - t0
    np.savez_compressed(RESULTS / f"{tag}.npz", max_d=max_d, msd=msd, mw=mw,
                        mw_heavy=mw_heavy, mde_heavy=mde_heavy,
                        msd_heavy=msd_heavy,
                        dt_fs=dt_fs, ep_len=ep_len, w_len=w_len,
                        n_atoms=sys_.getNumParticles(),
                        n_heavy=int(heavy.sum()),
                        box=box, grid=grid, hmr=use_hmr, seed=seed,
                        renew_steps=renew_steps,
                        renew_heavy_steps=renew_heavy_steps,
                        n_flags=n_flags,
                        d_primary=d_primary, openmm=openmm.__version__)
    summary = dict(tag=tag, n_atoms=sys_.getNumParticles(), dt_fs=dt_fs,
                   hmr=use_hmr, n_hmr=n_h, meas_ps=meas_ps, episode_ps=episode_ps,
                   equil_ps=equil_ps, wall_min=wall / 60,
                   n_renew=int(len(renew_steps)), n_flags=n_flags,
                   openmm=openmm.__version__)
    print(json.dumps(summary), flush=True)
    (RESULTS / f"{tag}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--box", type=float, default=2.4)
    p.add_argument("--dt", type=float, default=4.0)
    p.add_argument("--hmr", type=int, default=1)
    p.add_argument("--equil-ps", type=float, default=20.0)
    p.add_argument("--meas-ps", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.equil_ps, a.meas_ps = 2.0, 4.0
    run(a.tag, a.box, a.dt, bool(a.hmr), a.equil_ps, a.meas_ps, seed=a.seed)
