# 接口签名(黑盒红队可见的全部实现信息)

> 只有签名与类型,没有函数体。黑盒红队禁止请求/推测实现。

```python
# src/kernels.py (演示桩)
def shake_dof(n_atoms: int, n_constraints: int) -> int: ...
def virial_pressure(volume: float, kinetic: float, virial: float) -> float: ...
def pme_self_exclusion(net_q2: float, alpha: float) -> float: ...
def net_charge_finite_size_correction(q_net: float, box_L: float, alpha: float) -> float: ...
```

测试加载约定(黑盒测试必须遵守):

```python
import kernels   # harness 保证 sys.path 指向被测实现
def test_<name>() -> None: ...   # 纯函数,断言来自 spec.md
```
