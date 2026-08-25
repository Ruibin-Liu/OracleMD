# 雷区库

每域一个条目。作用:**①变异注入器的种子来源 ②黑盒红队的盲区提示 ③硬编码检查的登记处。**

条目格式:域 id + 为何脆弱 + 已知失效模式 + oracle + `## mutation` 块(注入器解析)
+ `## enforcement`(推荐处置:test / golden / hardcode / no-oracle)。

> 演示阶段公式为简化桩值(标注 DEMO);接入真实代码时替换为有文献/参考实现
> 出处的公式,并保留「适用条件」字段——那是本库最值钱的部分。

| 域 | 条目 |
|---|---|
| shake_settle | 001 |
| virial_pressure | 002 |
| pme_exclusions | 003 |
| net_charge | 004 |
| conditional_approximations | 005(演示 no-oracle) |
