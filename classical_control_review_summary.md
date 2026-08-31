
# 经典控制复习总结：从二阶系统到 Reaction Curve，再到 PID 初步整定

> 本文总结了我们今天从 **mass–spring–damper（二阶系统）**、到 **step response / reaction curve**、再到 **用 K–L–T 近似去理解 PID tuning** 的整条学习路线。  
> 重点不是死记公式，而是建立“**看到一条 reaction curve，就能大致解释系统发生了什么**”的直觉。

---

## 1. 我们一直在研究的直觉例子

我们主要围绕两个类比来理解控制系统：

1. **汽车 cruise control**
   - 目标速度：例如 50 km/h
   - 误差定义：
     \[
     e(t)=r(t)-y(t)
     \]
     其中 \(r\) 是 reference（目标），\(y\) 是输出（当前速度）
   - controller 根据误差决定 throttle / control input

2. **mass–spring–damper（弹簧-质量-阻尼）**
   - 一个质量块连接弹簧，并在水中运动
   - 是经典二阶系统的物理原型

---

## 2. PID 三项的物理直觉

### 2.1 P control

\[
u_P=K_p e
\]

直觉：

- 当前误差大，就更大力地推
- 当前误差小，就轻一点推

优点：

- 简单、直接
- 响应变快

缺点：

- 仅靠 P，通常不能完全消除 steady-state error
- \(K_p\) 太大时容易 overshoot、振荡

---

### 2.2 I control

\[
u_I=K_i\int e(t)\,dt
\]

直觉：

- I 项会“记账”
- 如果误差长期存在，即使每一刻都不大，积分也会慢慢累积出很大的 control action

作用：

- 消除对常值 reference 的 steady-state error

风险：

- \(K_i\) 太大时容易 overshoot、oscillation
- 还可能有 integral windup

---

### 2.3 D control

\[
u_D=K_d\frac{de}{dt}
\]

直觉：

- D 项看的是“误差变化速度”
- 像“提前刹车”

作用：

- 增加 damping
- 减少 overshoot
- 改善瞬态性能

注意：

- D 对高频噪声非常敏感
- 实际工程中通常会加低通滤波，而不是用理想微分

---

## 3. Mass–spring–damper：今天的核心 plant

系统方程：

\[
m\ddot x+b\dot x+kx=F(t)
\]

其中：

- \(m\)：质量（inertia）
- \(b\)：阻尼系数（damping coefficient）
- \(k\)：弹簧刚度（spring constant）
- \(F(t)\)：外部输入（external force）

物理含义：

- \(m\ddot x\)：惯性
- \(b\dot x\)：阻尼
- \(kx\)：弹簧回复力

如果施加一个恒定 step force：

\[
F(t)=F_0
\]

steady state 时：

\[
\dot x=0,\qquad \ddot x=0
\]

所以：

\[
kx_{ss}=F_0
\]

因此：

\[
\boxed{x_{ss}=\frac{F_0}{k}}
\]

---

## 4. Step response / reaction curve：到底在看什么？

Reaction curve 本质上就是：

> 给系统一个已知输入（最常见是 step input），观察输出随时间如何变化。

对 underdamped 二阶系统，曲线常见形状是：

- 先上升
- 冲过头（overshoot）
- 再来回振荡
- 最后收敛到 steady state

---

## 5. Step response 的常见指标

---

### 5.1 Rise time \(t_r\)

通常定义为 response **第一次** 从最终值的 10% 上升到 90% 所需时间：

\[
\boxed{t_r=t_{90}-t_{10}}
\]

直觉：

- 描述“第一次接近目标有多快”

注意：

- 它不等于“完全稳定下来用了多久”

---

### 5.2 Peak time \(t_p\)

定义：

\[
\boxed{t_p=\text{从输入开始到第一次 peak 的时间}}
\]

直觉：

- 第一次冲到最高点用了多久

注意：

- 在 peak 处，速度 \(\dot x=0\)
- 但这并不表示系统已经 steady state，因为通常 \(\ddot x\neq0\)

---

### 5.3 Percent overshoot（百分比超调）

\[
\boxed{
PO=\frac{x_{\text{peak}}-x_{ss}}{x_{ss}}\times100\%
}
\]

直觉：

- 描述“第一次冲过头有多严重”

---

### 5.4 Settling time \(t_s\)

最常用的是 **±2% criterion**：

如果最终值是 \(x_{ss}\)，则容许带为：

\[
0.98x_{ss}\le x(t)\le1.02x_{ss}
\]

settling time 定义为：

> 输出进入这个范围，并且之后不再离开该范围的时刻

常用近似（针对标准二阶 underdamped 系统）：

\[
\boxed{t_s\approx\frac{4}{\zeta\omega_n}}
\]

---

### 5.5 Steady-state error \(e_{ss}\)

对 reference \(r\) 和 steady-state output \(y_{ss}\)：

\[
\boxed{e_{ss}=r-y_{ss}}
\]

如果要写成百分比误差：

\[
\boxed{
\text{percent steady-state error}
=\frac{r-y_{ss}}{r}\times100\%
}
\]

注意区分：

- absolute steady-state error：例如 2
- percent steady-state error：例如 2%

---

## 6. 二阶系统的核心参数：\(\zeta,\omega_n,\omega_d\)

---

### 6.1 Natural frequency \(\omega_n\)

对 mass–spring–damper：

\[
\boxed{\omega_n=\sqrt{\frac{k}{m}}}
\]

直觉：

- \(k\) 越大，弹簧越硬，系统越“想快点拉回来”
- \(m\) 越大，惯性越大，系统越慢

所以：

- \(k\uparrow \Rightarrow \omega_n\uparrow\)
- \(m\uparrow \Rightarrow \omega_n\downarrow\)

---

### 6.2 Damping ratio \(\zeta\)

\[
\boxed{\zeta=\frac{b}{2\sqrt{mk}}}
\]

更深的理解是：

\[
\boxed{\zeta=\frac{b}{b_c}}
\]

其中 critical damping coefficient：

\[
\boxed{b_c=2\sqrt{mk}}
\]

所以 \(\zeta\) 是：

> 实际阻尼 / 临界阻尼

分类：

- \(0<\zeta<1\)：underdamped（会振荡、会 overshoot）
- \(\zeta=1\)：critically damped（不振荡条件下最快）
- \(\zeta>1\)：overdamped（不振荡，但可能变慢）

---

### 6.3 Damped natural frequency \(\omega_d\)

\[
\boxed{\omega_d=\omega_n\sqrt{1-\zeta^2}}
\]

直觉：

- \(\omega_n\) 是“如果没有阻尼，系统天然会以多快振荡”
- \(\omega_d\) 是“有阻尼后，实际观察到的振荡频率”

只要 \(0<\zeta<1\)，就有：

\[
\omega_d<\omega_n
\]

---

## 7. 二阶系统的标准形式

把方程除以 \(m\)：

\[
\ddot x+\frac{b}{m}\dot x+\frac{k}{m}x=\frac{F(t)}{m}
\]

齐次部分对应标准二阶形式：

\[
\boxed{
\ddot x+2\zeta\omega_n\dot x+\omega_n^2x=0
}
\]

对应关系：

\[
\boxed{\omega_n^2=\frac{k}{m}}
\]

\[
\boxed{2\zeta\omega_n=\frac{b}{m}}
\]

---

## 8. Pole（极点）与 reaction curve 的关系

标准二阶 underdamped 系统的 poles：

\[
\boxed{
s=-\zeta\omega_n\pm j\omega_d
}
\]

也就是：

\[
\boxed{
s=-\zeta\omega_n\pm j\,\omega_n\sqrt{1-\zeta^2}
}
\]

### 8.1 实部决定什么？

pole 的实部：

\[
-\zeta\omega_n
\]

控制的是指数衰减 envelope：

\[
e^{-\zeta\omega_n t}
\]

直觉：

- 实部越负（离 imaginary axis 越远）
- 衰减越快
- settling 越快

---

### 8.2 虚部决定什么？

pole 的虚部：

\[
\omega_d
\]

控制振荡快慢：

- 虚部绝对值越大
- 振荡越快
- peak 来得越早

---

### 8.3 Pole 到原点的距离为什么等于 \(\omega_n\)？

如果 pole 是：

\[
s=-\zeta\omega_n+j\omega_d
\]

则：

\[
|s|=\sqrt{(\zeta\omega_n)^2+\omega_d^2}=\omega_n
\]

所以：

\[
\boxed{|s|=\omega_n}
\]

这说明：

- pole 离原点的距离反映 \(\omega_n\)
- pole “朝左的程度”反映 \(\zeta\)

---

### 8.4 Critical damping 和 overdamping 的 poles

- \(\zeta=1\) 时：

\[
s^2+2\omega_n s+\omega_n^2=(s+\omega_n)^2
\]

所以是重复实根：

\[
\boxed{s_1=s_2=-\omega_n}
\]

- \(\zeta>1\) 时：
  - 两个不同负实根
  - 没有振荡
  - 通常由靠近 imaginary axis 的那个慢 pole 主导长期行为

---

## 9. 一些重要近似关系

### Peak time

\[
\boxed{
t_p=\frac{\pi}{\omega_d}
}
\]

### Damped oscillation period

\[
\boxed{
T_d=\frac{2\pi}{\omega_d}
}
\]

因此：

\[
\boxed{
t_p=\frac{T_d}{2}
}
\]

### Percent overshoot 与 \(\zeta\)

标准二阶 underdamped 系统：

\[
\boxed{
PO=e^{-\frac{\zeta\pi}{\sqrt{1-\zeta^2}}}\times100\%
}
\]

所以：

- \(\zeta\uparrow \Rightarrow PO\downarrow\)

### Settling time（2% 近似）

\[
\boxed{
t_s\approx\frac{4}{\zeta\omega_n}
}
\]

---

## 10. “\(\zeta\) 控形状，\(\omega_n\) 控时间尺度”

这是今天非常重要的一句总结。

对于标准二阶系统：

- **\(\zeta\)** 主要决定 response 的“形状”
  - 会不会 overshoot
  - overshoot 有多大
  - 振荡衰减得快不快
- **\(\omega_n\)** 主要决定 response 的“时间尺度”
  - 整体快还是慢
  - rise、peak、settle 是否更早发生

因此：

> 同样的 \(\zeta\)，更大的 \(\omega_n\) 常常意味着“同样风格，但时间轴压缩得更快”。

---

## 11. Reaction curve 的 K–L–T 视角

我们后来又学了另一套语言：**process reaction curve**。

它最常见于过程控制里，把系统近似成一阶加纯延迟（FOPDT）：

\[
\boxed{
G(s)\approx \frac{K e^{-Ls}}{Ts+1}
}
\]

这里：

- \(K\)：process gain
- \(L\)：dead time / delay
- \(T\)：time constant

### 11.1 这三个量的直觉

#### \(K\)：敏感度

\[
\boxed{K=\frac{\Delta y}{\Delta u}}
\]

表示：

> 输入改一点，最终输出会改多少

#### \(L\)：死区/延迟

表示：

> 输入变了以后，输出要等多久才开始明显动

#### \(T\)：开始动以后，主体响应有多慢

表示：

> 系统一旦开始响应，还要花多长时间去“爬向新稳态”

---

## 12. 如何从 reaction curve 上估计 \(K,L,T\)

这是今天你特别问到的重点。

---

### 12.1 先做 step experiment

给 plant 一个已知 step input \(\Delta u\)，观察 output。

设：

- 初始值：\(y_0\)
- 最终值：\(y_{ss}\)

则：

\[
\boxed{
K=\frac{y_{ss}-y_0}{\Delta u}
}
\]

---

### 12.2 用切线（tangent）估计 \(L\) 和 \(T\)

做法：

1. 找曲线最陡的点（最大斜率点）
2. 在该点画 tangent
3. 看 tangent 与两条水平线的交点

- tangent 与初始水平线 \(y=y_0\) 的交点时刻，定义为：

\[
\boxed{L}
\]

- tangent 与最终水平线 \(y=y_{ss}\) 的交点时刻，定义为：

\[
\boxed{L+T}
\]

因此：

\[
\boxed{
T=(L+T)-L
}
\]

如果最大斜率点是 \((t_i,y_i)\)，斜率是 \(m_s\)，则 tangent 为：

\[
y_{\text{tan}}(t)=y_i+m_s(t-t_i)
\]

于是：

\[
\boxed{
L=t_i+\frac{y_0-y_i}{m_s}
}
\]

\[
\boxed{
L+T=t_i+\frac{y_{ss}-y_i}{m_s}
}
\]

\[
\boxed{
T=\frac{y_{ss}-y_0}{m_s}
}
\]

---

## 13. K–L–T 方法适合什么系统？对弹簧系统有什么问题？

这是今天很重要的“反思”。

K–L–T / reaction-curve 方法最适合：

- 单调
- S-shaped
- 近似一阶加纯延迟

而我们的 mass–spring–damper 是一个真实二阶系统：

\[
\boxed{
G(s)=\frac{1}{ms^2+bs+k}
}
\]

它的问题在于：

1. 它没有真正的 physical dead time，严格来说 \(L=0\)
2. 它可能 overshoot、oscillate，不是单调 S-shape
3. 用 tangent 拟合出来的 \(L\) 和 \(T\) 只是 **equivalent KLT**，不是系统真正的物理参数

所以：

> 对 mass–spring–damper，用 KLT 是“可以做的近似”，但不是最自然、也不是最准确的建模方式。

---

## 14. Ziegler–Nichols reaction-curve tuning

一旦从 reaction curve 估出 \(K,L,T\)，就可以用经验公式给 P/PI/PID 一个初始整定值。

---

### 14.1 P controller

\[
\boxed{
K_p=\frac{T}{KL}
}
\]

---

### 14.2 PI controller

\[
\boxed{
K_p=0.9\frac{T}{KL}
}
\]

\[
\boxed{
T_i=3.33L
}
\]

如果改写成 \(K_i\) 形式：

\[
\boxed{
K_i=\frac{K_p}{T_i}
}
\]

---

### 14.3 PID controller

\[
\boxed{
K_p=1.2\frac{T}{KL}
}
\]

\[
\boxed{
T_i=2L
}
\]

\[
\boxed{
T_d=0.5L
}
\]

同样可以写成：

\[
\boxed{
K_i=\frac{K_p}{T_i}
}
\qquad
\boxed{
K_d=K_pT_d
}
\]

---

## 15. 为什么 \(T_i\) 和 \(T_d\) 是这些形式？

这个问题你也问到了。

### 15.1 \(L,T\) 是怎么来的？

- \(K,L,T\) 来自 reaction curve 的几何识别（最大斜率切线法）

### 15.2 \(T_i,T_d\) 是怎么来的？

- 不是从曲线几何直接推导出来的
- 而是 **Ziegler–Nichols 的经验 tuning rule**

也就是说：

- \(K,L,T\)：是 plant identification
- \(K_p,T_i,T_d\)：是 controller tuning heuristic

特别地：

\[
T_i=2L,\qquad T_d=0.5L
\]

是经验法则，不是像牛顿定律那样的“精确自然定律”。

直觉上：

- \(L\) 越大，系统越“后知后觉”
- 积分不能太激进，因此 \(T_i\) 应该随 \(L\) 变大
- D action 也会和 \(L\) 有关系，用来提前抑制即将发生的 overshoot

---

## 16. 用同一个弹簧系统做完整例子

我们用的真实 plant：

\[
m=1,\quad b=3,\quad k=9
\]

---

### 16.1 从二阶系统角度看

\[
\omega_n=\sqrt{\frac{k}{m}}=3
\]

\[
\zeta=\frac{b}{2\sqrt{mk}}=\frac{3}{6}=0.5
\]

\[
\omega_d=\omega_n\sqrt{1-\zeta^2}=3\sqrt{0.75}\approx2.598
\]

所以：

- underdamped
- 会 overshoot
- 会振荡后收敛

---

### 16.2 Open-loop reaction curve（单位 step force）

对单位 force step：

\[
\Delta u=1
\]

steady-state output：

\[
y_{ss}=\frac{1}{k}=\frac{1}{9}\approx0.1111
\]

因此：

\[
K=\frac{\Delta y}{\Delta u}=\frac{1/9}{1}=\frac{1}{9}\approx0.1111
\]

通过 tangent fit，我们近似得到：

\[
\boxed{
K\approx0.1111,\quad
L\approx0.126\text{ s},\quad
T\approx0.610\text{ s}
}
\]

注意：

> 这里的 \(L,T\) 是 equivalent KLT fit，不是这个二阶系统真实的“纯延迟 + 一阶时间常数”。

---

### 16.3 用 Ziegler–Nichols reaction-curve 公式算 PID

PID 规则：

\[
K_p=1.2\frac{T}{KL},\qquad
T_i=2L,\qquad
T_d=0.5L
\]

代入：

\[
K\approx0.1111,\quad
L\approx0.126,\quad
T\approx0.610
\]

得到：

\[
\boxed{K_p\approx52.21}
\]

\[
\boxed{T_i\approx0.252\text{ s}}
\]

\[
\boxed{T_d\approx0.063\text{ s}}
\]

进一步：

\[
K_i=\frac{K_p}{T_i}\approx206.8
\]

\[
K_d=K_pT_d\approx3.295
\]

---

### 16.4 这套 PID 的效果怎么样？

把这套 PID 接回真实的 mass–spring–damper plant，可以观察到：

- rise time 明显变快
- steady-state error 基本消除
- 但 overshoot 非常大

典型结果：

- 原 plant（归一化后）overshoot 大约 **16.3%**
- ZN-PID 后 overshoot 大约 **60.2%**

这说明：

> Ziegler–Nichols reaction-curve tuning 的风格本来就比较激进。

更重要的是，这也说明：

> 对一个本来就振荡的二阶 plant，直接用 KLT 拟合 + ZN PID，往往会给出很猛的参数。

这不是“算错了”，而是方法本身的风格和适用性决定的。

---

## 17. 今天得到的最重要直觉

### 17.1 看 reaction curve 的时候，你应该在脑中问什么？

1. rise time 快不快？
2. peak time 早不早？
3. overshoot 大不大？
4. settling time 长不长？
5. steady-state error 有没有？
6. 它看起来像 underdamped、critical 还是 overdamped？
7. pole 的实部和虚部大概意味着什么？
8. 如果把它近似成 KLT，它的 \(K,L,T\) 大概分别是什么？

---

### 17.2 今天两套语言其实在描述同一件事

**语言 A：二阶系统语言**

- \(\zeta\)
- \(\omega_n\)
- \(\omega_d\)
- poles

用来解释：

> 为什么 reaction curve 长成这样

**语言 B：过程控制 / reaction-curve 语言**

- \(K\)
- \(L\)
- \(T\)

用来做：

> 从实验曲线快速拟合一个简单模型，并据此给 PID 初值

---

### 17.3 不同方法的适用性不同

- 对真实二阶机械系统，\(\zeta,\omega_n,\) poles 通常更自然
- 对温度、流量、液位等过程控制系统，KLT / FOPDT 往往更常见
- KLT 不是普遍完美模型，而是一种非常实用的近似

---

## 18. 相关图片

### 18.1 弹簧系统的 KLT tangent fit

![KLT tangent fit](klt_tangent_fit.png)

### 18.2 同一个真实 plant：Open-loop vs P vs KLT-tuned PID

![Open-loop vs P vs PID](openloop_p_pid_comparison.png)

### 18.3 最初的 open-loop step response 示例图

![Open-loop step response](mass_spring_open_loop_step.png)

---

## 19. 后续最自然的下一步

如果继续往下学，最顺的一条路是：

1. 用 Python 自动标出
   - rise time
   - peak time
   - overshoot
   - settling time
2. 对同一个 plant 分别加
   - P
   - PI
   - PID
3. 系统比较改变 \(K_p,K_i,K_d\) 后，reaction curve 如何变化
4. 最后回到 cruise control，把“弹簧系统的直觉”迁移到汽车速度控制

---

## 20. 一句话总总结

> **\(\zeta,\omega_n,\omega_d\) 帮你理解二阶系统为什么这样动；K,L,T 帮你从实验曲线快速近似 plant；而 PID tuning 就是在这两种直觉之间搭桥。**

---
