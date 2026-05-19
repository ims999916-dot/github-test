import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# ============================================================
# 1. 초기 조건 및 패널 조건
# ============================================================
V_INIT = 15.0          # 시작 전압 [V]
DELTA_V = 0.5          # P&O 전압 변화량 [V]

IRR_STC = 1000.0       # 표준 일사량 [W/m²]

P_MAX = 250.0          # 최대 전력 [W]
V_MP = 30.0            # 최대전력점 전압 [V]
V_OC = 37.0            # 개방 전압 [V]
I_SC = 8.5             # 단락 전류 [A]

T_END = 10.0           # 전체 시뮬레이션 시간 [s]
MASTER_DT = 0.01       # 시뮬레이션 계산 간격 [s]
TAU = 0.03             # PV 전압이 기준 전압을 따라가는 응답 속도

REALTIME_PLAYBACK = True
PLAYBACK_SPEED = 2.0
LIVE_UPDATE_EVERY = 20

ZOOM_HALF_WINDOW = 0.35

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.titlesize": 10,
    }
)


# ============================================================
# 2. 일사량 변화 조건
# ============================================================
def irradiance(t: float) -> float:
    if 0.0 <= t < 3.0:
        return 1000.0
    if 3.0 <= t < 6.0:
        return 500.0
    return 1000.0


# ============================================================
# 3. 단순 PV 모델
# ============================================================
def solve_shape_exponent(vmp: float, voc: float, tol: float = 1e-12) -> float:
    r = vmp / voc

    def f(a):
        return (r**a) - (1.0 / (1.0 + a))

    lo, hi = 1e-8, 100.0
    flo, fhi = f(lo), f(hi)

    if flo * fhi > 0:
        raise RuntimeError("PV curve exponent solving failed.")

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)

        if abs(fmid) < tol:
            return mid

        if flo * fmid < 0:
            hi = mid
        else:
            lo = mid
            flo = fmid

    return 0.5 * (lo + hi)


A_EXP = solve_shape_exponent(V_MP, V_OC)
K_SCALE = P_MAX / (V_MP * I_SC * (1.0 - (V_MP / V_OC) ** A_EXP))


def pv_current(voltage: float, irr: float) -> float:
    v = np.clip(voltage, 0.0, V_OC)
    g_ratio = irr / IRR_STC

    current = K_SCALE * I_SC * g_ratio * (1.0 - (v / V_OC) ** A_EXP)
    return max(current, 0.0)


# ============================================================
# 4. P&O MPPT 알고리즘
# ============================================================
class PerturbAndObserveMPPT:
    def __init__(self, v_init: float, delta_v: float, v_min: float = 0.0, v_max: float = V_OC):
        self.v_ref = float(v_init)
        self.delta_v = float(delta_v)
        self.v_min = float(v_min)
        self.v_max = float(v_max)

        self.prev_v = None
        self.prev_p = None
        self.direction = +1.0

    def update(self, measured_v: float, measured_i: float) -> float:
        p = measured_v * measured_i

        if self.prev_v is None or self.prev_p is None:
            self.prev_v = measured_v
            self.prev_p = p
            self.v_ref = np.clip(self.v_ref + self.direction * self.delta_v, self.v_min, self.v_max)
            return self.v_ref

        d_v = measured_v - self.prev_v
        d_p = p - self.prev_p
        eps = 1e-12

        if abs(d_v) > eps and abs(d_p) > eps:
            if d_p > 0:
                self.direction = +1.0 if d_v > 0 else -1.0
            else:
                self.direction = -1.0 if d_v > 0 else +1.0

        self.v_ref = np.clip(self.v_ref + self.direction * self.delta_v, self.v_min, self.v_max)

        self.prev_v = measured_v
        self.prev_p = p

        return self.v_ref


# ============================================================
# 5. 하나의 P&O 시뮬레이션 케이스
# ============================================================
class POCase:
    def __init__(self, control_dt: float):
        self.control_dt = control_dt
        self.controller = PerturbAndObserveMPPT(V_INIT, DELTA_V, 0.0, V_OC)

        self.v_pv = V_INIT
        self.v_ref = V_INIT
        self.next_control_time = 0.0

    def step(self, t: float, irr: float, plant_dt: float):
        measured_i = pv_current(self.v_pv, irr)

        if t + 1e-12 >= self.next_control_time:
            self.v_ref = self.controller.update(self.v_pv, measured_i)
            self.next_control_time += self.control_dt

        alpha = np.clip(plant_dt / TAU, 0.0, 1.0)
        self.v_pv = np.clip(self.v_pv + alpha * (self.v_ref - self.v_pv), 0.0, V_OC)

        i_pv = pv_current(self.v_pv, irr)
        p_pv = self.v_pv * i_pv

        p_mpp = P_MAX * (irr / IRR_STC)
        i_mpp = p_mpp / V_MP

        return {
            "v_pv": self.v_pv,
            "v_ref": self.v_ref,
            "i_pv": i_pv,
            "i_mpp": i_mpp,
            "p_pv": p_pv,
            "p_mpp": p_mpp,
        }


# ============================================================
# 6. 그래프 관련 함수
# ============================================================
def style_axis(ax, y_label=None, x_label=None, x_major=1.0, x_minor=0.2):
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=8, labelpad=2)

    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=8)

    ax.grid(True, which="major", linewidth=0.8, alpha=0.85)
    ax.grid(True, which="minor", linewidth=0.35, alpha=0.45)
    ax.minorticks_on()

    ax.xaxis.set_major_locator(MultipleLocator(x_major))
    ax.xaxis.set_minor_locator(MultipleLocator(x_minor))

    ax.tick_params(axis="both", labelsize=8)


def set_window_title(fig, title):
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass


def set_window_geometry(fig, width, height, x, y):
    try:
        manager = fig.canvas.manager
        window = manager.window

        if hasattr(window, "wm_geometry"):
            window.wm_geometry(f"{width}x{height}+{x}+{y}")
        elif hasattr(window, "setGeometry"):
            window.setGeometry(x, y, width, height)
        elif hasattr(window, "SetSize") and hasattr(window, "SetPosition"):
            window.SetSize((width, height))
            window.SetPosition((x, y))
    except Exception:
        pass


def raise_window(fig):
    try:
        manager = fig.canvas.manager
        window = manager.window

        if hasattr(window, "lift"):
            window.lift()

        if hasattr(window, "attributes"):
            try:
                window.attributes("-topmost", True)
                window.after(100, lambda: window.attributes("-topmost", False))
            except Exception:
                pass

        if hasattr(window, "raise_"):
            window.raise_()

        if hasattr(window, "activateWindow"):
            window.activateWindow()
    except Exception:
        pass


# ============================================================
# 7. Overview 창 생성
# ============================================================
def create_overview_window():
    fig, axes = plt.subplots(4, 1, figsize=(8.2, 6.4), sharex=True)
    set_window_title(fig, "1. Overview - P&O MPPT")

    c_fast = "tab:blue"
    c_slow = "tab:orange"
    c_ref = "black"

    lines = {}

    lines["irr"], = axes[0].plot([], [], color=c_ref, linewidth=2.0, drawstyle="steps-post", label="Irradiance")

    lines["v01"], = axes[1].plot([], [], color=c_fast, linewidth=1.7, label="Vpv dt=0.01s")
    lines["vr01"], = axes[1].plot(
        [], [], color=c_fast, linestyle="--", linewidth=1.0, alpha=0.35, drawstyle="steps-post", label="Vref dt=0.01s"
    )

    lines["v05"], = axes[1].plot([], [], color=c_slow, linewidth=1.7, label="Vpv dt=0.05s")
    lines["vr05"], = axes[1].plot(
        [], [], color=c_slow, linestyle="--", linewidth=1.0, alpha=0.35, drawstyle="steps-post", label="Vref dt=0.05s"
    )

    axes[1].axhline(V_MP, color=c_ref, linestyle=":", linewidth=1.2, label="Target Vmp")

    lines["i01"], = axes[2].plot([], [], color=c_fast, linewidth=1.7, label="Ipv dt=0.01s")
    lines["i05"], = axes[2].plot([], [], color=c_slow, linewidth=1.7, label="Ipv dt=0.05s")
    lines["impp"], = axes[2].plot(
        [], [], color=c_ref, linestyle="--", linewidth=1.2, drawstyle="steps-post", label="Ideal Imp"
    )

    lines["p01"], = axes[3].plot([], [], color=c_fast, linewidth=1.7, label="Ppv dt=0.01s")
    lines["p05"], = axes[3].plot([], [], color=c_slow, linewidth=1.7, label="Ppv dt=0.05s")
    lines["pmpp"], = axes[3].plot(
        [], [], color=c_ref, linestyle="--", linewidth=1.2, drawstyle="steps-post", label="Ideal Pmp"
    )

    labels = ["Irradiance\n[W/m²]", "Voltage\n[V]", "Current\n[A]", "Power\n[W]"]

    for i, ax in enumerate(axes):
        style_axis(ax, y_label=labels[i], x_major=1.0, x_minor=0.2)
        ax.axvline(3.0, linestyle="--", linewidth=0.9, color="gray")
        ax.axvline(6.0, linestyle="--", linewidth=0.9, color="gray")
        ax.set_xlim(0, T_END)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

    axes[-1].set_xlabel("Time [s]", fontsize=8)

    axes[0].set_ylim(0, 1100)
    axes[1].set_ylim(10, 32)
    axes[2].set_ylim(0, I_SC + 1.0)
    axes[3].set_ylim(0, P_MAX + 30)

    axes[0].set_title("Irradiance: 1000 → 500 → 1000 W/m²", fontsize=9, pad=2)
    fig.suptitle("P&O MPPT Overview: dt 0.01s vs 0.05s", fontsize=11)
    fig.tight_layout(rect=[0.04, 0.06, 0.98, 0.93])

    return {"fig": fig, "axes": axes, "lines": lines}


# ============================================================
# 8. Zoom 창 생성
# ============================================================
def create_zoom_window(center_time: float, title: str):
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 3.3), sharex=True)
    set_window_title(fig, title)

    c_fast = "tab:blue"
    c_slow = "tab:orange"
    c_ref = "black"

    lines = {}

    lines["v01"], = axes[0].plot([], [], color=c_fast, linewidth=1.6, label="Vpv dt=0.01s")
    lines["v05"], = axes[0].plot([], [], color=c_slow, linewidth=1.6, label="Vpv dt=0.05s")
    axes[0].axhline(V_MP, color=c_ref, linestyle=":", linewidth=1.1, label="Target")

    lines["i01"], = axes[1].plot([], [], color=c_fast, linewidth=1.6)
    lines["i05"], = axes[1].plot([], [], color=c_slow, linewidth=1.6)
    lines["impp"], = axes[1].plot([], [], color=c_ref, linestyle="--", linewidth=1.1, drawstyle="steps-post")

    lines["p01"], = axes[2].plot([], [], color=c_fast, linewidth=1.6)
    lines["p05"], = axes[2].plot([], [], color=c_slow, linewidth=1.6)
    lines["pmpp"], = axes[2].plot([], [], color=c_ref, linestyle="--", linewidth=1.1, drawstyle="steps-post")

    labels = ["V\n[V]", "I\n[A]", "P\n[W]"]

    for i, ax in enumerate(axes):
        style_axis(ax, y_label=labels[i], x_major=0.1, x_minor=0.02)
        ax.axvline(center_time, linestyle="--", linewidth=0.9, color="gray")
        ax.set_xlim(center_time - ZOOM_HALF_WINDOW, center_time + ZOOM_HALF_WINDOW)
        ax.tick_params(axis="both", labelsize=8)
        ax.yaxis.labelpad = 2

    axes[0].legend(loc="upper right", fontsize=7, framealpha=0.85)
    axes[-1].set_xlabel("Time [s]", fontsize=8)

    axes[0].set_ylim(29.0, 31.0)
    axes[1].set_ylim(3.5, 8.8)
    axes[2].set_ylim(110, 260)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0.06, 0.08, 0.98, 0.90])

    return {"fig": fig, "axes": axes, "lines": lines, "center_time": center_time}


# ============================================================
# 9. 그래프 업데이트
# ============================================================
def update_overview(window, history, k):
    lines = window["lines"]

    r01 = history["P&O dt=0.01s"]
    r05 = history["P&O dt=0.05s"]

    t = np.array(r01["time"][: k + 1])
    irr = np.array(r01["irradiance"][: k + 1])

    lines["irr"].set_data(t, irr)
    lines["v01"].set_data(t, np.array(r01["v_pv"][: k + 1]))
    lines["vr01"].set_data(t, np.array(r01["v_ref"][: k + 1]))
    lines["v05"].set_data(t, np.array(r05["v_pv"][: k + 1]))
    lines["vr05"].set_data(t, np.array(r05["v_ref"][: k + 1]))

    lines["i01"].set_data(t, np.array(r01["i_pv"][: k + 1]))
    lines["i05"].set_data(t, np.array(r05["i_pv"][: k + 1]))
    lines["impp"].set_data(t, np.array(r01["i_mpp"][: k + 1]))

    lines["p01"].set_data(t, np.array(r01["p_pv"][: k + 1]))
    lines["p05"].set_data(t, np.array(r05["p_pv"][: k + 1]))
    lines["pmpp"].set_data(t, np.array(r01["p_mpp"][: k + 1]))

    window["fig"].canvas.draw_idle()
    window["fig"].canvas.flush_events()


def update_zoom(window, history, k):
    lines = window["lines"]
    center_time = window["center_time"]

    r01 = history["P&O dt=0.01s"]
    r05 = history["P&O dt=0.05s"]

    t = np.array(r01["time"][: k + 1])
    mask = (t >= center_time - ZOOM_HALF_WINDOW) & (t <= center_time + ZOOM_HALF_WINDOW)
    tz = t[mask]

    lines["v01"].set_data(tz, np.array(r01["v_pv"][: k + 1])[mask])
    lines["v05"].set_data(tz, np.array(r05["v_pv"][: k + 1])[mask])

    lines["i01"].set_data(tz, np.array(r01["i_pv"][: k + 1])[mask])
    lines["i05"].set_data(tz, np.array(r05["i_pv"][: k + 1])[mask])
    lines["impp"].set_data(tz, np.array(r01["i_mpp"][: k + 1])[mask])

    lines["p01"].set_data(tz, np.array(r01["p_pv"][: k + 1])[mask])
    lines["p05"].set_data(tz, np.array(r05["p_pv"][: k + 1])[mask])
    lines["pmpp"].set_data(tz, np.array(r01["p_mpp"][: k + 1])[mask])

    window["fig"].canvas.draw_idle()
    window["fig"].canvas.flush_events()


# ============================================================
# 10. 메인 실행
# ============================================================
def main():
    time_arr = np.arange(0.0, T_END + MASTER_DT * 0.5, MASTER_DT)

    cases = {"P&O dt=0.01s": POCase(0.01), "P&O dt=0.05s": POCase(0.05)}

    history = {
        label: {
            "time": [],
            "irradiance": [],
            "v_pv": [],
            "v_ref": [],
            "i_pv": [],
            "i_mpp": [],
            "p_pv": [],
            "p_mpp": [],
        }
        for label in cases
    }

    overview_window = create_overview_window()
    zoom3_window = create_zoom_window(3.0, "2. Zoom around 3s: irradiance drop")
    zoom6_window = create_zoom_window(6.0, "3. Zoom around 6s: irradiance recovery")

    plt.show(block=False)
    plt.pause(0.2)

    set_window_geometry(overview_window["fig"], 820, 660, 20, 40)
    set_window_geometry(zoom3_window["fig"], 660, 330, 860, 40)
    set_window_geometry(zoom6_window["fig"], 660, 330, 860, 410)

    raise_window(overview_window["fig"])

    wall_prev = time.perf_counter()

    for k, t in enumerate(time_arr):
        irr = irradiance(float(t))

        for label, case in cases.items():
            out = case.step(float(t), irr, MASTER_DT)

            history[label]["time"].append(float(t))
            history[label]["irradiance"].append(float(irr))
            history[label]["v_pv"].append(float(out["v_pv"]))
            history[label]["v_ref"].append(float(out["v_ref"]))
            history[label]["i_pv"].append(float(out["i_pv"]))
            history[label]["i_mpp"].append(float(out["i_mpp"]))
            history[label]["p_pv"].append(float(out["p_pv"]))
            history[label]["p_mpp"].append(float(out["p_mpp"]))

        if k % LIVE_UPDATE_EVERY == 0 or k == len(time_arr) - 1:
            update_overview(overview_window, history, k)
            update_zoom(zoom3_window, history, k)
            update_zoom(zoom6_window, history, k)
            plt.pause(0.001)

        if REALTIME_PLAYBACK and k < len(time_arr) - 1:
            target = MASTER_DT / max(PLAYBACK_SPEED, 1e-12)
            now = time.perf_counter()
            elapsed = now - wall_prev

            if elapsed < target:
                time.sleep(target - elapsed)

            wall_prev = time.perf_counter()

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
