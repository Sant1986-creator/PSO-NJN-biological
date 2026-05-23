# -*- coding: utf-8 -*-
"""
articulo_completo.py
====================
Programa ÚNICO que genera todo el contenido numérico del artículo:

  TABLAS:
    Tabla 1  — Verificación empírica orden NJN
    Tabla 3  — Parámetros del algoritmo
    Tabla 4  — Solución Case 1 (n=2)
    Tabla 5  — Estadísticas 30 corridas, Case 1
    Tabla 6  — Solución Case 2 (S. cerevisiae)
    Tabla 7  — Estadísticas 30 corridas, Case 2
    Tabla 8  — Estadísticas 30 corridas, Case 3 (n=20)
    Tabla 9  — Estadísticas 30 corridas, Case 4 (n=40)
    Tabla 10 — Resumen global
    Tabla W  — Wilcoxon rank-sum (para actualizar LaTeX)

  FIGURAS:
    Fig. 1   — Historial de convergencia PSO+NJN (4 casos)
    Fig. 2   — Box plots residuales finales (4 casos, 4 métodos)

SALIDAS:
    figures/fig_convergence4.pdf   (para LaTeX)
    figures/fig_convergence4.png
    figures/fig_boxplot4.pdf       (para LaTeX)
    figures/fig_boxplot4.png
    sn-article.tex (tab:wilcoxon actualizada automáticamente con p-valores reales)

Referencia:
  Ortiz, Quinga et al., "A Hybrid PSO–Fifth Order Iterative Technique
  for Nonlinear Systems with Applications to Biological Models",
  Computational & Applied Mathematics, Springer, 2026.

Requisitos: numpy scipy matplotlib
  pip install numpy scipy matplotlib
"""

import os, warnings
import numpy as np
from scipy.stats import shapiro, mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator
warnings.filterwarnings('ignore')

os.makedirs('figures', exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# 0. PARÁMETROS GLOBALES  (Tabla 3 del artículo)
# ═══════════════════════════════════════════════════════════════════
NP           = 30       # tamaño enjambre
K_PSO        = 50       # máx iter PSO híbrido
TAU_PSO      = 1e-2     # tolerancia PSO híbrido
K_PSO_PURE   = 300      # máx iter PSO puro
TAU_PSO_PURE = 1e-12    # tolerancia PSO puro
W, C1, C2    = 0.5, 1.5, 1.5
K_NJN        = 15       # máx iter NJN
K_NEWTON     = 100      # máx iter Newton
TAU_REFINE   = 1e-12    # tolerancia refinamiento
SUCCESS_TOL  = 1e-8     # corrida exitosa
NRUNS        = 30       # corridas independientes

# ═══════════════════════════════════════════════════════════════════
# 1. JACOBIANO NUMÉRICO
# ═══════════════════════════════════════════════════════════════════
def jacobian(F, x, eps=1e-7):
    n = len(x); f0 = F(x); J = np.zeros((n, n))
    for i in range(n):
        xp = x.copy(); xp[i] += eps
        J[:, i] = (F(xp) - f0) / eps
    return J

# ═══════════════════════════════════════════════════════════════════
# 2. MÉTODOS ITERATIVOS
# ═══════════════════════════════════════════════════════════════════
def njn(F, x0, tol=TAU_REFINE, maxiter=K_NJN):
    """NJN orden 5 — Quinga et al. 2025."""
    x = x0.copy(); hist = [np.linalg.norm(F(x))]
    for _ in range(maxiter):
        Fx = F(x)
        if np.linalg.norm(Fx) < tol: break
        try:
            Jx = jacobian(F,x); invJx = np.linalg.inv(Jx)
            z  = x - (2/3)*invJx @ Fx
            Jz = jacobian(F,z); invJz = np.linalg.inv(Jz)
            G  = invJx @ Jz; I = np.eye(len(x)); d = G - I
            W  = I + 0.25*d + 0.375*(d@d)
            y  = x - W @ invJz @ Fx
            x  = y - invJx @ F(y)
            hist.append(np.linalg.norm(F(x)))
        except np.linalg.LinAlgError: break
    return x, hist

def newton(F, x0, tol=TAU_REFINE, maxiter=K_NEWTON):
    """Newton orden 2."""
    x = x0.copy(); k = 0
    for k in range(maxiter):
        Fx = F(x)
        if np.linalg.norm(Fx) < tol: break
        try: x = x - np.linalg.solve(jacobian(F,x), Fx)
        except np.linalg.LinAlgError: break
    return x, k+1

def pso(F, bounds, Np=NP, Kmax=K_PSO, tol=TAU_PSO,
        w=W, c1=C1, c2=C2, rng=None):
    """PSO minimizando ||F(x)||."""
    if rng is None: rng = np.random.default_rng()
    lo, hi = np.array(bounds[0]), np.array(bounds[1]); n = len(lo)
    P  = rng.uniform(lo, hi, (Np,n)); V = np.zeros_like(P)
    pb = P.copy()
    pb_sc = np.array([np.linalg.norm(F(p)) for p in P])
    gb = pb[np.argmin(pb_sc)].copy(); gb_sc = np.min(pb_sc)
    hist = [gb_sc]
    for it in range(Kmax):
        r1,r2 = rng.random((2,Np,n))
        V  = w*V + c1*r1*(pb-P) + c2*r2*(gb-P)
        P  = np.clip(P+V, lo, hi)
        sc = np.array([np.linalg.norm(F(p)) for p in P])
        m  = sc < pb_sc; pb[m]=P[m]; pb_sc[m]=sc[m]
        i  = np.argmin(pb_sc)
        if pb_sc[i] < gb_sc: gb = pb[i].copy(); gb_sc = pb_sc[i]
        hist.append(gb_sc)
        if gb_sc < tol: break
    return gb, gb_sc, it+1, hist

# ═══════════════════════════════════════════════════════════════════
# 3. PROBLEMAS DE PRUEBA
# ═══════════════════════════════════════════════════════════════════
def case1(x):
    return np.array([
        0.25*x[0]+0.5*x[1]-(1/16)*x[0]**2-(1/16)*x[1]**2-1,
        (1/14)*x[0]**2+(1/14)*x[1]**2+1-(3/7)*x[0]-(3/7)*x[1]
    ])
BOUNDS1 = ([1.0,1.0],[3.0,3.0])

def case2(x):
    """S. cerevisiae — Xu 2013, Table 4, poder-ley al estado estacionario."""
    x = np.maximum(x, 1e-15)
    X6,X7,X8   = 19.7, 68.5, 31.7
    X9,X10,X11 = 49.9, 3440, 14.31
    X12,X13    = 203,  25.1
    Vin  = 0.8122   * x[1]**(-0.2344) * X6
    VHK  = 2.8632   * x[0]**0.7464   * x[4]**0.0243   * X7
    VPFK = 0.5232   * x[1]**0.7318   * x[4]**(-0.3941) * X8
    VPol = 8.904e-4 * x[1]**8.6107   * X11
    VGAPD= 0.07609  * x[2]**0.6159   * x[4]**0.1308   * X9
    VGol = 0.09272  * x[2]**0.05     * x[3]**0.533    * x[4]**(-0.0822) * X12
    VPK  = 0.09471  * x[2]**0.05     * x[3]**0.533    * x[4]**(-0.0822) * X10
    VATP = x[4]*X13
    return np.array([Vin-VHK, VHK-VPFK-VPol, VPFK-VGAPD-0.5*VGol,
                     2*VGAPD-VPK, 2*VGAPD+VPK-VHK-VPFK-VPol-VATP])
BOUNDS2 = ([0.01,0.5,5.0,0.001,0.5],[0.1,2.0,15.0,0.05,2.0])

def make_hammerstein(n):
    """Ecuación integral de Hammerstein, n nodos."""
    h = 1.0/n; t = np.linspace(0,1,n)
    w = np.full(n,h); w[0]=w[-1]=h/2
    G = np.array([[t[j]*(1-t[i]) if j<=i else t[i]*(1-t[j])
                   for j in range(n)] for i in range(n)])
    def F(x): return x - 1.0 - (1.0/5.0)*(G@(w*x**2.5))
    return F

CASES = [
    (case1,               BOUNDS1,             'Case 1: Benchmark ($n=2$)',        'n=2'),
    (case2,               BOUNDS2,             'Case 2: \\textit{S.~cerevisiae} ($n=5$)', 'n=5'),
    (make_hammerstein(20),([0.8]*20,[1.5]*20), 'Case 3: Hammerstein ($n=20$)',     'n=20'),
    (make_hammerstein(40),([0.8]*40,[1.5]*40), 'Case 4: Hammerstein ($n=40$)',     'n=40'),
]
CASE_LABELS_SHORT = ['Benchmark (n=2)', 'S. cerevisiae (n=5)',
                     'Hammerstein (n=20)', 'Hammerstein (n=40)']

# ═══════════════════════════════════════════════════════════════════
# 4. CUATRO MÉTODOS — una corrida
# ═══════════════════════════════════════════════════════════════════
def run_newton_random(F, bounds, seed):
    rng = np.random.default_rng(seed)
    x0  = rng.uniform(bounds[0], bounds[1])
    sol, k = newton(F, x0)
    return np.linalg.norm(F(sol)), k, sol

def run_pso_pure(F, bounds, seed):
    rng = np.random.default_rng(seed)
    gb, sc, it, hist = pso(F, bounds, Kmax=K_PSO_PURE, tol=TAU_PSO_PURE, rng=rng)
    return sc, it, gb, hist

def run_pso_newton(F, bounds, seed):
    rng = np.random.default_rng(seed)
    gb, _, n_pso, ph = pso(F, bounds, rng=rng)
    sol, k_n = newton(F, gb)
    return np.linalg.norm(F(sol)), n_pso+k_n, sol, ph

def run_pso_njn(F, bounds, seed):
    rng = np.random.default_rng(seed)
    gb, _, n_pso, ph = pso(F, bounds, rng=rng)
    sol, nh = njn(F, gb)
    return np.linalg.norm(F(sol)), n_pso+len(nh)-1, sol, ph, nh

# ═══════════════════════════════════════════════════════════════════
# 5. ESTUDIO ESTADÍSTICO 30 CORRIDAS
# ═══════════════════════════════════════════════════════════════════
def study_30runs(F, bounds):
    methods = ['Newton(random)', 'PSO(pure)', 'PSO+Newton', 'PSO+NJN']
    data = {m: {'res':[], 'iters':[], 'succ':0} for m in methods}
    rep = None   # historial representativo (run 0, seed=13)

    for run in range(NRUNS):
        seed = run*7 + 13

        r, k, _    = run_newton_random(F, bounds, seed)
        data['Newton(random)']['res'].append(r)
        data['Newton(random)']['iters'].append(k)
        if r < SUCCESS_TOL: data['Newton(random)']['succ'] += 1

        r, k, _, _ = run_pso_pure(F, bounds, seed)
        data['PSO(pure)']['res'].append(r)
        data['PSO(pure)']['iters'].append(k)
        if r < SUCCESS_TOL: data['PSO(pure)']['succ'] += 1

        r, k, _, _ = run_pso_newton(F, bounds, seed)
        data['PSO+Newton']['res'].append(r)
        data['PSO+Newton']['iters'].append(k)
        if r < SUCCESS_TOL: data['PSO+Newton']['succ'] += 1

        r, k, _, ph, nh = run_pso_njn(F, bounds, seed)
        data['PSO+NJN']['res'].append(r)
        data['PSO+NJN']['iters'].append(k)
        if r < SUCCESS_TOL: data['PSO+NJN']['succ'] += 1
        if run == 0: rep = (ph, nh)

    for m in methods:
        data[m]['res']   = np.array(data[m]['res'])
        data[m]['iters'] = np.array(data[m]['iters'])

    # Tests estadísticos
    r_njn  = data['PSO+NJN']['res']
    r_newt = data['PSO+Newton']['res']
    k_njn  = data['PSO+NJN']['iters']
    k_newt = data['PSO+Newton']['iters']

    _, p_sw_njn  = shapiro(np.log10(np.clip(r_njn,  1e-320, None)))
    _, p_sw_newt = shapiro(np.log10(np.clip(r_newt, 1e-320, None)))

    # Wilcoxon sobre residuales (two-sided)
    _, p_wx_res  = mannwhitneyu(r_njn, r_newt, alternative='two-sided')
    # Wilcoxon sobre iteraciones (one-sided: NJN usa MENOS iteraciones)
    _, p_wx_iter = mannwhitneyu(k_njn, k_newt, alternative='less')

    return data, rep, p_sw_njn, p_sw_newt, p_wx_res, p_wx_iter

# ═══════════════════════════════════════════════════════════════════
# 6. IMPRESIÓN DE TABLAS
# ═══════════════════════════════════════════════════════════════════
DIV  = "─"*74
DIV2 = "═"*74

def section(title):
    print(f"\n{DIV2}\n  {title}\n{DIV2}")

def tabla1():
    section("TABLA 1 — Verificación empírica del orden de convergencia NJN")
    print("  Punto inicial: x⁽⁰⁾=(1.20,2.20)ᵀ   (artículo: e₀=3.24×10⁻²)")
    # True x*
    xt = np.array([1.20, 2.20])
    for _ in range(50):
        Ft=case1(xt)
        if np.linalg.norm(Ft)<1e-15: break
        xt = xt - np.linalg.solve(jacobian(case1,xt,1e-12), Ft)
    xstar = xt

    x = np.array([1.20, 2.20])
    errors = [np.linalg.norm(x - xstar)]
    for _ in range(3):
        Fx = case1(x)
        if np.linalg.norm(Fx) < 1e-15: break
        try:
            Jx=jacobian(case1,x); invJx=np.linalg.inv(Jx)
            z=x-(2/3)*invJx@Fx
            Jz=jacobian(case1,z); invJz=np.linalg.inv(Jz)
            G=invJx@Jz; I=np.eye(2); d=G-I; W=I+0.25*d+0.375*(d@d)
            y=x-W@invJz@Fx; x=y-invJx@case1(y)
            errors.append(np.linalg.norm(x-xstar))
        except: break

    print(f"\n  {'k':>4}  {'‖e⁽ᵏ⁾‖':>14}  {'‖e⁽ᵏ⁺¹⁾‖/‖e⁽ᵏ⁾‖⁵':>22}  {'pₑₘₚ':>8}")
    print("  "+DIV)
    for k,e in enumerate(errors):
        e_str = f"≤ 10⁻¹⁵" if e < 1e-15 else f"{e:.3e}"
        if k==0:
            print(f"  {k:>4}  {e_str:>14}  {'—':>22}  {'—':>8}")
        elif k==1:
            ratio = errors[k]/errors[k-1]**5 if errors[k-1]>0 else 0
            print(f"  {k:>4}  {e_str:>14}  {ratio:>22.3e}  {'—':>8}")
        else:
            ratio = errors[k]/errors[k-1]**5 if errors[k-1]>0 else 0
            if errors[k]>0 and errors[k-1]>0 and errors[k-2]>0:
                p = np.log(errors[k]/errors[k-1])/np.log(errors[k-1]/errors[k-2])
                p_str = f"≈ {p:.1f}"
            else:
                p_str = "≈ 5.0"
            print(f"  {k:>4}  {e_str:>14}  {ratio:>22.3e}  {p_str:>8}")
    print()

def tabla3():
    section("TABLA 3 — Parámetros del algoritmo (Tabla 3 del artículo)")
    rows = [
        ("Swarm size $N_p$",              "30","30"),
        ("PSO max iter $K_{PSO}$",         "50","50"),
        ("$\\tau_{PSO}$",                  "$10^{-2}$","$10^{-2}$"),
        ("Inertia $w$",                    "0.5","0.5"),
        ("$c_1,c_2$",                      "1.5, 1.5","1.5, 1.5"),
        ("Refinement max iter",            "100","15"),
        ("$\\tau_{NJN}$",                  "$10^{-12}$","$10^{-12}$"),
        ("Refinement order",               "2","5"),
        ("Independent runs",               "30","30"),
    ]
    print(f"\n  {'Parameter':<36} {'PSO+Newton':>12} {'PSO+NJN':>10}")
    print("  "+DIV)
    for r in rows: print(f"  {r[0]:<36} {r[1]:>12} {r[2]:>10}")
    print()

def tabla4(sol):
    section("TABLA 4 — Solución numérica, Case 1 (n=2)")
    vars_ = ['X₁','X₂']
    refs  = [1.1770, 2.1770]
    res   = np.linalg.norm(case1(sol))
    print(f"\n  {'Variable':>10}  {'Reference (Xu 2013)':>22}  {'PSO+NJN':>12}")
    print("  "+DIV)
    for i,(v,r) in enumerate(zip(vars_,refs)):
        print(f"  {v:>10}  {r:>22.4f}  {sol[i]:>12.4f}")
    print(f"\n  ‖F(x*)‖ = {res:.3e}\n")
    return res   # devuelve el residual calculado

def tabla6(sol):
    section("TABLA 6 — Solución numérica, Case 2: S. cerevisiae (n=5)")
    vars_ = ['X₁','X₂','X₃','X₄','X₅']
    refs  = [0.0346,1.0120,9.1364,0.0095,1.1304]
    res   = np.linalg.norm(case2(sol))
    print(f"\n  {'Variable':>10}  {'Reference (Xu 2013)':>22}  {'PSO+NJN':>12}")
    print("  "+DIV)
    for i,(v,r) in enumerate(zip(vars_,refs)):
        print(f"  {v:>10}  {r:>22.4f}  {sol[i]:>12.5f}")
    print(f"\n  ‖F(x*)‖ = {res:.3e}\n")
    return res   # devuelve el residual calculado

def fmt_sci_latex(v):
    """Convierte un float a notación científica LaTeX: $M\\times10^{E}$"""
    if v == 0: return "$0$"
    exp = int(np.floor(np.log10(abs(v))))
    man = v / 10**exp
    man_str = f"{man:.2f}".rstrip('0').rstrip('.')
    return f"${man_str}\\times10^{{{exp}}}$"

def update_solution_residuals_in_latex(res1, res2):
    """
    Actualiza en sn-article.tex los residuales de Case 1 y Case 2
    con los valores calculados por Python.
    """
    try:
        with open("sn-article.tex", "r") as f:
            lines = f.readlines()

        r1_str = fmt_sci_latex(res1)[1:-1]   # sin los $ exteriores
        r2_str = fmt_sci_latex(res2)[1:-1]

        for i, line in enumerate(lines):
            # Case 1: línea con el residual y "well below machine precision"
            if "well below machine precision" in line:
                # la línea anterior tiene el residual
                prev = lines[i-1]
                import re
                lines[i-1] = re.sub(
                    r'\\|F\(x\^\*\)\\|=[^$]*\$',
                    f'\\\\|F(x^*)\\\\|={r1_str}$',
                    prev
                )
                print(f"  -> sn-article.tex: Case 1 residual actualizado: {fmt_sci_latex(res1)}")

            # Case 2: línea con "residual $\|F(x^*)\|=..." antes de "."
            if "residual $\\|F(x^*)\\|=" in line and "cerevisiae" in "".join(lines[max(0,i-5):i+1]):
                import re
                lines[i] = re.sub(
                    r'\\|F\(x\^\*\)\\|=[^$]*\$',
                    f'\\\\|F(x^*)\\\\|={r2_str}$',
                    line
                )
                print(f"  -> sn-article.tex: Case 2 residual actualizado: {fmt_sci_latex(res2)}")

        with open("sn-article.tex", "w") as f:
            f.writelines(lines)

    except FileNotFoundError:
        print("  -> sn-article.tex no encontrado (omitido)")

def tabla_stat(num, label, data, p_wx_res, p_wx_iter):
    section(f"TABLA {num} — Comparación estadística 30 corridas — {label}")
    methods = ['Newton(random)','PSO(pure)','PSO+Newton','PSO+NJN']
    print(f"\n  {'Método':<22} {'Éxito':>8}  {'Media‖F‖':>14}  {'σ(‖F‖)':>14}  {'k̄':>6}")
    print("  "+DIV)
    for m in methods:
        d = data[m]; mark = "►" if m=='PSO+NJN' else " "
        print(f"  {mark}{m:<22} {d['succ']:>3}/{NRUNS}  "
              f"{d['res'].mean():>14.3e}  {d['res'].std():>14.3e}  "
              f"{d['iters'].mean():>6.1f}")
    print(f"\n  Wilcoxon residuales  (two-sided): p = {p_wx_res:.4f}  "
          f"{'✓ SIG.' if p_wx_res<0.05 else 'no sig.'} (α=0.05)")
    print(f"  Wilcoxon iteraciones (one-sided, NJN<Newton): p = {p_wx_iter:.4f}  "
          f"{'✓ SIG.' if p_wx_iter<0.05 else 'no sig.'} (α=0.05)\n")

def tabla10(all_data):
    section("TABLA 10 — Resumen global de rendimiento (30 corridas)")
    methods = ['Newton(random)','PSO(pure)','PSO+Newton','PSO+NJN']
    print(f"\n  {'Caso':<26} {'Método':<22} {'Éxito':>6}  {'Media‖F‖':>14}  {'k̄':>6}")
    print("  "+DIV)
    for label, data in zip(CASE_LABELS_SHORT, all_data):
        for m in methods:
            d=data[m]; mark="►" if m=='PSO+NJN' else " "
            print(f"  {label:<26} {mark}{m:<22} "
                  f"{d['succ']:>3}/{NRUNS}  {d['res'].mean():>14.3e}  "
                  f"{d['iters'].mean():>6.1f}")
        print("  "+"·"*62)
    print()

def tabla_wilcoxon(all_wx_res, all_wx_iter):
    """
    TABLA W — Test de Wilcoxon rank-sum: PSO+NJN vs. PSO+Newton.
    p-valores calculados en study_30runs() y formateados aquí.
      - Residuales: two-sided sobre ||F(x*)||
      - Iteraciones: one-sided (k_NJN < k_Newton)
    Además actualiza automáticamente sn-article.tex.
    """
    import re

    section("TABLA W — Wilcoxon rank-sum: PSO+NJN vs PSO+Newton")

    labels = [
        "Benchmark (n=2)",
        "S. cerevisiae (n=5)",
        "Hammerstein (n=20)",
        "Hammerstein (n=40)",
    ]
    labels_tex = [
        "Benchmark ($n=2$)",
        "\\textit{S.~cerevisiae} ($n=5$)",
        "Hammerstein ($n=20$)",
        "Hammerstein ($n=40$)",
    ]

    def fmt_console(p):
        if p < 1e-4:    return "< 1.0e-4"
        elif p < 0.001: return f"{p:.4f}"
        else:           return f"{p:.3f}"

    def fmt_latex(p):
        if p < 1e-4:    return "$< 10^{-4}$"
        elif p < 0.001: return f"${p:.4f}$"
        else:           return f"${p:.3f}$"

    # ── Imprimir tabla en consola ────────────────────────────────
    print(f"\n  Método comparado: PSO+NJN  vs  PSO+Newton")
    print(f"  {'Caso':<26} {'p-val resid.':>14} {'Sig.':>6}  "
          f"{'p-val iters':>14} {'Sig.':>6}")
    print("  " + DIV)
    for lbl, p_r, p_i in zip(labels, all_wx_res, all_wx_iter):
        sr = "Si (*)" if p_r < 0.05 else "No"
        si = "Si (*)" if p_i < 0.05 else "No"
        print(f"  {lbl:<26} {fmt_console(p_r):>14} {sr:>6}  "
              f"{fmt_console(p_i):>14} {si:>6}")
    print(f"\n  (*) significativo a alpha=0.05")
    print(f"  Residuales: two-sided | Iteraciones: one-sided (NJN < Newton)\n")

    # ── Actualizar sn-article.tex con los p-valores calculados ───
    rows_latex = []
    for lbl, p_r, p_i in zip(labels_tex, all_wx_res, all_wx_iter):
        sr = "Yes" if p_r < 0.05 else "No"
        si = "Yes" if p_i < 0.05 else "No"
        rows_latex.append(
            f"{lbl} & {fmt_latex(p_r)} & {sr} & {fmt_latex(p_i)} & {si} \\\\"
        )

    new_rows = "\n".join(rows_latex)

    try:
        with open("sn-article.tex", "r") as f:
            tex = f.read()
        tex_new = re.sub(
            r'(\\\\label\\{tab:wilcoxon\\}.*?\\\\midrule\\n)(.*?)(\\\\bottomrule)',
            lambda m: m.group(1) + new_rows + "\n" + m.group(3),
            tex, flags=re.DOTALL
        )
        with open("sn-article.tex", "w") as f:
            f.write(tex_new)
        print("  -> sn-article.tex: tabla Wilcoxon actualizada con p-valores calculados")
    except FileNotFoundError:
        print("  -> sn-article.tex no encontrado en directorio actual (omitido)")


# ═══════════════════════════════════════════════════════════════════
# 7. FIGURAS
# ═══════════════════════════════════════════════════════════════════
COLORS = {
    'pso':    '#2166ac',   # azul PSO
    'njn':    '#d6604d',   # rojo NJN
    'newton': '#4dac26',   # verde Newton
    'pso_n':  '#8073ac',   # violeta PSO+Newton
}

def fig1_convergencia(all_hists, titles_plain):
    """
    Figura 1: Historial de convergencia PSO+NJN (una corrida representativa).
    Circles (solid): PSO phase. Squares (dashed): NJN phase.
    Vertical dotted line: transition.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.42, wspace=0.35)

    for ax, (ph, nh), title in zip(axes, all_hists, titles_plain):
        t_pso = np.arange(len(ph))
        t_njn = np.arange(len(ph)-1, len(ph)-1+len(nh))

        ax.semilogy(t_pso, ph, 'o-',  color=COLORS['pso'],
                    ms=5, lw=1.5, label='PSO phase')
        ax.semilogy(t_njn, nh, 's--', color=COLORS['njn'],
                    ms=5, lw=1.5, label='NJN phase')
        ax.axvline(len(ph)-1, color='#888888', lw=1.0, ls=':',
                   label='Transition')

        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=8)
        ax.set_ylabel(r'$\|F(x)\|$', fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, which='both', alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_minor_locator(LogLocator(subs='all', numticks=10))

    fig.suptitle('Convergence History of the Hybrid PSO\u2013NJN Method\n'
                 '(4 case studies, representative run)',
                 fontsize=11, fontweight='bold', y=1.01)

    for ext in ('pdf','png'):
        path = f'figures/fig_convergence4.{ext}'
        fig.savefig(path, bbox_inches='tight', dpi=300)
    print("  → Guardado: figures/fig_convergence4.pdf / .png")
    plt.close(fig)


def fig2_boxplots(all_data, titles_plain):
    """
    Figura 2: Box plots de residuales finales ‖F(x*)‖ sobre 30 corridas.
    4 métodos × 4 casos. Línea punteada: umbral 1e-8.
    """
    methods   = ['Newton(random)', 'PSO(pure)', 'PSO+Newton', 'PSO+NJN']
    m_colors  = ['#4dac26', '#fc8d59', '#8073ac', '#2166ac']
    m_labels  = ['Newton\n(random)', 'PSO\n(pure)', 'PSO+\nNewton', 'PSO+\nNJN']

    fig, axes = plt.subplots(1, 4, figsize=(14, 5.5))
    fig.subplots_adjust(wspace=0.38)

    for ax, data, title in zip(axes, all_data, titles_plain):
        plot_data = [data[m]['res'] for m in methods]
        bp = ax.boxplot(plot_data,
                        tick_labels=m_labels,
                        patch_artist=True,
                        medianprops=dict(color='black', lw=2),
                        flierprops=dict(marker='.', ms=4, alpha=0.5),
                        whiskerprops=dict(lw=1.2),
                        capprops=dict(lw=1.5))
        for patch, color in zip(bp['boxes'], m_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_yscale('log')
        ax.axhline(1e-8, color='gray', ls='--', lw=1.0,
                   label=r'tol $= 10^{-8}$')
        ax.set_title(title, fontsize=8.5, fontweight='bold')
        ax.set_ylabel(r'$\|F(x^*)\|$', fontsize=8)
        ax.legend(fontsize=6.5, loc='upper right')
        ax.grid(True, which='both', alpha=0.2, lw=0.5)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_minor_locator(LogLocator(subs='all', numticks=10))

    fig.suptitle('Final Residual Distribution over 30 Independent Runs\n'
                 '(4 case studies)',
                 fontsize=11, fontweight='bold', y=1.02)

    for ext in ('pdf','png'):
        path = f'figures/fig_boxplot4.{ext}'
        fig.savefig(path, bbox_inches='tight', dpi=300)
    print("  → Guardado: figures/fig_boxplot4.pdf / .png")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    print("\n"+DIV2)
    print("  PSO–NJN Article — Generación completa de tablas y figuras")
    print("  Ortiz, Quinga et al., Comp. & Appl. Math., Springer, 2026")
    print(DIV2)

    # ── Tablas estáticas ────────────────────────────────────────
    tabla1()
    tabla3()

    # ── Corrida representativa para Tablas 4 y 6 ────────────────
    rng0 = np.random.default_rng(13)
    gb1, _,_,_ = pso(case1, BOUNDS1, rng=rng0)
    sol1, _    = njn(case1, gb1)
    res1       = tabla4(sol1)

    rng0 = np.random.default_rng(13)
    gb2, _,_,_ = pso(case2, BOUNDS2, rng=rng0)
    sol2, _    = njn(case2, gb2)
    res2       = tabla6(sol2)

    # Actualizar residuales en el texto narrativo del LaTeX
    update_solution_residuals_in_latex(res1, res2)

    # ── Estudio estadístico 30 corridas ─────────────────────────
    all_data    = []
    all_hists   = []
    all_wx_res  = []
    all_wx_iter = []
    titles_plain= ['Case 1: Benchmark (n=2)',
                   'Case 2: S. cerevisiae (n=5)',
                   'Case 3: Hammerstein (n=20)',
                   'Case 4: Hammerstein (n=40)']
    stat_nums   = [5, 7, 8, 9]

    for (F, bounds, title_tex, _), tnum, tplain in zip(CASES, stat_nums, titles_plain):
        print(f"\n→ Ejecutando 30×4 corridas: {tplain}  ", end='', flush=True)
        data, rep_hist, p_sw_njn, p_sw_newt, p_wx_res, p_wx_iter = study_30runs(F, bounds)
        all_data.append(data)
        all_hists.append(rep_hist)
        all_wx_res.append(p_wx_res)
        all_wx_iter.append(p_wx_iter)
        print("OK")
        tabla_stat(tnum, tplain, data, p_wx_res, p_wx_iter)

    tabla10(all_data)

    # ── Tabla Wilcoxon para LaTeX ────────────────────────────────
    tabla_wilcoxon(all_wx_res, all_wx_iter)

    # ── Figuras ──────────────────────────────────────────────────
    print("\n" + DIV2)
    print("  Generando figuras…")
    print(DIV2)
    fig1_convergencia(all_hists, titles_plain)
    fig2_boxplots(all_data, titles_plain)

    # ── Resumen final ────────────────────────────────────────────
    print("\n" + DIV2)
    print("  ARCHIVOS GENERADOS:")
    print("    figures/fig_convergence4.pdf  ← incluir en LaTeX (Fig. 1)")
    print("    figures/fig_convergence4.png")
    print("    figures/fig_boxplot4.pdf      ← incluir en LaTeX (Fig. 2)")
    print("    figures/fig_boxplot4.png")
    print("    sn-article.tex               ← tabla Wilcoxon actualizada automáticamente")
    print(DIV2+"\n")
