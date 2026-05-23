# PSO–NJN: A Hybrid PSO–Fifth Order Iterative Technique for Nonlinear Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

Companion code for the article:

> Ortiz, N.<sup>1†</sup>, Quinga, S.<sup>1</sup>, Quinga, M.<sup>2</sup>, Castro, L.<sup>1</sup>, Socasi-Paucar, D.<sup>3</sup>
> **"A Hybrid PSO–Fifth Order Iterative Technique for Nonlinear Systems
> with Applications to Biological Models"**
> *Computational & Applied Mathematics*, Springer, 2026 (submitted).
>
> <sup>1</sup> Departamento de Ciencias Exactas, Universidad de las Fuerzas Armadas ESPE, Sangolquí, Ecuador  
> <sup>2</sup> School of Mathematical and Computational Sciences, Yachay Tech University, Urcuquí, Ecuador  
> <sup>3</sup> Departamento de Confiabilidad, Proyectos Integrales del Ecuador PIL S.A., Quito, Ecuador  
> <sup>†</sup> Corresponding author: ngortiz1@espe.edu.ec

---

## Description

This repository contains the Python implementation of the hybrid
PSO–NJN algorithm that combines:

- **Stage 1 — PSO**: Particle Swarm Optimization for global search
- **Stage 2 — NJN**: Fifth-order Newton–Jarratt iterative method for local refinement

The code reproduces all numerical tables and figures reported in the article,
including the 30-run statistical study across four test problems of increasing
dimension (n = 2, 5, 20, 40).

---

## Test Problems

| Case | Problem | Dimension |
|------|---------|-----------|
| 1 | Benchmark algebraic system (Xu, 2013) | n = 2 |
| 2 | *S. cerevisiae* metabolic network (Xu, 2013) | n = 5 |
| 3 | Hammerstein integral equation | n = 20 |
| 4 | Hammerstein integral equation | n = 40 |

---

## Requirements

```
Python >= 3.10
numpy
scipy
matplotlib
```

Install dependencies:

```bash
pip install numpy scipy matplotlib
```

---

## Usage

Run the main script to reproduce all tables and figures from the article:

```bash
python articulo_completo.py
```

This will generate:

- **Console output**: Tables 1, 3–10 and the Wilcoxon statistical tests
- `figures/fig_convergence4.pdf` — Figure 1 (convergence history, 4 cases)
- `figures/fig_boxplot4.pdf` — Figure 2 (residual box plots, 30 runs)
- `sn-article.tex` — LaTeX source updated automatically with computed values

All results are fully reproducible: fixed random seeds (`seed = run * 7 + 13`)
are used for each of the 30 independent runs.

---

## Repository Structure

```
PSO-NJN-biological/
│
├── articulo_completo.py      # Main script — generates all tables and figures
├── README.md                 # This file
└── figures/                  # Output directory (created automatically)
    ├── fig_convergence4.pdf
    └── fig_boxplot4.pdf
```

---

## Methods Compared

| Method | Order | Description |
|--------|-------|-------------|
| Newton (random) | 2 | Newton's method with random initialisation |
| PSO (pure) | — | Particle Swarm Optimization only |
| PSO + Newton | 2 | PSO global search + Newton local refinement |
| **PSO + NJN** | **5** | **PSO global search + NJN local refinement (proposed)** |

---

## Key Results

- PSO+NJN achieves **100% convergence** on all four test problems
- Pure PSO achieves **0% success** on both Hammerstein cases (n = 20, 40)
- PSO+NJN produces residuals **8.4× lower** than PSO+Newton on the metabolic model
- PSO+NJN requires significantly fewer iterations than PSO+Newton
  in **all four cases** (Wilcoxon p < 10⁻⁴)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ortiz2026psonjn,
  author  = {Ortiz, Nury and Quinga, Santiago and Quinga, Moisés and
             Castro, Lucía and Socasi-Paucar, Darwin},
  title   = {A Hybrid {PSO}--Fifth Order Iterative Technique for
             Nonlinear Systems with Applications to Biological Models},
  journal = {Computational \& Applied Mathematics},
  year    = {2026},
  note    = {Submitted}
}
```

---

## License

This project is licensed under the MIT License.

---

## Contact

Corresponding author: **Nury Ortiz** — ngortiz1@espe.edu.ec  
Departamento de Ciencias Exactas, Universidad de las Fuerzas Armadas ESPE, Sangolquí, Ecuador
