# Phenomenological Time Perception

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15533918.svg)](https://doi.org/10.5281/zenodo.15533918)
[![OSF](https://img.shields.io/badge/OSF-Preregistration-blue)](https://osf.io/8n7zg)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![LaTeX](https://img.shields.io/badge/Made%20with-LaTeX-1f425f.svg)](https://www.latex-project.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains the LaTeX sources, figures, and Python scripts for the study "A Multiscale ε–δ Metric Framework for Phenomenological Time Perception," preregistered on the Open Science Framework (OSF). The study validates a novel multiscale perceptual tick model that quantizes sensory inputs into discrete perceptual units (ticks) based on metric distances and temporal durations, using a simulation-based approach. The model is parameterized by ε (imperceptibility threshold) and δ (single-tick range), inspired by Cantor-like partitions and recursive algorithms. The repository includes scripts to generate theoretical figures and synthetic datasets, as well as LaTeX sources for the manuscript describing the model and simulation results. An appendix in the preregistration outlines a potential empirical protocol if funding is secured.

The preregistration is available on OSF at https://osf.io/8n7zg and will be updated with empirical data if human-subject data collection proceeds.

## Repository Structure

```
├── latex/
│   ├── Phenomenological_Time_Perception.tex  # Main LaTeX source for the manuscript
│   ├── metricchrono.bib                       # BibTeX file with cited references
│   ├── Phenomenological_Time_Perception.pdf  # Compiled manuscript
│   └── figures/                               # Directory for figure files
│       ├── epsilon_delta_ladder.png
│       ├── cantor_partition_depth5.png
│       ├── psychometric_curve.png
│       └── algorithm1_schematic.png
├── scripts/
│   ├── figures_generation.py   # Python script to generate theoretical figures
│   ├── analysis_plan.r         # Statistical models and visualisation pipeline
│   └── simulate_behaviour.py   # Python script to generate synthetic 2AFC data
├── Study Design/
│   ├── screening_survey.pdf    # Screening survey for potential empirical study
│   └── instructions.pdf        # Instructions for potential empirical study
├── README.md                   # This file
├── .gitignore                  # Git ignore file
└── LICENSE                     # License file (CC-BY 4.0)
```

## Dependencies

### Python Scripts

- **Python**: 3.8 or higher
- **Packages**:
  - `matplotlib` (>=3.5.0): For generating figures
  - `numpy` (>=1.21.0): For numerical computations
  - `pandas` (>=1.3.0): For synthetic data generation
- Install dependencies:
  ```bash
  pip install matplotlib numpy pandas
  ```

### LaTeX Compilation

- **LaTeX Distribution**: TeX Live (texlive-full, including texlive-fonts-extra) for PDFLaTeX
- **Packages** (included in preamble of `Phenomenological_Time_Perception.tex`):
  - `amsmath`, `amssymb`, `amsthm`, `mathtools`: Mathematical typesetting
  - `graphicx`: Figure inclusion
  - `natbib`: Bibliography management (using apalike style)
  - `geometry`: Page layout
  - `hyperref`: PDF hyperlinks
  - `tikz`: Graphics and diagrams
  - `algorithm`, `algpseudocode`: Algorithm typesetting
  - `booktabs`, `xcolor`: Table formatting
  - `enumitem`: List formatting
  - `microtype`: Typography improvements
- Fonts: Latin Modern (lmodern package)
- Install TeX Live (Ubuntu/Debian example):
  ```bash
  sudo apt-get install texlive-full texlive-fonts-extra
  ```
- For macOS with TeX Live Basic, install missing packages:
  ```bash
  sudo tlmgr update --self
  sudo tlmgr install enumitem algorithms algorithmicx
  ```

## Usage

### Generating Figures

The `figures_generation.py` script generates four theoretical figures visualizing the perceptual tick model:

- `epsilon_delta_ladder.png`: ε–δ ladder showing perceived ticks as a function of metric distance.
- `cantor_partition_depth5.png`: Cantor-like partition of sensory space (Σ).
- `psychometric_curve.png`: Psychometric curves for perceived units across durations and displacements.
- `algorithm1_schematic.png`: Schematic of the recursive multiscale tick computation (Algorithm 1).

To run:

```bash
cd scripts
python figures_generation.py
```

Output: Figures are saved in `latex/figures/`.

### Generating Synthetic Data

The `simulate_behaviour.py` script generates a synthetic dataset for a simulated two-alternative forced-choice (2AFC) task (30 virtual participants, 192 trials each, 4×4 displacement-duration grid).
To run:

```bash
cd scripts
python simulate_behaviour.py
```

Output: `simulated_2afc_data.csv` is saved in `latex/figures/`.

### Compiling the Manuscript

The LaTeX source (`Phenomenological_Time_Perception.tex`) compiles the manuscript, including figures and references.
To compile:

```bash
cd latex
pdflatex Phenomenological_Time_Perception.tex
bibtex Phenomenological_Time_Perception
pdflatex Phenomenological_Time_Perception.tex
pdflatex Phenomenological_Time_Perception.tex
```

Or using latexmk:

```bash
cd latex
latexmk -pdf Phenomenological_Time_Perception.tex
```

Output: `Phenomenological_Time_Perception.pdf` in `latex/`.

**Notes**:

- Ensure all figures are in `latex/figures/` before compiling.
- Use PDFLaTeX (not XeLaTeX or LuaLaTeX) to avoid fontspec issues.
- The preamble in `Phenomenological_Time_Perception.tex` includes all necessary packages.
- Bibliography uses `apalike` style for APA-format citations.

## Preregistration

The study is preregistered on OSF at https://osf.io/8n7zg. The preregistration focuses on simulation-based validation, with an appendix for a potential empirical protocol. Key hypotheses:

- H1: Perceived ticks follow a piecewise function based on ε and δ.
- H2: Sensory space partitioning resembles a Cantor-like structure.
- H3: Psychometric curves scale logarithmically with duration and inversely with displacement.
- H4 (Exploratory): Recursive algorithm predicts discrimination thresholds (r > 0.6).

## Contact

For questions, contact Adam Braun at mail.adam.braun@gmail.com. Issues can be reported via [GitHub Issues](https://github.com/AdamBraun/Phenomenological-Time-Perception/issues).

## Citation

If you use this work, please cite:

```bibtex
@misc{braun2025multiscale,
  author = {Braun, Adam},
  title = {A Multiscale ε–δ Metric Framework for Phenomenological Time Perception},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.15533918},
  url = {https://doi.org/10.5281/zenodo.15533918}
}
```

**APA Style:**
Braun, A. (2025). _A Multiscale ε–δ Metric Framework for Phenomenological Time Perception_ [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.15533918

**OSF Preregistration:**
Braun, A. (2025). A Multiscale ε–δ Metric Framework for Phenomenological Time Perception. OSF Preregistration. https://osf.io/8n7zg

## License

This work is licensed under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/). You are free to:

- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:

- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made

See the [LICENSE](LICENSE) file for the full license text.
