# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational course on machine learning model building ("Построение моделей машинного обучения") by Kalaidin. The repo contains instructor-provided notebooks and student solution notebooks organized by topic.

## Running Notebooks

```bash
# Run a notebook end-to-end and save output
/opt/homebrew/opt/python@3.10/bin/jupyter nbconvert --to notebook --execute <notebook.ipynb> --output <notebook.ipynb>

# Run Python scripts directly
/opt/homebrew/bin/python3.10 script.py

# Or use system python3
python3 script.py
```

## Repository Structure

```
Задания/
  Source/Модели построение анализ обучение/   # Instructor notebooks (read-only reference)
    1–15. <Topic>/                             # One folder per ML topic
  My/Модели построение анализ обучение/        # Student solutions (work here)
    4. Линейная регрессия/
    5. Логистическая регрессия/
    6. Обобщенные линейные модели/
    7. Метод опорных векторов/
```

## Course Topic Sequence

1. Loss functions & derivatives
2. Gradient descent
3. Variable relationships
4. Multiple linear regression
5. Logistic regression
6. Generalized linear models (GLM)
7. Support vector machines (SVM)
8. Generative classifiers
9. PCA
10. Artificial neural networks
11. Neural network training & gradient checking
12. Convolutional neural networks (CNN)
13. Recurrent neural networks (RNN) — uses `sholmes.txt` for text tasks
14. Regularization
15. Stochastic gradient descent (SGD)

## Workflow Pattern

- **Source notebooks** (`*_с_текстом.ipynb`) are instructor templates with embedded explanations — treat as read-only reference.
- **Student notebooks** follow two patterns:
  - Adapted source template (same filename prefix, in `My/`)
  - Custom dataset variant (filename includes dataset name, e.g. `04_my_data_MLR_california.ipynb`)
- Topics 4–7 have completed student solutions; topics 8–15 are not yet started.

## Dependencies

No `requirements.txt` exists. Notebooks use the standard ML stack: `numpy`, `scipy`, `sklearn`, `matplotlib`, `seaborn`, and for deep learning topics: `tensorflow`/`keras`. Install via pip if missing.
