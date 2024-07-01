# Trashbox-ARD
Thesis Title: Commpressed and Adversarially Robust Solid Waste Image Classification Model using Adversarially Robust Distillation (ARD)

![conceptual_framework](docs\conceptual_framework.png)

## Requirements
1. Powerful machine - since Adversarial Training and ARD really is computationally intensive. The TrashboxARD thesis proposal has been trained with RTX 3060 Ti.
2. Python and pip
3. Installed requirements - just follow the steps below to install all the dependencies. 
## Getting Started
To train your own efficient and adversarially robust image classification model, first you must install all the dependencies / requirements by creating an virtual environment and install all the dependencies

1. Create virtual environment 
```
python -m venv .venv 
```
2. Activate virtual environment
```
// For command prompt
.venv\Scripts\activate

// For powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
.venv\Scripts\Activate.ps1

// For other (Unix / Linux)
source .venv/bin/activate
```
3. Install the dependencies:
```
pip install -r requirements/default.txt
```