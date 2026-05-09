# Connect 4 AI GUI 🏆

<img width="1352" height="878" alt="Screenshot 2026-03-03 at 10 19 13 PM" src="https://github.com/user-attachments/assets/48b5d7b4-f8c0-4c68-9256-eda67ec89f10" />

![Connect 4](https://img.shields.io/badge/Connect-4-blue) ![Python](https://img.shields.io/badge/Python-3.13.9-green?logo=python&logoColor=white) ![Status](https://img.shields.io/badge/Status-1st_Place-gold) <a href="https://www.linkedin.com/company/ksu-cs-fair-maarad-aalimat-alhasib"><img src="https://raw.githubusercontent.com/ComputerScientistsPF/logos/main/badge.png" width="95"  style="vertical-align: middle; object-fit: contain; margin-bottom:-5px;" alt="Fair Badge"></a>


A sophisticated Connect 4 AI system that won first place in the King Saud University (KSU) Machine Learning course [Kaggle competition](https://www.kaggle.com/competitions/csc462-connect-4). This implementation features an advanced 4-model stacking ensemble approach for optimal gameplay decision-making. The goal of the competiton was to use pure ML to predict the best moves, exact solvers like minimax and MCTS were not allowed.

## Features
- **Advanced AI Engine**: 4-model stacking ensemble for optimal move prediction
- **Interactive GUI**: Visual gameplay interface with real-time AI decision display
- **Performance Analytics**: Real-time statistics and move evaluation (displayed in the terminal)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ghaidahiz/Connect4AI
   cd connect4-ai
   
2. **Install required dependencies**

```bash
pip install -r requirements.txt   
```

3. **Install the required font**

- **The GUI uses ByteBounce.ttf**

## Requirements Preview
See requirements.txt for complete list. Major dependencies include:

- **Python 3.13.9**

- **NumPy**

- **PyGame (for GUI)**

- **XGBoost**

- **CatBoost**

- **Scikit-learn**

- **PyTorch (for CNN model)**



## Usage
  Launch the Game Interface (game.py)


## AI Architecture
Our winning solution employs a 4-model stacking ensemble:

- **Feature Engineering:** Extensive board state feature extraction

- **Model Diversity:** Four distinct model architectures for robust predictions (xgboost, random forest, catboost, CNN (alphazero style))

- **Stacking Ensemble:** Meta-learner (xgboost) that combines predictions from all base models


## Key Technical Aspects
- **Positional Evaluation:** Advanced heuristics for board state assessment

- **Threat Detection:** Pattern recognition for offensive/defensive plays

- **Long-term Strategy:** Multi-move lookahead capabilities

- **Efficiency Optimized:** Fast inference for real-time gameplay


## Contributors
- **Ghaida AlZaidan**

- **Leena Alonayq**

- **Eman Ameen**

- **Shahad Aldamegh**


## Acknowledgments
Prof. Najwa Altwaijry

King Saud University Machine Learning Course | CSC462

All competitors who participated in the competition

Kaggle



  <a href="https://github.com/ComputerScientistsPF">
    <img src="https://raw.githubusercontent.com/ComputerScientistsPF/logos/main/300px.png" width="70" alt="شعار معرض العالمات" />
  </a>

 


