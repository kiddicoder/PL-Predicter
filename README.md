Premier League Match Outcome Predictor
This project predicts the outcomes of Premier League matches (win or non-win) using past match data and machine learning techniques. The predictor leverages historical match data from 2020 to 2022, and it uses a RandomForestClassifier to train on key features like venue, opponent, match time, and day of the week. Additionally, rolling averages of recent team performance are incorporated to improve predictive power.

Table of Contents
Introduction
Installation
Usage
Features
Results
Contributing
License
Introduction
This project applies machine learning to predict match outcomes in Premier League football matches. The model is trained on a dataset of matches from 2020 to 2022. Using Random Forests, the model predicts whether a team will win or not. By incorporating rolling averages of team statistics (e.g., goals for/against, shots), the model's precision improves, especially when accounting for team form over the last few games.

Installation
Requirements
Make sure you have the following packages installed:

Python 3.7 or higher
pandas
scikit-learn
Install Dependencies
To install the required packages, run:

bash
Copy code
pip install pandas scikit-learn
Data
The project expects a matches.csv file that contains the following columns:

date: The date the match was played.
team: The home team.
opponent: The away team.
venue: Either "Home" or "Away" depending on where the match was played.
result: The result for the home team ("W" for win, "L" for loss, "D" for draw).
gf: Goals scored by the team.
ga: Goals conceded by the team.
sh: Number of shots taken.
sot: Shots on target.
dist: Average distance covered.
fk: Number of free kicks.
pk: Number of penalty kicks.
pkatt: Penalty kick attempts.
Ensure the data is available in the same directory as the script.

Usage
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/pl-predictor.git
cd pl-predictor
Place your matches.csv file in the root directory of the project.

Run the script:

bash
Copy code
python PL_Predictor.py
The script will:

Preprocess the match data.
Train a Random Forest model.
Generate predictions for test data.
Calculate accuracy and precision of the model.
Output actual vs. predicted results and additional performance metrics.
Example Output
bash
Copy code
Model Accuracy: 0.612
Model Precision: 0.475
Updated Model Precision with Rolling Averages: 0.645
Combined Results (actual vs predictions):
    actual  prediction       date     team   opponent result
55       0           1 2022-01-23  Arsenal    Burnley      D
...
Features
Random Forest Classifier: Predicts match outcomes based on multiple match features.
Rolling Averages: Incorporates recent team form (e.g., goals, shots) to improve predictive power.
Custom Team Name Mapping: Handles inconsistent team name formatting.
Detailed Output: Displays actual vs. predicted outcomes and performance metrics like precision and accuracy.
Results
The model achieves an accuracy of approximately 61% and a precision of 64.6% after incorporating rolling averages of team performance. These metrics suggest that the model performs reasonably well in predicting wins and non-wins, though there is room for improvement.

Confusion Matrix:
Predicted No Win	Predicted Win
Actual No Win	140	32
Actual Win	75	29
Precision: 64.6% (after rolling averages)
Future Improvements
Adding additional features like player availability, injury reports, or detailed home/away form could enhance predictive performance.
Incorporating more advanced algorithms like Gradient Boosting or tuning the Random Forest hyperparameters could further optimize the model.
Expanding the model to handle draws explicitly (e.g., a multi-class classification problem) may also improve performance.
Contributing
Feel free to fork this project, submit issues, or contribute through pull requests.

License
This project is licensed under the MIT License. See the LICENSE file for details.
