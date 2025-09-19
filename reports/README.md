# 🚀 AI Team Builder - MBTI Prediction and Team Suggestion
This project is a web application prototype that uses machine learning to predict the 16 MBTI personality types from text, subsequently providing suggestions for roles and optimal team configurations.

# ✨ Key Features
4-Dimension MBTI Prediction: Analyzes English text to predict the four personality axes: Introversion/Extraversion (I/E), Intuition/Sensing (N/S), Thinking/Feeling (T/F), and Judging/Perceiving (J/P).

Role Suggestion: Automatically maps the 16 MBTI types to 8 common team roles (e.g., Leader, Planner, Executor...).

Explainable Results (XAI): Displays the keywords that most significantly influence the model's decision for each axis, enhancing transparency.

Team Building Algorithm: Integrates a greedy algorithm to propose a 4-person team optimized for diversity (based on Hamming distance) and balance of core roles.

Interactive Web Interface: Built with Gradio, allowing users to easily input names and texts for multiple members to receive analysis results.

# ⚙️ How It Works
The application is built on the following pipeline:

Text Embedding: The sentence-transformers/all-MiniLM-L6-v2 model is used to convert input text into semantic vectors.

Classification: Four separate pre-trained Logistic Regression models are used to predict the probability for each personality dimension.

Explanation: A "Leave-One-Out" method is applied to quickly identify the most influential keywords affecting the prediction outcomes.

Team Suggestion: A greedy algorithm is implemented to select the optimal team, starting with the member whose prediction has the highest confidence and iteratively adding members who maximize diversity.

# 🛠️ Installation and Setup
Prerequisites
Python 3.10+

Libraries listed in requirements.txt

Installation Guide
Clone this repository to your machine:

git clone [YOUR_REPO_URL]
cd [REPO_NAME]

Create and activate a virtual environment:

# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate on macOS/Linux
source .venv/bin/activate

Install the required libraries:
Note: Ensure you have a requirements.txt file. If not, run pip freeze > requirements.txt in your activated virtual environment after installing gradio, pandas, joblib, sentence-transformers, scikit-learn, torch, and emoji.

pip install -r requirements.txt

How to Use
Re-train the models (Optional):

Open and run the notebooks/notebook.ipynb file.

This process will re-train the 4 LogisticRegression models from scratch and save them to the models/ directory. You only need to do this if you want to train on new data.

Run the Web Application:

Open a terminal in the project's root directory.

Run the command:

python app.py

Open your browser and navigate to the local URL provided (usually http://127.0.0.1:7860).

# 📜 Ethical Considerations
Reference Tool: This application should only be considered a reference tool to suggest ideas and spark discussion.

Not for Critical Decisions: The prediction results are not a definitive judgment of a person and should absolutely not be used for critical decisions such as hiring, performance reviews, or promotions.

Privacy: The application is designed to run locally and does not store any text entered by the user.

# 🔮 Future Development
[ ] Improve accuracy by using larger language models.

[ ] Upgrade the team suggestion algorithm to handle more complex constraints (e.g., technical skills).

[ ] Collect and train on a specialized dataset from a corporate environment to increase practical applicability.

[ ] Integrate other personality models like the Big Five.
