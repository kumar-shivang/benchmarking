# Human Evaluation App Guide

This guide provides instructions for human evaluators to set up and run the Flask application for evaluating student answers.

## Prerequisites

- Python 3.8 or higher
- `pip` (Python package installer)

## Setup Instructions

### 1. Clone the Repository (if not already done)

```bash
git clone <repository-url>
cd benchmarking
```

### 2. Create a Virtual Environment

It is recommended to use a virtual environment to manage dependencies. Open PowerShell or Command Prompt in the project folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

Install the required Python packages for the human evaluation app:

```powershell
pip install -r human_eval_app/requirements.txt
```

## Running the Application

The application uses a pre-provided SQLite database (`human_eval_app\human_eval_source.db`). You do not need to set up any external databases.

1. Start the Flask server:

   ```powershell
   python human_eval_app/human_eval_app.py
   ```

2. Open your web browser and navigate to:
   [http://localhost:5000](http://localhost:5000)

## Evaluation Workflow

### 1. Set Your Name

On the home page, enter your name in the "Evaluator Name" field. This ensures all your evaluations are correctly attributed to you.

### 2. Start Evaluating

- Click on **"Start Evaluating"** or **"Evaluate Answers"** in the navigation bar.
- You will see a list of questions grouped by their source.
- Click on an answer that is marked as **"Pending"** to start evaluating it.

### 3. Other Features

- **Browse Questions**: View all questions and their solutions without evaluating.
- **View Stats**: Check the progress of evaluations and average scores.
- **Export**: Download the evaluation results as a CSV file.

### 4. Scoring Criteria

For each answer, you will be asked to provide scores (typically 1-5) and explanations for:

- **Correctness**: Is the answer factually correct?
- **Completeness**: Does it address all parts of the question?
- **Clarity**: Is the explanation easy to understand?
- **Overall Score**: Your general assessment of the answer.

### 4. Save and Continue

After filling out the scores and explanations, click **"Save Evaluation"**. You will be redirected back to the list to pick the next answer.

## Exporting Results

Once the evaluation is complete, the administrator can export the results to a CSV file by navigating to the `/export` route in the application or by clicking the **"Export"** link if available in the UI.

---
**Note:** All evaluation data is stored locally in `human_eval_app\human_eval.db`. Do not delete this file unless you want to reset all human evaluations.
