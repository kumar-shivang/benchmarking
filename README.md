# LLM Mathematics Benchmarking Tool

A comprehensive benchmarking system for evaluating Large Language Models (LLMs) on mathematical problem-solving tasks. The tool uses a student-evaluator paradigm where one LLM generates answers (student) and multiple LLMs evaluate those answers (evaluators), enabling objective comparison of model performance.

## Overview

This project provides a scalable framework to:

- Generate mathematical problem solutions using various LLMs
- Evaluate answer quality across multiple dimensions (correctness, completeness, clarity)
- Track costs and performance metrics across different models
- Support resumable execution for large-scale benchmarking
- Analyze and compare model performance with detailed statistics

## Features

- **Multi-Model Support**: Uses OpenRouter API to access 100+ LLMs
- **Parallel Processing**: Concurrent answer generation and evaluation with thread pooling
- **Robust Error Handling**: Authentication errors, API failures, and rate limiting management
- **Database Persistence**: SQLite database for storing questions, answers, evaluations, and costs
- **Resume Capability**: Continue interrupted benchmark runs from where they left off
- **Cost Tracking**: Detailed token usage and cost analysis per model and role
- **Interactive UI**: Streamlit web interface for manual testing and evaluation
- **Analysis Tools**: Rich CLI output with performance metrics and exportable reports
- **Configurable Prompts**: Customizable student and evaluator prompts via markdown files

## Project Structure

```
benchmarking/
├── main.py                      # CLI tool for running benchmarks
├── app.py                       # Streamlit web interface
├── llm.py                       # OpenRouter API integration
├── database.py                  # SQLAlchemy models and database operations
├── analyze.py                   # Performance analysis and reporting
├── student_prompt.md            # Prompt template for answer generation
├── evaluation_prompt.md         # Prompt template for evaluation
├── benchmarking.db             # SQLite database (generated)
├── benchmarking/               # Results directory
│   └── <model_name>_<timestamp>/ # Individual benchmark run data
└── benchmarking_data/          # Question datasets
    ├── Class 10 maths/         # Class 10 mathematics problems
    ├── class 11 maths/         # Class 11 mathematics problems
    ├── class 12 maths/         # Class 12 mathematics problems
    └── Class 9 maths/          # Class 9 mathematics problems
```

## Installation

### Prerequisites

- Python 3.8+
- OpenRouter API key ([Get one here](https://openrouter.ai/))

### Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd benchmarking
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   Create a `.env` file in the project root:

   ```env
   OPENROUTER_API_KEY=your_api_key_here
   ```

## Usage

### Command-Line Interface (CLI)

Run benchmarks using the CLI tool:

```bash
python main.py [OPTIONS]
```

#### Options

- `--student-model TEXT`: Model to generate answers (default: `deepseek/deepseek-v3`)
- `--evaluator-models TEXT`: Comma-separated evaluator models (default: `deepseek/deepseek-v3`)
- `--data-dir PATH`: Directory containing question JSON files (default: `benchmarking_data`)
- `--parallel-answers INT`: Number of concurrent answer generation threads (default: 5)
- `--parallel-evaluations INT`: Number of concurrent evaluation threads (default: 3)
- `--resume`: Resume from previous run

#### Examples

**Basic benchmark run:**

```bash
python main.py --student-model deepseek/deepseek-v3 \
               --evaluator-models "deepseek/deepseek-v3,anthropic/claude-3-opus"
```

**Resume interrupted run:**

```bash
python main.py --resume
```

**Custom data directory and parallelism:**

```bash
python main.py --data-dir ./my_questions \
               --parallel-answers 10 \
               --parallel-evaluations 5
```

### Web Interface

Launch the Streamlit app for interactive testing:

```bash
streamlit run app.py
```

Features:

- Select student and evaluator models via dropdowns
- Upload question JSON files
- View generated answers and evaluations in real-time
- Export results to CSV

### Analysis

Analyze benchmark results:

```bash
python analyze.py [--export]
```

Provides:

- Average scores by model (correctness, completeness, clarity, overall)
- Cost analysis (tokens used, total cost per model)
- Performance breakdown by evaluator
- Exportable CSV reports (with `--export` flag)

## Question Format

Questions should be stored as JSON files with the following structure:

```json
[
  {
    "question": "Solve for x: 2x + 5 = 15",
    "solution": "2x + 5 = 15\n2x = 10\nx = 5",
    "metadata": {
      "topic": "Linear Equations",
      "difficulty": "easy"
    }
  }
]
```

## Database Schema

### Tables

- **questions**: Question text, solutions, source files, metadata
- **runs**: Benchmark run information (student model, timestamp)
- **evaluations**: Evaluation scores and feedback
- **costs**: Token usage and cost tracking per model
- **student_answers**: Cached answers for resumability

## Evaluation Metrics

Each answer is evaluated on three dimensions (0-10 scale):

1. **Correctness**: Mathematical accuracy and reasoning
2. **Completeness**: Coverage of all question parts
3. **Clarity**: Explanation quality and notation standards

An overall score is calculated as: `(Correctness + Completeness + Clarity) / 3`

## Cost Tracking

The system tracks:

- Prompt tokens, completion tokens, total tokens
- API costs per request
- Breakdown by model and role (student/evaluator)
- Cumulative costs per benchmark run

## Architecture

### Workflow

1. **Question Loading**: Scan `benchmarking_data/` for JSON files
2. **Answer Generation**: Student model generates solutions (parallel)
3. **Evaluation**: Evaluator models score answers (parallel)
4. **Storage**: Results saved to SQLite database
5. **Analysis**: Aggregate statistics and export reports

### Parallelization

- **Answer Generation Pool**: Configurable thread pool for concurrent API calls
- **Evaluation Pool**: Separate thread pool for evaluation parallelism
- **Producer-Consumer**: Answers generated → queue → evaluated asynchronously

### Error Handling

- **Authentication Errors**: Graceful handling with user-friendly messages
- **Rate Limiting**: Exponential backoff with tenacity retry logic
- **Resumability**: Track evaluated questions to avoid reprocessing

## Configuration

### Prompts

Edit prompt templates:

- **student_prompt.md**: Customize how models generate answers
- **evaluation_prompt.md**: Customize evaluation criteria and scoring rubric

### Models

Access any OpenRouter-supported model:

```python
llm = LLM("anthropic/claude-3-opus")
llm = LLM("openai/gpt-4-turbo")
llm = LLM("google/gemini-pro-1.5")
```

## API Reference

### LLM Class

```python
from llm import LLM

llm = LLM("deepseek/deepseek-v3")
response, usage = llm.generate_response([
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello!"}
])
```

### Database Class

```python
from database import Database

db = Database("benchmarking.db")
db.save_question(question_id, question_text, solution, source_file, metadata)
db.get_or_create_run(student_model, timestamp)
```

## Troubleshooting

### Common Issues

1. **Authentication Error**
   - Verify API key in `.env` file
   - Check OpenRouter credit balance
   - Ensure API key has proper permissions

2. **Out of Memory**
   - Reduce `--parallel-answers` and `--parallel-evaluations`
   - Process smaller subsets of questions

3. **Database Locked**
   - Only one instance can write to the database at a time
   - Close other sessions or use `--resume` cautiously

## Contributing

Contributions are welcome! Areas for improvement:

- Additional evaluation metrics
- Support for other API providers (Anthropic, OpenAI direct)
- More sophisticated scoring algorithms
- Visualization dashboard
- Docker containerization

## License

[Add your license here]

## Acknowledgments

- OpenRouter for providing unified LLM API access
- SQLAlchemy for database management
- Rich and Click for beautiful CLI interfaces
- Streamlit for rapid web UI development
