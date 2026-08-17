# Exam Grader

Use Computer Vision to automatically grade multiple-choice exams from photos of student answer sheets.

## Prerequisites

- Python 3.13+
- pip

## Getting Started

1. Open a terminal and navigate to your project folder.

```sh
cd acv-app
```

2. Create a virtual environment.

```sh
python3 -m venv .venv
```

3. Activate your environment.

In your terminal, activate your environment with one of the following commands, depending on your operating system.

```sh
# Windows command prompt
.venv\Scripts\activate.bat

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS and Linux
source .venv/bin/activate
```

>[!NOTE]
> Once activated, you will see your environment name in parentheses before your prompt. `(.venv)`

4. Install dependencies.

```sh
pip install -r requirements.txt
```

5. Create your `.env` file from the example and fill in the values (e.g. `ANTHROPIC_API_KEY` if using the Claude vision provider).

```sh
cp .env.example .env
```

6. Run the app.

```sh
streamlit run app.py
```

7. You can now view the app in your browser: http://localhost:8501

## Linting

```sh
pyright src app.py
```

## Streamlit Setup

Follow these instructions to set up your Streamlit app:

- [Install Streamlit using command line - Streamlit Docs](https://docs.streamlit.io/get-started/installation/command-line)
