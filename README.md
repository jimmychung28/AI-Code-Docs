# AI Code Docs

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green)]()

AI Code Docs is an intelligent code documentation assistant designed to streamline and enhance the process of creating, maintaining, and exploring code documentation. By leveraging state-of-the-art AI models, this tool can generate precise, human-readable documentation for your codebase, making it easier for developers to onboard, collaborate, and maintain high-quality projects.

---

## Features

### 🚀 **Automated Documentation Generation**
- Parses your codebase to generate detailed function, class, and module-level documentation.
- Supports multiple programming languages (Python, JavaScript, Java, etc.).
- Extracts docstrings and adds meaningful comments where missing.

### 📚 **Natural Language Queries**
- Search your codebase using natural language questions.
- Get instant explanations for functions, classes, and modules.

### ✍️ **Customizable Output**
- Tailor the format, style, and depth of the documentation to match your team's conventions.
- Markdown, HTML, or plain text output options.

### 🔍 **Code Understanding**
- Analyzes code dependencies, structure, and purpose for accurate documentation.
- Includes diagrams for class hierarchies or workflows where applicable.

### 🤝 **Collaboration-Ready**
- Integrates with popular version control platforms like GitHub and GitLab.
- Comment and review AI-generated documentation as a team.

---

## Getting Started

### Prerequisites
- Python 3.8+ installed
- Access to OpenAI or other supported AI API keys (for AI generation)

### Installation
Clone the repository:
```bash
git clone https://github.com/jimmychung28/AI-Code-Docs.git
cd AI-Code-Docs
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Set up your API key:
```bash
export AI_API_KEY="your_api_key_here"
```

Run the application:
```bash
python ai_code_docs.py
```

---

## Usage

### Generate Documentation
Run the following command to document your code:
```bash
python ai_code_docs.py --path ./your-codebase --output ./docs
```

### Query Your Codebase
```bash
python ai_code_docs.py --query "What does the function `process_data` do?"
```

### Customize Configuration
Edit the `config.yaml` file to adjust the AI model, output format, or language preferences.

---

## Roadmap
- [ ] Expand support for additional programming languages.
- [ ] Add CI/CD integration for automated documentation updates.
- [ ] Improve natural language query accuracy and context awareness.
- [ ] Implement a web-based interface for real-time documentation browsing.

---

Let AI take the grunt work out of documenting your code. Happy coding! 😊
