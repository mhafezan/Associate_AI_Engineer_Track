# 📘 Working with the OpenAI API

This repository demonstrates practical usage of the OpenAI API through structured Python scripts. It focuses on building foundational skills for interacting with large language models, designing prompts, and developing interactive applications such as CLI tools and chatbots.

---

## 🚀 Project Overview

This project contains two main components:

1. **Fundamentals CLI Tool** – A command-line interface for generating structured responses (e.g., study plans) using configurable parameters.
2. **Interactive AI Chatbot** – A real-time terminal-based chatbot that maintains conversational context.

These implementations showcase best practices such as:
- Secure API key management
- Prompt engineering with system/user roles
- Token control and cost estimation
- Stateful conversation handling

---

## 📂 Project Structure

```
Working with the OpenAI API/
│
├── 1_Fundamentals.py     # CLI-based OpenAI API tool
├── 2_AI_ChatBot.py       # Interactive chatbot in terminal
└── README.md
```

---

## 🧠 1. Fundamentals CLI Tool

### Description
A flexible command-line tool that interacts with the OpenAI API to generate structured outputs (e.g., language learning plans). It demonstrates argument parsing, prompt structuring, and optional cost estimation.

### Key Features
- Argument parsing using `argparse`
- Configurable parameters:
  - Model selection
  - Token limits
  - Temperature
- Prompt engineering with system instructions
- Cost estimation based on token usage

### Example Usage

```bash
python3 1_Fundamentals.py   --model gpt-4o-mini   --user_content "I want to learn to speak Dutch. Create a study plan for me."   --max_completion_tokens 400   --temperature 0.7   --cost
```

---

## 🤖 2. Interactive AI Chatbot

### Description
A terminal-based chatbot that continuously accepts user input and responds using the OpenAI API. It maintains conversation history to enable context-aware interactions.

### Key Features
- Real-time interaction via terminal (PowerShell / VSCode)
- Persistent conversation memory
- Exit commands (`exit`, `quit`)
- Concise response generation

### Run the Chatbot

```bash
python3 2_AI_ChatBot.py
```

---

## 🔐 Environment Setup

### 1. Install Dependencies

```bash
pip install openai
```

### 2. Set API Key (Windows PowerShell)

```powershell
setx OPENAI_API_KEY "your-api-key-here"
```

Restart your terminal after setting the environment variable.

---

## ⚙️ Key Concepts Demonstrated

- Prompt Engineering
- Token Management
- Cost Estimation
- Stateful Conversations

---

## 📈 Potential Improvements

- Streaming responses (real-time token output)
- Dynamic pricing per model
- Logging and analytics
- GUI interface (e.g., Flask or Streamlit)

---

## 👨‍💻 Author

Mohammad Hafezan
