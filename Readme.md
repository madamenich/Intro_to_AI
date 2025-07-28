# AI Q&A Chatbot (Fine-tuned distilgpt2 on 50 Questions)

This project demonstrates how to fine-tune **distilgpt2** on a custom dataset of **50 AI-related questions and answers** and serve it as an interactive chatbot using **Streamlit**.

---

## Features
- Fine-tuning a pre-trained GPT-2 (DistilGPT2) on a small Q&A dataset.
- Streamlit web interface for asking questions and generating answers.
- Adjustable generation parameters (temperature, top-p, repetition penalty).
- Lightweight & fast (DistilGPT2 is smaller than standard GPT-2).

---

## Project Structure
```
fine-tune-gpt-2/
├── data/
│   └── ai_qa_50.txt          # 50 AI Q&A pairs
├── fine_tune_distilgpt2_ai.py # Script for fine-tuning
├── ai_chatbot_app.py          # Streamlit app for interactive Q&A
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## Setup Instructions

### 1. Clone or Copy the Project
```bash
git clone <https://github.com/madamenich/Intro_to_AI.git> 
cd Intro_to_AI
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
The key packages are:
- `torch==2.2.2`
- `transformers==4.40.2`
- `datasets==2.18.0`
- `streamlit==1.33.0`
- `numpy<2`

### 4. Fine-Tune the Model
```bash
python fine_tune_distilgpt2_ai.py
```
This will:
- Train `distilgpt2` on `data/ai_qa_50.txt`.
- Save the fine-tuned model in `./distilgpt2_ai_finetuned`.

### 5. Launch the Chatbot App
```bash
streamlit run ai_chatbot_app.py
```
Open the link (e.g., `http://localhost:8501`) to start asking AI-related questions.

---

## Usage Tips
- **Temperature:** Lower values (0.4–0.7) produce more focused answers; higher values add creativity but may cause repetition.
- **Repetition Penalty:** Already set to reduce loops.
- **Max New Tokens:** Controls the length of the answer.
- Try questions similar to the 50 training examples for best performance.

---

## Future Enhancements
- Add more Q&A data for improved accuracy and diversity.
- Use LoRA/PEFT for efficient fine-tuning of larger models.
- Integrate Retrieval-Augmented Generation (RAG) for dynamic, knowledge-grounded answers.
- Deploy the Streamlit app on Streamlit Cloud, Hugging Face Spaces, or Docker.

---

## Credits
- **Hugging Face Transformers** – [https://huggingface.co/transformers/](https://huggingface.co/transformers/)
- **Streamlit** – [https://streamlit.io/](https://streamlit.io/)
