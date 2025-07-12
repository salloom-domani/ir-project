# 📚 IR Project – Backend

This is the backend service for an Information Retrieval (IR) system designed for experimenting with various vector-based and traditional retrieval techniques. It provides APIs for indexing, querying, and evaluating a corpus of documents using different models.

🔗 **Frontend Repo**: [salloom-domani/ir-front](https://github.com/salloom-domani/ir-front)

---

## 🚀 Motivation

The goal of this project is to provide a flexible and educational platform for experimenting with and evaluating different IR algorithms. Whether you're a student, researcher, or just interested in search technologies, this backend is designed to help you:

- Upload and manage custom document collections.
- Apply different retrieval models like **TF-IDF**, **BM25**, and **HuggingFace Transformers**.
- Evaluate performance using standard IR metrics like **Precision**, **Recall**, **MAP**, and **MRR**.
- Understand the effect of vectorization, tokenization, and query strategies.

---

## 🛠️ Tech Stack

| Purpose       | Technology                                |
| ------------- | ----------------------------------------- |
| Web Framework | FastAPI                                   |
| ML & NLP      | scikit-learn, transformers, numpy, pandas |
| Server        | Uvicorn                                   |
| Evaluation    | Custom metrics scripts                    |

---

## 📂 Project Structure (Key Files)

```
ir-project/
│
├── main.py # Entry point with FastAPI app
├── models/ # Retrieval models (TF-IDF, BM25, etc.)
├── utils/ # Tokenizers, vectorizers, helpers
├── eval/ # Evaluation logic and IR metrics
├── data/ # Example documents / queries (optional)
└── requirements.txt # Dependencies
```

---

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/salloom-domani/ir-project.git
cd ir-project
```

2. Setup virtual environment
   bash
   Copy
   Edit
   python -m venv venv
   source venv/bin/activate
3. Install requirements
   bash
   Copy
   Edit
   pip install -r requirements.txt
4. Run the development server
   bash
   Copy
   Edit
   uvicorn main:app --reload
   You can now access the API at:
   📍 <http://localhost:8000>

💡 Example Use Cases
Upload a list of documents via POST

Send a query to retrieve top matching documents

Evaluate retrieval performance using ground truth

Switch between models (e.g., bm25, tfidf, transformer)

🧪 Sample API Endpoints
Endpoint Method Description
/documents/upload POST Upload a batch of documents
/query POST Retrieve results for a query
/evaluate POST Run evaluation using test queries
/models GET List available models

Use Swagger UI for exploring the API interactively:
🔍 <http://localhost:8000/docs>

🖼️ Frontend UI
The project has a separate frontend built with modern web tech that lets you:

Upload datasets

Run queries interactively

View ranked results

Compare evaluation metrics visually

👉 View Frontend Source

🤝 Contributing
Interested in improving the models, UI, or evaluation pipeline? Contributions are welcome!

Clone and fork the project

Use feature branches

Submit PRs with clear descriptions

Please include tests where applicable

📄 License
This project is licensed under the MIT License.
Feel free to use, modify, and share it as long as the license file is retained.

✨ Author
Developed by Salloom Domani
