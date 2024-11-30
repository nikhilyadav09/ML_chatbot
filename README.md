# Information Retrieval System with Generative AI

This project implements an advanced Information Retrieval (IR) system designed to process machine learning (ML)-related queries. It retrieves contextually relevant documents and summarizes them using a generative AI model, LLaMA 3.1 70B, providing a user-friendly interface for query handling, feedback, and performance analysis.

---

## Key Features

### User Functionality
- **Search Queries:** Retrieves relevant ML documents and generates concise summaries.
- **Chat History:** Tracks previous queries, allowing users to revisit and manage past conversations.
- **Feedback System:** Users can like, dislike, or comment on chatbot responses, improving system accuracy.
- **Text-to-Speech Integration:** Offers accessibility for diverse audiences.

### Admin Dashboard
- **Data Management:** Embed new documents via CSV upload, with automatic embedding generation.
- **Analytics:** Provides insights into user behavior, system performance, and feedback trends via visualizations.

### Advanced AI Capabilities
- **Generative Summarization:** LLaMA 3.1 70B generates summaries for retrieved results, improving response relevance.
- **Query Reranking:** Utilizes tools like Cohere Reranker to prioritize the most contextually relevant responses.
- **Sentiment Analysis:** Analyzes feedback sentiment to measure user satisfaction and refine retrieval accuracy.
- **Visualization:** Embedding visualization and clustering using PCA and KMeans for enhanced data insights.

---

## Project Structure
```
├── app.py                  # Core Flask application with routes and APIs
├── templates/              # HTML templates for user interface
├── static/                 # CSS, JS, and other frontend assets
├── .env                    # Environment variables for secure configurations
└── README.md               # Project documentation
```

---

## Technologies Used

- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Database:** PostgreSQL with pgvector extension for vectorized similarity search
- **AI Models:** LLaMA 3.1 70B, SentenceTransformers (`all-MiniLM-L6-v2`)
- **Libraries:**
  - `scikit-learn`: Clustering and dimensionality reduction
  - `nltk`: Sentiment analysis using VADER
  - `matplotlib`: Data visualizations
  - `werkzeug`: Secure authentication
  - `pandas`: Data preprocessing and analysis

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- PostgreSQL with pgvector extension
- `.env` file with the following variables:
  ```
  DB_NAME=<your_db_name>
  DB_USER=<your_db_user>
  DB_PASSWORD=<your_db_password>
  DB_HOST=<your_db_host>
  DB_PORT=5432
  ```

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/nikhilyadav09/ML_chatbot.git
   cd ML_chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the PostgreSQL database:
   ```sql
   CREATE DATABASE document_embeddings;
   CREATE EXTENSION vector;
   ```

4. Initialize the application:
   ```bash
   python app.py
   ```

5. Access the application at `http://localhost:5000`.

---

## API Endpoints

### User APIs
- **`POST /search`**: Submit a query to retrieve and summarize relevant documents.
- **`GET /get_chat/<chat_id>`**: Retrieve a specific chat entry.
- **`DELETE /delete_chat/<chat_id>`**: Delete a specific chat entry.
- **`POST /save_feedback`**: Submit feedback for a chat response.
- **`GET /api/sentiment-analysis`**: Analyze sentiment trends in user feedback.
- **`GET /api/query-classification`**: Classify historical user queries into categories.

### Admin APIs
- **`POST /admin`**: Upload a CSV file to embed new documents into the database.
- **`GET /api/embedding-visualization`**: Generate a visualization of embeddings and their clusters.
- **`GET /get_document_count`**: Get the total count of embedded documents.
- **`GET /api/top-users-chart`**: Generate a bar chart showing the top users based on search counts.

---

## How It Works

### User Workflow
1. Log in or register via the web interface.
2. Submit a query to search for ML-related information.
3. View the retrieved results with relevance scores and AI-generated summaries.
4. Provide feedback or manage chat history.

### Admin Workflow
1. Log in using admin credentials.
2. Embed new documents by uploading a CSV file.
3. Monitor user activity, feedback trends, and system performance on the dashboard.

---

## Challenges & Future Work

### Challenges
- Handling ambiguous or diverse queries effectively.
- Managing large-scale embeddings for performance optimization.

### Future Work
- Automate feedback-based dynamic learning.
- Implement domain-specific fine-tuning for better contextual accuracy.
- Introduce two-factor authentication for enhanced security.

---

## Authors
- **Nikhil Yadav** - Chatbot features, admin dashboard, book scraping, and embedding management.
- **Jitendra Kumar** - Data scraping, sentiment analysis, and generative AI integration.
- **Nikhil Raj Soni** - Authentication, user-specific alignment, and data cleaning.
- **Rashmi Kumari** - Book data cleaning and backend-frontend integration.

---

## License
This project is licensed under the MIT License.

