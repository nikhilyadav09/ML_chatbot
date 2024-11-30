import os
import csv
import torch
import psycopg2
import hashlib
import uuid
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from pgvector.psycopg2 import register_vector
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from groq import Groq
import psycopg2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json
import os
import numpy as np
import psycopg2
import base64
import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


class InformationRetrievalSystem:
    def __init__(self, db_params):
        # Initialize Sentence Transformer model
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Database connection
        self.conn = psycopg2.connect(**db_params)
        register_vector(self.conn)
        
        # Initialize database
        self.setup_database()
    
    def setup_database(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE EXTENSION IF NOT EXISTS vector;
            
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                search_count INTEGER DEFAULT 0,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                role VARCHAR(20) NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS document_embeddings (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                source VARCHAR(255),
                chapter_name VARCHAR(255),
                embedding vector(384)
            );
            
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                user_id INTEGER,
                query TEXT,
                response TEXT,
                source VARCHAR(255),
                chapter_name VARCHAR(255),
                similarity DOUBLE PRECISION,
                feedback BOOLEAN DEFAULT NULL, 
                feedback_comment TEXT DEFAULT NULL,        
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

        """)
        self.conn.commit()
        cursor.close()
    def register_admin(self, username, password):
        cursor = self.conn.cursor()
        hashed_password = generate_password_hash(password)
        
        try:
            cursor.execute(
                "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)",
                (username, hashed_password, 'admin')
            )
            self.conn.commit()
            return True
        except psycopg2.IntegrityError:
            self.conn.rollback()
            return False
        finally:
            cursor.close()
            
    def register_user(self, username, password):
        cursor = self.conn.cursor()
        hashed_password = generate_password_hash(password)
        
        try:
            cursor.execute(
                "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)",
                (username, hashed_password, 'user')
            )
            self.conn.commit()
            return True
        except psycopg2.IntegrityError:
            self.conn.rollback()
            return False
        finally:
            cursor.close()
    
    def validate_admin(self, username, password):
        cursor = self.conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username = %s AND role = 'admin'", (username,))
        result = cursor.fetchone()
        cursor.close()
        
        if result and check_password_hash(result[0], password):
            return True
        return False
    
    def validate_user(self, username, password):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, password FROM users WHERE username = %s AND role = 'user'", (username,))
        result = cursor.fetchone()
        cursor.close()
        
        if result and check_password_hash(result[1], password):
            return result[0]  # Return user ID
        return None

    
    def embed_csv_documents(self, csv_path):
        print(f"Starting to embed documents from {csv_path}")
        
        cursor = self.conn.cursor()
        
        # Ensure the table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_embeddings (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                source VARCHAR(255),
                chapter_name VARCHAR(255),
                embedding vector(384)
            )
        """)
        self.conn.commit()
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                headers = next(csv_reader)  # Capture headers for debugging
                print(f"CSV Headers: {headers}")
                
                # Prepare a new cursor for transaction
                cursor = self.conn.cursor()
                
                row_count = 0
                successful_rows = 0
                
                for row in csv_reader:
                    row_count += 1
                    
                    try:
                        # Validate row has enough columns
                        if len(row) < 3:
                            print(f"Warning: Skipping row {row_count} - insufficient columns")
                            continue
                        
                        chapter_name = row[1]  # First column is chapter name
                        source = row[2]  # Second column is source
                        text = row[3]  # Third column is text                        
                        # Truncate very long texts if necessary
                        
                        print(f"Embedding text: {text[:50]}...")  # Print first 50 chars
                        
                        embedding = self.model.encode(text).tolist()
                            
                        cursor.execute(
                            "INSERT INTO document_embeddings (text, source, chapter_name, embedding) VALUES (%s, %s, %s, %s)",
                            (text, source, chapter_name, embedding)
                        )
                        
                        successful_rows += 1
                    
                    except Exception as row_error:
                        print(f"Error embedding row {row_count}: {row_error}")
                        # Continue processing other rows instead of stopping entire process
                        continue
                
                self.conn.commit()
                print(f"Committed {successful_rows}/{row_count} rows to database")
            
            return f"Successfully embedded {successful_rows} documents from {csv_path}"
        
        except FileNotFoundError:
            print(f"Error: File not found - {csv_path}")
            return f"Failed to embed documents from {csv_path} - File not found"
        
        except Exception as e:
            print(f"Unexpected error during CSV processing: {e}")
            self.conn.rollback()
            return f"Failed to embed documents from {csv_path}"
        
        finally:
            if not cursor.closed:
                cursor.close()

    def search_documents(self, query, user_id, top_k=1):  # Default top_k to 1 for single best match
        query_embedding = self.model.encode(query).tolist()
        
        cursor = self.conn.cursor()
        try:
            # Execute SQL query to find top_k matches ordered by similarity
            cursor.execute("""
                SELECT text, source, chapter_name, 1 - (embedding <=> %s::vector) as similarity 
                FROM document_embeddings 
                ORDER BY similarity DESC 
                LIMIT %s
            """, (query_embedding, top_k))
            
            results = cursor.fetchall()
            if not results:
                # No matches found
                response_text = "No relevant matches found."
                response_data = []  # Empty response if no matches
            else:
                # Format results into structured response
                response_data = [
                    {
                        "text": text, 
                        "source": source, 
                        "chapter_name": chapter_name,
                        "similarity": float(similarity)
                    } 
                    for text, source, chapter_name, similarity in results

                ]
                print("response: recieved........")
                
                # Prepare chat history text for the best match
                top_result = response_data[0]  # The single best result
                response_text = (
                    f"{top_result['text']} "
                    f"(Similarity: {top_result['similarity']:.2f}, "
                    f"Chapter: {top_result['chapter_name']}, "
                    f"Source: {top_result['source']})"
                )
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER,
                        query TEXT,
                        response TEXT,
                        source VARCHAR(255),
                        chapter_name VARCHAR(255),
                        similarity DOUBLE PRECISION,
                        feedback BOOLEAN DEFAULT NULL, 
                        feedback_comment TEXT DEFAULT NULL,        
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id)
                    )
                """)
                # Save chat history with relevant context
                cursor.execute("""
                    INSERT INTO chat_history (
                        user_id, query, response, similarity, source, chapter_name, timestamp
                    ) VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """, (
                    user_id,
                    query,
                    (top_result['text']),  # Formatted response text
                    (top_result["similarity"]),  # Similarity score
                    top_result["source"],  # Source link
                    top_result["chapter_name"]  # Chapter name
                ))
                self.conn.commit()
            
            return response_data  # Return structured response data to the caller
        
        except Exception as e:
            # Error handling
            print(f"Error in search_documents: {str(e)}")
            self.conn.rollback()  # Rollback on failure
            return []
        
        finally:
            # Ensure cursor is always closed
            cursor.close()


    def get_user_chat_history(self, user_id):
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                user_id INTEGER,
                query TEXT,
                response TEXT,
                source VARCHAR(255),
                chapter_name VARCHAR(255),
                similarity DOUBLE PRECISION,
                feedback BOOLEAN DEFAULT NULL, 
                feedback_comment TEXT DEFAULT NULL,        
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
    
            # Fetch the chat history, including new fields
            cursor.execute("""
                SELECT id, query, response, similarity, source, chapter_name, timestamp 
                FROM chat_history 
                WHERE user_id = %s 
                ORDER BY timestamp DESC  
                LIMIT 20
            """, (user_id,))
            
            history = cursor.fetchall()
            
            # Return history with all required fields
            return [{
                "id": id,
                "query": query,
                "response": response,
                "similarity": float(similarity) if similarity is not None else None,
                "source": source,
                "chapter_name": chapter_name,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
            } for id, query, response, similarity, source, chapter_name, timestamp in history]
        
        except Exception as e:
            print(f"Error in get_user_chat_history: {str(e)}")
            return []
        finally:
            cursor.close()


# Flask Application Setup
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
CORS(app)

# Database Connection Parameters
DB_PARAMS = {
    'dbname': 'postgres',  # or your specific database name
    'user': 'postgres',
    'password': 9301,  # the password you just set
    'host': 'localhost',
    'port': 5432  # add port if not using default
}
# Initialize Information Retrieval System
ir_system = InformationRetrievalSystem(DB_PARAMS)

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    # Get user's chat history
    chat_history = ir_system.get_user_chat_history(session['user_id'])
    return render_template('index.html', chat_history=chat_history)  # Fixed variable name

# Function to establish database connection
def get_database_connection(DB_PARAMS):
    """Establish a connection to the PostgreSQL database."""
    return psycopg2.connect(**DB_PARAMS)

# Fetch embeddings from the database
def fetch_embeddings(db_params):
    """Fetch embeddings from the database."""
    try:
        with get_database_connection(db_params) as conn:
            with conn.cursor() as cur:
                # Fetch all embeddings
                cur.execute("""
                    SELECT embedding 
                    FROM document_embeddings 
                    WHERE embedding IS NOT NULL
                """)
                rows = cur.fetchall()
                
                # Return embeddings as numpy arrays
                if not rows:
                    print("No embeddings found in database")
                    return None
                
                embeddings = [np.frombuffer(row[0], dtype=np.float32) for row in rows]
                return np.array(embeddings)
    except Exception as e:
        print(f"Error fetching embeddings: {e}")
        return None

# Process embeddings with PCA and clustering
def process_embeddings(embeddings, n_clusters=3):
    """Process embeddings with dimensionality reduction and clustering."""
    if embeddings is None or len(embeddings) == 0:
        return None, None
    
    # Dimensionality Reduction using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Clustering using KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    return reduced_embeddings, clusters

# Generate embedding visualization data
def generate_embedding_visualization(db_params):
    """Generate visualization data for embeddings."""
    embeddings = fetch_embeddings(db_params)
    
    if embeddings is None:
        return None
    
    reduced_embeddings, clusters = process_embeddings(embeddings)
    
    # Prepare data for frontend
    visualization_data = {
        'embeddings': reduced_embeddings.tolist(),
        'clusters': clusters.tolist()
    }
    
    return visualization_data

# API endpoint to generate embedding visualization data
@app.route('/api/embedding-visualization', methods=['GET'])
def embedding_visualization():
    """API endpoint to generate embedding visualization data."""
    try:
        # Generate embedding visualization data
        visualization_data = generate_embedding_visualization(DB_PARAMS)
        
        if visualization_data is None:
            return jsonify({
                'status': 'error',
                'message': 'No embeddings found in the database.'
            }), 404
        
        return jsonify({
            'status': 'success',
            'visualizationData': visualization_data
        })
    except Exception as e:
        print(f"Error in embedding visualization API: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    
@app.route('/api/query-classification', methods=['GET'])
def query_classification():
    """API endpoint to classify queries from chat history."""
    try:
        # Establish database connection
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        # Fetch all queries from chat_history table
        cur.execute("""
            SELECT query FROM chat_history
        """)
        queries = cur.fetchall()

        if not queries:
            return jsonify({
                'status': 'error',
                'message': 'No queries found in chat history.'
            }), 404

        # Flatten the list of tuples to a list of strings
        queries = [query[0] for query in queries]

        # Example classification logic (you can replace this with a trained model)
        vectorizer = CountVectorizer()
        classifier = MultinomialNB()

        # Assume we have predefined training data
        training_data = [
            "how to train a machine learning model",
            "best practices for natural language processing",
            "spam email example",
            "example of phishing email"
        ]
        training_labels = ["ml_related", "ml_related", "spam", "spam"]

        # Fit the model with the training data
        vectors = vectorizer.fit_transform(training_data)
        classifier.fit(vectors, training_labels)

        # Classify all queries from chat history
        query_vectors = vectorizer.transform(queries)
        predicted_classes = classifier.predict(query_vectors)

        # Combine queries with their predicted classes
        results = [
            {"query": query, "predictedClass": predicted_class}
            for query, predicted_class in zip(queries, predicted_classes)
        ]

        # Close the database connection
        cur.close()
        conn.close()

        return jsonify({
            'status': 'success',
            'classifiedQueries': results
        })

    except Exception as e:
        print(f"Error in query classification API: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/sentiment-analysis', methods=['GET'])
def sentiment_analysis_visualization():
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    import matplotlib.pyplot as plt
    import io
    import base64

    # Ensure VADER lexicon is downloaded
    nltk.download('vader_lexicon', quiet=True)

    # Initialize VADER Sentiment Analyzer
    sia = SentimentIntensityAnalyzer()

    try:
        # Load the data
        file_path = 'analysis.csv'  # Ensure this path is correct
        df = pd.read_csv(file_path)

        # Feature Extraction
        df['answer_length'] = df['answer'].apply(lambda x: len(str(x).split()))
        df['question_length'] = df['question'].apply(lambda x: len(str(x).split()))
        df['sentiment_score'] = df['feedback'].apply(lambda feedback: sia.polarity_scores(str(feedback))['compound'])

        # Prepare data for Linear Regression
        X = df[['answer_length', 'question_length', 'sentiment_score']]
        y = df['sentiment_score']

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the Linear Regression model
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        # Make predictions
        y_pred = regressor.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Create visualization
        plt.figure(figsize=(10, 6))
        df['question_index'] = np.arange(len(df))
        plt.scatter(df['question_index'], df['sentiment_score'], color='blue', label='Actual Sentiment Score', alpha=0.7)
        plt.plot(df['question_index'], regressor.predict(X), color='red', label='Predicted Sentiment Score')

        plt.xlabel('Question Index')
        plt.ylabel('Sentiment Score')
        plt.title('Sentiment Analysis Trend Prediction')
        plt.legend()

        # Save plot to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        return jsonify({
            'status': 'success',
            'image': f'data:image/png;base64,{image_base64}',
            'metrics': {
                'mean_squared_error': mse,
                'r2_score': r2,
                'coefficients': regressor.coef_.tolist(),
                'intercept': regressor.intercept_
            },
            'top_insights': [
                f'Average Sentiment Score: {df["sentiment_score"].mean():.2f}',
                f'Sentiment Score Standard Deviation: {df["sentiment_score"].std():.2f}',
                f'Correlation between Answer Length and Sentiment: {df["answer_length"].corr(df["sentiment_score"]):.2f}'
            ]
        })

    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500



@app.route('/api/top-users-chart', methods=['GET'])
def top_users_chart():
    """Generate bar chart for top users."""
    try:
        # Retrieve top users data
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        cur.execute("""
            SELECT username, search_count 
            FROM users 
            ORDER BY search_count DESC 
            LIMIT 5
        """)
        top_users = cur.fetchall()
        cur.close()
        conn.close()

        # Create bar chart
        plt.figure(figsize=(10, 6))
        usernames = [user[0] for user in top_users]
        search_counts = [user[1] for user in top_users]

        plt.bar(usernames, search_counts, color='skyblue')
        plt.title('Top 5 Users by Search Count')
        plt.xlabel('Username')
        plt.ylabel('Search Count')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        return jsonify({
            'status': 'success',
            'chart': f'data:image/png;base64,{image_base64}'
        })

    except Exception as e:
        print(f"Error generating top users chart: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
@app.route('/admin', methods=['GET', 'POST'])
def admin_dashboard():
    if not session.get('is_admin'):
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        csv_path = request.form['csv_path']
        result = ir_system.embed_csv_documents(csv_path)
        return jsonify({"status": "success", "message": result})
    
    return render_template('admin.html')

def use_api(text, trial, question):
    key1 = "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"
    key2 = "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"
    key3 = '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'
    if trial%3==1:
        key = key1
    elif trial%3==2:
        key = key2
    else:
        key = key3
    client = Groq(
        # api_key="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"
        api_key=key
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f" this is asked question by a user :{question} and this is the text retreive from database on based of question :  {text}. Now you need to generate a answer based on given question and based on retrival answer from database. and you have to just give the answer, you don't need to",
            }
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content
 
@app.route('/save_feedback', methods=['POST'])
def save_feedback():
    # Check user session
    if 'user_id' not in session:
        app.logger.warning("Unauthorized access: No user session")
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
    
    # Retrieve request data
    data = request.json
    app.logger.info(f"Raw received data: {data}")

    # Extract parameters with logging
    chat_id = data.get('chat_id')
    response_type = data.get('response_type', '').strip()
    feedback = data.get('feedback', '').strip()
    
    app.logger.info(f"Parsed feedback data:")
    app.logger.info(f"- chat_id: {chat_id}")
    app.logger.info(f"- response_type: {response_type}")
    app.logger.info(f"- feedback: {feedback}")
    app.logger.info(f"- user_id from session: {session['user_id']}")

    try:
        cursor = ir_system.conn.cursor()
        
        # First, ensure the required columns exist in the chat_history table
        cursor.execute("""
            ALTER TABLE chat_history 
            ADD COLUMN IF NOT EXISTS feedback BOOLEAN DEFAULT NULL,
            ADD COLUMN IF NOT EXISTS feedback_comment TEXT DEFAULT NULL
        """)
        
        # Validate chat_id
        if not chat_id:
            app.logger.error("No chat_id provided")
            return jsonify({'status': 'error', 'message': 'Missing chat ID'}), 400
        
        # Determine update type
        if response_type in ['good', 'bad']:
            # Update the most recent chat history entry for this user
            cursor.execute("""
                UPDATE chat_history
                SET feedback = %s
                WHERE user_id = %s
                AND id = (
                    SELECT id 
                    FROM chat_history 
                    WHERE user_id = %s 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                )
            """, (response_type == 'good', session['user_id'], session['user_id']))
            
        if feedback:
            # Update the feedback comment for the most recent chat history entry
            cursor.execute("""
                UPDATE chat_history
                SET feedback_comment = %s
                WHERE user_id = %s
                AND id = (
                    SELECT id 
                    FROM chat_history 
                    WHERE user_id = %s 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                )
            """, (feedback, session['user_id'], session['user_id']))
        
        # Commit the transaction
        ir_system.conn.commit()
        
        return jsonify({
            'status': 'success', 
            'message': 'Feedback saved successfully'
        })
    
    except Exception as e:
        # Comprehensive error logging
        app.logger.error(f"Error saving feedback:", exc_info=True)
        ir_system.conn.rollback()
        return jsonify({
            'status': 'error', 
            'message': f'An error occurred: {str(e)}'
        }), 500
    
    finally:
        # Ensure cursor is closed
        if 'cursor' in locals() and not cursor.closed:
            cursor.close()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if admin
        if ir_system.validate_admin(username, password):
            # Create a session for the admin
            session['user_id'] = username
            session['is_admin'] = True
            return redirect(url_for('admin_dashboard'))
        
        # Check if regular user
        user_id = ir_system.validate_user(username, password)
        if user_id:
            session['user_id'] = user_id
            session['is_admin'] = False
            return redirect(url_for('index'))
        
        # If no match found, flash an error message
        error_message = "Invalid username or password. Please try again or register if you don't have an account."
        return render_template('login.html', error=error_message)
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Check if passwords match
        if password != confirm_password:
            return "Passwords do not match", 400
        
        if ir_system.register_user(username, password):
            return redirect(url_for('login'))
        else:
            return "Registration failed. Username may already exist.", 400
    
    return render_template('register.html')

def register_user(self, username, password):
    cursor = self.conn.cursor()
    hashed_password = generate_password_hash(password)
    
    try:
        cursor.execute(
            "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)",
            (username, hashed_password, 'user')
        )
        self.conn.commit()
        return True
    except psycopg2.IntegrityError:
        self.conn.rollback()
        return False
    finally:
        cursor.close()

def validate_user(self, username, password):
    cursor = self.conn.cursor()
    cursor.execute("SELECT id, password FROM users WHERE username = %s AND role = 'user'", (username,))
    result = cursor.fetchone()
    cursor.close()
    
    if result and check_password_hash(result[1], password):
        return result[0]  # Return user ID
    return None


@app.route('/search', methods=['POST'])
def search_documents():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    query = request.json.get('query', '')
    results = ir_system.search_documents(query, session['user_id'])
    
    # Check if results are empty
    if not results:
        return jsonify([])
    
    unique_results = list({result['text']: result for result in results}.values())
    
    # Initialize api_response to handle potential API call failures
    api_response = "No API response available"
    
    try:
        text = unique_results[0]["text"]
        trial = 0
        api_response = use_api(text, trial, query)
    except Exception as e:
        print(f"Error in API response: {e}")
        # If API call fails, keep the default "No API response available"
    
    # Add API response to the first result
    unique_results[0]["api_response"] = str(api_response)

    return jsonify(unique_results)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Add these routes to your app.py

@app.route('/get_chat/<int:chat_id>')
def get_chat(chat_id):
    print(f"\n--- GET CHAT DEBUG ---")
    print(f"Requested chat_id: {chat_id}")
    print(f"Current session: {session}")
    
    if 'user_id' not in session:
        print("Error: No user_id in session")
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # Connect to the database
        cursor = ir_system.conn.cursor()
        
        # Query to fetch the chat by ID and user_id
        query = """
            SELECT id, query, response, similarity, source, chapter_name, timestamp
            FROM chat_history
            WHERE id = %s AND user_id = %s
        """
        
        cursor.execute(query, (chat_id, session['user_id']))
        result = cursor.fetchone()
                
        if result:
            # Map the result to a dictionary
            response_data = {
                'id': result[0],
                'query': result[1],
                'response': result[2],
                'similarity': float(result[3]) if result[3] is not None else None,
                'source': result[4],
                'chapter_name': result[5],
                'timestamp': result[6].strftime("%Y-%m-%d %H:%M:%S") if result[6] else None
            }
            return jsonify(response_data)
        
        # Chat not found
        print("Error: Chat not found")
        return jsonify({'error': 'Chat not found'}), 404
        
    except Exception as e:
        # Handle unexpected errors
        print(f"Error occurred: {str(e)}")
        return jsonify({'error': 'An error occurred while fetching the chat'}), 500
    finally:
        # Ensure cursor is closed
        cursor.close()


@app.route('/delete_chat/<int:chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    print(f"\n--- DELETE CHAT DEBUG ---")
    print(f"Attempting to delete chat_id: {chat_id}")
    print(f"Current session: {session}")
    
    if 'user_id' not in session:
        print("Error: No user_id in session")
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        cursor = ir_system.conn.cursor()
        query = """
            DELETE FROM chat_history 
            WHERE id = %s AND user_id = %s
            RETURNING id
        """
        print(f"Executing delete query with params: id={chat_id}, user_id={session['user_id']}")
        
        cursor.execute(query, (chat_id, session['user_id']))
        deleted = cursor.fetchone()
        
        print(f"Delete result: {deleted}")
        
        ir_system.conn.commit()
        
        if deleted:
            print("Successfully deleted chat")
            return jsonify({'status': 'success', 'message': 'Chat deleted successfully'})
            
        print("Error: Chat not found")
        return jsonify({'error': 'Chat not found'}), 404
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        ir_system.conn.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()

@app.route('/get_document_count')
def get_document_count():
    cursor = ir_system.conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM document_embeddings")
        count = cursor.fetchone()[0]
        return jsonify({"count": count})
    except Exception as e:
        print(f"Error getting document count: {e}")
        return jsonify({"count": 0}), 500
    finally:
        cursor.close()
if __name__ == '__main__':
    # First-time admin registration (run once)
    ir_system.register_admin('nikhilyadav09', '9301')
    
    app.run(debug=True, host='0.0.0.0', port=5000)
