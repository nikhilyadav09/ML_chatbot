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
                role VARCHAR(20) NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS document_embeddings (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                source VARCHAR(255),
                embedding vector(384)
            );
            
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                user_id INTEGER,
                query TEXT,
                response TEXT,
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

    # ... (rest of the methods remain the same)
    def embed_csv_documents(self, csv_path):
        cursor = self.conn.cursor()
        
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip header
            
            for row in csv_reader:
                text = row[1]  # Assuming the content is in the second column
                embedding = self.model.encode(text).tolist()
                
                cursor.execute(
                    "INSERT INTO document_embeddings (text, source, embedding) VALUES (%s, %s, %s)",
                    (text, csv_path, embedding)
                )
        
        self.conn.commit()
        cursor.close()
        
        return f"Embedded documents from {csv_path}"
    
    def search_documents(self, query, user_id, top_k=5):
        query_embedding = self.model.encode(query).tolist()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT text, source, 1 - (embedding <=> %s::vector) as similarity 
            FROM document_embeddings 
            ORDER BY similarity DESC 
            LIMIT %s
        """, (query_embedding, top_k))
        
        results = cursor.fetchall()
        
        # Save chat history
        response_text = "\n\n".join([f"{text} (Similarity: {similarity:.2f})" for text, _, similarity in results])
        cursor.execute("""
            INSERT INTO chat_history (user_id, query, response) 
            VALUES (%s, %s, %s)
        """, (user_id, query, response_text))
        
        self.conn.commit()
        cursor.close()
        
        return [
            {
                "text": text, 
                "source": source, 
                "similarity": float(similarity)
            } 
            for text, source, similarity in results
        ]
    
    def get_user_chat_history(self, user_id):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT query, response, timestamp 
            FROM chat_history 
            WHERE user_id = %s 
            ORDER BY timestamp DESC 
            LIMIT 20
        """, (user_id,))
        
        history = cursor.fetchall()
        cursor.close()
        
        return [
            {
                "query": query,
                "response": response,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
            } 
            for query, response, timestamp in history
        ]

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

@app.route('/get_chat/<int:chat_id>')
def get_chat(chat_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    cursor = ir_system.conn.cursor()
    cursor.execute("""
        SELECT query, response 
        FROM chat_history 
        WHERE id = %s AND user_id = %s
    """, (chat_id, session['user_id']))
    
    result = cursor.fetchone()
    cursor.close()
    
    if result:
        return jsonify({
            'query': result[0],
            'response': result[1]
        })
    return jsonify({'error': 'Chat not found'}), 404

@app.route('/delete_chat/<int:chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    cursor = ir_system.conn.cursor()
    cursor.execute("""
        DELETE FROM chat_history 
        WHERE id = %s AND user_id = %s
    """, (chat_id, session['user_id']))
    
    ir_system.conn.commit()
    cursor.close()
    
    return jsonify({'status': 'success'})

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

@app.route('/admin', methods=['GET', 'POST'])
def admin_dashboard():
    if not session.get('is_admin'):
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        csv_path = request.form['csv_path']
        result = ir_system.embed_csv_documents(csv_path)
        return jsonify({"status": "success", "message": result})
    
    return render_template('admin.html')

@app.route('/search', methods=['POST'])
def search_documents():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    query = request.json.get('query', '')
    # Get only unique results
    results = ir_system.search_documents(query, session['user_id'])
    # Return only unique results
    unique_results = []
    seen = set()
    for result in results:
        result_text = result['text']
        if result_text not in seen:
            seen.add(result_text)
            unique_results.append(result)
    
    return jsonify(unique_results)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    # First-time admin registration (run once)
    ir_system.register_admin('nikhilyadav09', '9301')
    
    app.run(debug=True, host='0.0.0.0', port=5000)