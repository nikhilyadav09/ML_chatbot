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
                        
                        chapter_name = row[0]  # First column is chapter name
                        source = row[1]  # Second column is source
                        text = row[2]  # Third column is text                        
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
                    response_text,  # Formatted response text
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
    results = ir_system.search_documents(query, session['user_id'])
    unique_results = list({result['text']: result for result in results}.values())  
    # unique_results = {result['text']: result for result in results}.values()  # Ensure uniqueness

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
        print(f"Executing query with params: id={chat_id}, user_id={session['user_id']}")
        
        cursor.execute(query, (chat_id, session['user_id']))
        result = cursor.fetchone()
        
        print(f"Query result: {result}")
        
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
            print(f"Returning data: {response_data}")
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
    
    app.run(debug=True, host='0.0.0.0', port=8000)