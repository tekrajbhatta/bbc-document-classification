#!/usr/bin/env python3
"""
Flask Web Interface for BBC Document Classification System
Intelligent Information Retrieval Assignment - Softwarica College of IT and E-commerce | Coventry University
This creates a web-based front-end for the document classification system. Run this file after training the models with bbc_classifier.py
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import pickle
import os
import json
from datetime import datetime

# Import the required classes from the main module
from bbc_classifier import TextPreprocessor, DocumentClassifier

app = Flask(__name__)

# Global variables for the loaded model
classifier = None
model_loaded = False

def load_classification_model():
    """Load the pre-trained classification model"""
    global classifier, model_loaded
    
    model_path = 'document_classifier.pkl'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        print("Please run bbc_classifier.py first to train the models.")
        return False
    
    try:
        # Create a new classifier instance and load the saved model
        classifier = DocumentClassifier()
        classifier.load_model(model_path)
        
        model_loaded = True
        print("Classification model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

@app.route('/images/<filename>')
def serve_image(filename):
    """Serve images from the images directory"""
    return send_from_directory('images', filename)

@app.route('/')
def home():
    """Main page with classification form"""
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    """Handle classification requests"""
    global classifier, model_loaded
    
    if not model_loaded:
        return jsonify({
            'error': 'Classification model not loaded. Please train the model first.'
        }), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Please provide text to classify.'}), 400
        
        # Classify with both models in desired order
        results = {}
        model_order = ['naive_bayes', 'logistic_regression']
        
        for model_name in model_order:
            result = classifier.classify_document(text, model_name)
            results[model_name] = result
        
        # Format response
        response = {
            'input_text': text,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'classifications': results,
            'word_count': len(text.split())
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Classification error: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

# HTML Template (embedded for simplicity)
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BBC Document Classification System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <header class="header">
            <img src="/images/softwarica-logo-blue.svg" alt="Softwarica College Logo" class="logo">
            <h1>BBC Document Classifier</h1>
            <p class="subtitle">ST7071CEM: Intelligent Information Retrieval</p>
        </header>
        
        <div class="main-card">
            <div class="form-group">
                <label for="documentText">Enter text to classify:</label>
                <textarea 
                    id="documentText" 
                    placeholder="Paste or type any news article, headline, or text here to classify..."
                    maxlength="5000"></textarea>
                <div class="word-count" id="wordCount">0 words</div>
                <div class="help-text">
                    💡 <strong>Tips:</strong> Try headlines, articles, or paragraphs about politics, business, or health topics for best results.
                </div>
            </div>
            
            <button id="classifyBtn" class="classify-btn" onclick="classifyDocument()">
                🔍 Classify Document
            </button>
        </div>
        
        <div id="results" class="results">
            <div class="main-card">
                <h2>📈 Classification Results</h2>
                <div id="resultsContent"></div>
            </div>
        </div>
        
        <div class="examples">
            <h3>🚀 Quick Test Examples</h3>
            <div class="example-buttons">
                <div class="example-btn" onclick="useExample('The Prime Minister announced new economic policies during parliamentary session today')">
                    📊 Politics Example
                </div>
                <div class="example-btn" onclick="useExample('Stock market reaches record highs as investors react to quarterly earnings reports from major corporations')">
                    💼 Business Example
                </div>
                <div class="example-btn" onclick="useExample('New cancer treatment shows promising results in clinical trials involving 2000 patients across multiple hospitals')">
                    🏥 Health Example
                </div>
                <div class="example-btn" onclick="useExample('Government allocates additional healthcare budget for public hospital improvements')">
                    🔄 Mixed Topic Example
                </div>
            </div>
        </div>
        
        <footer class="footer">
            <div class="footer-content">
                <strong>Academic Project</strong> - Softwarica College of IT and E-commerce | Coventry University<br>
                <em>Data Source: BBC News RSS Feeds (Business, Politics, Health)</em><br><br>
                <strong>Student Name:</strong> Tek Raj Bhatt <br>
                <strong>Softwarica Student ID:</strong> 250069 <br>
                <strong>Coventry University Student ID:</strong> 16544288
            </div>
        </footer>
    </div>

    <script>
        let isClassifying = false;
        
        // Word counter
        const textarea = document.getElementById('documentText');
        const wordCount = document.getElementById('wordCount');
        
        function updateWordCount() {
            const text = textarea.value.trim();
            const words = text ? text.split(/\s+/).length : 0;
            wordCount.textContent = `${words} words`;
            wordCount.style.color = words > 10 ? 'var(--primary-color)' : 'var(--text-light)';
        }
        
        textarea.addEventListener('input', updateWordCount);
        
        function useExample(text) {
            document.getElementById('documentText').value = text;
            updateWordCount();
            
            // Add a subtle animation to show the text was loaded
            textarea.style.transform = 'scale(1.01)';
            setTimeout(() => {
                textarea.style.transform = 'scale(1)';
            }, 150);
        }
        
        async function classifyDocument() {
            if (isClassifying) return;
            
            const textArea = document.getElementById('documentText');
            const button = document.getElementById('classifyBtn');
            const results = document.getElementById('results');
            const resultsContent = document.getElementById('resultsContent');
            
            const text = textArea.value.trim();
            
            if (!text) {
                // Create a custom alert
                const alertDiv = document.createElement('div');
                alertDiv.className = 'error';
                alertDiv.innerHTML = '<strong>⚠️ Input Required:</strong> Please enter some text to classify!';
                textArea.parentNode.insertBefore(alertDiv, textArea);
                setTimeout(() => alertDiv.remove(), 3000);
                return;
            }
            
            // Update UI for loading state
            isClassifying = true;
            button.innerHTML = '<div class="spinner"></div> Analyzing Text...';
            button.disabled = true;
            results.style.display = 'none';
            
            try {
                const response = await fetch('/classify', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });
                
                if (!response.ok) {
                    throw new Error(`Server error: ${response.status}`);
                }
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                displayResults(data);
                
            } catch (error) {
                resultsContent.innerHTML = `
                    <div class="error">
                        <strong>❌ Classification Error:</strong> ${error.message}
                    </div>
                `;
                results.style.display = 'block';
            } finally {
                // Reset UI
                isClassifying = false;
                button.innerHTML = '🔍 Classify Document';
                button.disabled = false;
            }
        }
        
        function displayResults(data) {
            const resultsContent = document.getElementById('resultsContent');
            const results = document.getElementById('results');
            
            const stats = `
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-number">${data.word_count}</div>
                        <div class="stat-label">Words Analyzed</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">2</div>
                        <div class="stat-label">ML Models Used</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${new Date().toLocaleTimeString()}</div>
                        <div class="stat-label">Classified At</div>
                    </div>
                </div>
            `;
            
            let resultCards = '';
            
            // Ensure Naive Bayes appears first, then Logistic Regression
            const modelOrder = ['naive_bayes', 'logistic_regression'];
            
            for (const modelName of modelOrder) {
                const result = data.classifications[modelName];
                if (!result) continue;
                
                const confidence = result.confidence ? (result.confidence * 100).toFixed(1) : 'N/A';
                const confidenceWidth = result.confidence ? (result.confidence * 100) : 0;
                
                const modelDisplayName = modelName === 'naive_bayes' ? 'Naive Bayes' : 'Logistic Regression';
                
                resultCards += `
                    <div class="result-card">
                        <div class="model-name">${modelDisplayName}</div>
                        <div class="category ${result.predicted_category}">${result.predicted_category}</div>
                        <div class="confidence">
                            <strong>Confidence Score:</strong> ${confidence}%
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${confidenceWidth}%"></div>
                            </div>
                        </div>
                    </div>
                `;
            }
            
            resultsContent.innerHTML = stats + resultCards;
            results.style.display = 'block';
            
            // Smooth scroll to results with offset
            setTimeout(() => {
                results.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 100);
        }
        
        // Enhanced keyboard shortcuts
        document.getElementById('documentText').addEventListener('keydown', function(e) {
            // Enter to submit (Ctrl+Enter for new line)
            if (e.key === 'Enter' && !e.ctrlKey && !e.shiftKey) {
                e.preventDefault();
                classifyDocument();
            }
            
            // Escape to clear
            if (e.key === 'Escape') {
                this.value = '';
                updateWordCount();
                document.getElementById('results').style.display = 'none';
            }
        });
        
        // Initialize word count
        updateWordCount();
    </script>
</body>
</html>
"""

def create_html_template():
    """Create the HTML template file"""
    template_dir = 'templates'
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    
    with open(os.path.join(template_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(HTML_TEMPLATE)
    
    print("HTML template created successfully!")

if __name__ == '__main__':
    print("BBC Document Classification Web Interface")
    print("=" * 50)
    
    # Create HTML template
    create_html_template()
    
    # Load the trained model
    if not load_classification_model():
        print("\n❌ Cannot start web interface without trained model.")
        print("Please run 'python bbc_classifier.py' first to train the models.")
        exit(1)
    
    print("\n✅ Model loaded successfully!")
    print("\n🌐 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Start Flask production server
    app.run(host='localhost', port=5000)