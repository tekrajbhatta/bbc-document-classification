#!/usr/bin/env python3
"""
Document Classification System using BBC RSS Feeds
Intelligent Information Retrieval Assignment - Coventry University
Author: Tek Raj Bhatt
Student ID: 250069
Coventry ID: 16544288
Date: August 2025

This system crawls BBC RSS feeds for Politics, Business, and Health news,
then builds a classification model to categorize new documents.
"""

import feedparser
import pandas as pd
import numpy as np
import requests
import time
import re
from datetime import datetime
import pickle
import os
import logging
from urllib.parse import urlparse
from bs4 import BeautifulSoup

# Machine Learning Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier

# Text Processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

# Visualization 
import matplotlib.pyplot as plt
import seaborn as sns

# Debugging and pretty print
import pprint
import sys

class RSSDocumentCrawler:
    """Crawls RSS feeds and extracts document content"""
    
    def __init__(self):
        self.rss_feeds = {
            'business': 'https://feeds.bbci.co.uk/news/business/rss.xml',
            'politics': 'https://feeds.bbci.co.uk/news/politics/rss.xml',
            'health': 'https://feeds.bbci.co.uk/news/health/rss.xml'
        }
        self.documents = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Academic Research Bot) Educational Use Only'
        })
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def crawl_rss_feeds(self, max_docs_per_category=40):
        """
        Crawl RSS feeds and extract documents
        
        Args:
            max_docs_per_category (int): Maximum documents to collect per category
        
        Returns:
            list: List of document dictionaries
        """
        all_documents = []
        
        for category, feed_url in self.rss_feeds.items():
            self.logger.info(f"Crawling {category} RSS feed...")
            
            try:
                # Parse RSS feed
                feed = feedparser.parse(feed_url)
                
                # pprint.pprint(feed)
                # sys.exit()
                
                if feed.bozo:
                    self.logger.warning(f"RSS feed might have issues: {feed_url}")
                
                documents_collected = 0
                
                for entry in feed.entries:
                    
                    # pprint.pprint(feed)
                    # sys.exit()
                    
                    if documents_collected >= max_docs_per_category:
                        break
                        
                    # Extract basic information
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    link = entry.get('link', '')
                    published = entry.get('published', '')
                    
                    # Get full article content if possible
                    full_content = self._extract_full_content(link)
                    
                    # Combine text sources
                    document_text = f"{title}. {summary}"
                    if full_content:
                        document_text += f" {full_content}"
                    
                    # Clean and validate document
                    document_text = self._clean_text(document_text)
                    
                    if len(document_text.split()) >= 10:  # Minimum length requirement
                        doc = {
                            'text': document_text,
                            'category': category,
                            'title': title,
                            'url': link,
                            'published': published,
                            'length': len(document_text.split())
                        }
                        all_documents.append(doc)
                        documents_collected += 1
                        
                    # Rate limiting
                    time.sleep(1)
                
                self.logger.info(f"Collected {documents_collected} documents from {category}")
                
            except Exception as e:
                self.logger.error(f"Error crawling {category}: {str(e)}")
                
        self.documents = all_documents
        return all_documents
    
    def _extract_full_content(self, url):
        """
        Extract full article content from URL
        
        Args:
            url (str): Article URL
            
        Returns:
            str: Extracted content or empty string
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # BBC-specific content extraction
            content_selectors = [
                '[data-component="text-block"]',
                '.story-body__inner p',
                '[data-component="body-text"]',
                '.article__content p'
            ]
            
            content_parts = []
            for selector in content_selectors:
                elements = soup.select(selector)
                for elem in elements:
                    text = elem.get_text().strip()
                    if text and len(text) > 20:
                        content_parts.append(text)
            
            return ' '.join(content_parts[:5])  # Limit to first 5 paragraphs
            
        except Exception as e:
            self.logger.warning(f"Could not extract content from {url}: {str(e)}")
            return ""
    
    def _clean_text(self, text):
        """Clean and preprocess text"""
        # Remove HTML tags if any remain
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-]', ' ', text)
        return text.strip()
    
    def save_documents(self, filename='crawled_documents.csv'):
        """Save crawled documents to CSV file"""
        if self.documents:
            df = pd.DataFrame(self.documents)
            df.to_csv(filename, index=False)
            self.logger.info(f"Saved {len(self.documents)} documents to {filename}")
            return df
        return None

class TextPreprocessor:
    """Handles text preprocessing for classification"""
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.download('punkt')
            nltk.download('stopwords')
        except:
            pass
            
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
    def preprocess_text(self, text, remove_stopwords=True, use_stemming=True):
        """
        Preprocess text for classification
        
        Args:
            text (str): Input text
            remove_stopwords (bool): Whether to remove stop words
            use_stemming (bool): Whether to apply stemming
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words if requested
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming if requested
        if use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Remove empty tokens and join
        tokens = [token for token in tokens if len(token) > 1]
        
        return ' '.join(tokens)

class DocumentClassifier:
    """Main document classification system"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = None
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.training_data = None
        
    def load_data(self, data_source):
        """
        Load training data from CSV file or DataFrame
        
        Args:
            data_source: CSV filename (str) or DataFrame
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if isinstance(data_source, str):
            self.training_data = pd.read_csv(data_source)
        else:
            self.training_data = data_source.copy()
            
        self.logger.info(f"Loaded {len(self.training_data)} documents")
        return self.training_data
        
    def prepare_features(self, texts, fit_vectorizer=False):
        """
        Prepare text features using TF-IDF
        
        Args:
            texts (list): List of text documents
            fit_vectorizer (bool): Whether to fit the vectorizer
            
        Returns:
            scipy.sparse matrix: TF-IDF features
        """
        # Preprocess texts
        processed_texts = [
            self.preprocessor.preprocess_text(text) 
            for text in texts
        ]
        
        if fit_vectorizer:
            # Initialize and fit vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                min_df=2,
                max_df=0.8,
                ngram_range=(1, 2),  # Include bigrams
                strip_accents='unicode',
                lowercase=True
            )
            features = self.vectorizer.fit_transform(processed_texts)
        else:
            features = self.vectorizer.transform(processed_texts)
            
        return features
    
    def train_models(self):
        """Train two classification models"""
        
        if self.training_data is None:
            raise ValueError("No training data loaded. Call load_data() first.")
        
        # Prepare features and labels
        X = self.prepare_features(self.training_data['text'], fit_vectorizer=True)
        y = self.label_encoder.fit_transform(self.training_data['category'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define models to train
        models_to_train = {
            'naive_bayes': MultinomialNB(alpha=1.0),
            'logistic_regression': LogisticRegression(
                max_iter=1000, 
                random_state=42
            )
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store model and results
            self.models[name] = model
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'test_labels': y_test
            }
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            print(f"Classification Report for {name}:")
            print(classification_report(
                y_test, y_pred, 
                target_names=self.label_encoder.classes_
            ))
            
        return results
    
    def classify_document(self, text, model_name='logistic_regression'):
        """
        Classify a new document
        
        Args:
            text (str): Document text to classify
            model_name (str): Model to use for classification
            
        Returns:
            dict: Classification results with confidence scores
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet.")
        
        # Prepare features
        features = self.prepare_features([text])
        
        # Get model
        model = self.models[model_name]
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else None
        
        # Convert back to category name
        category = self.label_encoder.inverse_transform([prediction])[0]
        
        result = {
            'predicted_category': category,
            'confidence': max(probabilities) if probabilities is not None else None,
            'model_used': model_name
        }
        
        if probabilities is not None:
            # Add probability for each class
            class_probabilities = dict(zip(
                self.label_encoder.classes_, 
                probabilities
            ))
            result['all_probabilities'] = class_probabilities
            
        return result
    
    def save_model(self, filename='document_classifier.pkl'):
        """Save trained models and vectorizer"""
        model_data = {
            'models': self.models,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'preprocessor': self.preprocessor
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='document_classifier.pkl'):
        """Load pre-trained models"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.vectorizer = model_data['vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.preprocessor = model_data['preprocessor']
        
        print(f"Model loaded from {filename}")

class RobustnessEvaluator:
    """Evaluate system robustness with various test cases"""
    
    def __init__(self, classifier):
        self.classifier = classifier
        
    def test_robustness(self):
        """Test classification system with various challenging inputs"""
        
        test_cases = [
            # Short inputs
            {
                'text': "Stock market rises today.",
                'expected': 'business',
                'description': 'Short business text'
            },
            {
                'text': "Election results announced.",
                'expected': 'politics',
                'description': 'Short political text'
            },
            {
                'text': "New vaccine approved.",
                'expected': 'health',
                'description': 'Short health text'
            },
            
            # Long inputs
            {
                'text': """The pharmaceutical company announced breakthrough results from its Phase III clinical trial for a new cancer treatment drug. The randomized controlled trial involving 2,000 patients showed significant improvement in overall survival rates compared to existing standard treatments. Regulatory authorities are expected to review the submission for fast-track approval given the promising efficacy and safety profile demonstrated across multiple patient populations.""",
                'expected': 'health',
                'description': 'Long detailed health text'
            },
            
            # Text with stop words
            {
                'text': "The government announced that it will be implementing new policies regarding healthcare reform and the budget will be allocated accordingly.",
                'expected': 'politics',
                'description': 'Text heavy with stop words'
            },
            
            # Text without stop words
            {
                'text': "GDP growth unemployment rates inflation economic indicators financial markets trading volume",
                'expected': 'business',
                'description': 'Text without stop words'
            },
            
            # Mixed/challenging topics
            {
                'text': "Government announces new healthcare budget allocation for hospital infrastructure investment.",
                'expected': 'politics',  # Could be health, but government budget is political
                'description': 'Mixed politics/health topic'
            },
            {
                'text': "Pharmaceutical company stock prices surge following FDA approval announcement for new diabetes medication.",
                'expected': 'business',  # Stock focus makes it business despite health context
                'description': 'Mixed business/health topic'
            },
            
            # Ambiguous cases
            {
                'text': "Minister discusses economic impact of pandemic on healthcare sector.",
                'expected': 'politics',  # Minister discussing = political
                'description': 'Highly ambiguous case'
            }
        ]
        
        print("\n" + "="*80)
        print("ROBUSTNESS EVALUATION")
        print("="*80)
        
        correct_predictions = 0
        total_predictions = len(test_cases)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. {test_case['description']}")
            print(f"Text: {test_case['text'][:100]}{'...' if len(test_case['text']) > 100 else ''}")
            
            # Test with different models
            for model_name in self.classifier.models.keys():
                result = self.classifier.classify_document(test_case['text'], model_name)
                
                is_correct = result['predicted_category'] == test_case['expected']
                if is_correct:
                    correct_predictions += 1
                
                print(f"   {model_name}: {result['predicted_category']} " + 
                      f"(confidence: {result['confidence']:.3f}) " +
                      f"{'✓' if is_correct else '✗'}")
        
        accuracy = correct_predictions / (total_predictions * len(self.classifier.models))
        print(f"\nOverall Robustness Accuracy: {accuracy:.3f}")
        
        return accuracy

def generate_confusion_matrices(classifier, results):
    """Generate and display raw confusion matrices for all trained models"""
    
    print("\n3 Generating Confusion Matrices...")
    print("="*50)
    
    # Set up matplotlib styling
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create comparison plot - only one row for raw matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('BBC Document Classification - Confusion Matrices Comparison', 
                 fontsize=16, fontweight='bold', y=1.05)
    
    model_names = ['naive_bayes', 'logistic_regression']
    
    for idx, model_name in enumerate(model_names):
        if model_name not in results:
            continue
            
        y_true = results[model_name]['test_labels']
        y_pred = results[model_name]['predictions']
        
        # Raw confusion matrix only
        cm_raw = confusion_matrix(y_true, y_pred)
        ax = axes[idx]
        
        sns.heatmap(cm_raw, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=classifier.label_encoder.classes_,
                   yticklabels=classifier.label_encoder.classes_,
                   square=True,
                   ax=ax,
                   cbar=idx==1,  # Only show colorbar on the right
                   annot_kws={'size': 14})  # Larger font for numbers
        
        # Clean title without "Raw Counts"
        ax.set_title(f'{model_name.replace("_", " ").title()}', 
                    fontweight='bold', fontsize=14)
        ax.set_ylabel('True Label', fontweight='bold', fontsize=12)
        ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
        
        # Print detailed analysis
        print(f"\n{model_name.replace('_', ' ').title()} Confusion Matrix:")
        print("-" * 45)
        print("Raw Confusion Matrix:")
        print(cm_raw)
        
        # Calculate per-class metrics from confusion matrix
        classes = classifier.label_encoder.classes_
        print(f"\nPer-class Analysis:")
        for i, class_name in enumerate(classes):
            true_positives = cm_raw[i, i]
            total_actual = np.sum(cm_raw[i, :])
            total_predicted = np.sum(cm_raw[:, i])
            
            recall = true_positives / total_actual if total_actual > 0 else 0
            precision = true_positives / total_predicted if total_predicted > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"  {class_name.capitalize()}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1_score:.3f}")
            
            # Find most common misclassification
            if total_actual > true_positives:
                misclass_values = [cm_raw[i, j] for j in range(len(classes)) if j != i]
                if misclass_values:
                    max_misclass = max(misclass_values)
                    misclass_idx = None
                    for j in range(len(classes)):
                        if j != i and cm_raw[i, j] == max_misclass:
                            misclass_idx = j
                            break
                    
                    if misclass_idx is not None:
                        misclass_rate = cm_raw[i, misclass_idx] / total_actual
                        if misclass_rate > 0.1:  # Only show if > 10% confusion
                            print(f"    Most confused with: {classes[misclass_idx]} ({misclass_rate:.2%})")
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('confusion_matrices_raw_only.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    print(f"\nRaw confusion matrices saved as: confusion_matrices_raw_only.png")

def main():
    """Main execution function"""
    
    print("BBC RSS Document Classification System")
    print("Coventry University - Intelligent Information Retrieval Assignment")
    print("="*70)
    
    # Step 1: Crawl RSS feeds
    print("\n1. Crawling RSS feeds...")
    crawler = RSSDocumentCrawler()
    documents = crawler.crawl_rss_feeds(max_docs_per_category=40)  # Total ~120 docs
    
    if len(documents) < 100:
        print(f"Warning: Only collected {len(documents)} documents. Consider increasing max_docs_per_category.")
    
    # Save crawled data
    df = pd.DataFrame(documents)
    df.to_csv('crawled_documents.csv', index=False)
    
    # Display statistics
    print(f"\nCollected {len(documents)} documents:")
    category_counts = df['category'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count} documents")
    
    # Step 2: Train classification models
    print("\n2. Training classification models...")
    classifier = DocumentClassifier()
    classifier.load_data(df)
    results = classifier.train_models()
    
    # Step 2.1: Generate Confusion Matrices
    generate_confusion_matrices(classifier, results)
    
    # Step 3: Save trained model
    classifier.save_model()
    
    # Step 4: Robustness evaluation
    print("\n4. Evaluating system robustness...")
    evaluator = RobustnessEvaluator(classifier)
    robustness_score = evaluator.test_robustness()
    
    print("\nClassification system ready!")
    print("Start the web interface with: python flask_web_app.py")
    print(f"Data source: BBC RSS Feeds (Business, Politics, Health)")
    print(f"Total documents processed: {len(documents)}")

if __name__ == "__main__":
    # Install required packages if not present
    required_packages = [
        'feedparser', 'pandas', 'numpy', 'scikit-learn', 
        'nltk', 'beautifulsoup4', 'requests'
    ]
    
    print("Required packages:", ', '.join(required_packages))
    print("Install with: pip install", ' '.join(required_packages))
    print()
    
    main()