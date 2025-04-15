import os
import json
import argparse
import numpy as np
from flask import Flask, request, jsonify, render_template, current_app
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler
from urllib.parse import urlparse
import re

# Initialize Flask app
app = Flask(__name__)

# Argument parser to take HTML file as an argument
parser = argparse.ArgumentParser(description='Phishing URL Detection')
parser.add_argument('--html', type=str, default='index_3.html', help='HTML template file to use for the frontend')
args = parser.parse_args()

class PhishingUrlTest:

    def __init__(self, model_path=None, dataset_dir=None, sequence_length=200, embedding_dimension=100):
        self.model_path = model_path
        self.dataset_dir = dataset_dir
        self.sequence_length = sequence_length
        self.embedding_dimension = embedding_dimension
        self.model = None
        self.tokenizer = None

    def extract_domain_features(self, urls, pagerank_df):
    # """
    # 根据 URL 列表，从 PageRank CSV 中提取 pr_pos, pr_val, harmonicc_pos, harmonicc_val 特征
    # 若 URL 的主域名未命中 pagerank_df，则填充默认值（如 0）
    # """
        features = []

        # 将 CSV 转为字典，提升查找效率
        pagerank_dict = pagerank_df.set_index('domain').to_dict(orient='index')

        for url in urls:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # fallback: 移除端口号
            domain = domain.split(':')[0]

            if domain in pagerank_dict:
                row = pagerank_dict[domain]
                feature = [row['pr_pos'], row['pr_val'], row['harmonicc_pos'], row['harmonicc_val']]
                # feature = [row['pr_val'], row['harmonicc_val']]
            else:
                feature = [0, 0.0, 0, 0.0]
                # feature = [0.0, 0.0]  # 未命中时填默认值

            features.append(feature)

        return np.array(features)
    
    def extract_manual_features(self, urls):
        # # """
        # # 提取手动特征：URL 长度、特殊字符数、域名长度、数字比例等
        #     # """
        features = []

        for url in urls:
            feature = []

            # 1. URL 长度
            feature.append(len(url))

            # 2. 特殊字符数
            special_chars = re.findall(r'[-@./&]', url)  # 计算 URL 中的特殊字符数
            feature.append(len(special_chars))

            # 3. 数字比例
            digits = re.findall(r'\d', url)  # 提取 URL 中的所有数字
            feature.append(len(digits) / len(url) if len(url) > 0 else 0)

            # 4. 是否包含敏感词
            sensitive_keywords = ['login', 'bank', 'account', 'secure']
            contains_sensitive_word = any(word in url for word in sensitive_keywords)
            feature.append(int(contains_sensitive_word))

            # 5. 域名长度
            domain_match = re.search(r'//([^/]+)', url)
            domain_length = len(domain_match.group(1)) if domain_match else 0
            feature.append(domain_length)

            features.append(feature)

        return np.array(features)
    
    def load_model(self):
        # Dynamically load the model if a model path is provided
        if self.model_path:
            self.model = load_model(self.model_path)
            current_app.logger.debug("Model loaded from %s", self.model_path)
        else:
            current_app.logger.debug("No model path provided.")

    def prepare_data(self, url):
        # Tokenizer for text vectorization
        self.tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
        
        # Using the input URL
        raw_x_test = [url]
        
        self.tokenizer.fit_on_texts(raw_x_test)
        
        # Convert to sequence
        x_test_sequences = self.tokenizer.texts_to_sequences(raw_x_test)
        x_test = pad_sequences(x_test_sequences, maxlen=self.sequence_length, padding='post')
        x_test = np.array(x_test).astype('float32')

        pagerank_path = './dataset/top100k_cc.csv'
        pagerank_df = pd.read_csv(pagerank_path)

        pagerank_features = self.extract_domain_features(raw_x_test, pagerank_df)

        manual_features = np.hstack((self.extract_manual_features(raw_x_test), pagerank_features))

        scaler = StandardScaler()
        manual_features = scaler.fit_transform(manual_features)
 
        x_test = np.hstack((x_test, manual_features))

        return x_test

    def classify_url(self, url):
        # Prepare data
        x_test = self.prepare_data(url)
        current_app.logger.debug("Test data: %s", x_test)
        # Predict the category
        prediction = self.model.predict([x_test[:, :self.sequence_length], x_test[:, self.sequence_length:]])
        current_app.logger.debug("Model prediction: %s", prediction)
        
        prediction_class = np.argmax(prediction, axis=1)
        
        # Assuming '1' is phishing and '0' is legitimate
        label = 'phishing' if prediction_class == 1 else 'legitimate'
        confidence = np.max(prediction)  # Get the maximum confidence score
        
        current_app.logger.debug("Classified as: %s with confidence: %f", label, confidence)
        
        return label, confidence

# Define model paths
model_paths = {
    "MGCF_Net_cl": "./test_results/custom/balanced_dataset_new/char-level/MGCF_Net_1000bs_20e_0.001wd_50000nw_False_enhanced/20250405_031212/model_all.keras",
    "MGCF_Net_cl_enhanced": ".test_results/custom/balanced_dataset_new/char-level/MGCF_Net_1000bs_20e_0.001wd_50000nw_True_enhanced/20250405_031220/model_all.keras",
    "MGCF_Net_wl": "./test_results/custom/balanced_dataset_new/word-level/MGCF_Net_1000bs_20e_0.001wd_50000nw_False_enhanced/20250405_031228/model_all.keras",
    "MGCF_Net_wl_enhanced": "./test_results/custom/balanced_dataset_new/word-level/MGCF_Net_1000bs_20e_0.001wd_50000nw_True_enhanced/20250405_031236/model_all.keras",
    "cnn_base": "./test_results/custom/balanced_dataset_new/char-level/cnn_base_1000bs_20e_0.001wd_50000nw_False_enhanced/20250405_033756/model_all.keras"
}
@app.route('/')
def home():
    # Render the index page with available model options
    
    return render_template(args.html, models=model_paths.keys())

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    model_name = request.form['model']
    
    # Get model path from the model name
    model_path = model_paths.get(model_name, None)
    
    if not model_path:
        return render_template(args.html, prediction='Model not found.', url=url, confidence='N/A', model_name=model_name, models=model_paths.keys())
    
    # Initialize and load the selected model
    phishing_url_test = PhishingUrlTest(model_path=model_path)
    phishing_url_test.load_model()

    # Classify the URL
    result, confidence = phishing_url_test.classify_url(url)

    # Return the result to the web page
    return render_template(args.html, prediction=result, confidence=confidence, url=url, model_name=model_name, models=model_paths.keys())

if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=8001)
    
