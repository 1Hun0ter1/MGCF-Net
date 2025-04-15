import os
import datetime
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import re

class PhishingUrlDetection:
    
    def __init__(self, dataset_dir, result_dir="./results", sequence_length=200, categories=['phishing', 'legitimate']):
        self.params = {
            'dataset_dir': dataset_dir,
            'result_dir': result_dir,
            'sequence_length': sequence_length,
            'categories': categories
        }

    def load_and_vectorize_data(self):
        print("🔄 Loading CSV data...")

        # 构造路径
        train_path = os.path.join(self.params['dataset_dir'], "train.csv")
        val_path = os.path.join(self.params['dataset_dir'], "val.csv")
        test_path = os.path.join(self.params['dataset_dir'], "test.csv")

        # 用 pandas 加载
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        # 简洁变量名
        raw_x_train = train_df['url'].tolist()
        raw_y_train = train_df['label'].tolist()

        raw_x_val = val_df['url'].tolist()
        raw_y_val = val_df['label'].tolist()

        raw_x_test = test_df['url'].tolist()
        raw_y_test = test_df['label'].tolist()

        # tokenizer
        tokenizer = TfidfVectorizer(lowercase=True, max_features=1000)
        tokenizer.fit(raw_x_train + raw_x_val + raw_x_test)
        
        # 提取 TF-IDF 特征
        X_train_tfidf = tokenizer.transform(raw_x_train)
        X_val_tfidf = tokenizer.transform(raw_x_val)
        X_test_tfidf = tokenizer.transform(raw_x_test)

        # 自定义统计特征
        def extract_url_features(url):
            length = len(url)  # URL 长度
            num_slashes = url.count("/")  # URL 中的斜杠数
            num_query_params = url.count("=")  # URL 查询参数数
            has_https = 1 if url.startswith("https") else 0  # 是否是 https

            # 提取域名和子域名
            domain_match = re.search(r"https?://([A-Za-z_0-9.-]+).*", url)
            domain = domain_match.group(1) if domain_match else ''
            subdomain_count = domain.count('.') - 1  # 计算子域名数

            return [length, num_slashes, num_query_params, has_https, subdomain_count]

        # 提取所有URL的统计特征
        X_train_custom = np.array([extract_url_features(url) for url in raw_x_train])
        X_val_custom = np.array([extract_url_features(url) for url in raw_x_val])
        X_test_custom = np.array([extract_url_features(url) for url in raw_x_test])

        # 合并 TF-IDF 和自定义特征
        X_train = hstack([X_train_tfidf, X_train_custom])
        X_val = hstack([X_val_tfidf, X_val_custom])
        X_test = hstack([X_test_tfidf, X_test_custom])

        # Label 编码为 one-hot
        encoder = LabelEncoder()
        encoder.fit(self.params['categories'])

        y_train = encoder.transform(raw_y_train)
        y_val = encoder.transform(raw_y_val)
        y_test = encoder.transform(raw_y_test)

        print("✅ Data loaded successfully.")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def train_and_evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test, model_name='SVM', epochs=20):
        # 根据模型选择不同的机器学习模型
        if model_name == 'SVM':
            model = SVC(kernel='linear')
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_name == 'LogisticRegression':
            model = LogisticRegression(max_iter=1000)
        elif model_name == 'KNeighbors':
            model = KNeighborsClassifier(n_neighbors=5)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # 保存每个epoch的准确率
        train_accs = []
        val_accs = []

        # 模拟每个epoch
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}...")

            # 训练模型
            model.fit(X_train, y_train)

            # 在验证集上评估模型
            y_pred_val = model.predict(X_val)
            val_acc = accuracy_score(y_val, y_pred_val)
            val_accs.append(val_acc)
            print(f"Validation Accuracy: {val_acc:.4f}")

            # 训练集上的准确率
            y_pred_train = model.predict(X_train)
            train_acc = accuracy_score(y_train, y_pred_train)
            train_accs.append(train_acc)
            print(f"Training Accuracy: {train_acc:.4f}")

        # 绘制训练和验证准确率
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, epochs + 1), train_accs, label='train_acc')
        plt.plot(range(1, epochs + 1), val_accs, label='val_acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        result_dir = self.create_result_dir(model_name)
        plt.savefig(os.path.join(result_dir, 'accuracy_plot.png'))
        plt.close()

        # 在测试集上评估模型
        y_pred_test = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_test)
        print(f"Test Accuracy: {test_acc:.4f}")

        # 输出分类报告
        print("Classification Report (Test Set):")
        print(classification_report(y_test, y_pred_test))

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred_test)
        print("Confusion Matrix:")
        print(cm)

        # 绘制并保存混淆矩阵
        self.save_confusion_matrix(cm, model_name)

    def save_confusion_matrix(self, cm, model_name):
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.params['categories'], yticklabels=self.params['categories'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # 保存图像
        result_dir = self.create_result_dir(model_name)
        plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'))
        plt.close()

    def create_result_dir(self, model_name):
        # 生成模型名称文件夹
        model_dir = os.path.join(self.params['result_dir'], model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # 生成时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(model_dir, timestamp)

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        return result_dir


# 主程序
def main():
    parser = argparse.ArgumentParser(description="Train Phishing URL Detection model.")
    parser.add_argument("-dataset", "--dataset_dir", type=str, default="/mnt/nvme0n1/Tsinghua_Node11/hh/safety/dephides/dataset/balanced_dataset", help="Path to the dataset directory")
    parser.add_argument("-result", "--result_dir", type=str, default="./ml_results", help="Path to save the results")
    parser.add_argument("-model", "--model_name", type=str, default="SVM", choices=['SVM', 'RandomForest', 'LogisticRegression', 'KNeighbors'], help="Machine learning model name")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Number of epochs for training")
    args = parser.parse_args()

    phishing_url_detection = PhishingUrlDetection(dataset_dir=args.dataset_dir, result_dir=args.result_dir)
    
    # 加载并向量化数据
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = phishing_url_detection.load_and_vectorize_data()
    
    # 训练并评估模型
    phishing_url_detection.train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, model_name=args.model_name, epochs=args.epochs)

if __name__ == '__main__':
    main()
