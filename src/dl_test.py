import os
import datetime
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import argparse
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
# import numpy as np


# Assuming the DlModels class and other dependencies are imported as needed
# from dl_models import DlModels

class PhishingUrlTest:

    def __init__(self, model_path, result_dir):
        self.BASE_DIR = Path(__file__).resolve().parent
        self.params = {
            'sequence_length': 200,  # Adjust based on your training parameters
            'embedding_dimension': 100,
            'dataset_dir': "../dataset/small_dataset",  # Make sure this is the correct path to your dataset
            'model_path': model_path,  # Passed from command-line argument
            'result_dir': result_dir  # Passed from command-line argument
        }

        self.model = None
        self.tokenizer = None

    def load_model(self):
        # Load the trained model
        self.model = load_model(self.params['model_path'])
        print("Model loaded successfully")

    def load_and_vectorize_data(self):
        print("data loading")
        test_path = os.path.join(self.params['dataset_dir'], "test.txt")

        try:
            with open(test_path, "r") as f:
                test = [line.strip() for line in f.readlines()[:772]]  # small_dataset test set (15%)

        except FileNotFoundError as e:
            print(f"❌ 错误：{e}")
            return None

        # raw_x_test = [line.split("\t")[1] for line in test]
        # raw_y_test = [line.split("\t")[0] for line in test]

        def split_line(line):
            try:
                return line.split("\t")[1], line.split("\t")[0]  # 返回文本和标签
            except IndexError:
                return None  # 如果没有 '\t'，返回 None

        # raw_x_train, raw_y_train = zip(*[split_line(line) for line in train if split_line(line) is not None])
        # raw_x_val, raw_y_val = zip(*[split_line(line) for line in val if split_line(line) is not None])
        raw_x_test, raw_y_test = zip(*[split_line(line) for line in test if split_line(line) is not None])

        # Tokenizer for text vectorization
        self.tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
        self.tokenizer.fit_on_texts(raw_x_test)

        x_test_sequences = self.tokenizer.texts_to_sequences(raw_x_test)
        max_length = self.params['sequence_length']
        x_test = pad_sequences(x_test_sequences, maxlen=max_length, padding='post')
        x_test = np.array(x_test).astype('float32')

        encoder = LabelEncoder()
        encoder.fit(['phishing', 'legitimate'])

        y_test = to_categorical(encoder.transform(raw_y_test), num_classes=2)

        print("Test data loaded successfully.")
        return x_test, y_test

    def evaluate_model(self, x_test, y_test):
        # Evaluate model on test data
        score, acc = self.model.evaluate(x_test, y_test)
        print(f"Test loss: {score} | Test accuracy: {acc}")

        # Make predictions
        y_pred = np.argmax(self.model.predict(x_test), axis=1)
        y_true = np.argmax(y_test, axis=1)

        # 获取 classification_report 的字典格式
        report_dict = classification_report(y_true, y_pred, target_names=['phishing', 'legitimate'], output_dict=True)
        
        # 格式化报告中的每个小数，保留更高的精度
        for label, metrics in report_dict.items():
            if isinstance(metrics, dict):
                for metric_name, metric_value in metrics.items():
                    # 保留足够的小数位数
                    report_dict[label][metric_name] = f"{metric_value:.10f}"  # 可调整精度
        
        # 将格式化后的字典转换回字符串形式
        formatted_report = classification_report(y_true, y_pred, target_names=['phishing', 'legitimate'])
        print("Classification Report:")
        print(formatted_report)

        # Confusion Matrix
        test_confusion_matrix = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(test_confusion_matrix)

        # Save the classification report
        result_dir = self.params['result_dir']
        os.makedirs(result_dir, exist_ok=True)
        with open(f"{result_dir}/classification_report.txt", "w") as report_file:
            report_file.write(formatted_report)

        # Save confusion matrix plot
        self.plot_confusion_matrix(test_confusion_matrix, ['phishing', 'legitimate'], normalized=False)

    def plot_confusion_matrix(self, confusion_matrix, categories, normalized=False):
        # Plot confusion matrix
        sns.set()
        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(14.0, 7.0))

        if normalized:
            row_sums = np.array(confusion_matrix).sum(axis=1)
            matrix = confusion_matrix / row_sums[:, np.newaxis]
            matrix = [line.tolist() for line in matrix]
            g = sns.heatmap(matrix, annot=True, fmt='f', xticklabels=categories, yticklabels=categories)
        else:
            matrix = confusion_matrix
            g = sns.heatmap(matrix, annot=True, fmt='d', xticklabels=categories, yticklabels=categories)

        g.set_yticklabels(categories, rotation=0)
        g.set_xticklabels(categories, rotation=90)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        # Save confusion matrix plot
        plt.savefig(f"{self.params['result_dir']}/confusion_matrix.png")
        plt.close()

    def run(self):
        self.load_model()  # Load pre-trained model
        x_test, y_test = self.load_and_vectorize_data()  # Load and preprocess test data
        self.evaluate_model(x_test, y_test)  # Evaluate the model on the test data


def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("-r", "--result_dir", type=str, required=True, help="Directory to save results")
    
    args = parser.parse_args()

    return args


def main():
    args = argument_parsing()

    test_runner = PhishingUrlTest(model_path=args.model_path, result_dir=args.result_dir)
    test_runner.run()


if __name__ == "__main__":
    main()
