import os
import datetime
import json
import time
import pprint
import argparse
import datetime
import numpy as np
import seaborn as sns
# import keras.callbacks as ckbs
from tensorflow.keras import callbacks as ckbs
import matplotlib.pyplot as plt
# from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay, ExponentialDecay


from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
# from keras._tf_keras.keras.preprocessing.text import Tokenizer
from dl_models import DlModels
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path

import pandas as pd
import pdb
import re

from urllib.parse import urlparse
from sklearn.preprocessing import StandardScaler
import random

# sunucuda calismak icin
plt.switch_backend('agg')

pp = pprint.PrettyPrinter(indent=4)

TEST_RESULTS = {'data': {},
                "embedding": {},
                "hiperparameter": {},
                "test_result": {}}


class PhishingUrlDetection:
    
    def __init__(self):
        
        self.BASE_DIR = Path(__file__).resolve().parent
        self.params = {'loss_function': 'binary_crossentropy',
                    #    'optimizer': 'adam',
                       'lr_schedule': 'adam',
                       'sequence_length': 200,
                       'batch_train': 10000,
                       'batch_test': 10000,
                       'categories': ['phishing', 'legitimate'],
                       'char_index': None,
                       'weight_decay': 1e-4,
                       'epoch': 30,
                       'embedding_dimension': 100,
                       'result_dir': "../result/",
                       'dataset_dir': "../dataset/big_dataset"}

        self.ml_plotter = Plotter()
        self.dl_models = DlModels(self.params['categories'], self.params['embedding_dimension'], self.params['sequence_length'], self.params['weight_decay'])

    def set_params(self, args):
        self.params['test_case'] = args.test_case_name
        self.params['epoch'] = int(args.epoch)
        self.params['architecture'] = args.architecture
        self.params['weight_decay'] = args.weight_decay
        self.params['batch_train'] = args.batch_size
        self.params['batch_test'] = args.batch_size
        self.params['lr_schedule'] = args.lr_schedule
        self.params['feature_extraction_method'] = args.feature_extraction_method
        self.params['num_words'] = args.num_words
        self.params['data_name'] = args.data_name
        self.params['data_enhance'] = args.data_enhance
        # self.params['result_dir'] = "../test_results/custom/{}/".format(args.architecture)
        # self.params['dataset_dir'] = str((self.BASE_DIR / "../dataset/big_dataset/").resolve())
        # self.params['dataset_dir'] = (self.BASE_DIR / "../dataset/small_dataset/").resolve()
        self.params['dataset_dir'] = str((self.BASE_DIR / f"../dataset/{self.params['data_name']}/").resolve())
        # self.params['result_dir'] = str((self.BASE_DIR / "../test_results/custom/{}/".format(args.architecture)).resolve())

        # if not os.path.exists(self.params['result_dir'] ):
        #     os.mkdir(self.params['result_dir'] )
        #     print("Directory ", self.params['result_dir'] , " Created ")
        # else:
        #     print("Directory ", self.params['result_dir'] , " already exists")

        # 根据 args.feature 添加到 result_dir 中
        # if hasattr(args, 'feature') and args.feature:  # 检查 feature 参数
        #     self.params['result_dir'] = str((self.BASE_DIR / f"../test_results/custom/{args.feature}/{args.architecture}/").resolve())
        # else:
        #     self.params['result_dir'] = str((self.BASE_DIR / f"../test_results/custom/{args.architecture}/").resolve())
        # 根据 feature_extraction_method 添加到 result_dir 中
        feature_method = args.feature_extraction_method if args.feature_extraction_method else "one-hot"
        
        # self.params['result_dir'] = str((self.BASE_DIR / f"../test_results/custom/{feature_method}/{args.architecture}_{args.batch_size}_{args.epoch}/").resolve())
        
        architecture = args.architecture
        batch_size = args.batch_size
        epoch = args.epoch

        # 格式化字符串: architecture_batch_size_bs_epoch_e
        result_dir_name = f"{architecture}_{batch_size}bs_{epoch}e_{self.params['weight_decay']}wd_{self.params['num_words']}nw_{self.params['data_enhance']}_enhanced"
        
        # 生成时间戳，格式为：YYYYMMDD_HHMMSS
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 添加时间戳到 result_dir_name 中
        # result_dir_name = f"{result_dir_name}_{timestamp}"
        
        self.params['result_dir'] = str((self.BASE_DIR / f"../test_results/custom/{self.params['data_name']}/{feature_method}/{result_dir_name}/{timestamp}/").resolve())

        # 创建目录
        
        if not os.path.exists(self.params['result_dir']):
            os.makedirs(self.params['result_dir'])
            print(f"Directory {self.params['result_dir']} Created")
        else:
            print(f"Directory {self.params['result_dir']} already exists")
    
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


    # def scale_domain_features(self, train_features, val_features, test_features, scale_factor=10):
    #     """
    #     标准化并放大 domain 特征
    #     - train_features, val_features, test_features: numpy array
    #     - scale_factor: 放大的倍数（例如 10 表示放大 10 倍）
    #     """
    #     scaler = StandardScaler()
    #     train_scaled = scaler.fit_transform(train_features)
    #     val_scaled = scaler.transform(val_features)
    #     test_scaled = scaler.transform(test_features)

    #     # 放大特征
    #     train_scaled *= scale_factor
    #     val_scaled *= scale_factor
    #     test_scaled *= scale_factor

    #     return train_scaled, val_scaled, test_scaled

    

    # def generate_adversarial_sample(self, url):
    #     """
    #     对输入的 URL 生成对抗性样本，模拟钓鱼网站的常见攻击手段。
    #     """
    #     # 1. 随机替换部分字符
    #     url = list(url)
    #     num_changes = random.randint(1, 3)  # 每个URL随机改变1到3个字符

    #     for _ in range(num_changes):
    #         idx = random.randint(0, len(url) - 1)
    #         # 随机替换字符为其它字母或数字
    #         url[idx] = random.choice('abcdefghijklmnopqrstuvwxyz0123456789')
        
    #     return ''.join(url)

    def generate_adversarial_sample(self, url):
        # """
        # 模拟真实的钓鱼攻击 URL，对 URL 做策略性扰动
        # """
        # import random

        # 保留原始域名结构
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path
        scheme = parsed.scheme if parsed.scheme else "http"

        # 模拟攻击词列表
        phishing_keywords = ['login', 'secure', 'account', 'verify', 'update', 'webscr', 'signin', 'support']

        attack_type = random.choice(['typo', 'subdomain', 'hyphenation', 'keyword_injection'])

        if attack_type == 'typo':
            # 对主域名做字符重复或替换
            domain_parts = domain.split('.')
            if domain_parts:
                name = list(domain_parts[0])
                if len(name) >= 2:
                    i = random.randint(0, len(name) - 2)
                    name[i] = random.choice('abcdefghijklmnopqrstuvwxyz0123456789')
                domain_parts[0] = ''.join(name)
            domain = '.'.join(domain_parts)

        elif attack_type == 'subdomain':
            # 在前面加上目标品牌名
            sub = random.choice(['paypal', 'google', 'apple', 'bank', 'amazon'])
            domain = f"{sub}.{domain}"

        elif attack_type == 'hyphenation':
            domain = domain.replace('.', '-')
            domain = domain + '.com'

        elif attack_type == 'keyword_injection':
            kw = random.choice(phishing_keywords)
            path = f"/{kw}{path}"

        # 重新拼接成 URL
        fake_url = f"{scheme}://{domain}{path}"
        return fake_url



    def augment_data_with_adversarial_examples(self, x_train, y_train, augment_factor=0.2):
        # """
        # 仅对 phishing 类别样本生成对抗扰动。
        # """
        augmented_x = list(x_train)
        augmented_y = list(y_train)
        
        phishing_samples = [url for url, label in zip(x_train, y_train) if label == 'phishing']
        num_samples_to_generate = int(len(x_train) * augment_factor)

        for _ in range(num_samples_to_generate):
            idx = random.randint(0, len(phishing_samples) - 1)
            adversarial_url = self.generate_adversarial_sample(phishing_samples[idx])
            augmented_x.append(adversarial_url)
            augmented_y.append('phishing')
        
        return augmented_x, augmented_y

    def generate_adversarial_test_set(self, raw_x_test, raw_y_test, variant_per_url=2):
        adv_urls = []
        adv_labels = []

        phishing_count = 0  # 计数 phishing 样本
        legitimate_count = 0  # 计数 legitimate 样本

        for url, label in zip(raw_x_test, raw_y_test):
            if label == 'phishing':
                phishing_count += 1  # 统计 phishing 样本的数量
                # 对 phishing URL 生成攻击变体
                variants = self.generate_realistic_attack_variants(url)
                adv_urls.extend(variants[:variant_per_url])
                adv_labels.extend(['phishing'] * len(variants[:variant_per_url]))
            elif label == 'legitimate':
                legitimate_count += 1  # 统计 legitimate 样本的数量
                # 对 legitimate URL 生成攻击变体（如果需要）
                variants = self.generate_realistic_attack_variants(url)
                adv_urls.extend(variants[:variant_per_url])
                adv_labels.extend(['legitimate'] * len(variants[:variant_per_url]))

        print(f"Phishing samples in adversarial test set: {phishing_count}")
        print(f"Legitimate samples in adversarial test set: {legitimate_count}")
        
        return adv_urls, adv_labels





    def generate_realistic_attack_variants(self, url):
        """
        模拟几种实际钓鱼攻击方式来生成变体。
        """
        variants = []

        parsed = urlparse(url)
        base = parsed.netloc
        path = parsed.path or "/"
        
        # 方法1: 替换字符 (如 o -> 0, l -> 1)
        trans_table = str.maketrans({'o': '0', 'l': '1', 'i': '1', 'a': '4', 'e': '3'})
        variants.append(url.translate(trans_table))

        # 方法2: 拼接欺骗路径
        variants.append(f"http://{base}/secure/login/{path.strip('/')}")

        # 方法3: 添加诱导参数
        variants.append(f"http://{base}{path}?login=account&secure=1")

        return variants


    def model_sum(self, x):
        try:
            TEST_RESULTS['hiperparameter']["model_summary"] += x
        except:
            TEST_RESULTS['hiperparameter']["model_summary"] = x
    

    def analyze_token_coverage(self, json_path):
       

        with open(json_path, 'r', encoding='utf-8') as f:
            word_counts = json.load(f)

        sorted_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        freqs = np.array([count for _, count in sorted_counts])
        total = freqs.sum()
        cumulative = np.cumsum(freqs) / total

        def get_index_for_threshold(threshold):
            
            return int(np.searchsorted(cumulative, threshold) + 1)

        coverage_90 = get_index_for_threshold(0.90)
        coverage_95 = get_index_for_threshold(0.95)
        coverage_99 = get_index_for_threshold(0.99)

        print("📊 Token 覆盖率分析结果：")
        print(f"👉 覆盖 90% 的词数：{coverage_90}")
        print(f"👉 覆盖 95% 的词数：{coverage_95}")
        print(f"👉 覆盖 99% 的词数：{coverage_99}")

        result = {"90%": coverage_90, "95%": coverage_95, "99%": coverage_99}

        # 保存为 JSON 文件
        with open(os.path.join(self.params['result_dir'], 'coverage_result.json'), 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result
   
    
    def load_and_vectorize_data(self):
        
        print("🔄 Loading CSV data...")

        # 构造路径
        train_path = os.path.join(self.params['dataset_dir'], "train.csv")
        val_path = os.path.join(self.params['dataset_dir'], "val.csv")
        # val_path = os.path.join(self.params['dataset_dir'], "test.csv")
        test_path = os.path.join(self.params['dataset_dir'], "test.csv")

        # 用 pandas 加载
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        # 简洁变量名
        raw_x_train = train_df['url'].tolist()
        raw_y_train = train_df['label'].tolist()

        if self.params['data_enhance'] == "True":
            print("✅ 启用对抗性样本增强")

    # 调用已定义的增强方法
            raw_x_train, raw_y_train = self.augment_data_with_adversarial_examples(raw_x_train, raw_y_train, augment_factor=0.2)

            print(f"📈 增强后训练集数量：{len(raw_x_train)} 条")


        raw_x_val = val_df['url'].tolist()
        raw_y_val = val_df['label'].tolist()

        raw_x_test = test_df['url'].tolist()
        raw_y_test = test_df['label'].tolist()

        # 根据选定的特征提取方法进行处理
        feature_extraction_method = self.params['feature_extraction_method']
        
        if feature_extraction_method == 'char-level':
            print("⚙️ Using char-level features extraction")
            tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
            tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)
            self.params['char_index'] = tokenizer.word_index  # 字符表
            
            # 将文本转化为字符级序列并填充
            max_length = self.params['sequence_length']
            x_train = pad_sequences(tokenizer.texts_to_sequences(raw_x_train), maxlen=max_length, padding='post')
            x_val = pad_sequences(tokenizer.texts_to_sequences(raw_x_val), maxlen=max_length, padding='post')
            x_test = pad_sequences(tokenizer.texts_to_sequences(raw_x_test), maxlen=max_length, padding='post')

        elif feature_extraction_method == 'word-level':
            print("⚙️ Using word-level features extraction")
            tokenizer = Tokenizer(num_words=self.params['num_words'], lower=True, char_level=False, oov_token='-n-')
            tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)
            self.params['char_index'] = tokenizer.word_index
            print("💬 Total vocab size:", len(tokenizer.word_index))

            # 保存 tokenizer 的词频统计为 JSON
            word_counts_path = os.path.join(self.params['result_dir'], 'tokenizer_word_counts.json')
            with open(word_counts_path, 'w', encoding='utf-8') as f:
                json.dump(tokenizer.word_counts, f)

            # 可视化词频累积覆盖率图
            image_dir = os.path.join(self.params['result_dir'], 'images')
            os.makedirs(image_dir, exist_ok=True)
            self.ml_plotter.plot_token_coverage(tokenizer.word_counts, save_to=image_dir)

            # 自动分析覆盖率结果并保存
            self.analyze_token_coverage(word_counts_path)

            max_length = self.params['sequence_length']
            x_train = pad_sequences(tokenizer.texts_to_sequences(raw_x_train), maxlen=max_length, padding='post')
            x_val = pad_sequences(tokenizer.texts_to_sequences(raw_x_val), maxlen=max_length, padding='post')
            x_test = pad_sequences(tokenizer.texts_to_sequences(raw_x_test), maxlen=max_length, padding='post')

        elif feature_extraction_method == 'n-grams':
            print("⚙️ Using n-grams features extraction")
            # 使用 TfidfVectorizer 提取 n-grams 特征
            ngram_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3))  # 1-grams, 2-grams, 3-grams
            ngram_vectorizer.fit(raw_x_train + raw_x_val + raw_x_test)

            # 将 URL 文本转化为 n-grams 特征
            x_train_ngrams = ngram_vectorizer.transform(raw_x_train).toarray()
            x_val_ngrams = ngram_vectorizer.transform(raw_x_val).toarray()
            x_test_ngrams = ngram_vectorizer.transform(raw_x_test).toarray()

            # 用 n-grams 特征替换原始文本特征
            x_train = x_train_ngrams
            x_val = x_val_ngrams
            x_test = x_test_ngrams

        elif feature_extraction_method == 'TF-IDF':
            print("⚙️ Using TF-IDF features extraction")
            tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
            x_train_tfidf = tfidf_vectorizer.fit_transform(raw_x_train).toarray()
            x_val_tfidf = tfidf_vectorizer.transform(raw_x_val).toarray()
            x_test_tfidf = tfidf_vectorizer.transform(raw_x_test).toarray()

            # 用 TF-IDF 特征替换原始文本特征
            x_train = x_train_tfidf
            x_val = x_val_tfidf
            x_test = x_test_tfidf

        else:
            raise ValueError("Invalid feature extraction method")

        
        # 读取 PageRank 域名特征 CSV
        pagerank_path = '/mnt/nvme0n1/Tsinghua_Node11/hh/safety/dephides/dataset/top100k_cc.csv'
        pagerank_df = pd.read_csv(pagerank_path)

        # 生成 PageRank 特征（每条 URL 映射其主域名上的4个指标）
        pagerank_train_features = self.extract_domain_features(raw_x_train, pagerank_df)
        pagerank_val_features = self.extract_domain_features(raw_x_val, pagerank_df)
        pagerank_test_features = self.extract_domain_features(raw_x_test, pagerank_df)

        # # 放大 domain 特征重要性
        # pagerank_train_features, pagerank_val_features, pagerank_test_features = self.scale_domain_features(
        #     pagerank_train_features,
        #     pagerank_val_features,
        #     pagerank_test_features,
        #     scale_factor=10  # 可调
        # )

        # 再拼接上已有的手动特征
        manual_train_features = np.hstack((self.extract_manual_features(raw_x_train), pagerank_train_features))
        manual_val_features = np.hstack((self.extract_manual_features(raw_x_val), pagerank_val_features))
        manual_test_features = np.hstack((self.extract_manual_features(raw_x_test), pagerank_test_features))

        # 进行归一化（标准化）
        scaler = StandardScaler()
        manual_train_features = scaler.fit_transform(manual_train_features)
        manual_val_features = scaler.transform(manual_val_features)
        manual_test_features = scaler.transform(manual_test_features)


        # 拼接自动特征 + 手动特征
        x_train = np.hstack((x_train, manual_train_features))
        x_val = np.hstack((x_val, manual_val_features))
        x_test = np.hstack((x_test, manual_test_features))
                    
        # # 提取手动特征
        # manual_train_features = self.extract_manual_features(raw_x_train)
        # manual_val_features = self.extract_manual_features(raw_x_val)
        # manual_test_features = self.extract_manual_features(raw_x_test)
       
       
        # print(f"Manual features shape: {manual_train_features.shape}")  # Verify the shape of the manual features
        
        # # 合并手动特征和自动特征
        # x_train = np.hstack((x_train, manual_train_features))
        # x_val = np.hstack((x_val, manual_val_features))
        # x_test = np.hstack((x_test, manual_test_features))

        # Label 编码为 one-hot
        encoder = LabelEncoder()
        encoder.fit(self.params['categories'])

        y_train = to_categorical(encoder.transform(raw_y_train), num_classes=len(self.params['categories']))
        y_val = to_categorical(encoder.transform(raw_y_val), num_classes=len(self.params['categories']))
        y_test = to_categorical(encoder.transform(raw_y_test), num_classes=len(self.params['categories']))

        print("✅ Data loaded successfully.")

        # pdb.set_trace()
        # === 对抗性测试样本生成 ===
        
        if self.params['data_enhance'] == "True":
            self.adv_tokenizer = tokenizer  # 用于一致编码

            adv_x_raw, adv_y_raw = self.generate_adversarial_test_set(raw_x_test, raw_y_test)

            # 编码处理（与主 tokenizer 保持一致）
            adv_x_encoded = pad_sequences(
                tokenizer.texts_to_sequences(adv_x_raw), 
                maxlen=self.params['sequence_length'], 
                padding='post'
            )

            # 生成 domain + manual 特征
            pagerank_path = '/mnt/nvme0n1/Tsinghua_Node11/hh/safety/dephides/dataset/top100k_cc.csv'
            pagerank_df = pd.read_csv(pagerank_path)
            pagerank_adv_features = self.extract_domain_features(adv_x_raw, pagerank_df)
            manual_adv_features = np.hstack((self.extract_manual_features(adv_x_raw), pagerank_adv_features))

            # 归一化处理
            manual_adv_features = scaler.transform(manual_adv_features)
            adv_x_final = np.hstack((adv_x_encoded, manual_adv_features))
            adv_y_final = to_categorical(encoder.transform(adv_y_raw), num_classes=len(self.params['categories']))

            # 额外返回对抗测试集
            return (x_train, y_train), (x_val, y_val), (x_test, y_test), (adv_x_final, adv_y_final)
        else:
            return (x_train, y_train), (x_val, y_val), (x_test, y_test)
            
        
        
        # return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    # def load_and_vectorize_data(self):
    #     print("data loading")
    #     # train = [line.strip() for line in open("{}/train.txt".format(self.params['dataset_dir']), "r").readlines()[0:10]]
    #     # test = [line.strip() for line in open("{}/test.txt".format(self.params['dataset_dir']), "r").readlines()[0:10]]
    #     # val = [line.strip() for line in open("{}/val.txt".format(self.params['dataset_dir']), "r").readlines()[0:10]]
    #     train_path = os.path.join(self.params['dataset_dir'], "train.txt")
    #     test_path = os.path.join(self.params['dataset_dir'], "test.txt")
    #     val_path = os.path.join(self.params['dataset_dir'], "val.txt")

    #     # try:
    #     # # 选择70%的数据作为训练集，15%作为验证集，15%作为测试集
    #     #     with open(train_path, "r") as f:
    #     #         train = [line.strip() for line in f.readlines()[:254919]]  # small_dataset 训练集的70%
    #     #     with open(test_path, "r") as f:
    #     #         test = [line.strip() for line in f.readlines()[:7726]]  # small_dataset 测试集的15%
    #     #     with open(val_path, "r") as f:
    #     #         val = [line.strip() for line in f.readlines()[:15686]]  # small_dataset 验证集的15%

    #     try:
    #     # 选择70%的数据作为训练集，15%作为验证集，15%作为测试集
    #         with open(train_path, "r") as f:
    #             # train = [line.strip() for line in f.readlines()[:25491]]  # small_dataset 训练集的70%
    #             train = [line.strip() for line in f.readlines()[:160000]]  # small_dataset 训练集的70%
    #         with open(test_path, "r") as f:
    #             # test = [line.strip() for line in f.readlines()[:772]]  # small_dataset 测试集的15%
    #             test = [line.strip() for line in f.readlines()[:20000]]
    #         with open(val_path, "r") as f:
    #             # val = [line.strip() for line in f.readlines()[:1568]]  # small_dataset 验证集的15%
    #             val = [line.strip() for line in f.readlines()[:20000]]  # small_dataset 验证集的15%

                
    #     except FileNotFoundError as e:
    #         print(f"❌ 错误：{e}")
    #         return None
        

    #     TEST_RESULTS = {'data': {}}  # 确保 TEST_RESULTS 已定义
    #     TEST_RESULTS['data']['samples_train'] = len(train)
    #     TEST_RESULTS['data']['samples_test'] = len(test)
    #     TEST_RESULTS['data']['samples_val'] = len(val)
    #     TEST_RESULTS['data']['samples_overall'] = len(train) + len(test) + len(val)
    #     TEST_RESULTS['data']['name'] = self.params['dataset_dir']

    #     # raw_x_train = [line.split("\t")[1] for line in train]
    #     # raw_y_train = [line.split("\t")[0] for line in train]

    #     # raw_x_val = [line.split("\t")[1] for line in val]
    #     # raw_y_val = [line.split("\t")[0] for line in val]

    #     # raw_x_test = [line.split("\t")[1] for line in test]
    #     # raw_y_test = [line.split("\t")[0] for line in test]

    #     def split_line(line):
    #         try:
    #             return line.split("\t")[1], line.split("\t")[0]  # 返回文本和标签
    #         except IndexError:
    #             return None  # 如果没有 '\t'，返回 None

    #     raw_x_train, raw_y_train = zip(*[split_line(line) for line in train if split_line(line) is not None])
    #     raw_x_val, raw_y_val = zip(*[split_line(line) for line in val if split_line(line) is not None])
    #     raw_x_test, raw_y_test = zip(*[split_line(line) for line in test if split_line(line) is not None])

    #     tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    #     tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)
    #     self.params['char_index'] = tokenizer.word_index  # 记录字典

    #     # x_train = np.asanyarray(tokener.texts_to_sequences(raw_x_train))
    #     # x_val = np.asanyarray(tokener.texts_to_sequences(raw_x_val))
    #     # x_test = np.asanyarray(tokener.texts_to_sequences(raw_x_test))
    #     x_train_sequences = tokenizer.texts_to_sequences(raw_x_train)
    #     x_val_sequences = tokenizer.texts_to_sequences(raw_x_val)
    #     x_test_sequences = tokenizer.texts_to_sequences(raw_x_test)


    #     max_length = self.params.get('sequence_length', self.params['sequence_length'])  # 默认 200
    #     x_train = pad_sequences(x_train_sequences, maxlen=max_length, padding='post')
    #     x_val = pad_sequences(x_val_sequences, maxlen=max_length, padding='post')
    #     x_test = pad_sequences(x_test_sequences, maxlen=max_length, padding='post')

    #     x_train = np.asanyarray(x_train).astype('float32')
    #     x_val = np.asanyarray(x_val).astype('float32')
    #     x_test = np.asanyarray(x_test).astype('float32')

    
    #     encoder = LabelEncoder()
    #     encoder.fit(self.params['categories'])

    #     # y_train = np_utils.to_categorical(encoder.transform(raw_y_train), num_classes=len(self.params['categories']))
    #     # y_val = np_utils.to_categorical(encoder.transform(raw_y_val), num_classes=len(self.params['categories']))
    #     # y_test = np_utils.to_categorical(encoder.transform(raw_y_test), num_classes=len(self.params['categories']))

        
    #     y_train = to_categorical(encoder.transform(raw_y_train), num_classes=len(self.params['categories']))
    #     y_val = to_categorical(encoder.transform(raw_y_val), num_classes=len(self.params['categories']))
    #     y_test = to_categorical(encoder.transform(raw_y_test), num_classes=len(self.params['categories']))
    #     # ipdb.set_trace()
    #     print("Data are loaded.")

    #     return (x_train, y_train), (x_val, y_val), (x_test, y_test)



    def dl_algorithm(self, x_train, y_train, x_val, y_val, x_test, y_test, adv_x_test, adv_y_test):

        # x_train = sequence.pad_sequences(x_train, maxlen=self.params['sequence_length'])
        # x_test = sequence.pad_sequences(x_test, maxlen=self.params['sequence_length'])
        # x_val = sequence.pad_sequences(x_val, maxlen=self.params['sequence_length'])

        print("train sequences: {}  |  test sequences: {} | val sequences: {}\n"
              "x_train shape: {}  |  x_test shape: {} | x_val shape: {}\n"
              "Building Model....".format(len(x_train), len(x_test), len(x_val), x_train.shape, x_test.shape, x_val.shape))

        # model = self.dl_models.cnn_complex3(self.params['char_index'])
        
        # 拆分输入形状
        seq_len = self.params['sequence_length']
        text_input_shape = (seq_len,)
        manual_input_shape = (x_train.shape[1] - seq_len,)

        model_function = getattr(self.dl_models, self.params['architecture'])
        
        if self.params['architecture'] in [
            'DeepCNN_Light_Hybrid', 'MGCF_Net', 'MGCF_Net_NoCNN', 'MGCF_Net_NoBiLSTM', 
            'MGCF_Net_NoAttention', 'MGCF_Net_NoCNN_BiLSTM'
        ]:
            model = model_function(text_input_shape, manual_input_shape, self.params['char_index'])
        elif self.params['architecture'] in [
            'DeepCNN_Light_V2_2', 'DeepCNN_Light_V2_3', 'DeepCNN_Light_V2_1', 'AMR_CNN', 
            'AMR_CNN_2', 'Hybrid_cnn_brnn_att', 'DeepCNN_Light_TFIDF'
        ]:
            model = model_function(x_train, self.params['char_index'])
        else:    
            model = model_function(self.params['char_index'])


        # Build Deep Learning Architecture

        initial_lr = 0.001
        decay_steps = len(x_train) // self.params['batch_train'] * self.params['epoch']

        # 根据用户参数选择调度策略
        if self.params['lr_schedule'] == "cosine":
            print("🔁 使用 CosineDecay 学习率调度")
            lr_schedule = CosineDecay(initial_learning_rate=initial_lr, decay_steps=decay_steps, alpha=1e-6)
            optimizer = Adam(learning_rate=lr_schedule)

        elif self.params['lr_schedule'] == "exponential":
            print("🔁 使用 ExponentialDecay 学习率调度")
            lr_schedule = ExponentialDecay(initial_learning_rate=initial_lr, decay_steps=decay_steps, decay_rate=0.9)
            optimizer = Adam(learning_rate=lr_schedule)

        else:
            print("🚫 不使用学习率调度（使用固定学习率）")
            optimizer = Adam(learning_rate=initial_lr)

        # 用选定的优化器编译模型
        model.compile(loss=self.params['loss_function'], optimizer=optimizer, metrics=['accuracy'])

        # model.compile(loss=self.params['loss_function'], optimizer=self.params['optimizer'], metrics=['accuracy'])

        model.summary()
        model.summary(print_fn=lambda x: self.model_sum(x + '\n'))

        # hist = model.fit(x_train, y_train,
        #                  batch_size=int(self.params['batch_train']),
        #                  epochs=int(self.params['epoch']),
        #                  shuffle=True,
        #                  validation_data=(x_val, y_val),
        #                  callbacks=[CustomCallBack()])
        if self.params['architecture'] in [
            'DeepCNN_Light_Hybrid', 'MGCF_Net', 'MGCF_Net_NoCNN', 'MGCF_Net_NoBiLSTM', 
            'MGCF_Net_NoAttention', 'MGCF_Net_NoCNN_BiLSTM'
        ]:
            hist = model.fit(
                        [x_train[:, :seq_len], x_train[:, seq_len:]],
                        y_train,
                        batch_size=int(self.params['batch_train']),
                        epochs=int(self.params['epoch']),
                        shuffle=True,
                        validation_data=([x_val[:, :seq_len], x_val[:, seq_len:]], y_val),
                        callbacks=[CustomCallBack()])
        else:
            # self.params['sequence_length'] = x_train.shape[1]
            # print(f"The shape of x is {x_train.shape[1]}")
            # self.dl_models = DlModels(self.params['categories'], self.params['embedding_dimension'], self.params['sequence_length'], self.params['weight_decay'])
            hist = model.fit(
                        x_train,
                        y_train,
                        batch_size=int(self.params['batch_train']),
                        epochs=int(self.params['epoch']),
                        shuffle=True,
                        validation_data=(x_val, y_val),
                        callbacks=[CustomCallBack()])
            
        t = time.time()
        
        # score, acc = model.evaluate(x_test, y_test, batch_size=int(self.params['batch_test']))
        
        if self.params['architecture'] in [
            'DeepCNN_Light_Hybrid', 'MGCF_Net', 'MGCF_Net_NoCNN', 'MGCF_Net_NoBiLSTM', 
            'MGCF_Net_NoAttention', 'MGCF_Net_NoCNN_BiLSTM'
        ]:
            score, acc = model.evaluate(
                                        [x_test[:, :seq_len], x_test[:, seq_len:]],
                                        y_test,
                                        batch_size=int(self.params['batch_test']))
        else:
            score, acc = model.evaluate(
                                        x_test,
                                        y_test,
                                        batch_size=int(self.params['batch_test']))
        # ipdb.set_trace()
        TEST_RESULTS['test_result']['test_time'] = time.time() - t

        y_test = list(np.argmax(np.asanyarray(np.squeeze(y_test), dtype=int).tolist(), axis=1))
        # y_pred = model.predict_classes(x_test, batch_size=self.params['batch_test'], verbose=1).tolist()
        
        # y_pred = np.argmax(model.predict(x_test, batch_size=self.params['batch_test'], verbose=1), axis=1).tolist()

        if self.params['architecture'] in [
            'DeepCNN_Light_Hybrid', 'MGCF_Net', 'MGCF_Net_NoCNN', 'MGCF_Net_NoBiLSTM', 
            'MGCF_Net_NoAttention', 'MGCF_Net_NoCNN_BiLSTM'
        ]:
            y_pred = np.argmax(
                            model.predict([x_test[:, :seq_len], x_test[:, seq_len:]],
                                        batch_size=self.params['batch_test'],
                                        verbose=1),
                            axis=1
                        )
        else:
            y_pred = np.argmax(model.predict(x_test, batch_size=self.params['batch_test'], verbose=1), axis=1)

        
        report = classification_report(y_test, y_pred, target_names=self.params['categories'])
        print(report)
        TEST_RESULTS['test_result']['report'] = report
        TEST_RESULTS['epoch_history'] = hist.history
        TEST_RESULTS['test_result']['test_acc'] = acc
        TEST_RESULTS['test_result']['test_loss'] = score

        test_confusion_matrix = confusion_matrix(y_test, y_pred)
        TEST_RESULTS['test_result']['test_confusion_matrix'] = test_confusion_matrix.tolist()

        print('Test loss: {0}  |  test accuracy: {1}'.format(score, acc))
        

        # === 对抗测试集评估 ===
        if self.params['data_enhance'] == "True":
            print("⚔️ Evaluating on Adversarial Test Set...")
            adv_score, adv_acc = model.evaluate(
                [adv_x_test[:, :seq_len], adv_x_test[:, seq_len:]],
                adv_y_test,
                batch_size=int(self.params['batch_test']),
                verbose=1
            )

            adv_y_true = np.argmax(adv_y_test, axis=1)
            adv_y_pred = np.argmax(model.predict([adv_x_test[:, :seq_len], adv_x_test[:, seq_len:]], batch_size=self.params['batch_test']), axis=1)
            adv_report = classification_report(adv_y_true, adv_y_pred, target_names=self.params['categories'])

            print("⚔️ Adversarial Test Accuracy:", adv_acc)
            print("⚔️ Adversarial Test Loss:", adv_score)  # 输出对抗测试集损失
            print(adv_report)

            # 保存结果
            TEST_RESULTS['test_result']['adv_acc'] = adv_acc
            TEST_RESULTS['test_result']['adv_score'] = adv_score  # 保存对抗测试集损失
            TEST_RESULTS['test_result']['adv_report'] = adv_report

            # === Saving the confusion matrix for adversarial test results ===
            adv_confusion_matrix = confusion_matrix(adv_y_true, adv_y_pred)
            TEST_RESULTS['test_result']['adv_confusion_matrix'] = adv_confusion_matrix.tolist()


            with open(f"{'result_dir'}adv_classification_report.txt", "w") as adv_report_file:
                adv_report_file.write(TEST_RESULTS['test_result']['adv_report'])

        self.save_results(model)



    def traditional_ml(self, x_train, y_train, x_val, y_val, x_test, y_test, algorithm=None):
        print("traditional ML is running")
        if algorithm == "NB":
            gnb = GaussianNB()
            model = gnb.fit(x_train, y_train)
        elif algorithm == "RF":
            clf = RandomForestClassifier(n_estimators=10, random_state=0)
            model = clf.fit(x_train, y_train)
        elif algorithm == 'SVM':
            clf = svm.SVC(gamma='scale')
            model = clf.fit(x_train, y_train)
        else:
            gnb = GaussianNB()
            model = gnb.fit(x_train, y_train)

        model_prediction_val = model.predict(x_val)
        # model_probability = model.predict_proba(x_val)
        acc_val = accuracy_score(y_val, model_prediction_val)

        model_prediction_test = model.predict(x_test)
        acc_test = accuracy_score(y_test, model_prediction_test)
        print("acc_val: {} - acc_test: {}".format(acc_val, acc_test))
    
    
    def save_results(self, model):
        tm = str(datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-")
        tsm = tm.split("_")
        TEST_RESULTS['date'] = tsm[0]
        TEST_RESULTS['date_time'] = tsm[1]
        TEST_RESULTS['test_case'] = self.params['test_case']

        TEST_RESULTS['embedding']['vocabulary_size'] = len(self.params['char_index'])
        TEST_RESULTS["embedding"]['embedding_dimension'] = self.params['embedding_dimension']

        TEST_RESULTS['epoch_history']['epoch_time'] = TEST_RESULTS['epoch_times']
        TEST_RESULTS.pop('epoch_times')

        TEST_RESULTS['hiperparameter']['epoch'] = self.params['epoch']
        TEST_RESULTS['hiperparameter']['train_batch_size'] = self.params['batch_train']
        TEST_RESULTS['hiperparameter']['test_batch_size'] = self.params['batch_test']
        TEST_RESULTS['hiperparameter']['sequence_length'] = self.params['sequence_length']

        TEST_RESULTS['params'] = self.params

        model_json = model.to_json()

        # Ensure the directory path ends with a '/'
        result_dir = self.params['result_dir']
        if not result_dir.endswith('/'):
            result_dir += '/'

        # Save the entire model in .keras format
        model.save(f"{result_dir}model_all.keras")

        # Save the model structure (JSON format)
        with open(f"{result_dir}model.json", "w") as json_file:
            json_file.write(json.dumps(model_json))

        # Save weights in .weights.h5 format (using .h5 extension as required)
        model.save_weights(f"{result_dir}weights.weights.h5")

        # Save additional test results in JSON format
        with open(f"{result_dir}raw_test_results.json", "w") as test_results_file:
            test_results_file.write(json.dumps(TEST_RESULTS))

        # Capture model summary as plain text
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        model_summary_text = "\n".join(model_summary)

        # Save model summary in a text file with UTF-8 encoding
        with open(f"{result_dir}model_summary.txt", "w", encoding='utf-8') as summary_file:
            summary_file.write(model_summary_text)

        # Save classification report in a text file (standard test set)
        with open(f"{result_dir}classification_report.txt", "w") as report_file:
            report_file.write(TEST_RESULTS['test_result']['report'])

        # Save adversarial test results (adversarial accuracy and report)
        if self.params["data_enhance"] == "True":
            with open(f"{result_dir}adv_classification_report.txt", "w") as adv_report_file:
                adv_report_file.write(TEST_RESULTS['test_result']['adv_report'])

        # Save confusion matrix plots for both standard and adversarial test results
        img_result_dir = f"{result_dir}images/"
        
        if not os.path.exists(img_result_dir):
            os.mkdir(img_result_dir)
            print(f"Directory {img_result_dir} Created")
        else:
            print(f"Directory {img_result_dir} already exists")

        # Plot accuracy and loss graphs and save them in the result directory
        self.ml_plotter.plot_graphs(TEST_RESULTS['epoch_history']['accuracy'], 
                                    TEST_RESULTS['epoch_history']['val_accuracy'], 
                                    save_to=img_result_dir, name="accuracy")
        self.ml_plotter.plot_graphs(TEST_RESULTS['epoch_history']['loss'], 
                                    TEST_RESULTS['epoch_history']['val_loss'], 
                                    save_to=img_result_dir, name="loss")

        # Save confusion matrix plots for the standard test set
        self.ml_plotter.plot_confusion_matrix(TEST_RESULTS['test_result']['test_confusion_matrix'], 
                                                self.params['categories'], 
                                                save_to=img_result_dir)
        self.ml_plotter.plot_confusion_matrix(TEST_RESULTS['test_result']['test_confusion_matrix'], 
                                                self.params['categories'], 
                                                save_to=img_result_dir, normalized=True)

        # Save confusion matrix plots for the adversarial test set
        if self.params["data_enhance"] == "True":
            self.ml_plotter.plot_confusion_matrix(TEST_RESULTS['test_result']['adv_confusion_matrix'], 
                                                    self.params['categories'], 
                                                    save_to=img_result_dir)
            self.ml_plotter.plot_confusion_matrix(TEST_RESULTS['test_result']['adv_confusion_matrix'], 
                                                    self.params['categories'], 
                                                    save_to=img_result_dir, normalized=True)

        # Saving embedding
        embedding_weights = None
        for layer in model.layers:
            if 'embedding' in layer.name.lower():
                weights = layer.get_weights()
                if weights:
                    embedding_weights = weights[0]
                    break

        if embedding_weights is not None:
            words_embeddings = {w: embedding_weights[idx].tolist() 
                                for w, idx in self.params['char_index'].items() 
                                if idx < embedding_weights.shape[0]}
            
            with open(f"{result_dir}char_embeddings.json", "w") as embeddings_file:
                json.dump(words_embeddings, embeddings_file)
            print("✅ Embedding weights saved.")
        else:
            print("⚠️ No embedding layer found. Skipping embedding export.")


    # def save_results(self, model):
    #     tm = str(datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-")
    #     tsm = tm.split("_")
    #     TEST_RESULTS['date'] = tsm[0]
    #     TEST_RESULTS['date_time'] = tsm[1]
    #     TEST_RESULTS['test_case'] = self.params['test_case']

    #     TEST_RESULTS['embedding']['vocabulary_size'] = len(self.params['char_index'])
    #     TEST_RESULTS["embedding"]['embedding_dimension'] = self.params['embedding_dimension']

    #     TEST_RESULTS['epoch_history']['epoch_time'] = TEST_RESULTS['epoch_times']
    #     TEST_RESULTS.pop('epoch_times')

    #     TEST_RESULTS['hiperparameter']['epoch'] = self.params['epoch']
    #     TEST_RESULTS['hiperparameter']['train_batch_size'] = self.params['batch_train']
    #     TEST_RESULTS['hiperparameter']['test_batch_size'] = self.params['batch_test']
    #     TEST_RESULTS['hiperparameter']['sequence_length'] = self.params['sequence_length']

    #     TEST_RESULTS['params'] = self.params

    #     model_json = model.to_json()
    #     # model.save("{}model_all.h5".format(self.params['result_dir']))
    #     model.save("{}model_all.keras".format(self.params['result_dir']))

    #     open("{0}model.json".format(self.params['result_dir']), "w").write(json.dumps(model_json))
    #     ipdb.set_trace()
    #     # model.save_weights("{0}weights.h5".format(self.params['result_dir']))
    #     # model.save_weights("{0}/weights.h5".format(self.params['result_dir']))
    #     # model.save_weights("{0}/weights.keras".format(self.params['result_dir']))  # .keras 格式
    #     model.save_weights("{0}/weights.weights.h5".format(self.params['result_dir']))  # 使用 .weights.h5 格式
        
        
    #     open("{0}raw_test_results.json".format(self.params['result_dir']), "w").write(json.dumps(TEST_RESULTS))
    #     open("{0}model_summary.txt".format(self.params['result_dir']), "w").write(TEST_RESULTS['hiperparameter']["model_summary"])
    #     open("{0}classification_report.txt".format(self.params['result_dir']), "w").write(TEST_RESULTS['test_result']['report'])

    #     # self.ml_plotter.plot_graphs(TEST_RESULTS['epoch_history']['acc'], TEST_RESULTS['epoch_history']['val_acc'], save_to=self.params['result_dir'], name="accuracy")
    #     self.ml_plotter.plot_graphs(TEST_RESULTS['epoch_history']['accuracy'], TEST_RESULTS['epoch_history']['val_accuracy'], save_to=self.params['result_dir'], name="accuracy")
    #     self.ml_plotter.plot_graphs(TEST_RESULTS['epoch_history']['loss'], TEST_RESULTS['epoch_history']['val_loss'], save_to=self.params['result_dir'], name="loss")
    #     self.ml_plotter.plot_confusion_matrix(TEST_RESULTS['test_result']['test_confusion_matrix'], self.params['categories'], save_to=self.params['result_dir'])
    #     self.ml_plotter.plot_confusion_matrix(TEST_RESULTS['test_result']['test_confusion_matrix'], self.params['categories'], save_to=self.params['result_dir'], normalized=True)

    #     # saving embedding
    #     embeddings = model.layers[0].get_weights()[0]
    #     words_embeddings = {w: embeddings[idx].tolist() for w, idx in self.params['char_index'].items()}
    #     open("{0}char_embeddings.json".format(self.params['result_dir']), "w").write(json.dumps(words_embeddings))


class CustomCallBack(ckbs.Callback):

    def __init__(self):
        ckbs.Callback.__init__(self)
        TEST_RESULTS['epoch_times'] = []

    def on_epoch_begin(self, batch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        TEST_RESULTS['epoch_times'].append(time.time() - self.epoch_time_start)


class Plotter:

    def plot_graphs(self, train, val, save_to=None, name="accuracy"):

        if name == "accuracy":
            val, = plt.plot(val, label="val_acc")
            train, = plt.plot(train, label="train_acc")
        else:
            val, = plt.plot(val, label="val_loss")
            train, = plt.plot(train, label="train_loss")

        plt.ylabel(name)
        plt.xlabel("epoch")


        # Set the y-axis to have ticks from 0.97 to 1.0 with step size of 0.01
        if name == "accuracy":
            plt.ylim(0.9, 1.0)  # Adjust the y-axis range
            plt.yticks(np.arange(0.9, 1.01, 0.05))  # Set ticks at 0.95, 1.0


        plt.legend(handles=[val, train], loc=2)

        if save_to:
            plt.savefig("{0}/{1}.png".format(save_to, name))

        plt.close()

    def plot_confusion_matrix(self, confusion_matrix, categories, save_to=None, normalized=False):

        sns.set()
        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(14.0, 7.0))

        if normalized:
            row_sums = np.asanyarray(confusion_matrix).sum(axis=1)
            matrix = confusion_matrix / row_sums[:, np.newaxis]
            matrix = [line.tolist() for line in matrix]
            g = sns.heatmap(matrix, annot=True, fmt='f', xticklabels=True, yticklabels=True)

        else:
            matrix = confusion_matrix
            g = sns.heatmap(matrix, annot=True, fmt='d', xticklabels=True, yticklabels=True)

        g.set_yticklabels(categories, rotation=0)
        g.set_xticklabels(categories, rotation=90)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        if save_to:
            if normalized:
                plt.savefig("{0}/{1}.png".format(save_to, "normalized_confusion_matrix"))
            else:
                plt.savefig("{0}/{1}.png".format(save_to, "confusion_matrix"))
    
    def plot_token_coverage(self, word_counts, save_to):
   
        # 排序词频
        sorted_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        freqs = np.array([count for _, count in sorted_counts])
        cumulative = np.cumsum(freqs) / np.sum(freqs)

        # 可视化
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative) + 1), cumulative)
        plt.xlabel("Top-N Tokens")
        plt.ylabel("Cumulative Frequency Coverage")
        plt.title("Token Frequency Cumulative Coverage")
        plt.grid(True)

        # 保存图像
        if save_to:
            os.makedirs(save_to, exist_ok=True)
            plt.savefig(os.path.join(save_to, "token_coverage_curve.png"))
        plt.close()


def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ep", "--epoch", type=int, default=10, help='The number of epoch')
    parser.add_argument("-nm", "--test_case_name", help='Test Case Name')
    parser.add_argument("-arch", "--architecture", default="cnn", help='Architecture to be tested')
    parser.add_argument("-bs", "--batch_size", type=int, default=2000, help='batch size')
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-4, help='weight decay')
    parser.add_argument("-feature", "--feature_extraction_method", type=str, default="one-hot", help='Method of feature extraction')
    parser.add_argument("-lr", "--lr_schedule", type=str, default="none", choices=["none", "cosine", "exponential"], help="Learning rate schedule strategy")
    parser.add_argument("-nw", "--num_words", type=int, default=10000, help="The number of words to consider as features")
    parser.add_argument("-data", "--data_name", type=str, default="balanced_dataset", help="The type of dataset")
    # parser.add_argument("-enhanced", "--data_enhance", action="store_true", help="Enable adversarial data enhancement")
    parser.add_argument("-enhanced", "--data_enhance", type=str, choices=["True", "False"], default="False", help="Enable adversarial data enhancement (True/False)")

    args = parser.parse_args()

    return args


def main():

    args = argument_parsing()
    vc = PhishingUrlDetection()
    vc.set_params(args)

    # (x_train, y_train), (x_val, y_val), (x_test, y_test) = vc.load_and_vectorize_data()
    # vc.dl_algorithm(x_train, y_train, x_val, y_val, x_test, y_test)
    # pdb.set_trace()
    if args.data_enhance == "True":
        (x_train, y_train), (x_val, y_val), (x_test, y_test), (adv_x_test, adv_y_test) = vc.load_and_vectorize_data()
        vc.dl_algorithm(x_train, y_train, x_val, y_val, x_test, y_test, adv_x_test, adv_y_test)
    else:
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = vc.load_and_vectorize_data()
        vc.dl_algorithm(x_train, y_train, x_val, y_val, x_test, y_test, 0, 0)

if __name__ == '__main__':
    main()