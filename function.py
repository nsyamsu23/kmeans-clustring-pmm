import streamlit as st
import global_data 
import pandas as pd
import string
import re
from googletrans import Translator
import tensorflow_text as text
from nltk.corpus import stopwords
from wordcloud import WordCloud
import nltk
from nlp_id.tokenizer import Tokenizer
from nlp_id.postag import PosTag
from nlp_id.stopword import StopWord
from nlp_id.lemmatizer import Lemmatizer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
import g4f
from g4f.client import Client
import curl_cffi
import numpy as np
import tensorflow_hub as hub

from tensorflow_text import SentencepieceTokenizer

client = Client()
postagger = PosTag()
tokenizer = Tokenizer()
stopword_nlp_id = StopWord()
lemmatizer = Lemmatizer()
emoji_dict = global_data.emojiDict()
translator = Translator()
nltk.download('punkt')
# Fungsi untuk memplot distribusi skor berdasarkan cluster
# Fungsi untuk menganalisis dan merangkum cluster
def summarize_clusters(df_combined):
    cluster_summary = df_combined.groupby('cluster').agg({
        'reviewId': 'count',
        'score': 'mean',
        'review': lambda x: ' ,'.join(x[:300])  # Gabungkan contoh konten review
    }).reset_index()
    cluster_summary["Ringkasan ulasan"] = cluster_summary["review"].apply(g4f_search)
    return cluster_summary

def plot_score_distribution(df_combined):
    colors = ['blue', 'green', 'red']

    fig, ax = plt.subplots(figsize=(10, 6))
    bin_edges = [1, 2, 3,4,5,6]
    for i, cluster in enumerate(df_combined['cluster'].unique()):
        cluster_data = df_combined[df_combined['cluster'] == cluster]
        ax.hist(cluster_data['score'], bins=bin_edges, alpha=0.5, label=f'Cluster {cluster}', edgecolor='black', color=colors[i])
    
    ax.set_title('Distribusi Skor Berdasarkan Cluster')
    ax.set_xlabel('Skor')
    ax.set_ylabel('Frekuensi')
    ax.legend()
    ax.grid(True)

    ax.set_xticks([1.5, 2.5, 3.5, 4.5, 5.5])
    ax.set_xticklabels([1, 2, 3, 4, 5])

    st.pyplot(fig)
# Fungsi untuk merangkum ulasan menggunakan GPT-4
def g4f_search(tokens):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        provider=g4f.Provider.Blackbox,
        messages=[
            {"role": "assistant", "content": "Anda adalah seorang peneliti bahasa Indonesia profesional dan anda fasih berbahasa indonesia."},
            {"role": "user", "content": f"Ringkaskan dari Ulasan Pengguna Aplikasi Merdeka mengajar ini: \"{tokens}\" utamakan kritik!"}
        ],
    )
    # Fungsi untuk membersihkan pola tertentu dari teks
    def remove_pattern(text):
        pattern = r"\$@\$.+?-rv1\$@\$"
        cleaned_text = re.sub(pattern, "", text)
        cleaned_text = cleaned_text.lower()
        return cleaned_text
    def translate_text(text, src_language='auto', dest_language='id'):
        translator = Translator()
        translation = translator.translate(text, src=src_language, dest=dest_language)
        return translation.text
    return remove_pattern(translate_text(response.choices[0].message.content))
# Fungsi untuk menggabungkan DataFrame berdasarkan reviewId dan menyesuaikan kolom score
def combine_dataframes(df_reviews_all, df_reviews_all_modelling):
    df_reviews_all_modelling_filtered = df_reviews_all_modelling[df_reviews_all_modelling['reviewId'].isin(df_reviews_all['reviewId'])]
    df_combined = pd.merge(df_reviews_all, df_reviews_all_modelling_filtered, on='reviewId', how='right', suffixes=('_x', '_y'))
    df_combined['score_y'] = df_combined['score_x']
    df_combined['score'] = df_combined['score_x']
    df_combined.drop(columns=['score_x', 'score_y'], inplace=True)
    return df_combined
# Fungsi untuk memplot hasil clustering menggunakan PCA
def plot_pca_clusters(X_normalized, labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_normalized)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster in np.unique(labels):
        cluster_data = X_pca[labels == cluster]
        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster}', marker='o', alpha=0.6)
    
    ax.set_title('Hasil Clustering K-means (2D PCA)')
    ax.set_xlabel('PCA Komponen 1')
    ax.set_ylabel('PCA Komponen 2')
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)
# Fungsi untuk menampilkan Davies-Bouldin Index
def display_dbi_scores(X_normalized, clust_num):
    dbi_scores = []
    range_n_clusters = range(2, clust_num + 1)  # Tentukan rentang jumlah cluster yang ingin diuji

    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=5)
        kmeans.fit(X_normalized)
        labels = kmeans.labels_
        dbi = davies_bouldin_score(X_normalized, labels)
        dbi_scores.append(dbi)
        st.write(f'Number of Clusters: {n_clusters}, Davies-Bouldin Index: {dbi}')

    # Hitung threshold berdasarkan persentil
    good_threshold = np.percentile(dbi_scores, 25)
    moderate_threshold = np.percentile(dbi_scores, 50)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot DBI scores
    ax.plot(range_n_clusters, dbi_scores, marker='o', label='DBI Score')

    # Fill regions for quality ranges
    ax.axhspan(0, good_threshold, color='green', alpha=0.1, label='Good Quality')
    ax.axhspan(good_threshold, moderate_threshold, color='yellow', alpha=0.1, label='Moderate Quality')
    ax.axhspan(moderate_threshold, max(dbi_scores) + 0.5, color='red', alpha=0.1, label='Poor Quality')

    # Add titles and labels
    ax.set_title('Davies-Bouldin Index for Different Number of Clusters')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Davies-Bouldin Index')
    ax.grid(True)
    ax.legend()

    # Show plot
    st.pyplot(fig)

# Fungsi untuk menghitung frekuensi kata per cluster
def calculate_word_frequencies(df, clusters):
    overall_word_freq = {}
    for cluster in clusters:
        cluster_data = df[df['cluster'] == cluster]
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(cluster_data['review'])
        word_counts = dict(zip(vectorizer.get_feature_names_out(), np.asarray(X.sum(axis=0)).ravel()))
        for word, count in word_counts.items():
            if word in overall_word_freq:
                overall_word_freq[word].append((cluster, count))
            else:
                overall_word_freq[word] = [(cluster, count)]
    return overall_word_freq

# Fungsi untuk menetapkan ulang kata ke cluster dengan frekuensi tertinggi
def reassign_clusters(df, overall_word_freq, clusters):
    reassigned_clusters = {}
    for word, freq_list in overall_word_freq.items():
        max_freq_cluster = max(freq_list, key=lambda x: x[1])[0]
        reassigned_clusters[word] = max_freq_cluster

    reassigned_df = df.copy()
    for index, row in reassigned_df.iterrows():
        words = row['review'].split()
        cluster_counts = {cluster: 0 for cluster in clusters}
        for word in words:
            if word in reassigned_clusters:
                cluster_counts[reassigned_clusters[word]] += 1
        new_cluster = max(cluster_counts, key=cluster_counts.get)
        reassigned_df.at[index, 'cluster'] = new_cluster

    return reassigned_df

# Fungsi untuk menghasilkan Word Cloud dan Frekuensi Kata untuk setiap cluster
# Fungsi untuk menghasilkan Word Cloud dan Frekuensi Kata untuk setiap cluster
def generate_wordcloud_and_frequencies(df, clusters):
    for cluster in clusters:
        cluster_data = df[df['cluster'] == cluster]
        text = ' '.join(cluster_data['review'])
        
        # Word Cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Word Cloud for Cluster {cluster}')
        st.pyplot(fig)
        
        # Frekuensi kata
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(cluster_data['review'])
        word_counts = dict(zip(vectorizer.get_feature_names_out(), np.asarray(X.sum(axis=0)).ravel()))
        word_freq = pd.DataFrame(word_counts.items(), columns=['word', 'count']).sort_values(by='count', ascending=False)
        word_freq['cluster'] = cluster
        
        # Plot frekuensi kata
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(word_freq['word'][:10], word_freq['count'][:10])
        ax.set_title(f'Word Frequency for Cluster {cluster}')
        ax.set_xlabel('Words')
        ax.set_ylabel('Frequency')
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom')
        st.pyplot(fig)

# Main function to call from Streamlit app
def process_and_display_clusters(df_reviews_all_modelling):
    clusters = list(set(df_reviews_all_modelling['cluster']))
    
    st.write("### Frekuensi Kata dan Cluster ")
    
    overall_word_freq = calculate_word_frequencies(df_reviews_all_modelling, clusters)
    reassigned_df = reassign_clusters(df_reviews_all_modelling, overall_word_freq, clusters)
    generate_wordcloud_and_frequencies(reassigned_df, clusters)
def create_frequency_dict(texts):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    freqs = X.toarray().sum(axis=0)
    vocab = vectorizer.get_feature_names_out()
    frequency_dict = dict(zip(vocab, freqs))
    return frequency_dict

def replace_low_frequency_words(text, frequency_dict, threshold=5):
    words = text.split()
    replaced_words = [max(frequency_dict, key=frequency_dict.get) if frequency_dict.get(word, 0) < threshold else word for word in words]
    return ' '.join(replaced_words)

def pembobotan_kata(df, metode):
    texts = df['content_proses_stemming_nlp_id'].astype(str)
    num_documents = len(texts)
    
    # Create frequency dictionary
    frequency_dict = create_frequency_dict(texts)
    
    # Replace low frequency words
    texts = texts.apply(lambda x: replace_low_frequency_words(x, frequency_dict))
    
    if metode == 'bag_of_words':
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(texts).toarray()
    elif metode == 'tfidf':
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts).toarray()
    elif metode == 'word2vec':
        # Load Word2Vec model from TensorFlow Hub
        embed = hub.load("https://tfhub.dev/google/Wiki-words-250/2")

        def embed_text(text):
            tokens = text.split()  # Basic tokenization
            embeddings = [embed([token]) for token in tokens]
            embeddings = [embedding.numpy() for embedding in embeddings]
            if embeddings:
                return np.mean(embeddings, axis=0).flatten()
            else:
                return np.zeros(250)  # Assuming 250 dimensions for Wiki-words-250 model

        X = np.array([embed_text(doc) for doc in texts])
    else:
        raise ValueError("Metode pembobotan tidak dikenal.")

    X_normalized = normalize(X)
    return X, X_normalized


# Fungsi untuk melakukan clustering
def clustering_k_means(df, metode_pembobotan, clust_num):
    X, X_normalized = pembobotan_kata(df, metode_pembobotan)
    kmeans = KMeans(n_clusters=clust_num, random_state=5)
    kmeans.fit(X_normalized)
    labels = kmeans.labels_
    
    df_modelling = df[['reviewId', 'content_proses_stemming_nlp_id', 'score']].copy()
    df_modelling['cluster'] = labels
    df_modelling.rename(columns={'content_proses_stemming_nlp_id': 'review'}, inplace=True)
    
    return df_modelling
# Define data preparation function
def prepare_data(df,kamus_tidak_baku,chat_words_mapping):
    df_proses = df.copy()
    df_proses.drop(columns=['userName', 'userImage', 'replyContent', 'repliedAt', 'reviewCreatedVersion', 'thumbsUpCount', 'replyContent', 'repliedAt', 'appVersion', 'at'], inplace=True)
    df_proses = df_proses.loc[:, ['reviewId', 'score', 'content']]
    df_proses['content_cleaning'] = df_proses['content'].str.lower()
    df_proses['content_cleaning'] = df_proses['content_cleaning'].apply(replace_emojis_with_meanings)
    df_proses['content_cleaning'] = df_proses['content_cleaning'].apply(remove_urls)
    df_proses['content_cleaning'] = df_proses['content_cleaning'].apply(remove_html_tags)
    df_proses['content_cleaning'] = df_proses['content_cleaning'].apply(hapus_simbol)
    df_proses['content_cleaning'] = df_proses['content_cleaning'].apply(remove_pattern)

    # Normalisasi Kata
    df_proses['content_cleaning_normalized'] = df_proses['content_cleaning'].apply(lambda x: replace_taboo_words(x, kamus_tidak_baku)[0])
    df_proses['content_cleaning_normalized'] = df_proses['content_cleaning_normalized'].apply(lambda x: expand_chat_words(x, chat_words_mapping))
    df_proses['content_cleaning_normalized'] = df_proses['content_cleaning_normalized'].apply(remove_stop_words_nlp_id)
    df_proses = df_proses[df_proses['content_cleaning_normalized'].str.strip() != '']

    # Tokenisasi Teks
    df_proses['content_tokenizing'] = df_proses['content_cleaning_normalized'].apply(tokenizing_words)

    # Part of Speech (POS)
    df_proses['content_part_of_speech'] = df_proses['content_cleaning_normalized'].apply(pos_words)
    pos_tags_to_remove = ["PR", "RP", "UH", "SC", "SYM", "IN", "DT", "CC", "FW"]
    for tag in pos_tags_to_remove:
        df_proses['content_part_of_speech'] = df_proses['content_part_of_speech'].apply(lambda pos_list: remove_pronouns(pos_list, tag1=tag))
    df_proses['content_tokenizing'] = df_proses['content_part_of_speech'].apply(pos_to_tokens)

    # Stemming dan Lemmatisasi
    reset_total_changed_count()
    df_proses['content_proses_stemming_nlp_id'] = df_proses['content_tokenizing'].progress_apply(process_and_count_changes)
    df_proses['content_tokenizing'] = df_proses['content_proses_stemming_nlp_id'].apply(tokenizing_words)
    return df_proses
# Definisikan fungsi untuk lemmatisasi token
def lemmatize_wrapper(tokens):
    # Lemmatize each token
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    original_vs_lemmatized = list(zip(tokens, lemmatized_tokens))

    # Menghitung kata yang telah diubah
    changed_count = sum(1 for original, lemmatized in original_vs_lemmatized if original != lemmatized)

    return ' '.join(lemmatized_tokens), changed_count  # Mengembalikan token yang telah di-lemmatize dan jumlah kata yang diubah


# Variable untuk menghitung total kata yang diubah
total_changed_count = 0
def get_total_changed_count():
    global total_changed_count
    return total_changed_count

def reset_total_changed_count():
    global total_changed_count
    total_changed_count = 0

# Terapkan lemmatisasi dan hitung perubahan dengan progress bar
def process_and_count_changes(tokens):
    global total_changed_count
    lemmatized_tokens, changed_count = lemmatize_wrapper(tokens)
    total_changed_count += changed_count
    return lemmatized_tokens


def remove_pronouns(pos_list,tag1):
    return [(word, tag) for word, tag in pos_list if tag != tag1]

def pos_to_tokens(pos_list):
    # Mengubah list pasangan kata-tag menjadi kalimat
    sentence = ' '.join([word.lower() for word, tag in pos_list])
    # Tokenisasi kalimat
    tokens = tokenizer.tokenize(hapus_simbol(sentence))
    return tokens
def pos_words(text):
    tokens =postagger.get_pos_tag(text)
    return tokens
def tokenizing_words(text):
    tokens = tokenizer.tokenize(remove_stop_words_nlp_id(text))
    return tokens
# Definisikan fungsi untuk mengonversi angka ke huruf
def angka_ke_huruf(angka):
    satuan = ["", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan", "sepuluh", "sebelas"]

    if angka < 12:
        return satuan[angka]
    elif angka < 20:
        return satuan[angka - 10] + " belas"
    elif angka < 100:
        return satuan[angka // 10] + " puluh" + (" " + satuan[angka % 10] if (angka % 10 != 0) else "")
    elif angka < 200:
        return "seratus" + (" " + angka_ke_huruf(angka - 100) if (angka > 100) else "")
    elif angka < 1000:
        return satuan[angka // 100] + " ratus" + (" " + angka_ke_huruf(angka % 100) if (angka % 100 != 0) else "")
    elif angka < 2000:
        return "seribu" + (" " + angka_ke_huruf(angka - 1000) if (angka > 1000) else "")
    elif angka < 1000000:
        return angka_ke_huruf(angka // 1000) + " ribu" + (" " + angka_ke_huruf(angka % 1000) if (angka % 1000 != 0) else "")
    elif angka < 1000000000:
        return angka_ke_huruf(angka // 1000000) + " juta" + (" " + angka_ke_huruf(angka % 1000000) if (angka % 1000000 != 0) else "")
    else:
        return "Angka terlalu besar"

# Definisikan fungsi untuk mengonversi angka dalam teks menjadi huruf tanpa memperhatikan satuan
def remove_pattern(text):
    def ganti_angka(match):
        angka_str = match.group(0)
        angka = int(re.sub(r'\D', '', angka_str))  # Menghapus karakter non-digit
        return angka_ke_huruf(angka)

    return re.sub(r'\b\d+\b', ganti_angka, text)

# Fungsi penggantian kata tidak baku
def remove_stop_words_nltk(text):
    stop_words = stopwords.words('indonesian')
    stop_words.extend([
        "pmm","merdeka mengajar","nya","ok"
    ])
    stop_words = set(stop_words)
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)
def remove_stop_words_nlp_id(text):
    return stopword_nlp_id.remove_stopword(text)

def expand_chat_words(text, chat_words_mapping):
    words = text.split()
    expanded_words = [chat_words_mapping[word] if word in chat_words_mapping else word for word in words]
    return ' '.join(expanded_words)

def replace_taboo_words(text, kamus_tidak_baku):
    if isinstance(text, str) and isinstance(kamus_tidak_baku, dict):
        words = text.split()
        replaced_words = []
        kalimat_baku = []
        kata_diganti = []
        kata_tidak_baku_hash = []

        for word in words:
            if word in kamus_tidak_baku:
                baku_word = kamus_tidak_baku[word]
                if isinstance(baku_word, str) and all(char.isalpha() for char in baku_word):
                    replaced_words.append(baku_word)
                    kalimat_baku.append(baku_word)
                    kata_diganti.append(word)
                    kata_tidak_baku_hash.append(hash(word))
                else:
                    replaced_words.append(word)  # Append original word if baku_word is not valid
            else:
                replaced_words.append(word)

        replaced_text = ' '.join(replaced_words)
    else:
        replaced_text = ''
        kalimat_baku = []
        kata_diganti = []
        kata_tidak_baku_hash = []

    return replaced_text, kalimat_baku, kata_diganti, kata_tidak_baku_hash
def replace_emojis_with_meanings(text):
    emoji_dict = global_data.emojiDict()
    def replace(match):
        emoji_char = match.group()
        emoji_meaning = emoji_dict.get(emoji_char, "")
        return f" {emoji_meaning} "

    # Pola untuk menemukan semua emotikon dalam teks
    emoji_pattern = re.compile("|".join(map(re.escape, emoji_dict.keys())))
    # Mengganti semua emotikon yang ditemukan dengan artinya
    text_with_meanings = emoji_pattern.sub(replace, text)

    # Menghapus emotikon yang tidak dikenal
    non_known_emoji_pattern = re.compile(r'[^\w\s,.?!]')
    text_cleaned = non_known_emoji_pattern.sub('', text_with_meanings)

    # Menghapus spasi tambahan yang mungkin muncul setelah penggantian
    return ' '.join(text_cleaned.split())
def remove_html_tags(text):
    clean_text = re.sub('<.*>', '', text)
    return clean_text

# hapus simbol smbol
def hapus_simbol(teks):
    return teks.translate(str.maketrans('', '', string.punctuation))

# hapus url
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    clean_text = re.sub(url_pattern, '', text)
    return clean_text