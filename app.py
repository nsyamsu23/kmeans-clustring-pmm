import streamlit as st
from streamlit_navigation_bar import st_navbar
import global_data 
import pandas as pd
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from function import *
from tqdm import tqdm
tqdm.pandas()

# Set the page configuration
st.set_page_config(
    page_title="Merdeka Mengajar",
    page_icon="https://play-lh.googleusercontent.com/jBzUQv7vmYT_AUDOt-WQX3Uh4lupq6omQaL2nCdzlG4zNmZUJ2PaqCGpc_03-FBw7w",
    initial_sidebar_state="collapsed",
    layout="wide"
)

# Define the pages and styles for the navigation bar
pages = ["Dashboard", "Data Preparation", "Modeling dan Evaluasi", "About"]
styles = {
    "nav": {
        "background-color": "rgb(32, 30, 67)",
    },
    "div": {
        "max-width": "32rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(238, 238, 238)",
        "margin": "0 0.125rem",
        "padding": "0.4375rem 0.625rem",
    },
    "active": {
        "background-color": "rgba(255, 255, 255, 0.25)",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",
    },
}

# Load and process data
reviews_df = global_data.reviews_all_app()
reviews_df["at"] = pd.to_datetime(reviews_df["at"])
kamus_tidak_baku = global_data.data_dict()
chat_words_mapping = global_data.chat_words_mapping()

if 'df_reviews_all_proses' not in st.session_state:
    st.session_state.df_reviews_all_proses = prepare_data(reviews_df, kamus_tidak_baku, chat_words_mapping)
df_reviews_all_proses = st.session_state.df_reviews_all_proses

# Create the navigation bar
page = st_navbar(pages, styles=styles)

# Get user inputs for clustering
if 'clust_num' not in st.session_state:
    st.session_state.clust_num = 3
if 'metode_pembobotan' not in st.session_state:
    st.session_state.metode_pembobotan = 'word2vec'

clust_num = st.session_state.clust_num
metode_pembobotan = st.session_state.metode_pembobotan

X, X_normalized = pembobotan_kata(df_reviews_all_proses, metode_pembobotan)
df_hasil = clustering_k_means(df_reviews_all_proses, metode_pembobotan, clust_num)
df_combined = combine_dataframes(reviews_df, df_hasil)
cluster_counts = df_hasil['cluster'].value_counts()
cluster_summary = summarize_clusters(df_combined)

# Page: About
if page == "About":
    st.header("About")
    st.write("Ini adalah halaman tentang aplikasi ini, yang menjelaskan tujuan dan cara penggunaan aplikasi.")

elif page == "Data Preparation":
    st.header("Data Preparation")
    st.write("Ini adalah halaman persiapan data, di mana data mentah diproses sebelum dianalisis lebih lanjut.")
    df_reviews_all_proses = st.session_state.df_reviews_all_proses

    with st.expander("Data preview"):
        st.subheader("Data preview")
        st.write("Tampilan awal dari data ulasan yang telah dimuat.")
        st.dataframe(reviews_df.head(5), use_container_width=True)

    with st.expander("Case Folding"):
        st.subheader("Case Folding")
        st.write("Mengubah huruf besar ke kecil, mengubah emotikon ke teks, dan menghapus kode HTML, URL, dan simbol-simbol.")
        df_reviews_all_proses_cf = df_reviews_all_proses.loc[:, ['reviewId', 'content', 'content_cleaning']]
        st.dataframe(df_reviews_all_proses_cf.head(5), use_container_width=True)

    with st.expander("Normalisasi Kata"):
        st.subheader("Normalisasi Kata")
        st.write("Proses normalisasi kata untuk mengubah kata tidak baku menjadi kata baku.")
        df_reviews_all_proses_nk = df_reviews_all_proses.loc[:, ['reviewId', 'content', 'content_cleaning_normalized']]
        st.dataframe(df_reviews_all_proses_nk.head(5), use_container_width=True)

    with st.expander("Tokenisasi Teks"):
        st.subheader("Tokenisasi Teks")
        st.write("Proses memecah teks menjadi token-token yang lebih kecil.")
        df_reviews_all_proses_token = df_reviews_all_proses.loc[:, ['reviewId', 'content', 'content_tokenizing']]
        st.dataframe(df_reviews_all_proses_token.head(5), use_container_width=True)

    with st.expander("Part of Speech (POS)"):
        st.subheader("Part of Speech (POS)")
        st.write("Proses penandaan bagian dari ucapan (POS) pada setiap token dalam teks.")
        df_reviews_all_proses_pos = df_reviews_all_proses.loc[:, ['reviewId', 'content', 'content_part_of_speech']]
        st.dataframe(df_reviews_all_proses_pos.head(5), use_container_width=True)

    with st.expander("Stemming dan Lemmatisasi"):
        st.subheader("Stemming dan Lemmatisasi")
        st.write("Proses mengubah kata ke bentuk dasarnya.")
        df_reviews_all_proses_stem = df_reviews_all_proses.loc[:, ['reviewId', 'content', 'content_proses_stemming_nlp_id']]
        st.dataframe(df_reviews_all_proses_stem.head(5), use_container_width=True)

elif page == "Modeling dan Evaluasi":
    st.header("Modeling dan Evaluasi")
    st.write("Halaman ini digunakan untuk melakukan pemodelan dan evaluasi hasil clustering dari data ulasan.")
    col1input, col2input = st.columns(2)
    with col1input: 
        st.session_state.clust_num = st.number_input("Masukkan jumlah cluster:", min_value=2, value=3)
    with col2input:
        st.session_state.metode_pembobotan = st.selectbox("Pilih metode pembobotan:", ('bag_of_words', 'tfidf', 'word2vec'), index=2)

    df_reviews_all_proses = st.session_state.df_reviews_all_proses

    # Menampilkan data ulasan yang telah diproses
    st.subheader("Data Ulasan yang Telah Diproses")
    st.dataframe(df_reviews_all_proses, use_container_width=True)

    # Menampilkan distribusi cluster
    st.subheader("Distribusi Cluster")
    st.bar_chart(cluster_counts)

    # Membagi hasil clustering dan evaluasi clustering dalam beberapa kolom
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Skor dari Data yang Telah Dikombinasikan")
        plot_score_distribution(df_combined)
    
    with col2:
        st.write("### Davies-Bouldin Index untuk Berbagai Jumlah Cluster")
        display_dbi_scores(X_normalized, clust_num)
        st.write("### Plot PCA untuk Clusters")
        plot_pca_clusters(X_normalized, df_hasil['cluster'])

    # Membagi ringkasan cluster dan data yang telah dikombinasikan dalam beberapa kolom
    st.subheader("Data dengan Hasil Clustering")
    st.dataframe(df_combined, use_container_width=True)
    col3, col4 = st.columns(2)

    with col1:
        st.title("Data Klaster:")
        st.write("Hasil klasterisasi data ulasan pengguna.")
        st.dataframe(df_hasil, use_container_width=True)
        process_and_display_clusters(df_hasil)

    with col2:
        st.write("Ringkasan ulasan:")
        for i, summary in enumerate(cluster_summary["Ringkasan ulasan"]):
            st.subheader(f"Clustering {i}")
            st.write(summary)
    

elif page == "Dashboard":
    st.image(image="https://play-lh.googleusercontent.com/jBzUQv7vmYT_AUDOt-WQX3Uh4lupq6omQaL2nCdzlG4zNmZUJ2PaqCGpc_03-FBw7w",width=100,use_column_width=100)
    st.title('Dashboard Klasterisasi Ulasan Pengguna pada Aplikasi Merdeka Mengajar')

    # Calculate and display metrics
    st.write("Menampilkan beberapa metrik utama dari data ulasan.")
    average_rating = f"{reviews_df['score'].mean():.3f}"
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric(label="Rating Rata-rata", value=average_rating)
    col2.metric(label="Total Ulasan", value=len(reviews_df))
    col3.metric(label="Ulasan Terbaru", value=reviews_df['at'].max().strftime('%Y-%m-%d'))
    col4.metric(label="Ulasan Terlama", value=reviews_df['at'].min().strftime('%Y-%m-%d'))
    col5.metric(label="Ulasan Baru", value=len(reviews_df[reviews_df['at'] > (reviews_df['at'].max() - pd.Timedelta(days=5))]))
    col6.metric(label="Lebih dari 30 Hari", value=len(reviews_df[reviews_df['at'] < (reviews_df['at'].max() - pd.Timedelta(days=30))]))
    st.dataframe(reviews_df.head(5), use_container_width=True)

    col1input, col2input = st.columns(2)
    with col1input: 
        st.session_state.clust_num = st.number_input("Masukkan jumlah cluster:", min_value=2, value=3)
    with col2input:
        st.session_state.metode_pembobotan = st.selectbox("Pilih metode pembobotan:", ('bag_of_words', 'tfidf', 'word2vec'), index=2)

    # Ensure the score column is of type integer
    reviews_df['score'] = reviews_df['score'].astype(int)

    st.title('Analisis Ulasan Aplikasi')
    st.write("Analisis yang mendalam dari ulasan aplikasi berdasarkan berbagai metrik dan distribusi.")

    # Layout for Total Score and Distribution of 'at' (review times)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Total Score')
        st.write("Distribusi total skor dari ulasan pengguna.")
        score_counts = reviews_df['score'].value_counts().sort_index()
        fig, ax = plt.subplots()
        score_counts.plot(kind='bar', ax=ax)
        ax.set_title('Total Score')
        ax.set_xlabel('Skor')
        ax.set_ylabel('Jumlah')
        ax.set_xticks(range(5))
        ax.set_xticklabels([1, 2, 3, 4, 5])
        st.pyplot(fig)

    with col2:
        st.subheader("Distribusi Waktu Ulasan")
        st.write("Distribusi ulasan berdasarkan waktu.")
        fig, ax = plt.subplots(figsize=(10, 6))
        reviews_df['at'].hist(bins=50, edgecolor='k', alpha=0.7, ax=ax)
        ax.set_title('Distribusi Waktu Ulasan')
        ax.set_xlabel('Waktu')
        ax.set_ylabel('Frekuensi')
        ax.grid(True)
        st.pyplot(fig)

    # Distribusi Skor menurut Versi Aplikasi dan Trend Ulasan dari Waktu ke Waktu
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Distribusi Skor menurut Versi Aplikasi')
        st.write("Distribusi skor ulasan berdasarkan versi aplikasi.")
        app_version_scores = reviews_df.groupby(['appVersion', 'score']).size().unstack().fillna(0)
        fig, ax = plt.subplots(figsize=(14, 8))
        app_version_scores.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title('Distribusi Skor berdasarkan Versi Aplikasi')
        ax.set_xlabel('Versi Aplikasi')
        ax.set_ylabel('Jumlah')
        ax.legend(title='Skor')
        ax.grid(True)
        plt.xticks(rotation=90)
        st.pyplot(fig)

    with col2:
        st.subheader('Trend Ulasan dari Waktu ke Waktu')
        st.write("Tren jumlah ulasan dari waktu ke waktu.")
        fig, ax = plt.subplots(figsize=(12, 6))
        reviews_df.set_index('at').resample('M').size().plot(marker='o', ax=ax)
        ax.set_title('Trend of Reviews Over Time')
        ax.set_xlabel('Waktu')
        ax.set_ylabel('Jumlah Ulasan')
        ax.grid(True)
        st.pyplot(fig)

    st.title("Cluster:")
    st.write("Distribusi dari setiap cluster.")
    st.bar_chart(cluster_counts)
    # Data Cluster and Ringkasan Ulasan
    col1, col2 = st.columns(2)

    with col1:
        st.title("Data Klaster:")
        st.write("Hasil klasterisasi data ulasan pengguna.")
        st.dataframe(df_hasil, use_container_width=True)
        process_and_display_clusters(df_hasil)

    with col2:
        st.write("Ringkasan ulasan:")
        for i, summary in enumerate(cluster_summary["Ringkasan ulasan"]):
            st.subheader(f"Clustering {i}")
            st.write(summary)

st.write("Selesai")