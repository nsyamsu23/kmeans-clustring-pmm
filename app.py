import streamlit as st
from streamlit_navigation_bar import st_navbar
import global_data 
import pandas as pd

from nltk.corpus import stopwords
import matplotlib as plt
from function import *
from tqdm import tqdm
tqdm.pandas()
# Set the page configuration
st.set_page_config(
    page_title="Dashboard",
    page_icon=":bar_chart:",
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

# Create the navigation bar
page = st_navbar(pages, styles=styles)

# Page: about
if page == "About":
    st.header("About")
    st.write("Ini adalah halaman about.")

elif page == "Data Preparation":
    st.header("Data Preparation")
    st.write("Ini adalah halaman persiapan data.")
    df_reviews_all_proses = st.session_state.df_reviews_all_proses
    with st.expander("Data preview"):
        st.subheader("Data preview")
        st.write("Ini adalah halaman persiapan data.")
        st.dataframe(reviews_df.head(5), selection_mode="multi-row", use_container_width=True)

    with st.expander("Case Folding"):
        st.subheader("Case Folding")
        st.write("Mengubah huruf besar ke kecil, mengubah emotikon ke teks, dan menghapus kode HTML, URL, dan simbol-simbol.")
        df_reviews_all_proses_cf = df_reviews_all_proses.loc[:, ['reviewId','content', 'content_cleaning']]
        st.dataframe(df_reviews_all_proses_cf.head(5), selection_mode="multi-row", use_container_width=True)

    with st.expander("Normalisasi Kata"):
        st.subheader("Normalisasi Kata")
        st.write("Melakukan normalisasi kata.")
        df_reviews_all_proses_nk = df_reviews_all_proses.loc[:, ['reviewId','content', 'content_cleaning_normalized']]
        st.dataframe(df_reviews_all_proses_nk.head(5), selection_mode="multi-row", use_container_width=True)

    with st.expander("Tokenisasi Teks"):
        st.subheader("Tokenisasi Teks")
        st.write("Melakukan normalisasi kata.")
        df_reviews_all_proses_token = df_reviews_all_proses.loc[:, ['reviewId','content', 'content_tokenizing']]
        st.dataframe(df_reviews_all_proses_token.head(5), selection_mode="multi-row", use_container_width=True)

    with st.expander("Part of Speech (POS)"):
        st.subheader("Part of Speech (POS)")
        st.write("Melakukan normalisasi kata.")
        df_reviews_all_proses_pos = df_reviews_all_proses.loc[:, ['reviewId','content', 'content_part_of_speech']]
        st.dataframe(df_reviews_all_proses_pos.head(5), selection_mode="multi-row", use_container_width=True)

    with st.expander("Stemming dan Lemmatisasi"):
        st.subheader("Stemming dan Lemmatisasi")
        st.write("Melakukan normalisasi kata.")
        df_reviews_all_proses_stem = df_reviews_all_proses.loc[:, ['reviewId','content', 'content_proses_stemming_nlp_id']]
        st.dataframe(df_reviews_all_proses_stem.head(5), selection_mode="multi-row", use_container_width=True)

# Page: Modeling dan Evaluasi
elif page == "Modeling dan Evaluasi":
    st.header("Modeling dan Evaluasi")
    df_reviews_all_proses = st.session_state.df_reviews_all_proses
    st.dataframe(df_reviews_all_proses, selection_mode="multi-row", use_container_width=True)
    clust_num = st.number_input("Masukkan jumlah cluster:",min_value=2, value=3)
    metode_pembobotan = st.selectbox("Pilih metode pembobotan:", ('bag_of_words', 'tfidf', 'word2vec'))
    # Menjalankan clustering jika tombol ditekan
    X, X_normalized = pembobotan_kata(df_reviews_all_proses,metode_pembobotan)
    if st.button("Jalankan Clustering"):
        df_hasil = clustering_k_means(df_reviews_all_proses, metode_pembobotan, clust_num)
        st.dataframe(df_hasil, selection_mode="multi-row", use_container_width=True)
        # Menampilkan distribusi cluster
        cluster_counts = df_hasil['cluster'].value_counts()
        st.write("Distribusi Cluster:")
        st.bar_chart(cluster_counts)
        process_and_display_clusters(df_hasil)
        st.write("### Davies-Bouldin Index for Different Number of Clusters")
        display_dbi_scores(X_normalized, clust_num)
        st.write("### PCA Plot for Clusters")
        plot_pca_clusters(X_normalized,df_hasil['cluster'])
        df_combined = combine_dataframes(reviews_df,df_hasil)
        st.dataframe(df_combined, selection_mode="multi-row", use_container_width=True)
        plot_score_distribution(df_combined)
        cluster_summary = summarize_clusters(df_combined)
        st.dataframe(cluster_summary, selection_mode="multi-row", use_container_width=True)
# Page: Dashboard
elif page == "Dashboard":
    st.title('User Reviews Clustering Dashboard')
    st.write("This is a simple example of a dashboard created using Streamlit.")
    df_reviews_all_proses = st.session_state.df_reviews_all_proses

    # Interactive date filter
    colSD, col2ED = st.columns(2, vertical_alignment="top")
    startDate = reviews_df["at"].min()
    endDate = reviews_df["at"].max()
    with colSD:
        date1 = pd.to_datetime(st.date_input("Start Date", startDate))
    with col2ED:
        date2 = pd.to_datetime(st.date_input("End Date", endDate))

    
    filtered_reviews_df = reviews_df[(reviews_df["at"] >= date1) & (reviews_df["at"] <= date2)].copy()
    # Calculate and display metrics
    average_rating = f"{filtered_reviews_df['score'].mean():.3f}"
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric(label="Rating Avg", value=average_rating)
    col2.metric(label="Total Reviews", value=len(filtered_reviews_df))
    col3.metric(label="Newest Review", value=filtered_reviews_df['at'].max().strftime('%Y-%m-%d'))
    col4.metric(label="Oldest Review", value=filtered_reviews_df['at'].min().strftime('%Y-%m-%d'))
    col5.metric(label="Recent Reviews", value=len(filtered_reviews_df[filtered_reviews_df['at'] > (filtered_reviews_df['at'].max() - pd.Timedelta(days=5))]))
    col6.metric(label="Over 30 Days", value=len(filtered_reviews_df[filtered_reviews_df['at'] < (filtered_reviews_df['at'].max() - pd.Timedelta(days=30))]))
    st.dataframe(reviews_df.head(5), selection_mode="multi-row", use_container_width=True)
    
    col1input,col2input = st.columns(2, vertical_alignment="top") 
    with col1input: 
        clust_num = st.number_input("Masukkan jumlah cluster:",min_value=2, value=3)
    with col2input:
        metode_pembobotan = st.selectbox("Pilih metode pembobotan:", ('bag_of_words', 'tfidf', 'word2vec'))
    # Assuming df_reviews_all is already loaded
    # Convert 'at' and 'repliedAt' to datetime
    reviews_df['at'] = pd.to_datetime(reviews_df['at'])
    reviews_df['repliedAt'] = pd.to_datetime(reviews_df['repliedAt'])

    # Ensure the score column is of type integer
    reviews_df['score'] = reviews_df['score'].astype(int)

    st.title('Analisis Ulasan Aplikasi')

    # Layout for Total Score and Distribution of 'at' (review times)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Total Score')
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
        st.subheader("Distribution of 'at' (review times)")
        fig, ax = plt.subplots(figsize=(10, 6))
        reviews_df['at'].hist(bins=50, edgecolor='k', alpha=0.7, ax=ax)
        ax.set_title('Distribution of Review Times')
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        ax.grid(True)
        st.pyplot(fig)

    # Trend Ulasan dari Waktu ke Waktu
    st.subheader('Trend Ulasan dari Waktu ke Waktu')
    fig, ax = plt.subplots(figsize=(12, 6))
    reviews_df.set_index('at').resample('M').size().plot(marker='o', ax=ax)
    ax.set_title('Trend of Reviews Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of Reviews')
    ax.grid(True)
    st.pyplot(fig)

    # Distribusi Score menurut Versi Aplikasi
    st.subheader('Distribusi Score menurut Versi Aplikasi')
    app_version_scores = reviews_df.groupby(['appVersion', 'score']).size().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(14, 8))
    app_version_scores.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Distribution of Scores by App Version')
    ax.set_xlabel('App Version')
    ax.set_ylabel('Count')
    ax.legend(title='Score')
    ax.grid(True)
    plt.xticks(rotation=90)
    st.pyplot(fig)
    X, X_normalized = pembobotan_kata(df_reviews_all_proses,metode_pembobotan)
    df_hasil = clustering_k_means(df_reviews_all_proses, metode_pembobotan, clust_num)
    cluster_counts = df_hasil['cluster'].value_counts()
    df_combined = combine_dataframes(reviews_df,df_hasil)
    cluster_summary = summarize_clusters(df_combined)
    #clustering
    col1display,col2display = st.columns(2, vertical_alignment="top",)
    col1disA,col1disB = st.columns(2,vertical_alignment="top")
    col1disAa,col1disBa = st.columns(2,vertical_alignment="top")
    with col1display:
        with col1disA:
            st.title(" Data Cluster:")
            st.dataframe(df_hasil, selection_mode="multi-row", use_container_width=True)
            process_and_display_clusters(df_hasil)
        with col1disB:
            st.title(" Cluster:")
            st.bar_chart(cluster_counts)
            st.write("Ringkasan ulasan:")
            for i, summary in enumerate(cluster_summary["Ringkasan ulasan"]):
                st.subheader(f"Clustering {i}")
                st.write(summary)
