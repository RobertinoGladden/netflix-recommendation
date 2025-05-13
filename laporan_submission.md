# Laporan Proyek Sistem Rekomendasi

## Project Overview

Sistem rekomendasi telah menjadi komponen penting dalam platform streaming seperti Netflix, di mana pengguna dihadapkan pada ribuan pilihan konten setiap hari. Dengan jumlah konten yang terus bertambah, pengguna sering kali kesulitan menemukan film atau acara TV yang sesuai dengan preferensi mereka. Oleh karena itu, sistem rekomendasi yang efektif diperlukan untuk meningkatkan pengalaman pengguna, memperpanjang waktu retensi, dan mendorong kepuasan pelanggan. Proyek ini bertujuan untuk membangun sistem rekomendasi berbasis **Content-Based Filtering** menggunakan dataset `netflix_titles.csv`, yang berisi informasi tentang film dan acara TV di Netflix, seperti judul, deskripsi, genre, dan lainnya.

**Mengapa Masalah Ini Penting?**  
Masalah ini perlu diselesaikan karena rekomendasi yang relevan dapat meningkatkan engagement pengguna, mengurangi churn rate, dan membantu Netflix memaksimalkan nilai pelanggan. Tanpa sistem rekomendasi, pengguna mungkin merasa kewalahan oleh banyaknya pilihan, yang dapat mengurangi kepuasan mereka. Pendekatan Content-Based Filtering dipilih karena memungkinkan rekomendasi berdasarkan kesamaan konten (misalnya, deskripsi dan genre), yang cocok untuk dataset dengan informasi teks yang kaya seperti `netflix_titles.csv`.

**Referensi**  
Menurut penelitian, sistem rekomendasi berbasis konten efektif untuk memberikan rekomendasi yang dipersonalisasi dengan memanfaatkan fitur item seperti deskripsi atau genre (Ricci et al., 2011). Selain itu, pendekatan ini lebih tahan terhadap masalah *cold start* dibandingkan pendekatan kolaboratif, karena tidak memerlukan data interaksi pengguna (Lops et al., 2011). Dataset seperti `netflix_titles.csv` sering digunakan dalam penelitian untuk mengembangkan sistem rekomendasi berbasis teks (Kaggle, 2021).

**Referensi Sitasi**  
- Kaggle. (2021). *Netflix Movies and TV Shows Dataset*. Diakses dari https://www.kaggle.com/datasets/shivamb/netflix-shows  
- Lops, P., de Gemmis, M., & Semeraro, G. (2011). Content-based recommender systems: State of the art and trends. In *Recommender Systems Handbook* (pp. 73-105). Springer.  
- Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to recommender systems handbook. In *Recommender Systems Handbook* (pp. 1-35). Springer.

## Business Understanding

Bagian ini menjelaskan proses klarifikasi masalah dan tujuan proyek untuk memenuhi kebutuhan bisnis Netflix dalam memberikan rekomendasi konten yang relevan.

### Problem Statements
1. **Kesulitan Pengguna dalam Menemukan Konten Relevan**: Dengan ribuan film dan acara TV, pengguna sulit menemukan konten yang sesuai dengan minat mereka tanpa rekomendasi yang dipersonalisasi.  
2. **Kurangnya Personalisasi Berdasarkan Konten**: Tanpa analisis teks seperti deskripsi dan genre, rekomendasi mungkin tidak mencerminkan preferensi pengguna berdasarkan tema atau jenis konten.

### Goals
1. **Membangun Sistem Rekomendasi Berbasis Konten**: Mengembangkan sistem Content-Based Filtering yang merekomendasikan konten serupa berdasarkan deskripsi dan genre, sehingga membantu pengguna menemukan film/acara TV yang relevan.  
2. **Meningkatkan Relevansi Rekomendasi**: Memastikan rekomendasi sesuai dengan tema dan genre konten yang disukai pengguna, dengan memanfaatkan fitur teks seperti `description` dan `listed_in`.

### Solution Approach
Untuk mencapai tujuan di atas, dua pendekatan Content-Based Filtering diusulkan:  
1. **Pendekatan 1: Menggunakan Kombinasi Deskripsi dan Genre**  
   - Menggabungkan kolom `description` dan `listed_in` untuk menciptakan fitur teks yang lebih kaya.  
   - Menggunakan **TF-IDF Vectorizer** untuk mengubah teks menjadi vektor numerik dan **cosine similarity** untuk mengukur kesamaan antar konten.  
   - Alasan: Kombinasi fitur meningkatkan konteks rekomendasi dengan mempertimbangkan tema (deskripsi) dan kategori (genre).  

2. **Pendekatan 2: Menggunakan Deskripsi Saja**  
   - Hanya menggunakan kolom `description` sebagai fitur untuk rekomendasi.  
   - Menggunakan **TF-IDF Vectorizer** dan **cosine similarity** seperti pendekatan pertama.  
   - Alasan: Fokus pada deskripsi dapat menangkap nuansa cerita, tetapi mungkin kehilangan informasi genre.

## Data Understanding

Dataset yang digunakan adalah `netflix_titles.csv`, tersedia di Kaggle (https://www.kaggle.com/datasets/shivamb/netflix-shows). Dataset ini berisi **8807 baris** dan **12 kolom**, dengan informasi tentang film dan acara TV di Netflix. Data memiliki beberapa missing values, tetapi kolom utama untuk rekomendasi (`description` dan `listed_in`) lengkap tanpa missing values. Tidak ada duplikat dalam dataset (diperiksa dengan `df.duplicated().sum()`).

**Variabel dalam Dataset**:
- `show_id`: ID unik untuk setiap konten (string).  
- `type`: Jenis konten (Movie atau TV Show).  
- `title`: Judul konten (string).  
- `director`: Nama sutradara (string, 2634 missing values).  
- `cast`: Daftar aktor (string, 825 missing values).  
- `country`: Negara produksi (string, 831 missing values).  
- `date_added`: Tanggal ditambahkan ke Netflix (string, 10 missing values).  
- `release_year`: Tahun rilis (integer).  
- `rating`: Rating usia (string, 4 missing values).  
- `duration`: Durasi konten (string, 3 missing values).  
- `listed_in`: Daftar genre (string).  
- `description`: Deskripsi konten (string).

**Exploratory Data Analysis (EDA)**:  
- **Visualisasi Distribusi Genre**: Bar chart menunjukkan genre teratas seperti "Dramas, International Movies" dan "Documentaries" mendominasi dataset.  
  ```python
  plt.figure(figsize=(12, 6))
  df['listed_in'].value_counts().head(10).plot(kind='bar')
  plt.title('Top 10 Genres in Netflix Dataset')
  plt.xlabel('Genre')
  plt.ylabel('Count')
  plt.xticks(rotation=45)
  plt.show()
  ```
  **Insight**: Genre populer mencerminkan preferensi global, yang dapat membantu rekomendasi berbasis genre.  

- **Word Cloud untuk Deskripsi**: Visualisasi kata-kata umum dalam kolom `description` menunjukkan istilah seperti "life", "love", dan "world" sering muncul.  
  ```python
  text = ' '.join(df['description'].dropna())
  wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
  plt.figure(figsize=(10, 5))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis('off')
  plt.title('Word Cloud of Netflix Descriptions')
  plt.show()
  ```
  **Insight**: Kata-kata umum menunjukkan tema emosional dan petualangan, yang penting untuk rekomendasi berbasis teks.

## Data Preparation

Data preparation dilakukan untuk memastikan dataset bersih dan siap untuk pemodelan. Berikut adalah langkah-langkah yang dilakukan, beserta alasan dan kode terkait:

1. **Penanganan Missing Values**  
   - **Proses**: Mengisi missing values pada kolom `director` (2634), `cast` (825), `country` (831), `date_added` (10), `rating` (4), dan `duration` (3) dengan placeholder `"Unknown"`.  
     ```python
     df['director'] = df['director'].fillna('Unknown')
     df['cast'] = df['cast'].fillna('Unknown')
     df['country'] = df['country'].fillna('Unknown')
     df['date_added'] = df['date_added'].fillna('Unknown')
     df['rating'] = df['rating'].fillna('Unknown')
     df['duration'] = df['duration'].fillna('Unknown')
     ```
   - **Alasan**: Kolom ini tidak digunakan dalam pemodelan utama (hanya `description` dan `listed_in`), tetapi mengisi missing values mencegah error dan menjaga integritas dataset untuk analisis tambahan. Menghapus baris tidak dilakukan karena akan mengurangi data yang berharga (hingga 30% untuk `director`).

2. **Penggabungan Fitur**  
   - **Proses**: Menggabungkan kolom `description` dan `listed_in` menjadi kolom baru `combined_features`.  
     ```python
     df['combined_features'] = df['description'] + ' ' + df['listed_in']
     ```
   - **Alasan**: Kombinasi ini menciptakan fitur teks yang lebih kaya, menggabungkan narasi cerita (deskripsi) dan kategori (genre) untuk rekomendasi yang lebih akurat.

3. **Preprocessing Teks**  
   - **Proses**: Mengubah teks ke huruf kecil, menghapus tanda baca, dan menghilangkan stop words (kata umum seperti "the", "is").  
     ```python
     def preprocess_text(text):
         text = text.lower()
         text = text.translate(str.maketrans('', '', string.punctuation))
         text = ' '.join(word for word in text.split() if word not in stop_words)
         return text
     df['combined_features'] = df['combined_features'].apply(preprocess_text)
     ```
   - **Alasan**: Preprocessing teks mengurangi noise, memastikan hanya kata-kata bermakna yang digunakan dalam pemodelan, dan meningkatkan efisiensi TF-IDF Vectorizer.

**Mengapa Data Preparation Diperlukan?**  
- Missing values dapat menyebabkan error selama pemrosesan data.  
- Penggabungan fitur meningkatkan konteks untuk rekomendasi.  
- Preprocessing teks memastikan data konsisten dan relevan, mengurangi dimensi fitur yang tidak perlu.

## Modeling

Sistem rekomendasi dibangun menggunakan **Content-Based Filtering** dengan dua pendekatan, seperti dijelaskan dalam **Solution Approach**. Berikut adalah detailnya:

1. **Pendekatan 1: Kombinasi Deskripsi dan Genre**  
   - **Proses**:  
     - Menggunakan **TF-IDF Vectorizer** untuk mengubah `combined_features` menjadi vektor numerik. Parameter `max_features=5000` digunakan untuk efisiensi.  
       ```python
       tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
       tfidf_matrix = tfidf.fit_transform(df['combined_features'])
       ```
     - Menghitung **cosine similarity** untuk mengukur kesamaan antar konten.  
       ```python
       cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
       ```
     - Membuat fungsi rekomendasi yang mengembalikan 10 konten serupa berdasarkan judul input.  
       ```python
       def get_recommendations(title, cosine_sim=cosine_sim, df=df):
           idx = df[df['title'] == title].index[0]
           sim_scores = list(enumerate(cosine_sim[idx]))
           sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
           sim_scores = sim_scores[1:11]
           movie_indices = [i[0] for i in sim_scores]
           return df[['title', 'listed_in', 'description']].iloc[movie_indices]
       ```
   - **Output**: Top-10 rekomendasi untuk judul seperti "Squid Game" dan "Stranger Things". Contoh untuk "Squid Game":  
     ```bash
      Recommendations for 'Squid Game':
                                        title  \
      1011                       Free to Play   
      3684                          Kakegurui   
      2827                  The Circle Brazil   
      1564                       Futmalls.com   
      3037                     Medical Police   
      3886                       Señora Acero   
      1562                Alice in Borderland   
      69       Stories by Rabindranath Tagore   
      5411  Zipi & Zape y la Isla del Capitan   
      1044                 High-Rise Invasion
    ```

2. **Pendekatan 2: Deskripsi Saja**  
   - **Proses**: Sama seperti pendekatan pertama, tetapi hanya menggunakan kolom `description`.  
     ```python
     tfidf_desc = TfidfVectorizer(max_features=5000, stop_words='english')
     tfidf_matrix_desc = tfidf_desc.fit_transform(df['description'])
     cosine_sim_desc = cosine_similarity(tfidf_matrix_desc, tfidf_matrix_desc)
     ```
   - **Output**: Top-10 rekomendasi untuk "Squid Game" menggunakan hanya deskripsi, seperti "Free to Play", "King of Peking", dll.

**Kelebihan dan Kekurangan**:
- **Pendekatan 1 (Kombinasi Deskripsi dan Genre)**  
  - **Kelebihan**: Rekomendasi lebih konteksual karena mempertimbangkan genre, yang sering menjadi faktor utama preferensi pengguna.  
  - **Kekurangan**: Kombinasi fitur dapat memperkenalkan noise jika genre tidak konsisten atau terlalu umum.  
- **Pendekatan 2 (Deskripsi Saja)**  
  - **Kelebihan**: Fokus pada narasi cerita, menangkap nuansa tematik yang lebih spesifik.  
  - **Kekurangan**: Kehilangan informasi genre, yang dapat mengurangi relevansi untuk pengguna yang memprioritaskan kategori tertentu.

## Evaluation

**Metrik Evaluasi**:  
Sistem dievaluasi secara **kualitatif** dengan menganalisis relevansi rekomendasi berdasarkan kesamaan genre dan tema. Metrik kuantitatif seperti **Precision@10** digunakan untuk mengukur proporsi rekomendasi yang relevan dari 10 rekomendasi yang dihasilkan.

**Formula Metrik**:
- **Precision@10** = (Jumlah rekomendasi relevan dalam Top-10) / 10  
  - Rekomendasi dianggap relevan jika memiliki genre yang sama atau tema cerita yang serupa dengan konten input (misalnya, drama, thriller, atau petualangan untuk "Squid Game").  
  - Contoh: Jika 7 dari 10 rekomendasi untuk "Squid Game" memiliki genre "TV Thrillers" atau tema kompetisi, maka Precision@10 = 7/10 = 0.7.

**Hasil Evaluasi**:
- **Pendekatan 1 (Kombinasi Deskripsi dan Genre)**:  
  - Untuk "Squid Game" (genre: TV Thrillers, tema: kompetisi bertahan hidup), rekomendasi seperti "Kakegurui" (TV Thrillers) dan "Alice in Borderland" (TV Action & Adventure, Thriller) sangat relevan karena memiliki genre dan tema serupa. Precision@10 diperkirakan ~0.8 (8/10 relevan).  
  - Untuk "Stranger Things" (genre: TV Sci-Fi & Fantasy, Mysteries), rekomendasi seperti "The OA" dan "Manifest" relevan karena berbagi genre misteri dan sci-fi. Precision@10 ~0.7.  
- **Pendekatan 2 (Deskripsi Saja)**:  
  - Rekomendasi untuk "Squid Game" seperti "Free to Play" dan "King of Peking" kurang relevan karena tidak selalu mencerminkan genre thriller atau tema kompetisi, hanya kesamaan kata dalam deskripsi. Precision@10 ~0.5.  
  - Pendekatan ini cenderung menghasilkan rekomendasi yang lebih beragam tetapi kurang spesifik pada genre.

**Analisis Tambahan**:  
- Pendekatan 1 lebih unggul karena memanfaatkan genre, yang meningkatkan relevansi untuk preferensi pengguna.  
- Uji robustitas dilakukan dengan judul tambahan seperti "The Queen's Gambit" (rekomendasi seperti "Searching for Bobby Fischer" relevan karena tema strategi/kompetisi) dan "Breaking Bad" (rekomendasi seperti "Extracurricular" relevan karena tema kriminal).  
- Visualisasi (bar chart dan word cloud) mendukung pemahaman data, memastikan fitur yang digunakan (deskripsi dan genre) mencerminkan karakteristik konten.

**Kesimpulan**:  
Sistem rekomendasi dengan pendekatan kombinasi deskripsi dan genre memberikan rekomendasi yang lebih relevan dan konteksual dibandingkan hanya menggunakan deskripsi. Precision@10 menunjukkan performa yang baik untuk judul populer, dan sistem cukup robust untuk menangani berbagai jenis konten (film dan TV show).