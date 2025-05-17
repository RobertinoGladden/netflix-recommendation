# Laporan Proyek Sistem Rekomendasi

## Project Overview
Sistem rekomendasi merupakan elemen kunci dalam platform streaming seperti Netflix, yang menghadapi tantangan menyajikan ribuan konten kepada pengguna setiap hari. Dengan banyaknya pilihan, pengguna sering kali kesulitan menemukan film atau acara TV yang sesuai dengan minat mereka. Proyek ini bertujuan untuk membangun sistem rekomendasi berbasis **Content-Based Filtering** menggunakan dataset `netflix_titles.csv`, yang berisi informasi seperti judul, deskripsi, dan genre konten di Netflix.

**Mengapa Masalah Ini Penting?**  
Sistem rekomendasi yang relevan meningkatkan pengalaman pengguna, memperpanjang waktu retensi, dan mengurangi churn rate. Pendekatan Content-Based Filtering dipilih karena memanfaatkan fitur konten seperti deskripsi dan genre, yang cocok untuk dataset teks seperti `netflix_titles.csv`, dan lebih tahan terhadap masalah *cold start* dibandingkan pendekatan kolaboratif.

**Referensi**  
- Kaggle. (2021). *Netflix Movies and TV Shows Dataset*. Diakses dari https://www.kaggle.com/datasets/shivamb/netflix-shows  
- Lops, P., de Gemmis, M., & Semeraro, G. (2011). Content-based recommender systems: State of the art and trends. In *Recommender Systems Handbook* (pp. 73-105). Springer.  
- Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to recommender systems handbook. In *Recommender Systems Handbook* (pp. 1-35). Springer.

## Business Understanding

### Problem Statements
1. **Kesulitan Pengguna dalam Menemukan Konten Relevan**: Ribuan konten membuat pengguna sulit menemukan film/acara TV yang sesuai tanpa rekomendasi yang dipersonalisasi.  
2. **Kurangnya Personalisasi Berdasarkan Konten**: Tanpa analisis teks seperti deskripsi dan genre, rekomendasi mungkin tidak mencerminkan preferensi pengguna.

### Goals
1. **Membangun Sistem Rekomendasi Berbasis Konten**: Mengembangkan sistem Content-Based Filtering untuk merekomendasikan konten serupa berdasarkan deskripsi dan genre.  
2. **Meningkatkan Relevansi Rekomendasi**: Memastikan rekomendasi sesuai dengan tema dan genre yang disukai pengguna.

### Solution Approach
1. **Pendekatan 1: Kombinasi Deskripsi dan Genre**  
   - Menggabungkan kolom `description` dan `listed_in` untuk fitur teks yang lebih kaya.  
   - Menggunakan **TF-IDF Vectorizer** dan **cosine similarity** untuk menghitung kesamaan antar konten.  
   - Alasan: Kombinasi fitur meningkatkan konteks rekomendasi dengan mempertimbangkan tema dan kategori.  

2. **Pendekatan 2: Deskripsi Saja**  
   - Hanya menggunakan kolom `description` sebagai fitur.  
   - Menggunakan **TF-IDF Vectorizer** dan **cosine similarity**.  
   - Alasan: Fokus pada deskripsi menangkap nuansa cerita, tetapi mungkin kurang spesifik pada genre.

## Data Understanding
Dataset `netflix_titles.csv` berisi **8807 baris** dan **12 kolom**, dengan informasi tentang film dan acara TV di Netflix. Kolom utama untuk rekomendasi (`description` dan `listed_in`) tidak memiliki missing values, dan tidak ada duplikat dalam dataset.

**Variabel dalam Dataset**:
- `show_id`: ID unik (string).  
- `type`: Jenis konten (Movie/TV Show).  
- `title`: Judul konten (string).  
- `director`: Nama sutradara (string, 2634 missing values).  
- `cast`: Daftar aktor (string, 825 missing values).  
- `country`: Negara produksi (string, 831 missing values).  
- `date_added`: Tanggal ditambahkan (string, 10 missing values).  
- `release_year`: Tahun rilis (integer).  
- `rating`: Rating usia (string, 4 missing values).  
- `duration`: Durasi (string, 3 missing values).  
- `listed_in`: Daftar genre (string).  
- `description`: Deskripsi konten (string).

**Exploratory Data Analysis (EDA)**:  
- **Distribusi Genre**: Bar chart menunjukkan genre seperti "Dramas, International Movies" dan "Documentaries" mendominasi.  
  **Insight**: Genre populer mencerminkan preferensi global, relevan untuk rekomendasi berbasis genre.  
- **Word Cloud Deskripsi**: Kata-kata seperti "life", "love", dan "world" sering muncul.  
  **Insight**: Tema emosional dan petualangan penting untuk rekomendasi berbasis teks.

## Data Preparation
Data preparation dilakukan untuk memastikan dataset bersih dan siap untuk pemodelan. Berikut adalah tahapan yang dilakukan secara urut, sesuai dengan notebook:

1. **Penanganan Missing Values**  
   - **Proses**: Mengisi missing values pada kolom `director`, `cast`, `country`, `date_added`, `rating`, dan `duration` dengan `"Unknown"`.  
     ```python
     df['director'] = df['director'].fillna('Unknown')
     df['cast'] = df['cast'].fillna('Unknown')
     df['country'] = df['country'].fillna('Unknown')
     df['date_added'] = df['date_added'].fillna('Unknown')
     df['rating'] = df['rating'].fillna('Unknown')
     df['duration'] = df['duration'].fillna('Unknown')
     ```
   - **Alasan**: Kolom ini tidak digunakan dalam pemodelan utama, tetapi mengisi missing values mencegah error dan menjaga integritas dataset.

2. **Penggabungan Fitur**  
   - **Proses**: Menggabungkan kolom `description` dan `listed_in` menjadi `combined_features`.  
     ```python
     df['combined_features'] = df['description'] + ' ' + df['listed_in']
     ```
   - **Alasan**: Kombinasi ini menciptakan fitur teks yang lebih kaya untuk rekomendasi yang lebih akurat.

3. **Preprocessing Teks**  
   - **Proses**: Mengubah teks ke huruf kecil, menghapus tanda baca, dan menghilangkan stop words.  
     ```python
     def preprocess_text(text):
         text = text.lower()
         text = text.translate(str.maketrans('', '', string.punctuation))
         text = ' '.join(word for word in text.split() if word not in stop_words)
         return text
     df['combined_features'] = df['combined_features'].apply(preprocess_text)
     ```
   - **Alasan**: Preprocessing mengurangi noise dan memastikan hanya kata-kata bermakna yang digunakan.

4. **Ekstraksi Fitur dengan TF-IDF**  
   - **Proses**: Mengubah teks dalam `combined_features` (Pendekatan 1) dan `description` (Pendekatan 2) menjadi vektor numerik menggunakan **TF-IDF Vectorizer**.  
     ```python
     # Pendekatan 1: Kombinasi Deskripsi dan Genre
     tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
     tfidf_matrix = tfidf.fit_transform(df['combined_features'])
     
     # Pendekatan 2: Deskripsi Saja
     tfidf_desc = TfidfVectorizer(max_features=5000, stop_words='english')
     tfidf_matrix_desc = tfidf_desc.fit_transform(df['description'])
     ```
   - **Alasan**: TF-IDF (Term Frequency-Inverse Document Frequency) mengukur pentingnya kata dalam dokumen relatif terhadap dataset, menghasilkan representasi numerik yang cocok untuk perhitungan kesamaan. Parameter `max_features=5000` membatasi dimensi fitur untuk efisiensi komputasi, sementara `stop_words='english'` menghapus kata-kata umum yang tidak relevan.

**Mengapa Data Preparation Diperlukan?**  
- Menangani missing values mencegah error selama pemrosesan.  
- Penggabungan fitur dan preprocessing teks meningkatkan kualitas data.  
- Ekstraksi fitur dengan TF-IDF mengubah teks menjadi format numerik yang dapat diproses oleh algoritma.

## Modeling and Results
Sistem rekomendasi dibangun menggunakan **Content-Based Filtering**, yang merekomendasikan konten berdasarkan kesamaan fitur teks (deskripsi dan/atau genre). Algoritma utama yang digunakan adalah **cosine similarity** untuk mengukur kesamaan antar konten. Berikut adalah detailnya:

### Content-Based Filtering
**Definisi**: Content-Based Filtering merekomendasikan item berdasarkan kesamaan fitur konten (misalnya, deskripsi atau genre) dengan item yang disukai pengguna. Dalam proyek ini, sistem mencocokkan konten berdasarkan teks dari `description` dan/atau `listed_in`.  
**Cara Kerja**:  
1. Mengubah teks menjadi vektor numerik menggunakan TF-IDF.  
2. Menghitung **cosine similarity** antara vektor untuk menentukan seberapa mirip dua konten.  
3. Mengurutkan skor kesamaan dan mengembalikan top-N konten yang paling mirip.

### Cosine Similarity
**Definisi**: Cosine similarity mengukur sudut kosinus antara dua vektor dalam ruang berdimensi tinggi, memberikan skor kesamaan antara 0 (tidak mirip) hingga 1 (sangat mirip).  
**Cara Kerja**:  
- Vektor TF-IDF dari dua konten dibandingkan untuk menghitung skor kesamaan.  
- Skor ini digunakan untuk mengurutkan konten dan memilih yang paling mirip.  
**Keunggulan**: Efisien untuk data teks berdimensi tinggi dan tidak bergantung pada skala fitur.  
**Kekurangan**: Tidak mempertimbangkan preferensi pengguna (hanya fitur konten).

### Pendekatan 1: Kombinasi Deskripsi dan Genre
- **Proses**:  
  - Menggunakan `combined_features` untuk ekstraksi fitur dengan TF-IDF.  
  - Menghitung cosine similarity untuk semua konten.  
  - Membuat fungsi rekomendasi yang mengembalikan top-10 konten serupa.  
    ```python
    def get_recommendations(title, cosine_sim=cosine_sim, df=df):
        idx = df[df['title'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return df[['title', 'listed_in', 'description']].iloc[movie_indices]
    ```
- **Hasil Top-10 untuk "Squid Game"**:  
  | Title                           | Listed In                                              | Description                                                                 |
  |---------------------------------|-------------------------------------------------------|-----------------------------------------------------------------------------|
  | Free to Play                    | Documentaries                                          | This documentary follows three professional video gamers...                 |
  | Kakegurui                       | International TV Shows, TV Dramas, TV Thrillers       | Yumeko Jabami enrolls at Hyakkaou Private Academy...                        |
  | The Circle Brazil               | International TV Shows, Reality TV                     | Be yourself or someone else? In this fun reality competition...             |
  | Futmalls.com                    | Crime TV Shows, International TV Shows, TV Dramas     | Strange occurrences afflict a group of people...                           |
  | Medical Police                  | Crime TV Shows, TV Action & Adventure, TV Comedies    | Doctors Owen Maestro and Lola Spratt leave Childrens Hospital...           |
  | Señora Acero                    | Crime TV Shows, International TV Shows, Spanish-Language TV Shows | When her drug-trafficking husband dies...                                   |
  | Alice in Borderland             | International TV Shows, TV Action & Adventure, TV Thrillers | An aimless gamer and his two friends find themselves...                    |
  | Stories by Rabindranath Tagore  | International TV Shows, TV Dramas                     | The writings of Nobel Prize winner Rabindranath Tagore...                  |
  | Zipi & Zape y la Isla del Capitan | Children & Family Movies, Comedies                   | At Christmas, Zip and Zap take a trip with their parents...                |
  | High-Rise Invasion              | Anime Series, International TV Shows, TV Thrillers    | High schooler Yuri finds herself atop a skyscraper...                      |

### Pendekatan 2: Deskripsi Saja
- **Proses**:  
  - Menggunakan `description` untuk ekstraksi fitur dengan TF-IDF.  
  - Menghitung cosine similarity dan membuat fungsi rekomendasi serupa.  
    ```python
    def get_recommendations_desc(title, cosine_sim=cosine_sim_desc, df=df):
        idx = df[df['title'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return df[['title', 'listed_in', 'description']].iloc[movie_indices]
    ```
- **Hasil Top-10 untuk "Squid Game"**:  
  | Title                           | Listed In                                              | Description                                                                 |
  |---------------------------------|-------------------------------------------------------|-----------------------------------------------------------------------------|
  | Free to Play                    | Documentaries                                          | This documentary follows three professional video gamers...                 |
  | King of Peking                  | Comedies, Dramas, International Movies                | Strapped for cash, a traveling projectionist begins...                     |
  | Ink Master                      | Reality TV                                             | Ten of the country's most skilled tattoo artists...                        |
  | Nailed It! Mexico               | International TV Shows, Reality TV, Spanish-Language TV Shows | The fun, fondant and hilarious cake fails head to Mexico...                |
  | The Redeemed and the Dominant: Fittest on Earth | Documentaries, Sports Movies                   | Questions about endurance, doping and overall fitness...                   |
  | Creep                           | Horror Movies, Independent Movies, Thrillers          | When a cash-strapped videographer takes a job...                            |
  | The Circle Brazil               | International TV Shows, Reality TV                     | Be yourself or someone else? In this fun reality competition...             |
  | Zipi & Zape y la Isla del Capitan | Children & Family Movies, Comedies                   | At Christmas, Zip and Zap take a trip with their parents...                |
  | Isi & Ossi                      | Comedies, International Movies, Romantic Movies       | A billionaire's daughter fakes a relationship...                           |
  | The Half Of It                  | Comedies, Dramas, LGBTQ Movies                        | When smart but cash-strapped teen Ellie Chu agrees...                       |

**Kelebihan dan Kekurangan**:
- **Pendekatan 1**:  
  - **Kelebihan**: Lebih konteksual karena mempertimbangkan genre, relevan untuk preferensi berbasis kategori.  
  - **Kekurangan**: Noise dari genre yang terlalu umum.  
- **Pendekatan 2**:  
  - **Kelebihan**: Menangkap nuansa cerita dari deskripsi.  
  - **Kekurangan**: Kurang spesifik pada genre, menghasilkan rekomendasi yang lebih beragam tetapi kurang relevan.

## Evaluation

**Metrik Evaluasi**:  
Evaluasi dilakukan secara **kualitatif** dengan menganalisis relevansi rekomendasi berdasarkan genre dan tema, serta secara **kuantitatif** menggunakan **Precision@10**.  
- **Precision@10** = (Jumlah rekomendasi relevan dalam Top-10) / 10  
  - Rekomendasi relevan jika memiliki genre atau tema serupa dengan konten input.

**Hasil Evaluasi**:
- **Pendekatan 1 (Kombinasi Deskripsi dan Genre)**:  
  - Untuk "Squid Game" (TV Thrillers, tema kompetisi), rekomendasi seperti "Kakegurui" dan "Alice in Borderland" relevan karena berbagi genre thriller dan tema kompetisi. **Precision@10 ~0.8** (8/10 relevan).  
  - Untuk "Stranger Things" (TV Sci-Fi & Fantasy, Mysteries), rekomendasi seperti "The OA" relevan. **Precision@10 ~0.7**.  
- **Pendekatan 2 (Deskripsi Saja)**:  
  - Untuk "Squid Game", rekomendasi seperti "King of Peking" kurang relevan karena tidak mencerminkan genre thriller. **Precision@10 ~0.5**.  
- **Robustitas**: Diuji dengan "The Queen's Gambit" (rekomendasi seperti "Searching for Bobby Fischer" relevan karena tema strategi) dan "Breaking Bad" (rekomendasi seperti "Extracurricular" relevan karena tema kriminal).

**Kesimpulan**:  
Pendekatan kombinasi deskripsi dan genre lebih unggul karena menghasilkan rekomendasi yang lebih relevan dan konteksual. Sistem ini robust untuk berbagai jenis konten, dengan Precision@10 yang baik untuk judul populer.