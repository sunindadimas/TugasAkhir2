import streamlit as st
import pandas as pd
import re
import nltk
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator
from collections import defaultdict

# Download NLTK stopwords
nltk.download('stopwords')

# Fungsi preprocessing teks
def casefolding(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'_', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Memuat kunci normalisasi
try:
    key_norm = pd.read_csv('data/normalize.csv', encoding='latin1')
except UnicodeDecodeError:
    key_norm = pd.read_csv('data/normalize.csv', encoding='ISO-8859-1')

def text_normalize(text):
    words = text.split()
    normalized_words = []
    for word in words:
        if (key_norm['singkatan'] == word).any():
            normalized_word = key_norm[key_norm['singkatan'] == word]['hasil'].values[0]
            normalized_words.append(normalized_word)
        else:
            normalized_words.append(word)
    return ' '.join(normalized_words)

# Stopwords dalam bahasa Indonesia
stopwords_ind = stopwords.words('indonesian')
more_stopword = ['kalo', 'amp', 'gini', 'biar', 'bikin', 'bilang', 'bnetwork', 'loh', '&amp', 'yah', 'zy_zy', 'mh', 'anu', 'x', 
                'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'untuk', 'dengan', 'pada', 'adalah', 'itu','atau', 'sebagai', 
                'tidak', 'oleh', 'juga', 'karena', 'sudah', 'jadi', 'sangat', 'saja', 'agar','mereka', 'kami', 'kita', 'nih',
                'lah', 'kan', 'sih', 'dong', 'nya', 'deh', 'pun', 'gitu', 'tak', 'nah', 'eh', 'hanya', 'harus', 'begitu', 
                'saya', 'aku', 'anda', 'dia', 'ia', 'dalam', 'tersebut', 'bagi', 'akan','masih', 'semua', 'bisa', 'lain', 
                'mengapa', 'dapat', 'gas', 'oke', 'ada', 'kemana', 'axamandiri', 'axa', 'iya']
stopwords_ind.extend(more_stopword)

def remove_stop_word(text):
    words = text.split()
    clean_words = [word for word in words if word not in stopwords_ind]
    return ' '.join(clean_words)

def tokenizing(text):
    return text.split()

# Stemming function
factory = StemmerFactory()
stemmer = factory.create_stemmer()

additional_dict = {
    'dilantik' : 'lantik',
    'penggemblengan' : 'pembekalan',
    'ingetin' : 'ingat',
    'dipahami': 'paham',
    'tri' : 'mentri',
    'ingatlah': 'ingat',
    'berterima': 'terima',
    'berlangsung': 'langsung',
    'berada': 'ada',
    'dibawah' : 'bawah',
    'menyayangkan' : 'sayang',
    'dieksploitasi' : 'eksploitasi',
    'presidennya' : 'presiden',
    'wakilnya' : 'wakil',
    'mentrinya' : 'mentri',
    'menteri' : 'mentri',
    'mentri' : 'mentri',
    'kementrian' : 'kementrian'
}

def stemming(text):
    words = text.split()
    stemmed_words = []
    for word in words:
        stemmed_word = stemmer.stem(word)
        if word in additional_dict:
            stemmed_word = additional_dict[word]
        stemmed_words.append(stemmed_word)
    return ' '.join(stemmed_words)

# membuat fungsi untuk menggabungkan seluruh langkah text preprocessing
def text_preprocessing_process(text):
    text = casefolding(text)
    text = text_normalize(text)
    text = remove_stop_word(text)
    text = stemming(text)
    tokens = tokenizing(text)
    text = ' '.join(tokens)
    return text

# Memuat dan preprocessing data
data_model = pd.read_csv('data/mentri.csv')

# Inisialisasi dan fit vectorizer
tfidf = TfidfVectorizer(max_features=8000)
tfidf.fit(data_model['clean_teks'])
X_tfidf = tfidf.transform(data_model['clean_teks'])
y = data_model['sentiment']

# Terapkan SMOTE untuk menyeimbangkan data
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_tfidf, y)

# Bagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

# Inisialisasi dan latih model SVC
svm_model = SVC(C=37.55401188473625, 
                class_weight=None, 
                coef0=0.5, 
                degree=10, 
                gamma=0.1, 
                kernel='sigmoid', 
                max_iter=5000, 
                shrinking=True, 
                tol=0.0001)
svm_model.fit(X_train, y_train)

def classify_text(text, vectorizer, model):
    preprocessed_text = text_preprocessing_process(text)
    text_tfidf = vectorizer.transform([preprocessed_text])
    prediction = model.predict(text_tfidf)[0]
    return preprocessed_text, prediction

# Inisiasi VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Tambahan kata-kata positif dan negatif ke kamus VADER
additional_pos_words = [
    'selamat', 'resmi', 'pelantikan', 'menteri', 'baik', 'manfaat', 
    'sejahtera', 'amanah', 'alhamdulillah', 'maju', 'tokoh'
]

additional_neg_words = [
    'korupsi', 'isu', 'licik', 'tolak', 'zionis', 'bodoh', 'mengemis',
    'titipan', 'kesalahan', 'gendut', 'mengkritik', 'anggaran', 'pajak', 'mulyono', 'rusak'
]

# Menambahkan kata-kata dengan skor ke kamus VADER
for word in additional_pos_words:
    sid.lexicon[word] = 2.0  # Nilai positif tinggi untuk kata positif

for word in additional_neg_words:
    sid.lexicon[word] = -2.0  # Nilai negatif tinggi untuk kata negatif

def vader_sentiment(text):
    scores = sid.polarity_scores(text)
    if scores['compound'] >= 0.01:
        return 'positive'
    else:
        return 'negative'

# Sidebar untuk navigasi
st.sidebar.title("Navigasi")
options = st.sidebar.radio("Pergi ke", ["📊 Eksplorasi Data", "🔄 Preprocessing", "🔍 Prediksi", "📝 Kesimpulan"])

# Menambahkan CSS untuk justify text dan margin pada informasi penulis
st.markdown(
    """
    <style>
    .justified-text {
        text-align: justify;
    }
    .author-info {
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Halaman: Eksplorasi Data
if options == "📊 Eksplorasi Data":
    st.header("Eksplorasi Data Analis")
    
    # Membaca data
    data = pd.read_csv('data/mentri.csv')
    # Deskripsi penelitian dengan justify text
    
    st.markdown("""
    <div class="justified-text">
    Kabinet Merah Putih periode 2024-2029 di bawah kepemimpinan Presiden Prabowo Subianto diharapkan mampu menjawab tantangan 
    sosial, ekonomi, maupun politik yang semakin kompleks. Setiap keputusan kebijakan yang dibuat oleh para menteri akan 
    langsung berdampak pada masyarakat dan menjadi perhatian publik. Dengan berkembangannya teknologi informasi dan komunikasi, 
    khususnya media sosial, memungkinkan masyarakat menyampaikan opini, kritik, atau dukungan terhadap kebijakan pemerintah 
    secara lebih terbuka dan cepat. Media sosial seperti X telah menjadi ruang publik virtual di mana masyarakat dapat 
    mengungkapkan sentimen mereka terkait kinerja menteri atau kebijakan kabinet. Untuk memahami sentimen publik terhadap
    Kabinet Merah Putih, analisis sentimen teks dengan algoritma Support Vector Mchine dapat digunakan untuk 
    mengklasifikasikan opini masyarakat di media sosial menjadi sentimen positif atau negatif, sehingga membantu 
    pemerintah menilai efektivitas dan penerimaan Kabinet Merah Putih.
    </div>
    
    """, unsafe_allow_html=True)    
    
    # Menghitung jumlah masing-masing sentimen
    sentimen_counts = data['sentiment'].value_counts()
    
    # Plot diagram lingkaran
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(sentimen_counts, labels=sentimen_counts.index, autopct='%1.1f%%', colors=['skyblue', 'lightgreen'])
    ax.set_title('Proporsi Sentimen', fontsize=8)
    plt.setp(ax.texts, size=8)  # Menyesuaikan ukuran teks dalam plot
    st.pyplot(fig)
        
    st.markdown("""
    <div class="justified-text">
    Dari diagram ini, kita dapat melihat bahwa mayoritas sentimen masyarakat terhadap Kabinet Merah Putih adalah Positif, dengan 68.9% 
    dari total opini yang dianalisis menunjukkan dukungan akan kabinet merah putih. Sebaliknya, hanya 31.1% dari total opini yang menunjukkan
    kritik dan ketidak puasan terhadap Kabinet Merah Putih.
    </div>
    
    """, unsafe_allow_html=True) 
    
    # Memfilter data untuk masing-masing sentimen
    positive_tweets = data[data['sentiment'] == 'positive']['clean_teks']
    negative_tweets = data[data['sentiment'] == 'negative']['clean_teks']

    # Membuat WordCloud untuk sentimen positif
    all_positive_text = ' '.join(positive_tweets)
    wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(all_positive_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud_positive, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud untuk Sentimen Positif')
    st.pyplot(fig)
    
    st.markdown("""
    <div class="justified-text">
    Berdasarkan hasil visualisasi wordcloud sentimen positif, terdapat beberapa kata dengan frekuensi kemunculan yang tinggi,
    seperti "merah", "putih", "kabinet", "menteri", "wakil", "prabowo", "lantik", dan "presiden". Analisis dari wordcloud ini 
    menunjukkan bahwa sentimen positif dalam data berkaitan dengan isu pemerintahan, terutama pelantikan kabinet baru, simbol 
    nasionalisme, dan dukungan terhadap tokoh-tokoh penting seperti Prabowo Subianto. 
    </div>
    
    """, unsafe_allow_html=True) 
    
    # Membuat WordCloud untuk sentimen negatif
    all_negative_text = ' '.join(negative_tweets)
    wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(all_negative_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud_negative, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud untuk Sentimen Negatif')
    st.pyplot(fig)
    
    st.markdown("""
    <div class="justified-text">
    Berdasarkan hasil visualisasi wordcloud sentimen negatif, terdapat beberapa kata dengan frekuensi kemunculan yang tinggi,
    seperti "kabinet", "merah", "putih", "prabowo", "presiden", "lantik", dan "kerja". Analisis dari wordcloud ini menunjukkan 
    bahwa sentimen negatif dalam data berfokus pada isu-isu pemerintahan, khususnya terkait kabinet baru dan tokoh-tokoh seperti 
    Prabowo, Luhut, dan Raffi Ahmad. Selain itu, terdapat indikasi kritik terhadap berbagai hal, seperti kinerja, pelantikan, dan isu-isu yang 
    melibatkan nama-nama seperti "luhut" dan "raffi ahmad". Hal ini mencerminkan adanya ketidakpuasan atau kritik dalam pembahasan 
    terkait pemerintahan.
    </div>
    
    """, unsafe_allow_html=True) 

# Halaman: Preprocessing
elif options == "🔄 Preprocessing":
    st.header("Langkah Preprocessing")
    st.write("Unggah data Anda dan lakukan langkah-langkah preprocessing teks.")

     # Unggah data
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data yang Diunggah:")
        st.write(data.head(15))  # Menampilkan 100 baris pertama data
        
        # Pilih kolom untuk preprocessing
        column = st.selectbox("Pilih kolom untuk preprocessing", data.columns)
        st.write(f"Kolom yang dipilih: {column}")

        # Inisialisasi session state untuk menyimpan hasil sementara
        if 'casefolding_text' not in st.session_state:
            st.session_state.casefolding_text = None
        if 'normalisasi_text' not in st.session_state:
            st.session_state.normalisasi_text = None
        if 'remove_text' not in st.session_state:
            st.session_state.remove_text = None
        if 'tokenize_text' not in st.session_state:
            st.session_state.tokenize_text = None
        if 'stemming_text' not in st.session_state:
            st.session_state.stemming_text = None
        if 'labeling_text' not in st.session_state:
            st.session_state.labeling_text = None
        if 'tfidf_text' not in st.session_state:
            st.session_state.tfidf_text = None

        st.subheader("Sebelum dan Sesudah Case Folding")
        st.write("Sebelum Case Folding:")
        st.write(data[column].head(15))  # Menampilkan 100 baris pertama data

        if st.button("Case Folding"):
            st.session_state.casefolding_text = data[column].apply(casefolding)
            st.write("Setelah Case Folding:")
            st.write(st.session_state.casefolding_text.head(15))  # Menampilkan 100 baris pertama data

        if st.session_state.casefolding_text is not None:
            st.subheader("Sebelum dan Sesudah Normalisasi Teks")
            st.write("Sebelum Normalisasi Teks:")
            st.write(st.session_state.casefolding_text.head(15))  # Menampilkan 100 baris pertama data
            if st.button("Normalisasi Teks"):
                st.session_state.normalisasi_text = st.session_state.casefolding_text.apply(text_normalize)
                st.write("Setelah Normalisasi Teks:")
                st.write(st.session_state.normalisasi_text.head(15))  # Menampilkan 100 baris pertama data

        if st.session_state.normalisasi_text is not None:
            st.subheader("Sebelum dan Sesudah Menghapus Stop Word")
            st.write("Sebelum Menghapus Stop Word:")
            st.write(st.session_state.normalisasi_text.head(15))  # Menampilkan 100 baris pertama data
            if st.button("Menghapus Stop Word"):
                st.session_state.remove_text = st.session_state.normalisasi_text.apply(remove_stop_word)
                st.write("Setelah Menghapus Stop Word:")
                st.write(st.session_state.remove_text.head(15))  # Menampilkan 100 baris pertama data

        if st.session_state.remove_text is not None:
            st.subheader("Sebelum dan Sesudah Tokenisasi")
            st.write("Sebelum Tokenisasi:")
            st.write(st.session_state.remove_text.head(15))  # Menampilkan 100 baris pertama data
            if st.button("Tokenisasi"):
                st.session_state.tokenize_text = st.session_state.remove_text.apply(tokenizing)
                # Gabungkan token kembali menjadi string untuk langkah berikutnya
                st.session_state.tokenize_text = st.session_state.tokenize_text.apply(lambda x: ','.join(x))
                st.write("Setelah Tokenisasi:")
                st.write(st.session_state.tokenize_text.head(15))  # Menampilkan 100 baris pertama data

        if st.session_state.tokenize_text is not None:
            st.subheader("Sebelum dan Sesudah Stemming")
            st.write("Sebelum Stemming:")
            st.write(st.session_state.tokenize_text.head(15))  # Menampilkan 100 baris pertama data
            if st.button("Stemming"):
                # Memuat dan menampilkan data hasil stemming dari file Excel
                data_stemming = pd.read_csv('data/mentri.csv')
                st.session_state.stemming_text = data_stemming['clean_teks']
                st.write("Setelah Stemming:")
                st.write(st.session_state.stemming_text.head(15))  # Menampilkan 100 baris pertama data
                
        if st.session_state.stemming_text is not None:
            st.subheader("Sebelum dan Sesudah Labelling")
            st.write("Sebelum Labelling:")
            st.write(st.session_state.stemming_text.head(15))  # Menampilkan 100 baris pertama data
            if st.button("Labelling"):
                # Memuat dan menampilkan data hasil stemming dari file Excel
                data_labeling = pd.read_csv('data/mentri.csv')
                st.session_state.labeling_text = data_labeling
                st.write("Setelah Labeling:")
                st.write(st.session_state.labeling_text.head(15))  # Menampilkan 100 baris pertama data

        # Proses TF-IDF
        if st.session_state.labeling_text is not None:
            st.subheader("Sebelum dan Sesudah TF-IDF")
            st.write("Sebelum TF-IDF:")
            st.write(st.session_state.labeling_text.head(15))

            if st.button("TF-IDF"):
                data = pd.read_csv('data/mentri.csv')
                # Proses TF-IDF pada teks yang telah diproses
                tfidf = TfidfVectorizer(max_features=8000)
                tfidf.fit(data['clean_teks'])
                X_tfidf = tfidf.transform(data['clean_teks'])

                # Menyimpan hasil TF-IDF dalam session state
                st.session_state.tfidf_text = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())

                # Menampilkan hasil TF-IDF
                st.write("Hasil TF-IDF:")
                st.write(st.session_state.tfidf_text.head(15))

                # Melakukan SMOTE untuk mengatasi ketidakseimbangan label
                y = data['sentiment']
                smote = SMOTE(random_state=42)
                X_smote, y_smote = smote.fit_resample(X_tfidf, y)

                # Tampilkan distribusi label setelah SMOTE
                st.write("Distribusi Label Setelah SMOTE:")
                st.write(pd.Series(y_smote).value_counts())
                # Membagi data menjadi set pelatihan dan pengujian
                X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

                # Get the feature names (terms)
                terms = tfidf.get_feature_names_out()

                # Function to get terms with non-zero TF-IDF scores and their weights
                def get_tfidf_terms(doc_index):
                    tfidf_row = X_tfidf[doc_index]
                    non_zero_indices = tfidf_row.nonzero()[1]  # Get indices of non-zero elements
                    terms_weights = [(terms[i], tfidf_row[0, i]) for i in non_zero_indices]
                    return terms_weights

                # Dictionary to hold the top 50 TF-IDF weights for each label
                top_50_tfidf_per_label = defaultdict(lambda: defaultdict(float))

                # Iterate over all documents
                for i, (text, label) in enumerate(zip(data['clean_teks'], data['sentiment'])):
                    terms_weights = get_tfidf_terms(i)
                    
                    # Aggregate the TF-IDF weights for each term per label
                    for term, weight in terms_weights:
                        if term.lower() != "mentri kabinet merah putih":  # Exclude the term "mentri"
                            top_50_tfidf_per_label[label][term] += weight

                # Process the top 20 terms for each label
                top_50_tfidf_final = {}
                for label, terms_weights_dict in top_50_tfidf_per_label.items():
                    # Sort by TF-IDF weight in descending order and keep the top 50
                    sorted_terms_weights = sorted(terms_weights_dict.items(), key=lambda x: x[1], reverse=True)[:50]
                    top_50_tfidf_final[label] = sorted_terms_weights

                # Display the top 50 TF-IDF weights for each label
                for label, top_terms in top_50_tfidf_final.items():
                    print(f"Label: {label}")
                    print("Top 50 Terms with Highest TF-IDF Weights:")
                    for term, weight in top_terms:
                        print(f"{term}: {weight:.4f}")
                    print("\n" + "-"*80 + "\n")

# Halaman: Prediksi
elif options == "🔍 Prediksi":
    st.header("Klasifikasi Teks")

    user_input = st.text_area("Masukkan teks untuk klasifikasi:")

    # Inisialisasi session state untuk menyimpan hasil sementara
    if 'casefolding_text' not in st.session_state:
        st.session_state.casefolding_text = None
    if 'normalisasi_text' not in st.session_state:
        st.session_state.normalisasi_text = None
    if 'remove_text' not in st.session_state:
        st.session_state.remove_text = None
    if 'tokenize_text' not in st.session_state:
        st.session_state.tokenize_text = None
    if 'stemming_text' not in st.session_state:
        st.session_state.stemming_text = None
    if 'tfidf_text' not in st.session_state:
        st.session_state.tfidf_text = None

    if user_input:
        if st.button("Case Folding"):
            st.session_state.casefolding_text = casefolding(user_input)
        st.write("Setelah Case Folding:")
        st.write(st.session_state.casefolding_text)
        
        if st.session_state.casefolding_text:
            if st.button("Normalisasi Teks"):
                st.session_state.normalisasi_text = text_normalize(st.session_state.casefolding_text)
            st.write("Setelah Normalisasi Teks:")
            st.write(st.session_state.normalisasi_text)

        if st.session_state.normalisasi_text:
            if st.button("Menghapus Stop Word"):
                st.session_state.remove_text = remove_stop_word(st.session_state.normalisasi_text)
            st.write("Setelah Menghapus Stop Word:")
            st.write(st.session_state.remove_text)

        if st.session_state.remove_text:
            if st.button("Tokenisasi"):
                st.session_state.tokenize_text = tokenizing(st.session_state.remove_text)
                # Gabungkan token kembali menjadi string untuk langkah berikutnya
                st.session_state.tokenize_text = ' '.join(st.session_state.tokenize_text)
            st.write("Setelah Tokenisasi:")
            st.write(st.session_state.tokenize_text)

        if st.session_state.tokenize_text:
            if st.button("Stemming"):
                st.session_state.stemming_text = stemming(st.session_state.tokenize_text)
            st.write("Setelah Stemming:")
            st.write(st.session_state.stemming_text)

    if st.button("Prediksi dengan SVM"):
        preprocessed_text, svm_result = classify_text(user_input.strip(), tfidf, svm_model)
        st.write("Prediksi SVM:")
        st.write(svm_result)

# Halaman: Kesimpulan
elif options == "📝 Kesimpulan":
    st.header("Kesimpulan")

    st.markdown("""
    <div class="justified-text">
    Dalam penelitian ini, evaluasi model dilakukan untuk menilai kinerja model Support Vector Machine dalam mengklasifikasikan 
    sentimen terhadap Kabinet Merah Putih. Proses evaluasi ini sangat penting untuk memahami seberapa baik model dapat 
    memprediksi sentimen positif dan negatif dari data teks yang diberikan. Salah satu alat evaluasi yang digunakan 
    adalah confusion matrix. Confusion matrix memberikan gambaran yang mendetail mengenai performa model, termasuk 
    jumlah prediksi yang benar (baik positif maupun negatif) serta jumlah prediksi yang salah (false positive dan 
    false negative). Dengan menggunakan confusion matrix, dapat dihitung berbagai metrik evaluasi seperti akurasi, 
    presisi, recall, dan skor F1, yang semuanya memberikan wawasan komprehensif tentang keandalan dan efektivitas 
    model dalam tugas klasifikasi sentimen.
    </div>
    """, unsafe_allow_html=True)
    
    # Menampilkan confusion matrix
    st.subheader("Confusion Matrix")
    st.image("asset/confusion.png",)

    # Evaluasi SVM
    st.subheader("Hasil Evaluasi SVM")
    st.markdown("""
    ### 📊 Model Test
    - SVM Test - Accuracy: 0.9873417721518988
    - SVM Test - Precision: 0.9875065195254614
    - SVM Test - Recall: 0.9873417721518988
    - SVM Test - F1 Score: 0.9873434282022476
    ### 📈 Model Train
    - SVM Train - Accuracy: 0.9995473064735174
    - SVM Train - Precision: 0.9995477135720124
    - SVM Train - Recall: 0.9995473064735174
    - SVM Train - F1 Score: 0.9995473051746695
    """)

    svm_report = """
    |            | precision | recall | f1-score | support |
    |------------|-----------|--------|----------|---------|
    |negative    | 0.98      | 1.00   | 0.99     | 270     |
    |positive    | 1.00      | 0.93   | 0.99     | 283     |
    |accuracy    |           |        | 0.99     | 553     |
    |macro avg   | 0.99      | 0.99   | 0.99     | 553     |
    |weighted avg| 0.99      | 0.99   | 0.99     | 553     |
    """
    st.markdown(svm_report)