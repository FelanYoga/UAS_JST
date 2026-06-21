# ============================================================================
# SISTEM PEMILIHAN SSD BERDASARKAN KEBUTUHAN LAPTOP DAN BUDGET
# MENGGUNAKAN JARINGAN SYARAF TIRUAN (JST)
#
# Proyek UAS Mata Kuliah Jaringan Syaraf Tiruan
# Model 1 : Perceptron (Klasifikasi Kompatibilitas SSD - Laptop)
# Model 2 : Learning Vector Quantization / LVQ (Rekomendasi Kategori SSD)
#
# Catatan: Kedua model JST diimplementasikan murni dari nol (from scratch)
# menggunakan operasi matriks NumPy, tanpa pustaka machine learning
# eksternal seperti TensorFlow, Keras, PyTorch, atau Scikit-Learn.
# ============================================================================

import io
import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------------------------------
# KONFIGURASI HALAMAN STREAMLIT
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sistem Pemilihan SSD - JST",
    page_icon="💽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# BAGIAN 1 : TEMA DAN STYLING GLASSMORPHISM
# ============================================================================
# Daftar konfigurasi warna untuk masing masing tema glassmorphism.
# Setiap tema memiliki kombinasi warna background gradient, warna kartu kaca,
# warna border transparan, dan warna accent neon yang berbeda.
THEME_CONFIG = {
    "Aurora Glass": {
        "bg_gradient": "linear-gradient(135deg, #1e1b4b 0%, #312e81 35%, #4c1d95 70%, #1e1b4b 100%)",
        "card_bg": "rgba(255, 255, 255, 0.08)",
        "card_border": "rgba(0, 229, 255, 0.35)",
        "accent": "#22d3ee",
        "accent_soft": "rgba(34, 211, 238, 0.18)",
        "text_main": "#f1f5f9",
        "text_sub": "#cbd5e1",
        "sidebar_bg": "rgba(30, 27, 75, 0.55)",
        "shadow": "0 8px 32px rgba(0, 0, 0, 0.45)",
    },
    "Emerald Glass": {
        "bg_gradient": "linear-gradient(135deg, #022c22 0%, #064e3b 35%, #0d9488 70%, #022c22 100%)",
        "card_bg": "rgba(255, 255, 255, 0.10)",
        "card_border": "rgba(16, 185, 129, 0.40)",
        "accent": "#34d399",
        "accent_soft": "rgba(52, 211, 153, 0.18)",
        "text_main": "#f0fdf4",
        "text_sub": "#d1fae5",
        "sidebar_bg": "rgba(2, 44, 34, 0.55)",
        "shadow": "0 8px 32px rgba(0, 0, 0, 0.40)",
    },
    "Midnight Glass": {
        "bg_gradient": "linear-gradient(135deg, #020617 0%, #0f172a 35%, #1e293b 70%, #020617 100%)",
        "card_bg": "rgba(255, 255, 255, 0.06)",
        "card_border": "rgba(59, 130, 246, 0.35)",
        "accent": "#3b82f6",
        "accent_soft": "rgba(59, 130, 246, 0.16)",
        "text_main": "#e2e8f0",
        "text_sub": "#94a3b8",
        "sidebar_bg": "rgba(2, 6, 23, 0.65)",
        "shadow": "0 8px 32px rgba(0, 0, 0, 0.55)",
    },
}


def apply_theme(theme_name: str) -> None:
    """
    Menyuntikkan CSS custom melalui st.markdown untuk menerapkan gaya
    glassmorphism sesuai tema yang dipilih pengguna. Seluruh komponen utama
    (kartu, sidebar, tombol, input, tabel) diberi efek blur, border
    transparan, rounded corner, dan animasi hover sesuai ketentuan desain.
    """
    t = THEME_CONFIG[theme_name]
    css = f"""
    <style>
    .stApp {{
        background: {t['bg_gradient']};
        background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}

    [data-testid="stSidebar"] {{
        background: {t['sidebar_bg']};
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border-right: 1px solid {t['card_border']};
    }}

    [data-testid="stSidebar"] * {{
        color: {t['text_main']} !important;
    }}

    h1, h2, h3, h4, h5, h6, p, span, label, div {{
        color: {t['text_main']};
    }}

    /* ---------- Kartu Kaca Utama ---------- */
    .glass-card {{
        background: {t['card_bg']};
        border: 1px solid {t['card_border']};
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: {t['shadow']};
        transition: transform 0.25s ease, box-shadow 0.25s ease, border 0.25s ease;
        margin-bottom: 1rem;
    }}
    .glass-card:hover {{
        transform: translateY(-4px);
        border: 1px solid {t['accent']};
        box-shadow: 0 12px 40px rgba(0,0,0,0.5), 0 0 18px {t['accent_soft']};
    }}

    /* ---------- Kartu Metrik Dashboard ---------- */
    .metric-card {{
        background: {t['card_bg']};
        border: 1px solid {t['card_border']};
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        text-align: center;
        transition: transform 0.25s ease, border 0.25s ease;
        box-shadow: {t['shadow']};
    }}
    .metric-card:hover {{
        transform: translateY(-3px) scale(1.01);
        border: 1px solid {t['accent']};
    }}
    .metric-value {{
        font-size: 2.0rem;
        font-weight: 800;
        color: {t['accent']};
        margin: 0;
    }}
    .metric-label {{
        font-size: 0.85rem;
        color: {t['text_sub']};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 0;
    }}

    /* ---------- Badge Kategori ---------- */
    .badge {{
        display: inline-block;
        padding: 0.25rem 0.85rem;
        border-radius: 999px;
        background: {t['accent_soft']};
        border: 1px solid {t['accent']};
        color: {t['accent']};
        font-weight: 600;
        font-size: 0.85rem;
    }}

    /* ---------- Tombol ---------- */
    .stButton > button {{
        background: {t['accent_soft']};
        border: 1px solid {t['accent']};
        color: {t['text_main']};
        border-radius: 12px;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
        backdrop-filter: blur(10px);
        transition: all 0.25s ease;
    }}
    .stButton > button:hover {{
        background: {t['accent']};
        color: #0b1020;
        transform: translateY(-2px);
        box-shadow: 0 6px 18px {t['accent_soft']};
        border: 1px solid {t['accent']};
    }}

    /* ---------- Input, Select, Number ---------- */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div,
    .stSlider, textarea {{
        background: {t['card_bg']} !important;
        border-radius: 10px !important;
        border: 1px solid {t['card_border']} !important;
        color: {t['text_main']} !important;
    }}

    /* ---------- Tabel / Dataframe ---------- */
    [data-testid="stDataFrame"] {{
        background: {t['card_bg']};
        border-radius: 14px;
        border: 1px solid {t['card_border']};
        padding: 0.4rem;
    }}

    /* ---------- Tab ---------- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: {t['card_bg']};
        border-radius: 12px 12px 0 0;
        border: 1px solid {t['card_border']};
        color: {t['text_main']};
    }}
    .stTabs [aria-selected="true"] {{
        background: {t['accent_soft']};
        border-bottom: 2px solid {t['accent']};
    }}

    /* ---------- Judul Halaman ---------- */
    .page-title {{
        font-size: 2.1rem;
        font-weight: 800;
        color: {t['text_main']};
        margin-bottom: 0.1rem;
    }}
    .page-subtitle {{
        color: {t['text_sub']};
        margin-bottom: 1.4rem;
    }}

    /* ---------- Divider tipis ---------- */
    hr {{
        border-color: {t['card_border']};
    }}

    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-thumb {{ background: {t['accent']}; border-radius: 8px; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def glass_card_open(extra_class: str = "") -> None:
    st.markdown(f'<div class="glass-card {extra_class}">', unsafe_allow_html=True)


def glass_card_close() -> None:
    st.markdown('</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str) -> str:
    return f"""
    <div class="metric-card">
        <p class="metric-value">{value}</p>
        <p class="metric-label">{label}</p>
    </div>
    """

@st.cache_data
def load_ssd_dataset() -> pd.DataFrame:
    """
    Membuat dataset SSD secara manual menggunakan DataFrame pandas.
    Harga bersifat estimasi untuk keperluan simulasi pelatihan model JST
    dan dapat berbeda dengan harga pasar aktual saat aplikasi dijalankan.
    Kolom Kategori SSD pada dataset ini digunakan sebagai label rujukan,
    sedangkan keputusan akhir rekomendasi tetap dihasilkan oleh model LVQ.
    """
    data = [
        # Nama SSD, Jenis SSD, Kapasitas (GB), Harga (Rp), Read (MB/s), Write (MB/s), Kategori
        ("Kingston A400 240GB",        "SATA", 240,  399000, 500,  450,  "Entry Level"),
        ("Crucial BX500 240GB",        "SATA", 240,  419000, 540,  500,  "Entry Level"),
        ("SanDisk SSD Plus 240GB",     "SATA", 240,  429000, 530,  440,  "Entry Level"),
        ("WD Green SN350 240GB",       "NVMe", 240,  449000, 2400, 900,  "Entry Level"),
        ("Crucial BX500 480GB",        "SATA", 480,  599000, 540,  500,  "Entry Level"),
        ("Crucial MX500 500GB",        "SATA", 500,  699000, 560,  510,  "Mid Range"),
        ("Samsung 870 EVO 500GB",      "SATA", 500,  1015000,560,  530,  "Mid Range"),
        ("Kingston NV2 500GB",         "NVMe", 500,  699000, 3500, 2100, "Mid Range"),
        ("WD Blue SN570 500GB",        "NVMe", 500,  749000, 3500, 2300, "Mid Range"),
        ("Samsung 980 500GB",          "NVMe", 500,  899000, 3100, 2600, "Mid Range"),
        ("Crucial P3 1TB",             "NVMe", 1000, 999000, 3500, 3000, "Mid Range"),
        ("WD Blue SN580 1TB",          "NVMe", 1000, 1199000,4150, 4150, "High Performance"),
        ("Samsung 970 EVO Plus 1TB",   "NVMe", 1000, 1599000,3500, 3300, "High Performance"),
        ("WD Black SN770 1TB",         "NVMe", 1000, 1549000,5150, 4900, "High Performance"),
        ("Kingston KC3000 1TB",        "NVMe", 1000, 1699000,7000, 6000, "High Performance"),
        ("Samsung 980 PRO 1TB",        "NVMe", 1000, 1799000,7000, 5000, "High Performance"),
        ("Seagate FireCuda 530 1TB",   "NVMe", 1000, 1899000,7300, 6000, "High Performance"),
        ("Kingston KC3000 2TB",        "NVMe", 2000, 3099000,7000, 7000, "Professional"),
        ("Samsung 990 PRO 2TB",        "NVMe", 2000, 3199000,7450, 6900, "Professional"),
        ("WD Black SN850X 2TB",        "NVMe", 2000, 3399000,7300, 6600, "Professional"),
        ("Crucial T700 2TB",           "NVMe", 2000, 4299000,12400,11800,"Professional"),
        ("Samsung 990 PRO 4TB",        "NVMe", 4000, 6499000,7450, 6900, "Professional"),
    ]
    df = pd.DataFrame(
        data,
        columns=[
            "Nama SSD", "Jenis SSD", "Kapasitas (GB)", "Harga (Rp)",
            "Kecepatan Read (MB/s)", "Kecepatan Write (MB/s)", "Kategori SSD",
        ],
    )
    return df


# Batas normalisasi global, dipakai konsisten oleh dataset dan input pengguna
BUDGET_MIN, BUDGET_MAX = 300_000, 7_000_000
KAPASITAS_MIN, KAPASITAS_MAX = 120, 4000

KEBUTUHAN_MAP = {"Perkantoran / Office": 1, "Multimedia": 2, "Gaming": 3, "Content Creation / Profesional": 4}
INTENSITAS_MAP = {"Ringan": 1, "Sedang": 2, "Berat": 3}
KATEGORI_LIST = ["Entry Level", "Mid Range", "High Performance", "Professional"]
KATEGORI_TO_IDX = {k: i for i, k in enumerate(KATEGORI_LIST)}
IDX_TO_KATEGORI = {i: k for i, k in enumerate(KATEGORI_LIST)}


def normalize(value: float, vmin: float, vmax: float) -> float:
    """Normalisasi min-max sederhana ke rentang 0 sampai 1."""
    if vmax == vmin:
        return 0.0
    return float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))


@st.cache_data
def generate_lvq_training_data(seed: int = 42, n_per_class: int = 25) -> tuple:
    """
    Membentuk data latih sintetis untuk model LVQ berdasarkan profil
    karakteristik tiap kategori SSD (budget, kebutuhan, kapasitas,
    intensitas). Profil ini diturunkan dari rentang harga dan spesifikasi
    yang terdapat pada dataset SSD manual di atas, kemudian diberi variasi
    acak terbatas (noise) agar model belajar dari distribusi yang realistis
    dan tidak hanya menghafal satu titik data per kelas.
    """
    rng = np.random.default_rng(seed)
    profiles = {
        "Entry Level":      dict(budget=(350_000, 650_000),   kapasitas=(120, 480),  kebutuhan=[1, 1, 1, 2], intensitas=[1, 1, 2]),
        "Mid Range":        dict(budget=(650_000, 1_300_000), kapasitas=(480, 1000), kebutuhan=[1, 2, 2, 3], intensitas=[2, 2, 1]),
        "High Performance": dict(budget=(1_300_000, 2_200_000), kapasitas=(960, 1000), kebutuhan=[3, 3, 4, 2], intensitas=[2, 3, 3]),
        "Professional":     dict(budget=(2_200_000, 6_800_000), kapasitas=(1900, 4000), kebutuhan=[4, 4, 3, 4], intensitas=[3, 3, 2]),
    }
    X, y = [], []
    for kategori, prof in profiles.items():
        for _ in range(n_per_class):
            budget = rng.uniform(*prof["budget"])
            kapasitas = rng.uniform(*prof["kapasitas"])
            kebutuhan = rng.choice(prof["kebutuhan"])
            intensitas = rng.choice(prof["intensitas"])
            X.append([
                normalize(budget, BUDGET_MIN, BUDGET_MAX),
                float(kebutuhan),
                normalize(kapasitas, KAPASITAS_MIN, KAPASITAS_MAX),
                float(intensitas),
            ])
            y.append(KATEGORI_TO_IDX[kategori])
    return np.array(X, dtype=float), np.array(y, dtype=int)


@st.cache_data
def generate_perceptron_training_data() -> tuple:
    """
    Membentuk data latih untuk model Perceptron berdasarkan tabel kebenaran
    logika kompatibilitas SSD terhadap laptop.

    x1 = Jenis Slot Laptop (1 = M.2 NVMe tersedia, 0 = hanya SATA/M.2 SATA)
    x2 = Generasi Laptop / dukungan protokol NVMe pada motherboard (1 = mendukung, 0 = tidak)
    x3 = Dukungan NVMe pada SSD (1 = SSD berjenis NVMe, 0 = SSD berjenis SATA)

    Aturan logika yang dipelajari:
    - SSD SATA (x3 = 0) selalu kompatibel pada laptop apapun.
    - SSD NVMe (x3 = 1) hanya kompatibel apabila laptop memiliki slot M.2 NVMe
      (x1 = 1) DAN motherboard mendukung protokol NVMe (x2 = 1).
    Aturan ini terbukti bersifat linearly separable (lihat Bab IV Laporan
    Akademis) sehingga dapat dipelajari dengan baik oleh Perceptron satu lapis.
    """
    combos = [
        (0, 0, 0, 1), (0, 0, 1, 0),
        (0, 1, 0, 1), (0, 1, 1, 0),
        (1, 0, 0, 1), (1, 0, 1, 0),
        (1, 1, 0, 1), (1, 1, 1, 1),
    ]
    # Setiap kombinasi diduplikasi beberapa kali agar proses belajar lebih stabil
    combos_repeated = combos * 6
    X = np.array([[c[0], c[1], c[2]] for c in combos_repeated], dtype=float)
    y = np.array([c[3] for c in combos_repeated], dtype=float)
    return X, y

class PerceptronFromScratch:
    """
    Implementasi Perceptron satu lapis dari prinsip dasar menggunakan
    operasi matriks NumPy murni, mengikuti model yang diperkenalkan oleh
    Rosenblatt (1958).

    Rumus utama:
        net    = sum(wi * xi) + b
        output = 1 jika net >= 0, selain itu 0  (fungsi aktivasi step)
        dw     = alpha * (target - output) * xi
        db     = alpha * (target - output)
    """

    def __init__(self, n_inputs: int, learning_rate: float = 0.1, epochs: int = 100, seed: int = 1):
        rng = np.random.default_rng(seed)
        # Inisialisasi bobot kecil acak agar pembelajaran tidak simetris dari awal
        self.weights = rng.uniform(-0.5, 0.5, size=n_inputs)
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.history = {"epoch": [], "error": [], "weights": [], "bias": []}

    @staticmethod
    def _step(net: np.ndarray) -> np.ndarray:
        """Fungsi aktivasi step (Heaviside)."""
        return np.where(net >= 0, 1.0, 0.0)

    def net_input(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return self._step(self.net_input(X))

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Melatih Perceptron menggunakan aturan pembelajaran Rosenblatt.
        Mengembalikan riwayat pelatihan (jumlah error per epoch) untuk
        keperluan visualisasi grafik training pada aplikasi.
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples = X.shape[0]

        for epoch in range(1, self.epochs + 1):
            total_error = 0
            for i in range(n_samples):
                xi = X[i]
                target = y[i]
                net = np.dot(xi, self.weights) + self.bias
                output = 1.0 if net >= 0 else 0.0
                error = target - output
                if error != 0:
                    total_error += 1
                    self.weights += self.learning_rate * error * xi
                    self.bias += self.learning_rate * error

            self.history["epoch"].append(epoch)
            self.history["error"].append(total_error)
            self.history["weights"].append(self.weights.copy())
            self.history["bias"].append(self.bias)

            # Early stopping apabila seluruh data latih sudah terklasifikasi benar
            if total_error == 0:
                break

        return self.history

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == np.array(y, dtype=float)))

class LVQFromScratch:
    """
    Implementasi Learning Vector Quantization (LVQ1) dari prinsip dasar
    menggunakan operasi matriks NumPy murni, mengikuti konsep pembelajaran
    kompetitif yang dikembangkan oleh Kohonen (1986, 1990).

    Setiap kelas direpresentasikan oleh satu vektor prototipe (codebook
    vector). Prototipe pemenang (winner) adalah prototipe dengan jarak
    Euclidean terkecil terhadap data input.

    Rumus utama:
        d(x, w)  = sqrt(sum((xi - wi)^2))
        w_baru   = w_lama + alpha * (x - w_lama)   jika prediksi benar
        w_baru   = w_lama - alpha * (x - w_lama)   jika prediksi salah
    """

    def __init__(self, n_classes: int, n_features: int, learning_rate: float = 0.2,
                 epochs: int = 50, decay: float = 0.5, decay_every: int = 10, seed: int = 7):
        self.n_classes = n_classes
        self.n_features = n_features
        self.initial_lr = learning_rate
        self.epochs = epochs
        self.decay = decay
        self.decay_every = decay_every
        self.seed = seed
        self.prototypes = None  # shape: (n_classes, n_features)
        self.history = {"epoch": [], "avg_distance": [], "accuracy": [], "learning_rate": []}

    def _init_prototypes(self, X: np.ndarray, y: np.ndarray) -> None:
        """Inisialisasi prototipe dari rata rata data latih pada setiap kelas."""
        self.prototypes = np.zeros((self.n_classes, self.n_features))
        for c in range(self.n_classes):
            mask = (y == c)
            if np.any(mask):
                self.prototypes[c] = X[mask].mean(axis=0)
            else:
                rng = np.random.default_rng(self.seed)
                self.prototypes[c] = rng.uniform(0, 1, size=self.n_features)

    @staticmethod
    def _euclidean(x: np.ndarray, w: np.ndarray) -> float:
        return float(np.sqrt(np.sum((x - w) ** 2)))

    def _find_winner(self, x: np.ndarray) -> int:
        distances = [self._euclidean(x, w) for w in self.prototypes]
        return int(np.argmin(distances))

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.array([self._find_winner(x) for x in X])

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Melatih prototipe LVQ menggunakan aturan kompetitif Kohonen.
        Mengembalikan riwayat pelatihan (rata rata jarak dan akurasi
        per epoch) untuk keperluan visualisasi grafik training.
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        self._init_prototypes(X, y)

        alpha = self.initial_lr
        n_samples = X.shape[0]

        for epoch in range(1, self.epochs + 1):
            distances_epoch = []
            correct = 0
            indices = np.random.default_rng(self.seed + epoch).permutation(n_samples)

            for i in indices:
                xi, target = X[i], y[i]
                winner = self._find_winner(xi)
                d = self._euclidean(xi, self.prototypes[winner])
                distances_epoch.append(d)

                if winner == target:
                    self.prototypes[winner] += alpha * (xi - self.prototypes[winner])
                    correct += 1
                else:
                    self.prototypes[winner] -= alpha * (xi - self.prototypes[winner])

            self.history["epoch"].append(epoch)
            self.history["avg_distance"].append(float(np.mean(distances_epoch)))
            self.history["accuracy"].append(correct / n_samples)
            self.history["learning_rate"].append(alpha)

            # Skema peluruhan learning rate setiap beberapa epoch
            if epoch % self.decay_every == 0:
                alpha *= self.decay

        return self.history

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == np.array(y, dtype=int)))

@st.cache_resource
def train_perceptron_model(learning_rate: float = 0.1, epochs: int = 100):
    """Melatih model Perceptron dan menyimpan hasilnya di cache resource."""
    X, y = generate_perceptron_training_data()
    model = PerceptronFromScratch(n_inputs=3, learning_rate=learning_rate, epochs=epochs, seed=1)
    history = model.fit(X, y)
    acc = model.accuracy(X, y)
    return model, history, acc, X, y


@st.cache_resource
def train_lvq_model(learning_rate: float = 0.2, epochs: int = 50):
    """Melatih model LVQ dan menyimpan hasilnya di cache resource."""
    X, y = generate_lvq_training_data()
    model = LVQFromScratch(
        n_classes=len(KATEGORI_LIST), n_features=4,
        learning_rate=learning_rate, epochs=epochs, decay=0.5, decay_every=10, seed=7,
    )
    history = model.fit(X, y)
    acc = model.accuracy(X, y)
    return model, history, acc, X, y

def format_rupiah(value: float) -> str:
    return f"Rp {value:,.0f}".replace(",", ".")


def init_session_state() -> None:
    """Inisialisasi seluruh variabel session_state yang dibutuhkan aplikasi."""
    if "theme" not in st.session_state:
        st.session_state.theme = "Aurora Glass"
    if "riwayat" not in st.session_state:
        st.session_state.riwayat = []  # list of dict, riwayat rekomendasi selama aplikasi berjalan
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"


SLOT_OPTIONS = {
    "M.2 NVMe (mendukung SSD SATA maupun NVMe)": 1,
    "M.2 SATA / 2.5 inci SATA saja (tidak ada slot NVMe)": 0,
}
MOBO_OPTIONS = {
    "Mendukung protokol NVMe (PCIe Gen3 ke atas)": 1,
    "Tidak mendukung protokol NVMe (hanya AHCI/SATA)": 0,
}


def check_compatibility(perceptron: PerceptronFromScratch, slot_x1: int, mobo_x2: int, ssd_jenis: str) -> int:
    """
    Mengecek kompatibilitas satu SSD kandidat terhadap kondisi laptop
    pengguna menggunakan model Perceptron yang telah dilatih.
    Mengembalikan 1 (Kompatibel) atau 0 (Tidak Kompatibel).
    """
    x3 = 1 if ssd_jenis == "NVMe" else 0
    x = np.array([[slot_x1, mobo_x2, x3]], dtype=float)
    pred = perceptron.predict(x)
    return int(pred[0])


def recommend_ssd(df: pd.DataFrame, kategori: str, slot_x1: int, mobo_x2: int,
                   perceptron: PerceptronFromScratch, kapasitas_diinginkan: float) -> pd.DataFrame:
    """
    Menyaring dataset SSD berdasarkan kategori hasil prediksi LVQ, lalu
    memvalidasi kompatibilitas setiap kandidat menggunakan model Perceptron.
    Hasil diurutkan berdasarkan kedekatan kapasitas dengan permintaan
    pengguna agar rekomendasi paling relevan ditampilkan lebih dahulu.
    """
    kandidat = df[df["Kategori SSD"] == kategori].copy()
    kandidat["Kompatibel"] = kandidat["Jenis SSD"].apply(
        lambda jenis: check_compatibility(perceptron, slot_x1, mobo_x2, jenis)
    )
    kandidat = kandidat[kandidat["Kompatibel"] == 1].copy()
    if kandidat.empty:
        return kandidat
    kandidat["Selisih Kapasitas"] = (kandidat["Kapasitas (GB)"] - kapasitas_diinginkan).abs()
    kandidat = kandidat.sort_values("Selisih Kapasitas").drop(columns=["Selisih Kapasitas"])
    return kandidat

def page_dashboard(df: pd.DataFrame) -> None:
    st.markdown('<p class="page-title">📊 Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Ringkasan data SSD dan aktivitas rekomendasi pada sistem.</p>', unsafe_allow_html=True)

    total_riwayat = len(st.session_state.riwayat)
    total_kompatibel = sum(1 for r in st.session_state.riwayat if r.get("Status Kompatibilitas") == "Kompatibel")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card("Total SSD Tersedia", f"{len(df)}"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("Total Riwayat Rekomendasi", f"{total_riwayat}"), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("Rata-rata Harga SSD", format_rupiah(df["Harga (Rp)"].mean())), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card("Jumlah Kategori SSD", f"{df['Kategori SSD'].nunique()}"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        glass_card_open()
        st.markdown("#### Distribusi Kategori SSD")
        dist = df["Kategori SSD"].value_counts().reindex(KATEGORI_LIST).fillna(0).reset_index()
        dist.columns = ["Kategori", "Jumlah"]
        fig = px.pie(
            dist, names="Kategori", values="Jumlah", hole=0.45,
            color="Kategori",
            color_discrete_sequence=px.colors.sequential.Tealgrn if st.session_state.theme == "Emerald Glass"
            else (px.colors.sequential.Blues if st.session_state.theme == "Midnight Glass" else px.colors.sequential.Plasma),
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color=THEME_CONFIG[st.session_state.theme]["text_main"],
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig, width='stretch')
        glass_card_close()

    with col_b:
        glass_card_open()
        st.markdown("#### Distribusi Jenis SSD (SATA vs NVMe)")
        dist_jenis = df["Jenis SSD"].value_counts().reset_index()
        dist_jenis.columns = ["Jenis", "Jumlah"]
        fig2 = px.bar(
            dist_jenis, x="Jenis", y="Jumlah", color="Jenis", text="Jumlah",
            color_discrete_sequence=[THEME_CONFIG[st.session_state.theme]["accent"], "#94a3b8"],
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color=THEME_CONFIG[st.session_state.theme]["text_main"],
            showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig2, width='stretch')
        glass_card_close()

    glass_card_open()
    st.markdown("#### Sebaran Harga dan Kapasitas SSD")
    fig3 = px.scatter(
        df, x="Kapasitas (GB)", y="Harga (Rp)", color="Kategori SSD", symbol="Jenis SSD",
        size="Kecepatan Read (MB/s)", hover_name="Nama SSD",
        category_orders={"Kategori SSD": KATEGORI_LIST},
    )
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color=THEME_CONFIG[st.session_state.theme]["text_main"],
        legend=dict(orientation="h", yanchor="bottom", y=-0.35),
        margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(fig3, width='stretch')
    glass_card_close()

    glass_card_open()
    st.markdown("#### Tabel Dataset SSD")
    st.dataframe(df, width='stretch', hide_index=True)
    glass_card_close()

def plot_perceptron_training(history: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["epoch"], y=history["error"], mode="lines+markers",
        name="Jumlah Kesalahan", line=dict(color=THEME_CONFIG[st.session_state.theme]["accent"], width=3),
    ))
    fig.update_layout(
        title="Grafik Training Perceptron (Jumlah Error per Epoch)",
        xaxis_title="Epoch", yaxis_title="Jumlah Kesalahan Klasifikasi",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color=THEME_CONFIG[st.session_state.theme]["text_main"],
        margin=dict(t=50, b=10, l=10, r=10),
    )
    return fig


def plot_lvq_training(history: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["epoch"], y=history["avg_distance"], mode="lines+markers",
        name="Rata-rata Jarak ke Prototipe", yaxis="y1",
        line=dict(color=THEME_CONFIG[st.session_state.theme]["accent"], width=3),
    ))
    fig.add_trace(go.Scatter(
        x=history["epoch"], y=history["accuracy"], mode="lines+markers",
        name="Akurasi Training", yaxis="y2",
        line=dict(color="#f59e0b", width=3, dash="dot"),
    ))
    fig.update_layout(
        title="Grafik Training LVQ (Jarak Prototipe dan Akurasi per Epoch)",
        xaxis_title="Epoch",
        yaxis=dict(title="Rata-rata Jarak Euclidean"),
        yaxis2=dict(title="Akurasi", overlaying="y", side="right", range=[0, 1.05]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color=THEME_CONFIG[st.session_state.theme]["text_main"],
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        margin=dict(t=50, b=10, l=10, r=10),
    )
    return fig


def page_recommendation(df: pd.DataFrame) -> None:
    st.markdown('<p class="page-title">🧠 Rekomendasi SSD</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Masukkan kebutuhan Anda, sistem akan memprediksi kategori SSD terbaik dan memvalidasi kompatibilitasnya dengan laptop Anda.</p>', unsafe_allow_html=True)

    tab_rekomendasi, tab_training = st.tabs(["🎯 Form Rekomendasi", "📈 Visualisasi Training Model"])

    # ------------------------------------------------------------------
    # TAB 1 : FORM REKOMENDASI
    # ------------------------------------------------------------------
    with tab_rekomendasi:
        glass_card_open()
        st.markdown("#### Masukkan Profil Kebutuhan Anda")
        with st.form("form_rekomendasi"):
            c1, c2 = st.columns(2)
            with c1:
                budget = st.number_input(
                    "Budget yang Dimiliki (Rp)", min_value=300_000, max_value=7_000_000,
                    value=1_000_000, step=50_000,
                )
                kebutuhan_label = st.selectbox("Kebutuhan Penggunaan Utama", list(KEBUTUHAN_MAP.keys()))
                kapasitas = st.selectbox(
                    "Kapasitas Penyimpanan yang Diinginkan (GB)",
                    [240, 480, 500, 1000, 2000, 4000], index=2,
                )
            with c2:
                intensitas_label = st.selectbox("Intensitas Penggunaan Harian", list(INTENSITAS_MAP.keys()))
                slot_label = st.selectbox("Jenis Slot pada Laptop", list(SLOT_OPTIONS.keys()))
                mobo_label = st.selectbox("Dukungan Motherboard Laptop", list(MOBO_OPTIONS.keys()))

            submitted = st.form_submit_button("🔍 Cari Rekomendasi SSD")
        glass_card_close()

        if submitted:
            perceptron_model, _, _, _, _ = train_perceptron_model()
            lvq_model, _, _, _, _ = train_lvq_model()

            slot_x1 = SLOT_OPTIONS[slot_label]
            mobo_x2 = MOBO_OPTIONS[mobo_label]
            kebutuhan_code = KEBUTUHAN_MAP[kebutuhan_label]
            intensitas_code = INTENSITAS_MAP[intensitas_label]

            x_lvq = np.array([[
                normalize(budget, BUDGET_MIN, BUDGET_MAX),
                float(kebutuhan_code),
                normalize(kapasitas, KAPASITAS_MIN, KAPASITAS_MAX),
                float(intensitas_code),
            ]])
            kategori_idx = lvq_model.predict(x_lvq)[0]
            kategori_hasil = IDX_TO_KATEGORI[kategori_idx]

            kompatibel_nvme = check_compatibility(perceptron_model, slot_x1, mobo_x2, "NVMe")
            kompatibel_sata = check_compatibility(perceptron_model, slot_x1, mobo_x2, "SATA")

            hasil_rekomendasi = recommend_ssd(df, kategori_hasil, slot_x1, mobo_x2, perceptron_model, kapasitas)

            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                glass_card_open()
                st.markdown("##### 🧩 Hasil Klasifikasi Kompatibilitas (Perceptron)")
                st.write(f"SSD berjenis **NVMe** pada konfigurasi laptop ini: "
                         f"<span class='badge'>{'Kompatibel' if kompatibel_nvme else 'Tidak Kompatibel'}</span>", unsafe_allow_html=True)
                st.write(f"SSD berjenis **SATA** pada konfigurasi laptop ini: "
                         f"<span class='badge'>{'Kompatibel' if kompatibel_sata else 'Tidak Kompatibel'}</span>", unsafe_allow_html=True)
                glass_card_close()
            with col2:
                glass_card_open()
                st.markdown("##### 🏷️ Hasil Rekomendasi Kategori (LVQ)")
                st.markdown(f"<span class='badge' style='font-size:1.1rem;'>{kategori_hasil}</span>", unsafe_allow_html=True)
                st.write(f"Berdasarkan budget {format_rupiah(budget)}, kebutuhan **{kebutuhan_label}**, "
                         f"kapasitas **{kapasitas} GB**, dan intensitas **{intensitas_label}**.")
                glass_card_close()

            glass_card_open()
            st.markdown("##### 💽 SSD yang Direkomendasikan")
            status_riwayat = "Kompatibel" if (kompatibel_nvme or kompatibel_sata) else "Tidak Kompatibel"

            if hasil_rekomendasi.empty:
                st.warning(
                    "Tidak ditemukan SSD pada kategori dan jenis slot yang sesuai. "
                    "Coba ubah jenis slot laptop atau pertimbangkan kategori SSD di sekitar kebutuhan Anda."
                )
                ssd_terpilih_nama = "Tidak ada SSD yang sesuai"
            else:
                st.dataframe(hasil_rekomendasi, width='stretch', hide_index=True)
                ssd_terpilih_nama = hasil_rekomendasi.iloc[0]["Nama SSD"]
                st.success(f"SSD yang paling direkomendasikan: **{ssd_terpilih_nama}**")
            glass_card_close()

            # Simpan ke riwayat rekomendasi (session_state)
            st.session_state.riwayat.append({
                "Waktu": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Budget": budget,
                "Kebutuhan": kebutuhan_label,
                "Kapasitas Diinginkan (GB)": kapasitas,
                "Intensitas": intensitas_label,
                "Jenis Slot Laptop": slot_label,
                "Kategori Rekomendasi (LVQ)": kategori_hasil,
                "Status Kompatibilitas": status_riwayat,
                "SSD Direkomendasikan": ssd_terpilih_nama,
            })

    # ------------------------------------------------------------------
    # TAB 2 : VISUALISASI TRAINING MODEL
    # ------------------------------------------------------------------
    with tab_training:
        st.markdown("Atur parameter pelatihan untuk melihat pengaruh **learning rate** dan **epoch** terhadap proses belajar masing masing model.")

        col_p, col_l = st.columns(2)
        with col_p:
            glass_card_open()
            st.markdown("##### ⚙️ Parameter Perceptron")
            lr_p = st.slider("Learning Rate Perceptron", 0.01, 1.0, 0.1, 0.01, key="lr_p")
            ep_p = st.slider("Jumlah Epoch Perceptron", 5, 200, 100, 5, key="ep_p")
            model_p, hist_p, acc_p, X_p, y_p = train_perceptron_model(lr_p, ep_p)
            st.plotly_chart(plot_perceptron_training(hist_p), width='stretch')
            st.markdown(f"**Akurasi pada data latih:** {acc_p * 100:.2f}% &nbsp;|&nbsp; **Konvergen pada epoch:** {hist_p['epoch'][-1]}")
            glass_card_close()

        with col_l:
            glass_card_open()
            st.markdown("##### ⚙️ Parameter LVQ")
            lr_l = st.slider("Learning Rate Awal LVQ", 0.01, 1.0, 0.2, 0.01, key="lr_l")
            ep_l = st.slider("Jumlah Epoch LVQ", 5, 200, 50, 5, key="ep_l")
            model_l, hist_l, acc_l, X_l, y_l = train_lvq_model(lr_l, ep_l)
            st.plotly_chart(plot_lvq_training(hist_l), width='stretch')
            st.markdown(f"**Akurasi pada data latih:** {acc_l * 100:.2f}% &nbsp;|&nbsp; **Epoch total:** {hist_l['epoch'][-1]}")
            glass_card_close()

        glass_card_open()
        st.markdown("##### 📌 Catatan Analisis")
        st.write(
            "Learning rate yang terlalu besar cenderung membuat proses belajar berosilasi, "
            "sedangkan learning rate yang terlalu kecil membuat konvergensi berjalan lambat. "
            "Geser slider di atas untuk mengamati langsung pengaruhnya terhadap grafik error "
            "Perceptron maupun grafik jarak prototipe dan akurasi pada LVQ."
        )
        glass_card_close()

def page_history() -> None:
    st.markdown('<p class="page-title">🕘 Riwayat Rekomendasi</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Seluruh hasil rekomendasi yang telah dihasilkan selama aplikasi berjalan pada sesi ini.</p>', unsafe_allow_html=True)

    riwayat = st.session_state.riwayat

    if not riwayat:
        glass_card_open()
        st.info("Belum ada riwayat rekomendasi. Silakan buka halaman **Rekomendasi SSD** dan lakukan pencarian terlebih dahulu.")
        glass_card_close()
        return

    df_riwayat = pd.DataFrame(riwayat)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(metric_card("Total Riwayat", f"{len(df_riwayat)}"), unsafe_allow_html=True)
    with col2:
        jumlah_kompatibel = int((df_riwayat["Status Kompatibilitas"] == "Kompatibel").sum())
        st.markdown(metric_card("Rekomendasi Kompatibel", f"{jumlah_kompatibel}"), unsafe_allow_html=True)
    with col3:
        kategori_terbanyak = df_riwayat["Kategori Rekomendasi (LVQ)"].mode()
        kategori_terbanyak = kategori_terbanyak.iloc[0] if not kategori_terbanyak.empty else "-"
        st.markdown(metric_card("Kategori Paling Sering", kategori_terbanyak), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    glass_card_open()
    st.markdown("#### Tren Kategori Rekomendasi dari Waktu ke Waktu")
    tren = df_riwayat["Kategori Rekomendasi (LVQ)"].value_counts().reindex(KATEGORI_LIST).fillna(0).reset_index()
    tren.columns = ["Kategori", "Jumlah"]
    fig = px.bar(
        tren, x="Kategori", y="Jumlah", color="Kategori", text="Jumlah",
        category_orders={"Kategori": KATEGORI_LIST},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color=THEME_CONFIG[st.session_state.theme]["text_main"],
        showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(fig, width='stretch')
    glass_card_close()

    glass_card_open()
    st.markdown("#### Tabel Riwayat Rekomendasi")
    st.dataframe(
        df_riwayat.sort_values("Waktu", ascending=False),
        width='stretch', hide_index=True,
    )

    if st.button("🗑️ Hapus Seluruh Riwayat"):
        st.session_state.riwayat = []
        st.rerun()
    glass_card_close()

def to_excel_bytes(df_dataset: pd.DataFrame, df_riwayat: pd.DataFrame) -> bytes:
    """
    Mengekspor dataset SSD dan riwayat rekomendasi ke dalam satu file Excel
    dengan dua sheet terpisah, menggunakan engine openpyxl.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_dataset.to_excel(writer, sheet_name="Dataset SSD", index=False)
        if df_riwayat.empty:
            pd.DataFrame({"Info": ["Belum ada riwayat rekomendasi pada sesi ini."]}).to_excel(
                writer, sheet_name="Riwayat Rekomendasi", index=False
            )
        else:
            df_riwayat.to_excel(writer, sheet_name="Riwayat Rekomendasi", index=False)
    buffer.seek(0)
    return buffer.getvalue()


def page_export(df: pd.DataFrame) -> None:
    st.markdown('<p class="page-title">📤 Export Data</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Unduh dataset SSD dan riwayat rekomendasi dalam format Excel (.xlsx).</p>', unsafe_allow_html=True)

    df_riwayat = pd.DataFrame(st.session_state.riwayat)

    glass_card_open()
    st.markdown("#### Pratinjau Dataset SSD")
    st.dataframe(df.head(5), width='stretch', hide_index=True)
    st.caption(f"Total {len(df)} baris data SSD akan disertakan pada file Excel.")
    glass_card_close()

    glass_card_open()
    st.markdown("#### Pratinjau Riwayat Rekomendasi")
    if df_riwayat.empty:
        st.info("Belum ada riwayat rekomendasi untuk diekspor. Sheet riwayat akan tetap dibuat namun kosong.")
    else:
        st.dataframe(df_riwayat.tail(5), width='stretch', hide_index=True)
        st.caption(f"Total {len(df_riwayat)} baris riwayat akan disertakan pada file Excel.")
    glass_card_close()

    glass_card_open()
    excel_bytes = to_excel_bytes(df, df_riwayat)
    st.download_button(
        label="⬇️ Export ke Excel (.xlsx)",
        data=excel_bytes,
        file_name=f"laporan_ssd_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    glass_card_close()

def main() -> None:
    init_session_state()
    apply_theme(st.session_state.theme)

    df = load_ssd_dataset()

    with st.sidebar:
        st.markdown("## 💽 SSD Advisor JST")
        st.caption("Sistem Pemilihan SSD Berbasis Jaringan Syaraf Tiruan")
        st.markdown("---")

        st.markdown("#### 🎨 Pilih Tema Tampilan")
        theme_choice = st.selectbox(
            "Tema Glassmorphism", list(THEME_CONFIG.keys()),
            index=list(THEME_CONFIG.keys()).index(st.session_state.theme),
            label_visibility="collapsed",
        )
        if theme_choice != st.session_state.theme:
            st.session_state.theme = theme_choice
            st.rerun()

        st.markdown("---")
        st.markdown("#### 🧭 Navigasi")
        page = st.radio(
            "Pilih Halaman",
            ["Dashboard", "Rekomendasi SSD", "Riwayat", "Export"],
            index=["Dashboard", "Rekomendasi SSD", "Riwayat", "Export"].index(st.session_state.page),
            label_visibility="collapsed",
        )
        st.session_state.page = page

        st.markdown("---")
        st.caption("Proyek UAS Jaringan Syaraf Tiruan")
        st.caption("Model: Perceptron (Kompatibilitas) & LVQ (Rekomendasi Kategori)")

    if st.session_state.page == "Dashboard":
        page_dashboard(df)
    elif st.session_state.page == "Rekomendasi SSD":
        page_recommendation(df)
    elif st.session_state.page == "Riwayat":
        page_history()
    elif st.session_state.page == "Export":
        page_export(df)


if __name__ == "__main__":
    main()