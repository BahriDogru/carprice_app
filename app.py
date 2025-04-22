import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn.preprocessing import FunctionTransformer
import Preprocessing
from datetime import datetime
from collections import defaultdict
import ast
import warnings
import plotly.express as px # type: ignore
import json
import unidecode # type: ignore
import os
import requests


# Sayfa ayarları
st.set_page_config(page_title="Araç Fiyat Tahmincisi", page_icon="🚗", layout="wide")
warnings.filterwarnings("ignore")
# Sayfa stil ayarları
st.markdown("""
    <style>
        /* Ana arkaplan */
        .main {
            background-color: rgba(0, 0, 0, 0.8);
            color: white;
        }

        /* Arka plan resmi */
        .stApp {
            background-image: url('https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8Y2FyJTIwYmxhY2slMjBhbmQlMjB3aGl0ZXxlbnwwfHwwfHx8MA%3D%3D&auto=format&fit=crop&w=1500&q=80');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-blend-mode: overlay;
            background-color: rgba(0, 0, 0, 0.85);
        }

        /* Başlık stilleri */
        h1 {
            color: #FFD700;
            text-align: left;
            font-weight: bold;
            padding: 20px 0;
            text-shadow: 2px 2px 4px #000000;
            margin-left: 2rem;
        }

        h2, h3 {
            color: #FFD700;
            margin-top: 20px;
            text-align: left;
            margin-left: 2rem;
        }

        

        /* Buton stili */
        .stButton > button {
            background-color: #FFD700;
            color: black;
            font-weight: bold;
            border: none;
            border-radius: 5px;
            padding: 0.6rem 1rem;
        }

        .stButton > button:hover {
            background-color: #E6C200;
        }

        /* Bilgi kutusu stili */
        .stInfo {
            background-color: rgba(70, 70, 70, 0.7);
            color: white;
            border: 1px solid #FFD700;
        }

        /* Başarı mesajı stili */
        .stSuccess {
            background-color: rgba(0, 0, 0, 0.7);
            color: #FFD700;
            font-weight: bold;
            border: 2px solid #FFD700;
            padding: 20px;
            text-align: left;
        }

        /* Expander stili */
        .streamlit-expanderHeader {
            color: #FFD700;
            font-weight: bold;
            background-color: rgba(50, 50, 50, 0.7);
        }

        /* Radio butonları */
        .row-widget.stRadio > div {
            flex-direction: row;
        }

        .stRadio > label {
            color: white;
        }

        /* Seçenekler için stil */
        div[data-baseweb="select"] > div {
            background-color: #333;
            color: white;
            border: 1px solid #555;
        }

        /* Etiketler için stil */
        label {
            color: white !important;
            font-weight: 500;
        }

        /* İnput kutuları için stil */
        input, textarea {
            background-color: #333 !important;
            color: white !important;
            border: 1px solid #555 !important;
        }



        /* Bölücü çizgi stili */
        hr {
            border-color: #FFD700;
            margin: 20px 0;
            margin-left: 2rem;
        }

        /* Container genişliği */
        .css-ocqkz7, .css-1kyxreq, .css-12w0qpk {
            max-width: 100%;
        }


    </style>
""", unsafe_allow_html=True)


# Veriyi yükle
df = pd.read_csv("files\Car_price_clean_data.csv")  # CSV dosyasını yükle

# Model ve train columns'u yükle
@st.cache_resource(show_spinner="🔄 Model yükleniyor, lütfen bekleyin...")
def load_model_from_drive():
    url = "https://drive.google.com/uc?id=1MJS1RCRXZGKVwQ0xlDr4oXZdh6NZZNRx"
    model_path = "model_cache/voting_clf.pkl"

    os.makedirs("model_cache", exist_ok=True)

    if not os.path.exists(model_path):
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)

    return joblib.load(model_path)
model = load_model_from_drive()
train_columns = joblib.load("features_names.pkl")

st.title("Araç Fiyat Tahmini")
st.write("Bu uygulama, aracınızın bilgilerine göre tahmini fiyat sunar.")

# Marka, seri ve model verisi tutan sözlük yapsı
def model_map_load(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
        model_map = json.loads(data)
    return model_map
model_map = model_map_load('files\car_dict.txt')

# Şehirlerin isimlerini tutan liste
def city_district_load(file_path):
    with open(file_path,'r', encoding='utf-8') as file:
        data = file.read()
        data_city_district = ast.literal_eval(data)
    return data_city_district
city_district = city_district_load('files\province_district_map.txt')

# Arçların rengini tutan liste
colors = ["Beyaz","Gri","Siyah","Gümüş/Metalik Gri","Mavi","Kırmızı","Turuncu","Kahverengi","Yeşil","Sarı","Bej/Krem","Bordo","Antrasit/Koyu Gri", "Diğer"]
# Araçların body_type listesi
body_type_list = ["Sedan","Coupe","Cabrio","Hatchback/3","Hatchback/5","Station Wagon","SUV","Crossover","MPV/Minibüs","Roadster","Pick-up","Van / Panelvan","Liftback"]



col3, col4, col5= st.columns([3, 1, 4])  # sol:input, sağ:görselleştirme

with col3:
    # Kullanıcı inputları
    input_col1, input_col2 = st.columns(2)
    bugun = datetime.now()
    tarih_formatı = "%d %B %Y"
    formatted_tarih = bugun.strftime(tarih_formatı)

    with input_col1:
        brand = st.selectbox("Marka", list(model_map.keys()))
        series_options = list(model_map[brand].keys())
        series = st.selectbox("Seri", series_options)
        model_options = model_map[brand][series]
        model_name = st.selectbox("Model", model_options)
        year = st.number_input("Yıl", min_value=1900, max_value=2025, value=2018)
        engine_size = st.number_input("Motor Hacmi (cc)", min_value=800, max_value=6000, value=1600)
        engine_power = st.number_input("Motor Gücü (HP)", min_value=50, max_value=600, value=110)
        city = st.selectbox("Şehir", city_district.keys())
        
    with input_col2:
        mileage = st.number_input("Kilometre", min_value=0, max_value=500000,value=100000)
        gear_type = st.selectbox("Vites Tipi", ['Otomatik', 'Yarı Otomatik', 'Manuel'])
        fuel_type = st.selectbox("Yakıt Tipi", ['Benzin', 'Dizel', 'Hibrit', 'LPG & Benzin', 'Elektrik'])
        drive_type = st.selectbox("Çekiş Tipi", ['Önden Çekiş', 'Arkadan İtiş', '4WD (Sürekli)'])
        color = st.selectbox("Renk", colors)
        body_type = st.selectbox("Kasa Tipi",body_type_list)
        address = st.selectbox("İlçe/Semt", city_district.get(city, []))

        
    # Boya ve Parça Durumu Bilgisi alma
    st.markdown("### Boya & Parça Durumu")
    with st.expander("Parçaları Düzenle"):
        parts = [
            "Sağ Ön Çamurluk", "Sol Ön Çamurluk", "Sağ Arka Çamurluk", "Sol Arka Çamurluk", "Sağ Ön Kapı",
            "Sol Ön Kapı","Sağ Arka Kapı","Sol Arka Kapı","Arka Kaput",
            "Motor Kaputu", "Ön Tampon", "Arka Tampon", "Tavan"
        ]
        options = ["Belirtilmemiş", "Orijinal", "Lokal Boyalı", "Boyalı", "Değişmiş"]

        results = defaultdict(list)
        selections = {}

        # 3 sütunluk yapı
        cols = st.columns(3)

        for idx, part in enumerate(parts):
            with cols[idx % 3]:
                status = st.selectbox(f"{part}", options, key=part)
                selections[part] = status
                if status != "Belirtilmemiş":
                    results[status].append(part)

    # Özet çıktıyı tek satırda göstermerk
    summary = ""
    for status in ["Orijinal", "Lokal Boyalı", "Boyalı", "Değişmiş"]:
        if results[status]:
            summary += f"{status}: {', '.join(results[status])} | "

    if summary == "":
        summary = "Tüm parçalar belirtilmemiş."

    st.write("**Seçilen Durumlar:**")
    st.info(summary)

    # Tramer bilgisi
    tramer_condition = st.radio("Tramer Bilgisi", ["Yok", "Belirtilmemiş", "Var", "Ağır Hasar kayıtlı"])
    tramer_amount = 0
    if tramer_condition == "Var":
        tramer_amount = st.number_input("Tramer Tutarı (TL)", min_value=0, value=1000)
    
    
with col4:
    ""
    


with col5:
    # Grafiklerin gösterileceği alan
    st.markdown("### Araç Fiyat Dağılımı")
    # Kullanıcının seçtiği değerler
    selected_brand = brand
    selected_series = series
    selected_model = model_name    

    # Fiyat sütununu sayısala çevir
    df["Price(TL)"] = df["Price(TL)"].astype(str).str.replace(".", "").str.replace(" TL", "").astype(float)

    # Grafik çizim fonksiyonu
    def plot_price_distribution(data, title, avg_price, col):
        fig, ax = plt.subplots(figsize=(5, 2.5))
        sns.histplot(data["Price(TL)"], bins=30, kde=True, ax=ax)
        ax.axvline(avg_price, color='red', linestyle='--', label=f'Ortalama: {avg_price:,.0f} TL')
        ax.set_title(title)
        ax.set_xlabel("Fiyat (TL)")
        ax.legend()
        col.pyplot(fig)

    # 3 sütun oluştur
    col_a, col_b = st.columns(2)

    # 1. Marka bazlı ortalama fiyat
    if selected_brand:
        brand_prices = df[df["Brand"] == selected_brand]
        avg_brand = brand_prices["Price(TL)"].mean()
        plot_price_distribution(brand_prices, f"{selected_brand} Araçların Fiyat Dağılımı", avg_brand, col_a)

    # 2. Marka + Seri bazlı
    if selected_series:
        series_prices = df[(df["Brand"] == selected_brand) & (df["Series"] == selected_series)]
        avg_series = series_prices["Price(TL)"].mean()
        plot_price_distribution(series_prices, f"{selected_brand} {selected_series} Araçların Fiyat Dağılımı", avg_series, col_b)

    # 3. Marka + Seri + Model bazlı
    if selected_model:
        model_prices = df[(df["Brand"] == selected_brand) & (df["Series"] == selected_series) & (df["Model"] == selected_model)]
        if not model_prices.empty:
            avg_model = model_prices["Price(TL)"].mean()
            plot_price_distribution(model_prices, f"{selected_brand} {selected_series} - {selected_model} Fiyat Dağılımı", avg_model, col_a)
        else:
            col_a.info("Seçilen model için yeterli örnek veri bulunamadı.")
    
    with st.expander("### 🗺️ Türkiye'de Araç Sayıları (Seçilen Şehre Odaklı)", expanded=True):
        # 1. GeoJSON dosyasını yükle
        with open("files/tr.json", encoding="utf-8") as f:
            turkey_geojson = json.load(f)

        # 2. GeoJSON şehir adlarını upper yap
        for feature in turkey_geojson["features"]:
            feature["properties"]["name"] = feature["properties"]["name"].upper()

        # 3. Araç sayılarını hesapla

        df["City"] = df["City"].apply(lambda x: unidecode.unidecode(x).upper())

        city_counts = df["City"].str.upper().value_counts().reset_index()
        city_counts.columns = ["City", "AracSayisi"]

        # 4. Koordinatlar
        city_coords  = {
            "TÜM TÜRKİYE": {"lat": 39.0, "lon": 35.0, "zoom": 5},
            "ADANA": {"lat": 37.0000, "lon": 35.3213},
            "ADIYAMAN": {"lat": 37.7643, "lon": 38.2789},
            "AFYONKARAHISAR": {"lat": 38.7586, "lon": 30.5578},
            "AGRI": {"lat": 39.7192, "lon": 43.0500},
            "AKSARAY": {"lat": 38.3733, "lon": 34.0367},
            "AMASYA": {"lat": 40.6539, "lon": 35.8344},
            "ANKARA": {"lat": 39.9208, "lon": 32.8541},
            "ANTALYA": {"lat": 36.8969, "lon": 30.7133},
            "ARDAHAN": {"lat": 41.1150, "lon": 42.7000},
            "ARTVIN": {"lat": 41.1800, "lon": 41.8200},
            "AYDIN": {"lat": 37.8494, "lon": 27.8358},
            "BALIKESIR": {"lat": 39.6475, "lon": 27.8850},
            "BARTIN": {"lat": 41.6383, "lon": 32.3447},
            "BATMAN": {"lat": 37.8833, "lon": 41.1333},
            "BAYBURT": {"lat": 40.2500, "lon": 40.2167},
            "BILECIK": {"lat": 40.1444, "lon": 30.0033},
            "BINGOL": {"lat": 38.8850, "lon": 40.5000},
            "BITLIS": {"lat": 38.4000, "lon": 42.1167},
            "BOLU": {"lat": 40.7400, "lon": 31.6167},
            "BURDUR": {"lat": 37.7200, "lon": 30.2900},
            "BURSA": {"lat": 40.1828, "lon": 29.0663},
            "CANAKKALE": {"lat": 40.1533, "lon": 26.4100},
            "CANKIRI": {"lat": 40.6000, "lon": 33.6167},
            "CORUM": {"lat": 40.5500, "lon": 34.9500},
            "DENIZLI": {"lat": 37.7750, "lon": 29.0850},
            "DIYARBAKIR": {"lat": 37.9100, "lon": 40.2300},
            "DUZCE": {"lat": 40.8378, "lon": 31.1511},
            "EDIRNE": {"lat": 41.6781, "lon": 26.5600},
            "ELAZIG": {"lat": 38.6700, "lon": 39.2200},
            "ERZINCAN": {"lat": 39.7500, "lon": 39.4900},
            "ERZURUM": {"lat": 39.9000, "lon": 41.2700},
            "ESKISEHIR": {"lat": 39.7767, "lon": 30.5206},
            "GAZIANTEP": {"lat": 37.0662, "lon": 37.3833},
            "GIRESUN": {"lat": 40.9167, "lon": 38.3833},
            "GUMUSHANE": {"lat": 40.2500, "lon": 39.4833},
            "HAKKARI": {"lat": 37.5700, "lon": 43.7300},
            "HATAY": {"lat": 36.2000, "lon": 36.1667},
            "IGDIR": {"lat": 39.9167, "lon": 44.0333},
            "ISPARTA": {"lat": 37.7656, "lon": 30.5589},
            "ISTANBUL": {"lat": 41.0082, "lon": 28.9784},
            "IZMIR": {"lat": 38.4192, "lon": 27.1287},
            "KAHRAMANMARAS": {"lat": 37.5800, "lon": 36.9500},
            "KARABUK": {"lat": 41.2000, "lon": 32.6333},
            "KARAMAN": {"lat": 37.1767, "lon": 33.0200},
            "KARS": {"lat": 40.6000, "lon": 43.1000},
            "KASTAMONU": {"lat": 41.3700, "lon": 33.7800},
            "KAYSERI": {"lat": 38.7300, "lon": 35.4800},
            "KILIS": {"lat": 36.7167, "lon": 37.1333},
            "KIRIKKALE": {"lat": 39.8467, "lon": 33.5000},
            "KIRKLARELI": {"lat": 41.7333, "lon": 27.2167},
            "KIRSEHIR": {"lat": 39.1500, "lon": 34.1667},
            "KOCAELI": {"lat": 40.7667, "lon": 29.9167},
            "KONYA": {"lat": 37.8746, "lon": 32.4932},
            "KUTAHYA": {"lat": 39.4167, "lon": 29.9833},
            "MALATYA": {"lat": 38.3500, "lon": 38.3300},
            "MANISA": {"lat": 38.6200, "lon": 27.4200},
            "MARDIN": {"lat": 37.3300, "lon": 40.7300},
            "MERSIN": {"lat": 36.8000, "lon": 34.6333},
            "MUGLA": {"lat": 37.2167, "lon": 28.3667},
            "MUS": {"lat": 38.7333, "lon": 41.4833},
            "NEVSEHIR": {"lat": 38.6500, "lon": 34.7167},
            "NIGDE": {"lat": 37.9667, "lon": 34.6833},
            "ORDU": {"lat": 40.9833, "lon": 37.8833},
            "OSMANIYE": {"lat": 37.0667, "lon": 36.2500},
            "RIZE": {"lat": 41.0100, "lon": 40.5300},
            "SAKARYA": {"lat": 40.7833, "lon": 30.4000},
            "SAMSUN": {"lat": 41.2900, "lon": 36.3400},
            "SIIRT": {"lat": 37.9333, "lon": 41.9500},
            "SINOP": {"lat": 42.0200, "lon": 35.1500},
            "SIVAS": {"lat": 39.7500, "lon": 37.0200},
            "SANLIURFA": {"lat": 37.1500, "lon": 38.7900},
            "SIRNAK": {"lat": 37.3333, "lon": 43.0500},
            "TEKIRDAG": {"lat": 40.9833, "lon": 27.5167},
            "TOKAT": {"lat": 40.3200, "lon": 36.5500},
            "TRABZON": {"lat": 41.0015, "lon": 39.7178},
            "TUNCELI": {"lat": 39.1000, "lon": 39.5333},
            "USAK": {"lat": 38.6800, "lon": 29.4100},
            "VAN": {"lat": 38.4800, "lon": 42.7000},
            "YALOVA": {"lat": 40.6500, "lon": 29.2833},
            "YOZGAT": {"lat": 39.8200, "lon": 34.8100},
            "ZONGULDAK": {"lat": 41.4500, "lon": 31.7900}
        }

        # 5. Seçilen şehir
        selected_city = city.upper().replace("İ", "I").replace("Ş", "S").replace("Ç", "C").replace("Ğ", "G").replace("Ü", "U").replace("Ö", "O")
       
        city_counts["Highlight"] = np.where(city_counts["City"] == selected_city, 1, 0)
        center = city_coords.get(selected_city, city_coords["TÜM TÜRKİYE"])
        # 6. Harita
        fig = px.choropleth_mapbox(
            data_frame=city_counts,
            geojson=turkey_geojson,
            featureidkey="properties.name",
            locations="City",
            color="Highlight",
            color_continuous_scale="Blues",
            mapbox_style="carto-positron",
            center={"lat": center["lat"], "lon": center["lon"]},
            zoom=5,
            opacity=0.7,
            height=500,
            labels={"AracSayisi": "Araç Sayısı"},
            hover_name="City",
            hover_data={"AracSayisi": True, "Highlight": False},
        )

        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig)


# Tahmin butonu
if st.button("Fiyatı Tahmin Et"):
    input_dict = {
        'Title': 'Başlık',
        'Address': address,
        'City': city,
        'Price(TL)': '',
        'ListingID': '',
        'ListingDate': formatted_tarih,
        'Brand': brand,
        'Series': series,
        'Model': model_name,
        'Year': year,
        'Kilometers': mileage,
        'GearType': gear_type,
        'FuelType': fuel_type,
        'BodyType': body_type,
        'Color': color,
        'EngineSize': engine_size,
        'EnginePower': engine_power,
        'DriveType': drive_type,
        'PaintAndPartsCondition': summary.strip(),
        'TradeInStatus': 'Takasa Uygun',
        'SellerType': 'Sahibinden',
        'VehicleTax(TL)': '',
        'TramerCondition': tramer_condition + str(tramer_amount)
    }
    user_df = pd.DataFrame([input_dict])
    user_df = Preprocessing.preprocessing(dataframe=user_df)

    # train_columns = [col.upper() for col in train_columns]  # Gerekirse bunu aç
    user_df.columns = user_df.columns.str.upper()  # Gerekirse bunu da aç

    # Eksik kolonları sıfırla doldur
    for col in train_columns:
        if col not in user_df.columns:
            user_df[col] = 0


    user_df = user_df[train_columns]
    predicted_price = model.predict(user_df)[0] 
    
    # Göster
    if avg_model:
        
        model_fark = avg_model - predicted_price
        print(f"model fark {model_fark}")
        print(f"tahmin:{predicted_price}")
        if model_fark > 100000:
            predicted_price = predicted_price + (model_fark * 80/100)
            st.success(f"### Tahmini Araç Fiyatı: {predicted_price:,.0f} TL")
        elif model_fark < - 100000:
            predicted_price = predicted_price + (model_fark * 80/100)
            st.success(f"### Tahmini Araç Fiyatı: {predicted_price:,.0f} TL")
        else:
            st.success(f"### Tahmini Araç Fiyatı: {predicted_price:,.0f} TL")
    else:
       st.success(f"### Tahmini Araç Fiyatı: {predicted_price:,.0f} TL")

    

