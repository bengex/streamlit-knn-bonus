import streamlit as st
import pandas as pd
import joblib
import time
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

df = pd.read_csv('Data_Cleaned.csv')
x = df.drop('Bonus', axis=1)
y = df['Bonus']

#smote
sm = SMOTE(random_state=30)
x, y = sm.fit_resample(x,y)

# Load pipeline dan encoder
loaded_pipeline = joblib.load('pipeline_knn.sav')
encoder = joblib.load('encoder_dataset.sav')  

#model knn
y_pred = loaded_pipeline.predict(x)
akurasi = accuracy_score(y, y_pred)
akurasi = round((akurasi * 100), 2)

# Judul aplikasi
st.title('Implemtasi Metode K-Nearest Neighbor (K-NN) Untuk Menentukan Bonus Karyawan Pada Industri Perbankan Nasional')
st.markdown(f"**Akurasi Model Sebesar:** :green[{akurasi:.2f}%]")
st.write('')
st.write("Gunakan data karyawan untuk memprediksi kelayakan bonus tahunan.")

# Sidebar logo
st.sidebar.image("logo upy.png")

tab1, tab2 = st.tabs(['Input Data', 'Unggah File'])
with tab1 :
    # Input Data
    st.header("Input Data Karyawan")
    col1, col2 = st.columns(2)

    with col1:
        Level = st.selectbox("Level", list(range(10, 20)))
        PendidikanTerakhir = st.selectbox("Pendidikan Terakhir", ['SD', 'SMP', 'SMA', 'SMK', 'D1', 'D2', 'D3', 'D4', 'S1', 'S2'])
        Kinerja = st.number_input("Nilai Kinerja", min_value=0.0, step=0.1)
        HAV = st.selectbox("HAV", list(range(1, 10)))

    with col2:
        LamaBekerja = st.number_input("Lama Bekerja (bulan)", min_value=0)
        Sanksi = st.selectbox("Sanksi", ['Aktif', 'Tidak Aktif'])
        Loyalitas = st.number_input("Nilai Loyalitas", min_value=0.0, step=0.1)
        KonfirmasiAtasan = st.selectbox("Konfirmasi Atasan", list(range(1, 10)))

    # Tombol prediksi
    if st.button("MULAI"):
        try:
            # Menyiapkan DataFrame untuk input
            input_df = pd.DataFrame([{
                'Level': Level,
                'LamaBekerja': LamaBekerja,
                'PendidikanTerakhir': PendidikanTerakhir,
                'Sanksi': Sanksi,
                'Kinerja': Kinerja,
                'Loyalitas': Loyalitas,
                'HAV': HAV,
                'Konfirmasi Atasan': KonfirmasiAtasan
            }])

            #proses transform data
            input_encoded = encoder.transform(input_df)

            # Prediksi menggunakan pipeline
            y_pred = loaded_pipeline.predict(input_encoded)
            prediction = y_pred[0]
            
            #bar progres
            bar = st.progress(0)
            status = st.empty()

            for i in range(1, 101):
                status.text(f'{i}% MEMPROSES')
                bar.progress(i)
                time.sleep(0.01)
                if i == 100:
                    time.sleep(1)
                    status.empty()
                    bar.empty()

             
            # Tampilkan hasil prediksi
            if prediction == 0:
                st.success("✅ Selamat Karyawan Tersebut LAYAK Mendapat Bonus Tahunan")
            else:
                st.error("❌ Mohon Maaf Karyawan Tersebut TIDAK LAYAK Mendapat Bonus Tahunan")
                

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")

with tab2 :
    # Membaca dataset
    st.header('Unggah File')

    st.write('Silahkan mengunduh dataset untuk mencoba')
    
    dataset = pd.read_csv('Dataset-coba.csv')
    csv = dataset.to_csv(index=False)

    #tombol download dataset
    st.download_button(
        label="Unduh Dataset", 
        data=csv, 
        file_name="Dataset-coba.csv", 
        mime="text/csv")
    
    st.write('')
    
    #mengunggah dataset
    unggah = st.file_uploader(label="Silahkan unggah file .csv", type="csv")
    
    dataset = None
    if unggah is not None:
        dataset = pd.read_csv(unggah)
        st.success('File berhasil diunggah')

        if st.button("CEK"):
            if dataset is None:
                st.warning('Silahkan unggah file terlebih dahulu.')
            else:

                input_encode = encoder.transform(dataset)
    

            # Prediksi menggunakan pipeline
                y_pred = loaded_pipeline.predict(input_encode)
                prediction = y_pred[0]
                bar = st.progress(0)
                status = st.empty()

                for i in range(1, 101):
                    status.text(f'{i}% MEMPROSES')
                    bar.progress(i)
                    time.sleep(0.01)
                    if i == 100:
                        time.sleep(1)
                        status.empty()
                        bar.empty()
                
                hasil = []
                for pred in y_pred:
                    if pred == 0:
                        hasil.append('Layak')
                    else:
                        hasil.append('Tidak Layak')
                
                tabel_hasil = pd.DataFrame({'Hasil':hasil})
                col1, col2 = st.columns([1, 2])
                with col1 :
                    st.dataframe(tabel_hasil)
                with col2 :
                    st.dataframe(dataset)