# Submission 2: Stroke Prediction
Nama: Guntur Aji Pratama

Username dicoding: gatama

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) |
| Masalah | Bagaimana cara memprediksi pasien di rumah sakit atau semacamnya terkena penyakit stroke berdasarkan jenis kelamin, umur, penyakit lain yang dimiliki, dan status sebagai perokok? |
| Solusi machine learning | Dari permasalahan diatas dengan memanfaatkan teknologi, machine learning menjadi salah satu solusi untuk membantu mengurangi tingkat kematian yang cukup tinggi akibat penyakit ini. Dengan sebuah sistem prediksi penyakit stroke, diharapkan para tenaga medis maupun masyarakat dapat terbantu untuk dapat mendeteksi penyakit ini lebih awal. |
| Metode pengolahan | Metode pengolahan data menggunakan komponen ExampleGen, StatisticGen, SchemaGen, ExampleValidator dari TensorFlow Extended sebagai bagian dari proses Data Validation dan komponen Transform dari TensorFlow Extended sebagai bagian dari proses Data Preprocessing. |
| Arsitektur model | Arsitektur model machine learning secara umum menggunakan Deep Learning dengan fungsi Dense dari TensorFlow. |
| Metrik evaluasi | Metrik yang digunakan untuk mengevaluasi model machine learning ialah metrik BinaryAccuracy yang dapat mengkalkulasi akurasi model dalam mengklasifikasikan 2 buah kategori, AUC, Recall, dan Precision. |
| Performa model | Performa model machine learning secara umum cukup baik dikarenakan mencapai akurasi lebih dari 92%. |
| Opsi deployment | Opsi deployment sistem machine learning opearations ini ialah menggunakan Railway.app |
| Web app | https://gatama-mlops-production.up.railway.app/v1/models/cc-model/metadata |
| Monitoring | Dari hasil monitoring model serving, belum ada kekacauan yang terjadi terkait performa model dan proses calling API terbilang cukup stabil. |