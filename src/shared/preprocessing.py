import string, json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from pathlib import Path
import csv

# fungsi untuk load dan preprocess caption.txt dari dataset Flickr8k
def load_captions(filepath: str) -> dict[str, list[str]]:
    # dictionary untuk menyimpan mapping image_id dan list of captions
    mapping = {}

    # membaca file caption.txt
    with open(filepath, "r", encoding="utf-8", newline="") as file:
        # menggunakan CSV reader agar caption yang punya koma tetap terbaca utuh
        # contoh: "A dog shakes its head near the shore , a red ball next to it ."
        rows = csv.reader(file, delimiter='\t')  # delimiter tab karena file menggunakan tab sebagai pemisah

        # skip header: image,caption
        next(rows, None)  

        # iterasi tiap baris
        for row in rows:
            # validasi jumlah kolom (minimal 2) agar tidak error saat split
            if len(row) < 2:
                continue
            
            # ekstrak nama file gambar dan caption
            image_id = row[0].strip()
            caption = row[1].strip()

            # lewati baris kosong atau data hilang
            if not image_id or not caption:
                continue

            # simpan caption ke list per image (1 image bisa punya > 1 caption)
            mapping.setdefault(image_id, []).append(caption)

    return mapping


# fungsi untuk membersihkan caption serta menambahkan token <start> dan <end>
def clean_and_wrap_captions(caption_mapping: dict[str, list[str]]) -> dict[str, list[str]]:
    # dictionary baru untuk menyimpan hasil caption yang sudah dibersihkan dan ditambahkan token
    cleaned_mapping = {}

    # tabel translasi untuk menghapus tanda baca
    table = str.maketrans('', '', string.punctuation)
    
    # iterasi tiap image_id dan list of captions
    for key, captions in caption_mapping.items():
        # iterasi tiap caption untuk image tersebut
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower() # lowercase
            caption = caption.translate(table) # hapus tanda baca menggunakan tabel translasi
            caption = f"<start> {caption} <end>" # wrap dengan start & end token
            
            # jika image_id belum ada di cleaned_mapping, buat list baru
            if key not in cleaned_mapping:
                cleaned_mapping[key] = []

            # tambahkan caption yang sudah dibersihkan dan ditambahkan token ke list untuk image tersebut
            cleaned_mapping[key].append(caption)
            
    return cleaned_mapping


# fungsi untuk membangun vocabolary & tokenizer dari seluruh caption yang sudah dibersihkan
def build_tokenizer(cleaned_mapping, output_path="tokenizer.json"):
    # kumpulkan semua caption menjadi satu list panjang
    all_captions = []
    for key in cleaned_mapping:
        for caption in cleaned_mapping[key]:
            all_captions.append(caption)
            
    # menghapus tanda selain < dan > pada filter agar token <start> dan <end> tetap terjaga
    filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n' 
    
    # Tokenizer Keras otomatis menambahkan <pad> sebagai token 0 (Zero Padding)
    tokenizer = Tokenizer(filters=filters, oov_token="<unk>")
    tokenizer.fit_on_texts(all_captions)
    
    # simpan tokenizer dalam format JSON (Keras to_json)
    tokenizer_json = tokenizer.to_json()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))
    
    # hitung ukuran vocabulary (jumlah token unik + 1 untuk padding)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Ukuran Vocabulary: {vocab_size}")
    print(f"Tokenizer berhasil disimpan di {output_path}")
    
    return tokenizer, all_captions


# fungsi untuk menghitung panjang maksimum dari seluruh caption (untuk keperluan padding)
def calculate_max_length(all_captions):
    # iterasi seluruh caption dan hitung jumlah kata lalu cari yang paling panjang
    max_length = max(len(caption.split()) for caption in all_captions)
    print(f"Maximum Caption Length: {max_length}")
    
    return max_length


# fungsi untuk mengubah caption menjadi sekuens integer dan padding
def captions_to_sequences_and_pad(tokenizer, all_captions, max_length):
    # ubah caption menjadi urutan integer menggunakan tokenizer
    sequences = tokenizer.texts_to_sequences(all_captions)

    # padding semua caption agar memiliki panjang yang sama
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding="post")

    return sequences, padded


# fungsi untuk menyimpan word_index ke JSON utk dipakai di pipeline NumPy
def save_tokenizer_json(tokenizer, output_path="tokenizer.json"):
    # ambil string JSON yang merepresentasikan keseluruhan objek Tokenizer
    tokenizer_json_str = tokenizer.to_json()
    
    with open(output_path, "w", encoding="utf-8") as f:
        # parse lalu dump lagi agar JSON mudah dibaca
        json.dump(json.loads(tokenizer_json_str), f, ensure_ascii=False, indent=2)
        
    print(f"Tokenizer berhasil disimpan di {output_path}")
