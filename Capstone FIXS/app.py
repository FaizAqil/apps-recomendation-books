from flask import Flask, request, jsonify
from google.cloud import firestore, storage
import tensorflow as tf
import requests
import os
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)

db = firestore.Client()
storage_client = storage.Client()

model_url = 'https://storage.googleapis.com/online-book-borrowing-cloudrun/model-user-like/rekomendasi_buku_CF_weights.h5'

model_path = 'rekomendasi_buku_CF_weights.h5'
response = requests.get(model_url)
with open(model_path, 'wb') as f:
    f.write(response.content)

model = tf.keras.models.load_model(model_path)

def is_image_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/upload', methods=['POST'])
def upload():
    user_data = request.json
    file = request.files.get('file')

    # Periksa apakah file ada dan merupakan gambar
    if file and is_image_file(file.filename):
        filename = secure_filename(file.filename)

        # Upload file ke Google Cloud Storage
        bucket_name = 'online-book-borrowing-cloudrun'
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(filename)
        blob.upload_from_file(file)

        # Simpan informasi buku ke Firestore
        book_data = {
            'user_id': user_data['user_id'],
            'title': user_data['title'],
            'review': user_data['review'],
            'file_url': blob.public_url
        }
        db.collection('books').add(book_data)

        return jsonify({"message": "Book uploaded successfully", "file_url": blob.public_url}), 201
    else:
        return jsonify({"error": "Invalid file type. Only image files are allowed."}), 400

@app.route('/get_buku', methods=['GET'])
def get_buku():
    books_ref = db.collection('books')
    books = books_ref.stream()

    book_list = []
    for book in books:
        book_data = book.to_dict()
        book_data['id'] = book.id
        book_list.append(book_data)

    return jsonify({"books": book_list})

@app.route('/rekomendasi', methods=['POST'])
def rekomendasi():
    user_id = request.json.get('user_id')

    similar_users = get_similar_users(user_id) 
    user_based_recommendations = get_user_based_recommendations(similar_users) 

    combined_recommendations = list(set(user_based_recommendations))

    return jsonify({"recommendations": combined_recommendations})

def get_user_data(user_id):
    user_ref = db.collection('users').document(user_id)
    user_data = user_ref.get().to_dict()
    return user_data

def prepare_input_for_model(user_data):
    # Siapkan input untuk model
    # Misalnya, jika model Anda memerlukan array dengan fitur tertentu
    # Ganti dengan logika yang sesuai untuk menyiapkan input
    input_data = np.array([user_data['feature1'], user_data['feature2'], user_data['feature3']])  # Contoh
    return input_data.reshape(1, -1) 

def get_similar_users(user_id):
    user_data = get_user_data(user_id)

    input_data = prepare_input_for_model(user_data)

    similar_users = model.predict(input_data)  

    return similar_users.tolist()  

def get_user_based_recommendations(similar_users):
    recommendations = []
    
    for similar_user in similar_users:
        user_books_ref = db.collection('books').where('user_id', '==', similar_user).stream()
        
        for book in user_books_ref:
            book_data = book.to_dict()
            if book_data['file_url'] not in recommendations:
                recommendations.append(book_data['file_url'])

    return recommendations

@app.route('/rating', methods=['POST'])
def rating():
    rating_data = request.json
    return jsonify({"message": "Rating submitted successfully"}), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
