from django.shortcuts import render
from django.http import JsonResponse
from .prediction_model import model, labelencoder
import numpy as np
import librosa
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.core.files.storage import default_storage
import os

def normalize_spectrogram(spec):
    mean = np.mean(spec)
    std = np.std(spec)
    normalized_spec = (spec - mean) / std
    return normalized_spec

def load_audio_and_extract_spectrogram(audio_file, duration=5, n_mels=128, hop_length=512):
    y, sr = librosa.load(audio_file, duration=duration)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    norm_spectrogram = normalize_spectrogram(mel_spectrogram_db)
    norm_spectrogram = norm_spectrogram.T
    return norm_spectrogram

def predict_song_logic(audio_file):
    spectrogram = load_audio_and_extract_spectrogram(audio_file)

    spectrogram = np.expand_dims(spectrogram, axis=0)
    # Get the prediction probabilities
    predictions = model.predict(spectrogram)
    
    # Round the predictions to four decimal places and convert it to %
    predictions = [(round(p, 4))*100 for p in predictions[0]]
    
    # Get the indices of the top 5 predictions
    top_5_indices = np.argsort(predictions)[-5:][::-1]
    
    # Decode the top 5 predictions
    top_5_songs = labelencoder.inverse_transform(top_5_indices)
    
    # Pair the top 5 predictions with their probabilities
    top_5_with_probs = [(song, predictions[idx]) for song, idx in zip(top_5_songs, top_5_indices)]
    
    return top_5_with_probs


@csrf_exempt
def predict_song(request):
    if request.method == "POST":
        audio_file = request.FILES.get('audio')
        if not audio_file:
            return JsonResponse({'error': 'Audio file not found'}, status=400)
        
        try:
            predicted_song = predict_song_logic(audio_file)
            return JsonResponse({'song': predicted_song})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)

@csrf_exempt
def predict_songView(request):
    if request.method == "POST" and 'audioFile' in request.FILES:
        audio_file = request.FILES['audioFile']
        temp_file_path = default_storage.save('temp_audio.wav', audio_file)
        temp_file_full_path = os.path.join(settings.MEDIA_ROOT, temp_file_path)
        
        try:
            predicted_song = predict_song_logic(temp_file_full_path)
            os.remove(temp_file_full_path)  # Clean up the temporary file
            return render(request, "record.html", {'songs_list': predicted_song})
        except Exception as e:
            os.remove(temp_file_full_path)  # Clean up the temporary file
            return render(request, "error.html", {'error': str(e)})
    else:
        return render(request, "record.html")
