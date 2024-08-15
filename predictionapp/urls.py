from django.urls import path
from . import views

urlpatterns = [
    path("api/predict-song",views.predict_song,name="predict-song"),
    path("predict-song",views.predict_songView,name="predic_songView")
]
