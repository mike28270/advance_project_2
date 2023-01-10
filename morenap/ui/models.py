from django.db import models
from django.conf import settings

# Create your models here.
class Hotel(models.Model):
    name = models.CharField(max_length=50)
    hotel_Main_Img = models.ImageField(upload_to="images/")
