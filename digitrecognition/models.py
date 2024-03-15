from django.db import models

class Digit(models.Model):
    image = models.ImageField(upload_to='digits/')
    prediction = models.IntegerField(null=True, blank=True)
