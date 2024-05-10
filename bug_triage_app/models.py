from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    ROLES = [
        ('0', 'Guest'),
        ('1', 'Developer'),
        ('2', 'Bug Reporter'),
    ]
    
    GENDER_CHOICES = [
        ('0', '--- Select Your Gender ---'),
        ('1', 'Male'),
        ('2', 'Female'),
    ]
    
    mobile = models.CharField(max_length=10, null=True, blank=True)
    gender = models.CharField(choices=GENDER_CHOICES, default='0', max_length=5)
    role = models.CharField(choices=ROLES, default='0', max_length=5)

class BugFeature(models.Model):
    summary = models.CharField(max_length=255)
    product = models.CharField(max_length=255)
    component = models.CharField(max_length=255)


