from .models import User,BugFeature
from django.contrib.auth.forms import UserCreationForm
from django import forms

class UserCreform(UserCreationForm):
    password1 = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(attrs={"class": "form-control my-2", "placeholder": "Password"})
    )
    password2 = forms.CharField(
        label="Password Again",
        widget=forms.PasswordInput(attrs={"class": "form-control my-2", "placeholder": "Password Again"})
    )

    class Meta:
        model = User
        fields = ["username", "role", "gender"] 
        widgets = {
            "username": forms.TextInput(attrs={"class": "form-control my-2", "placeholder": "Username"}),
            "role": forms.Select(attrs={"class": "form-control my-2"}),
            "gender": forms.Select(attrs={"class": "form-control my-2"}),
        }
class BugFeatureForm(forms.ModelForm):
    class Meta:
        model = BugFeature
        fields = ['summary', 'product', 'component']
        widgets = {
            "summary": forms.Textarea(attrs={"class": "form-control my-2", "placeholder": "Summary"}),
            "product": forms.TextInput(attrs={"class": "form-control my-2", "placeholder": "Product"}),
            "component": forms.TextInput(attrs={"class": "form-control my-2", "placeholder": "Component"}),
            

        }