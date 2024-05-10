from django.urls import path
from bug_triage_app import views
from django.contrib.auth import views as g

urlpatterns=[
    path('',views.home,name="hm"),
    path('lgn/',g.LoginView.as_view(template_name="html/login.html"),name="lg"),
	path('lgot/',g.LogoutView.as_view(template_name="html/logout.html"),name="lgt"),
    path('reg/',views.register,name="rg"),
    path('rep/',views.report_features,name="bg"),
]