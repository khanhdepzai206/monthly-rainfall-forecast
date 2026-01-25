from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('chart-data/', views.get_chart_data, name='chart_data'),
    path('history/', views.prediction_history, name='prediction_history'),
    path('comparison/', views.comparison, name='comparison'),
    path('compare-two/', views.compare_two, name='compare_two'),
    path('login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),
]
