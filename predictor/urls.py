from django.urls import path
from django.contrib.auth import views as auth_views
from . import views
from . import api_views

urlpatterns = [
    # simple Flask‑style pages using root templates
    path('', views.index, name='index'),
    path('metrics/', views.model_metrics, name='metrics'),      # formerly model-metrics/
    path('predict/', views.flask_predict, name='predict'),       # render predict.html
    path('compare/', views.comparison, name='compare'),          # alias to comparison

    # legacy/advanced pages (kept for reference)
    path('predict-daily/', views.predict_daily, name='predict_daily'),
    path('model-metrics-old/', views.model_metrics, name='model_metrics_old'),
    path('history/', views.prediction_history, name='prediction_history'),
    path('comparison-old/', views.comparison, name='comparison_old'),
    path('compare-two/', views.compare_two, name='compare_two'),

    # authentication
    path('login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),
    
    # API endpoints (remain unchanged)
    path('chart-data/', views.get_chart_data, name='chart_data'),
    path('api/predict/', api_views.predict_daily_api, name='api_predict'),
    path('api/predict-range/', api_views.predict_range_api, name='api_predict_range'),
    path('api/predict-compare/', api_views.predict_compare_models_api, name='api_predict_compare'),
    path('api/model-info/', api_views.model_info_api, name='api_model_info'),
    path('api/model-metrics/', api_views.model_metrics_api, name='api_model_metrics'),
]
