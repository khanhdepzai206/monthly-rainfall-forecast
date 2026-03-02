from django.shortcuts import render
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
import sys
import os
import pickle
import pandas as pd
import numpy as np
import json
from datetime import datetime
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Thêm DuBao vào path để import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'DuBao'))

from src.predict import predict_rainfall, predict_rainfall_daily, predict_rainfall_daily_two_stage
from .models import RainfallPrediction

# Cấu hình đường dẫn mô hình và metrics
MODEL_CONFIG = {
    'gradient_boosting_weather': {
        'path': 'rainfall_model.pkl',
        'name': 'Gradient Boosting với Weather Data',
        'description': 'Sử dụng nhiệt độ, độ ẩm, gió làm features để dự đoán lượng mưa',
    },
    'random_forest_weather': {
        'path': 'rainfall_model_rf.pkl',
        'name': 'Random Forest với Weather Data',
        'description': 'Sử dụng Random Forest với tất cả weather features',
    },
    'sarimax': {
        'path': 'sarimax_model.pkl',
        'name': 'SARIMA (Seasonal Average)',
        'description': 'Dùng trung bình theo mùa cho từng tháng từ dữ liệu lịch sử',
    },
}

def _get_model_metrics_from_pickle(model_path, avg_rainfall=180):
    """Đọc metrics từ file pickle. Trả về dict với mae, rmse, r2_score, accuracy_percent."""
    result = {'mae': None, 'rmse': None, 'r2_score': None, 'accuracy_percent': None}
    if not os.path.exists(model_path):
        return result
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        m = data.get('metrics') if isinstance(data, dict) else {}
        result['mae'] = round(float(m.get('mae', 0)), 2) if m.get('mae') is not None else None
        result['rmse'] = round(float(m.get('rmse', 0)), 2) if m.get('rmse') is not None else None
        r2 = m.get('r2_score')
        if r2 is not None:
            result['r2_score'] = round(float(r2), 4)
            result['accuracy_percent'] = round(max(0, min(100, float(r2) * 100)), 1)
        elif result['mae'] is not None and avg_rainfall and avg_rainfall > 0:
            result['accuracy_percent'] = round(max(0, 100 - (result['mae'] / avg_rainfall) * 100), 1)
    except Exception as e:
        print(f"Error loading metrics from {model_path}: {e}")
    return result


def _get_daily_metrics(project_base_path):
    """Đọc metrics cho các mô hình theo ngày từ model_metrics.json."""
    path = os.path.join(project_base_path, 'DuBao', 'models', 'model_metrics.json')
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Return all daily metrics
        return {k: v for k, v in data.items() if k.startswith('daily_')}
    except Exception:
        return {}


def _load_fallback_metrics(project_base_path):
    """Đọc metrics dự phòng từ file JSON khi pickle không có."""
    path = os.path.join(project_base_path, 'DuBao', 'models', 'model_metrics.json')
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading model_metrics.json: {e}")
        return {}


def get_all_models_metrics(project_base_path, avg_rainfall=180):
    """Trả về dict metrics cho tất cả mô hình. Ưu tiên từ pickle, không có thì lấy từ model_metrics.json."""
    fallback = _load_fallback_metrics(project_base_path)
    out = {}
    for key, cfg in MODEL_CONFIG.items():
        path = os.path.join(project_base_path, 'DuBao', 'models', cfg['path'])
        m = _get_model_metrics_from_pickle(path, avg_rainfall)
        # Nếu pickle không có số liệu thì dùng từ JSON
        fb = fallback.get(key, {})
        if m['mae'] is None and fb:
            m['mae'] = round(float(fb.get('mae', 0)), 2) if fb.get('mae') is not None else None
        if m['rmse'] is None and fb:
            m['rmse'] = round(float(fb.get('rmse', 0)), 2) if fb.get('rmse') is not None else None
        if m['r2_score'] is None and fb.get('r2_score') is not None:
            m['r2_score'] = round(float(fb['r2_score']), 4)
        if m['accuracy_percent'] is None and fb.get('accuracy_percent') is not None:
            m['accuracy_percent'] = round(float(fb['accuracy_percent']), 1)
        elif m['accuracy_percent'] is None and m['r2_score'] is not None:
            m['accuracy_percent'] = round(max(0, min(100, m['r2_score'] * 100)), 1)
        out[key] = {
            'name': cfg['name'],
            'description': cfg['description'],
            'mae': m['mae'],
            'rmse': m['rmse'],
            'r2_score': m['r2_score'],
            'accuracy_percent': m['accuracy_percent'],
        }

    # Add daily models
    daily_models = _get_daily_metrics(project_base_path)
    for key, metrics in daily_models.items():
        out[key] = {
            'name': key.replace('daily_', '').replace('_', ' ').title(),
            'description': f'Mô hình {key.replace("daily_", "").replace("_", " ").title()} cho dự đoán theo ngày.',
            'mae': metrics.get('mae'),
            'rmse': metrics.get('rmse'),
            'r2_score': metrics.get('r2_score'),
            'accuracy_percent': metrics.get('accuracy_percent'),
        }

    return out

def index(request):
    """Trang chủ giản lược – dùng template giống Flask (templates/index.html)."""
    # đơn giản chỉ render trang chính viết sẵn bên thư mục templates/
    return render(request, 'index.html')

# NOTE: the old `predict` view served as a JSON API and is no longer used by the simplified
# Flask‑style front-end. Predictions are now handled by `predictor/api_views.py` under `/api/...`.
# The URL `/predict/` will be mapped to `flask_predict` below instead.
#
# If you need the legacy API, you can rename this function and update URLs accordingly.


def flask_predict(request):
    """Trang dự đoán: GET render template, POST trả JSON từ 3 mô hình 2 giai đoạn."""
    if request.method == 'POST':
        return _predict_daily_two_stage_api(request)
    return render(request, 'predict.html')


def _predict_daily_two_stage_api(request):
    """API: Dự đoán theo ngày với 3 mô hình 2 giai đoạn (Gradient Boosting, Random Forest, Extra Trees)."""
    try:
        year = int(request.POST.get('year', request.GET.get('year', 0)))
        month = int(request.POST.get('month', request.GET.get('month', 0)))
        day = int(request.POST.get('day', request.GET.get('day', 0)))
        if not (1979 <= year <= 2100) or not (1 <= month <= 12) or not (1 <= day <= 31):
            return JsonResponse({'success': False, 'error': 'Ngày không hợp lệ'})
        project_root = os.path.join(os.path.dirname(__file__), '..')
        models_dir = os.path.join(project_root, 'DuBao', 'models')
        daily_path = os.path.join(project_root, 'DuBao', 'data', 'daily_combined.csv')
        model_names = [
            ('gradient_boosting', 'Gradient Boosting'),
            ('random_forest', 'Random Forest'),
            ('extra_trees', 'Extra Trees'),
        ]
        models_result = []
        chart_data = {'labels': [], 'cls_accuracy': [], 'cls_f1': [], 'reg_r2': [], 'reg_mae': []}
        for key, label in model_names:
            path = os.path.join(models_dir, f'daily_two_stage_{key}.pkl')
            if not os.path.exists(path):
                models_result.append({
                    'model': label,
                    'key': key,
                    'has_rain': False,
                    'amount_mm': 0,
                    'error': 'Mô hình chưa được train',
                    'metrics': {},
                })
                continue
            has_rain, amount_mm, metrics = predict_rainfall_daily_two_stage(
                path, year, month, day, daily_path
            )
            models_result.append({
                'model': label,
                'key': key,
                'has_rain': has_rain,
                'amount_mm': round(amount_mm, 2),
                'metrics': metrics,
            })
            chart_data['labels'].append(label)
            chart_data['cls_accuracy'].append(round(metrics.get('cls_accuracy', 0) * 100, 1))
            chart_data['cls_f1'].append(round(metrics.get('cls_f1', 0) * 100, 1))
            chart_data['reg_r2'].append(round(metrics.get('reg_r2', 0) * 100, 1))
            chart_data['reg_mae'].append(round(metrics.get('reg_mae', 0), 2))
        date_label = f'{day:02d}/{month:02d}/{year}'
        return JsonResponse({
            'success': True,
            'date_label': date_label,
            'year': year,
            'month': month,
            'day': day,
            'models': models_result,
            'chart_data': chart_data,
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

# legacy predict API removed – predictions are now served via predictor/api_views.py

@csrf_exempt
def get_chart_data(request):
    """API lấy dữ liệu cho biểu đồ"""
    try:
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'DuBao', 'data', 'monthly_rainfall.csv')
        df = pd.read_csv(csv_path)
        
        chart_type = request.GET.get('type', 'yearly')
        
        if chart_type == 'monthly':
            # Biểu đồ trung bình mưa theo tháng
            monthly_avg = df.groupby('month')['rainfall'].mean()
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            return JsonResponse({
                'success': True,
                'labels': months,
                'data': [round(x, 2) for x in monthly_avg.fillna(0).tolist()],
                'title': 'Average Monthly Rainfall'
            })
        else:
            # Biểu đồ tổng lượng mưa theo năm
            yearly = df.groupby('year')['rainfall'].sum().reset_index()
            yearly = yearly.sort_values('year')
            return JsonResponse({
                'success': True,
                'labels': yearly['year'].astype(str).tolist(),
                'data': yearly['rainfall'].fillna(0).tolist(),
                'title': 'Yearly Total Rainfall'
            })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@csrf_exempt  
def prediction_history(request):
    """API lấy lịch sử dự đoán"""
    try:
        if request.user.is_authenticated:
            predictions = RainfallPrediction.objects.filter(user=request.user).values(
                'year', 'month', 'day', 'predicted_rainfall', 'historical_avg', 'created_at'
            )[:50]
            
            data = []
            for pred in predictions:
                if pred.get('day') is not None:
                    date_str = f"{pred['day']:02d}/{pred['month']:02d}/{pred['year']}"
                else:
                    date_str = f"{pred['month']}/{pred['year']}"
                data.append({
                    'date': date_str,
                    'predicted': round(pred['predicted_rainfall'], 2),
                    'historical_avg': round(pred['historical_avg'], 2) if pred['historical_avg'] else 'N/A',
                    'created': pred['created_at'].strftime('%d/%m/%Y %H:%M') if pred['created_at'] else ''
                })
            
            return JsonResponse({'success': True, 'data': data})
        else:
            return JsonResponse({'success': False, 'error': 'Not authenticated'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

def comparison(request):
    """Trang so sánh đơn giản – render template root giống Flask."""
    return render(request, 'compare.html')


def compare_two(request):
    """Simple page to compare any two models side-by-side"""
    try:
        # Build same models_comparison dict as `comparison`
        hyperparams_path = os.path.join(os.path.dirname(__file__), '..', 'DuBao', 'models', 'hyperparameters.json')

        models_comparison = {
            'gradient_boosting_weather': {
                'name': 'Gradient Boosting + Weather', 'mae': 0.42, 'rmse': 0.55, 'r2_score': 1.000, 'color': '#FF6B6B'
            },
            'gradient_boosting': {
                'name': 'Gradient Boosting (Rainfall)', 'mae': 42.15, 'rmse': 54.32, 'r2_score': 0.7285, 'color': '#4ECDC4'
            },
            'sarima': {
                'name': 'SARIMA', 'mae': 39.87, 'rmse': 51.45, 'r2_score': 0.7512, 'color': '#95E1D3'
            },
            'lstm': {
                'name': 'LSTM', 'mae': 45.92, 'rmse': 57.83, 'r2_score': 0.6945, 'color': '#FFE66D'
            }
        }

        if os.path.exists(hyperparams_path):
            try:
                with open(hyperparams_path, 'r') as f:
                    hyper = json.load(f)
                    best = hyper.get('best_result', {})
                    if best:
                        models_comparison['gradient_boosting']['mae'] = round(best.get('test_mae', models_comparison['gradient_boosting']['mae']), 2)
                        models_comparison['gradient_boosting']['rmse'] = round(best.get('test_rmse', models_comparison['gradient_boosting']['rmse']), 2)
                        models_comparison['gradient_boosting']['r2_score'] = round(best.get('test_r2', models_comparison['gradient_boosting']['r2_score']), 4)
            except Exception:
                pass

        models_json = json.dumps(models_comparison)

        return render(request, 'predictor/compare_two.html', {
            'models': models_comparison,
            'models_json': models_json
        })

    except Exception as e:
        return render(request, 'predictor/compare_two.html', {'error': str(e), 'models': {}})


def predict_daily(request):
    """
    View: Trang dự đoán mưa theo ngày (2-Step Model)
    Template: predict_daily.html
    """
    return render(request, 'predictor/predict_daily.html')


def model_metrics(request):
    """
    View: Trang hiển thị độ chính xác của các mô hình
    Template: model_metrics.html
    """
    return render(request, 'predictor/model_metrics.html')