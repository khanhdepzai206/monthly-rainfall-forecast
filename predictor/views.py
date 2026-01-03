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

from src.predict import predict_rainfall
from .models import RainfallPrediction

def index(request):
    """Trang chủ - hiển thị thống kê và biểu đồ"""
    # Lấy thông tin bộ dữ liệu
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'DuBao', 'data', 'raw_daily.csv')
    df = pd.read_csv(csv_path, skiprows=2)
    df.columns = ["date", "rainfall"]
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["date"])
    df["rainfall"] = pd.to_numeric(df["rainfall"], errors="coerce")
    
    # Tạo dữ liệu monthly series
    monthly_path = os.path.join(os.path.dirname(__file__), '..', 'DuBao', 'data', 'monthly_rainfall.csv')
    yearly_labels = []
    yearly_values = []
    monthly_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_avg = [0] * 12
    
    try:
        mdf = pd.read_csv(monthly_path)
        if {'year', 'month', 'rainfall'}.issubset(mdf.columns):
            # Yearly data
            yearly = mdf.groupby('year')['rainfall'].sum().reset_index()
            yearly = yearly.sort_values('year')
            yearly_labels = yearly['year'].astype(str).tolist()
            yearly_values = yearly['rainfall'].fillna(0).tolist()
            
            # Monthly average
            monthly_avg = mdf.groupby('month')['rainfall'].mean().reindex(range(1, 13)).fillna(0)
            monthly_avg = [round(x, 2) for x in monthly_avg.tolist()]
    except Exception as e:
        print(f"Error loading monthly data: {e}")
    
    # Get historical predictions
    recent_predictions = RainfallPrediction.objects.all()[:10] if request.user.is_authenticated else []
    
    context = {
        'total_records': len(df),
        'start_date': df['date'].min().strftime('%d/%m/%Y'),
        'end_date': df['date'].max().strftime('%d/%m/%Y'),
        'avg_rainfall': round(df['rainfall'].mean(), 2),
        'max_rainfall': round(df['rainfall'].max(), 2),
        'min_rainfall': round(df['rainfall'].min(), 2),
        'yearly_labels': yearly_labels,
        'yearly_values': yearly_values,
        'monthly_labels': monthly_labels,
        'monthly_avg': monthly_avg,
        'recent_predictions': recent_predictions,
    }
    
    return render(request, 'predictor/index.html', context)

@csrf_exempt
def predict(request):
    """API dự đoán lượng mưa"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body) if request.content_type == 'application/json' else request.POST
            year = int(data.get('year'))
            month = int(data.get('month'))
            
            # Kiểm tra giá trị hợp lệ
            if not (1979 <= year <= 2100) or not (1 <= month <= 12):
                return JsonResponse({
                    'success': False,
                    'error': 'Năm phải từ 1979-2100 và tháng từ 1-12'
                })
            
            # Lấy dữ liệu lịch sử
            csv_path = os.path.join(os.path.dirname(__file__), '..', 'DuBao', 'data', 'monthly_rainfall.csv')
            monthly_df = pd.read_csv(csv_path)
            
            # Tìm dữ liệu cùng tháng trong quá khứ
            same_month_data = monthly_df[monthly_df['month'] == month]['rainfall']
            
            metrics = {
                'mae': None,
                'rmse': None,
                'accuracy_percent': None,
            }
            
            if len(same_month_data) > 0:
                base_prediction = same_month_data.mean()
                std_dev = same_month_data.std()
                
                # Thêm random variation
                random_factor = np.random.normal(0, std_dev * 0.2)
                prediction = max(0, base_prediction + random_factor)
                
                # Tính metrics
                try:
                    vals = same_month_data.values.astype(float)
                    n = len(vals)
                    if n > 1:
                        preds = []
                        s = vals.sum()
                        for i in range(n):
                            preds.append((s - vals[i]) / (n - 1))
                        preds = np.array(preds)
                        actuals = vals
                        
                        abs_errors = np.abs(preds - actuals)
                        mae = float(np.nanmean(abs_errors))
                        rmse = float(np.sqrt(np.nanmean((preds - actuals) ** 2)))
                        
                        # Accuracy: % predictions within 20% of actual
                        good = 0
                        total_nonzero = 0
                        for a, p in zip(actuals, preds):
                            if a != 0:
                                total_nonzero += 1
                                if abs(p - a) / abs(a) <= 0.2:
                                    good += 1
                        accuracy_pct = (good / total_nonzero * 100.0) if total_nonzero > 0 else 50.0
                        
                        metrics = {
                            'mae': round(mae, 2),
                            'rmse': round(rmse, 2),
                            'accuracy_percent': round(accuracy_pct, 2),
                        }
                except Exception:
                    pass
            else:
                prediction = monthly_df['rainfall'].mean()
            
            # Lấy dữ liệu lịch sử tháng để vẽ chart so sánh
            historical_data = None
            historical_labels = None
            if len(same_month_data) > 0:
                historical_years = monthly_df[monthly_df['month'] == month]['year'].values
                historical_data = same_month_data.values.tolist()
                historical_labels = [str(int(y)) for y in historical_years]
            
            # Save prediction to database
            if request.user.is_authenticated:
                RainfallPrediction.objects.create(
                    user=request.user,
                    year=year,
                    month=month,
                    predicted_rainfall=prediction,
                    historical_avg=same_month_data.mean() if len(same_month_data) > 0 else None
                )
            
            return JsonResponse({
                'success': True,
                'year': year,
                'month': month,
                'rainfall': round(prediction, 2),
                'metrics': metrics,
                'historical_data': historical_data,
                'historical_labels': historical_labels
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request'})

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
                'year', 'month', 'predicted_rainfall', 'historical_avg', 'created_at'
            )[:50]
            
            data = []
            for pred in predictions:
                data.append({
                    'date': f"{pred['month']}/{pred['year']}",
                    'predicted': round(pred['predicted_rainfall'], 2),
                    'historical_avg': round(pred['historical_avg'], 2) if pred['historical_avg'] else 'N/A',
                    'created': pred['created_at'].strftime('%d/%m/%Y %H:%M') if pred['created_at'] else ''
                })
            
            return JsonResponse({'success': True, 'data': data})
        else:
            return JsonResponse({'success': False, 'error': 'Not authenticated'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})
