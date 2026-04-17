from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
import sys
import os
import pickle
import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime, timedelta
from .models import RainfallPrediction, DailyPrediction, ActualRainfall
from .forms import ActualRainfallForm
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Thêm DuBao vào path để import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'DuBao'))

from src.predict import predict_rainfall, predict_rainfall_daily, predict_rainfall_daily_two_stage
from src.rolling_forecast import rolling_forecast
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


@csrf_exempt
def flask_predict(request):
    """Trang dự đoán: GET render template, POST trả JSON từ 3 mô hình 2 giai đoạn."""
    if request.method == 'POST':
        return _predict_daily_two_stage_api(request)
    return render(request, 'predict.html')


def _predict_daily_two_stage_api(request):
    """API: Dự đoán theo ngày với 3 mô hình RF, XGB, LR (cùng models như daily-predict)."""
    try:
        year = int(request.POST.get('year', request.GET.get('year', 0)))
        month = int(request.POST.get('month', request.GET.get('month', 0)))
        day = int(request.POST.get('day', request.GET.get('day', 0)))
        
        if not (1979 <= year <= 2026) or not (1 <= month <= 12) or not (1 <= day <= 31):
            return JsonResponse({'success': False, 'error': 'Ngày không hợp lệ'})
        
        # Tạo date object
        from datetime import datetime as dt
        predict_date = dt(year, month, day).date()
        
        project_root = os.path.join(os.path.dirname(__file__), '..')
        models_dir = os.path.join(project_root, 'DuBao', 'models')
        features_path = os.path.join(project_root, 'DuBao', 'data', 'daily_features.csv')
        
        # Load features cho ngày cụ thể
        df = pd.read_csv(features_path)
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        row = df[df['date'] == predict_date]
        if row.empty:
            return JsonResponse({'success': False, 'error': f'Không có dữ liệu cho ngày {day:02d}/{month:02d}/{year}'})
        
        # Features cần thiết (giống như trong training - loại bỏ các cột không phải feature)
        features_df = row.copy()
        exclude_cols = {'date', 'target', 'datetime', 'rainfall'}
        feature_cols = [c for c in features_df.columns if c not in exclude_cols]
        features = features_df[feature_cols]
        
        # Load models
        rf_path = os.path.join(models_dir, 'rf_daily_model.pkl')
        xgb_path = os.path.join(models_dir, 'xgb_daily_model.pkl')
        lr_path = os.path.join(models_dir, 'lr_daily_model.pkl')
        
        models_result = []
        
        # Predict với 3 models
        model_configs = [
            (rf_path, 'RandomForest', 'RF'),
            (xgb_path, 'XGBoost', 'XGB'),
            (lr_path, 'LinearRegression', 'LR'),
        ]
        
        for model_path, label, key in model_configs:
            if not os.path.exists(model_path):
                models_result.append({
                    'model': label,
                    'key': key,
                    'amount_mm': 0,
                    'error': 'Mô hình chưa được train',
                })
                continue
            
            try:
                model = joblib.load(model_path)
                pred = float(model.predict(features)[0])
                models_result.append({
                    'model': label,
                    'key': key,
                    'amount_mm': round(pred, 2),
                })
            except Exception as e:
                models_result.append({
                    'model': label,
                    'key': key,
                    'amount_mm': 0,
                    'error': str(e),
                })
        
        date_label = f'{day:02d}/{month:02d}/{year}'
        return JsonResponse({
            'success': True,
            'date_label': date_label,
            'year': year,
            'month': month,
            'day': day,
            'models': models_result,
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@csrf_exempt
def predict_multi_day_forecast(request):
    """
    API: Rolling forecast cho 4-5 ngày tiếp theo.
    Sử dụng dữ liệu thời tiết dự báo + rolling prediction.
    """
    try:
        year = int(request.POST.get('year', request.GET.get('year', 0)))
        month = int(request.POST.get('month', request.GET.get('month', 0)))
        day = int(request.POST.get('day', request.GET.get('day', 0)))
        
        # Validate: phải là ngày hợp lệ
        from datetime import datetime as dt
        try:
            start_date = dt(year, month, day).date()
        except ValueError:
            return JsonResponse({'success': False, 'error': 'Ngày không hợp lệ'})
        
        # Check nếu start_date trước ngày 01/04/2026, adjust
        last_available = dt(2026, 4, 1).date()
        if start_date <= last_available:
            # Đổi sang ngày tiếp theo
            start_date = last_available + timedelta(days=1)
        
        # Call rolling forecast
        project_root = os.path.join(os.path.dirname(__file__), '..')
        predictions = rolling_forecast(
            start_date.strftime('%Y-%m-%d'),
            num_days=5,
            models_dir=os.path.join(project_root, 'DuBao', 'models'),
            data_dir=os.path.join(project_root, 'DuBao', 'data')
        )
        
        if predictions is None:
            return JsonResponse({'success': False, 'error': 'Không thể tạo forecast'})
        
        return JsonResponse({
            'success': True,
            'start_date': start_date.strftime('%d/%m/%Y'),
            'forecast': predictions
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
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


@login_required
def update_actual(request):
    """Route cũ giữ lại để tương thích, dùng chung logic với actual_input."""
    if request.method == 'POST':
        try:
            actual_date = request.POST.get('actual_date')
            actual_rainfall = float(request.POST.get('actual_rainfall', 0))

            if not actual_date:
                return JsonResponse({'success': False, 'error': 'Chưa chọn ngày'})

            actual, _ = ActualRainfall.objects.update_or_create(
                date=actual_date,
                defaults={'actual_rainfall': actual_rainfall}
            )

            try:
                prediction = DailyPrediction.objects.get(date=actual.date)
                actual.prediction = prediction
                actual.save()
                actual.evaluate_error()

                retrained = False
                retrain_info = actual.retrain_summary()

                if retrain_info['needs_retrain']:
                    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'DuBao'))
                    from run_pipeline import retrain_models
                    retrain_models()
                    actual.retrained = True
                    actual.save()
                    retrained = True

                message = f'Đã cập nhật actual rainfall cho ngày {actual.date}. '
                if retrained:
                    message += f"Sai lệch lớn ({retrain_info['max_pct']:.1f}%), hệ thống đang retrain."
                else:
                    message += f"Sai lệch nhỏ ({retrain_info['max_pct']:.1f}%); không cần retrain."

                message += ' ' + retrain_info['details']

                return JsonResponse({'success': True, 'message': message})

            except DailyPrediction.DoesNotExist:
                return JsonResponse({'success': False, 'error': f'Không tìm thấy prediction cho ngày {actual.date}'})

        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})

    return redirect('actual_input')


def daily_predict(request):
    """Trang dự đoán lượng mưa ngày mai bằng 3 mô hình ML (2 bước: có mưa? + mm)."""
    context = {
        'predictions': None,
        'best_model': None,
        'error': None
    }

    if request.method == 'POST':
        try:
            # Dùng hệ daily_ml_system (2-stage) để dự đoán ngày mai
            dubao_src = os.path.join(os.path.dirname(__file__), '..', 'DuBao', 'src')
            sys.path.insert(0, dubao_src)
            from daily_ml_system.predict_daily import predict_tomorrow
            from daily_ml_system.evaluate import full_report
            from daily_ml_system import config as ml_cfg

            _target_from_data, prediction_date, preds, probs = predict_tomorrow()
            # Hiển thị "ngày mai" theo lịch thật (nút UI), không phải theo ngày cuối trong CSV
            calendar_tomorrow = datetime.now().date() + timedelta(days=1)
            rain_th = 0.5
            try:
                # bundle lưu threshold nếu có
                import pickle
                if os.path.exists(ml_cfg.MODEL_BUNDLE_PATH):
                    with open(ml_cfg.MODEL_BUNDLE_PATH, 'rb') as f:
                        b = pickle.load(f)
                    rain_th = float(b.get('rain_prob_threshold', rain_th))
            except Exception:
                pass

            # Best model dựa trên MAE 7 ngày gần nhất (nếu có actual), fallback XGB
            best_model = None
            try:
                rep = full_report()
                best_model = (rep.get('best_model') or '').upper()
            except Exception:
                best_model = None

            context.update({
                'predictions': {
                    'rf': round(float(preds.get('rf', 0.0)), 2),
                    'xgb': round(float(preds.get('xgb', 0.0)), 2),
                    'et': round(float(preds.get('et', 0.0)), 2),
                    'rf_prob': round(float(probs.get('rf', 0.0)) * 100, 1),
                    'xgb_prob': round(float(probs.get('xgb', 0.0)) * 100, 1),
                    'et_prob': round(float(probs.get('et', 0.0)) * 100, 1),
                    'rf_has_rain': bool(probs.get('rf', 0.0) >= rain_th),
                    'xgb_has_rain': bool(probs.get('xgb', 0.0) >= rain_th),
                    'et_has_rain': bool(probs.get('et', 0.0) >= rain_th),
                    'rain_threshold': round(float(rain_th) * 100, 0),
                },
                'best_model': best_model,
                'date': calendar_tomorrow.strftime('%d/%m/%Y'),
                'data_as_of': pd.Timestamp(prediction_date).strftime('%d/%m/%Y'),
            })

        except Exception as e:
            context['error'] = f"Lỗi khi dự đoán: {str(e)}"
            print(f"Error in daily_predict: {e}")

    return render(request, 'daily_predict.html', context)


@login_required
def actual_input(request):
    """Trang nhập lượng mưa thực tế."""
    form = ActualRainfallForm(request.POST or None)

    if request.method == 'POST':
        actual_date = request.POST.get('date')
        actual_rainfall = request.POST.get('actual_rainfall')

        if actual_date and actual_rainfall not in [None, '']:
            try:
                actual, created = ActualRainfall.objects.update_or_create(
                    date=actual_date,
                    defaults={'actual_rainfall': float(actual_rainfall)}
                )

                # Tìm prediction tương ứng
                try:
                    prediction = DailyPrediction.objects.get(date=actual.date)
                    actual.prediction = prediction
                    actual.save()

                    # Tính sai số
                    actual.evaluate_error()

                    # Kiểm tra retrain
                    retrain_info = actual.retrain_summary()
                    action = 'cập nhật' if not created else 'lưu'

                    if retrain_info['needs_retrain']:
                        try:
                            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'DuBao'))
                            from run_pipeline import retrain_models
                            retrain_models()
                            actual.retrained = True
                            actual.save()
                            messages.success(request, (
                                f"Đã {action} actual rainfall cho ngày {actual.date}. "
                                f"Sai lệch lớn ({retrain_info['max_pct']:.1f}%), hệ thống đang retrain. "
                                f"{retrain_info['details']}"
                            ))
                        except Exception as e:
                            messages.warning(request, (
                                f"Đã {action} actual rainfall nhưng lỗi retrain: {str(e)}. "
                                f"{retrain_info['details']}"
                            ))
                    else:
                        messages.success(request, (
                            f"Đã {action} actual rainfall cho ngày {actual.date}. "
                            f"Sai lệch nhỏ ({retrain_info['max_pct']:.1f}%), không cần retrain. "
                            f"{retrain_info['details']}"
                        ))

                except DailyPrediction.DoesNotExist:
                    messages.warning(request, f'Đã lưu actual rainfall nhưng không tìm thấy prediction cho ngày {actual.date}')

                return redirect('actual_input')

            except ValueError:
                messages.error(request, 'Lượng mưa thực tế không hợp lệ')
        else:
            messages.warning(request, 'Vui lòng nhập đầy đủ ngày và lượng mưa thực tế.')

    # Hiển thị actuals gần đây
    recent_actuals = ActualRainfall.objects.select_related('prediction').order_by('-date')[:10]

    context = {
        'form': form,
        'recent_actuals': recent_actuals
    }

    return render(request, 'actual_input.html', context)