"""
API Dự đoán Mưa theo Ngày (2-Step Model)
Tích hợp mô hình Machine Learning vào Django
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import logging
import traceback
from datetime import date as _date
from datetime import datetime as _dt

logger = logging.getLogger(__name__)

# Thêm DuBao vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'DuBao'))

from src.predict_best_model import predict_with_best_model
from src.predict import predict_rainfall_daily_two_stage
from .models import DailyPrediction

# Cấu hình đường dẫn mô hình
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'DuBao', 'models')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'DuBao', 'data')


@csrf_exempt
@require_http_methods(["GET"])
def predict_tomorrow_compare_api(request):
    """
    API: Trả về dự đoán ngày mai cho 3 mô hình (XGB/ExtraTrees/RF) của daily_ml_system (2-stage).

    Response:
    {
      "success": true,
      "date": "YYYY-MM-DD",
      "models": [
        {"model":"XGBoost","has_rain":true,"rain_probability":0.62,"predicted_rainfall":3.1},
        ...
      ]
    }
    """
    try:
        # legacy behavior: daily_ml_system may not be installed/configured
        # Keep endpoint but degrade gracefully if module is missing.
        dubao_src = os.path.join(os.path.dirname(__file__), '..', 'DuBao', 'src')
        sys.path.insert(0, os.path.abspath(dubao_src))

        from daily_ml_system.predict_daily import predict_tomorrow_detail  # type: ignore

        d = predict_tomorrow_detail()
        target_date = d["target_date"]
        prediction_date = d["prediction_date"]
        prob = d["prob"]
        mm_if_rain = d["mm_if_rain"]
        expected_mm = d["expected_mm"]

        # UI wants tomorrow by calendar date
        import datetime as _dt2
        calendar_tomorrow = _dt2.date.today() + _dt2.timedelta(days=1)

        models = [
            ("XGBoost", "xgb"),
            ("Extra Trees", "et"),
            ("Random Forest", "rf"),
        ]
        models_data = []
        models_map = {}
        for label, key in models:
            pr = float(prob.get(key, 0.0) or 0.0)
            mm_rain = float(mm_if_rain.get(key, 0.0) or 0.0)
            mm_exp = float(expected_mm.get(key, 0.0) or 0.0)
            payload = {
                "model": label,
                "rain_probability": round(pr, 4),
                # chuẩn chuyên nghiệp: dùng expected mm làm predicted_rainfall
                "predicted_rainfall": round(mm_exp, 2),
                "rainfall_if_rain": round(mm_rain, 2),
            }
            models_data.append(payload)
            models_map[label] = payload

        return JsonResponse({
            "success": True,
            # date: giữ lại target_date theo dữ liệu để debug/đối chiếu
            "date": target_date.strftime("%Y-%m-%d"),
            # date_label: dùng để hiển thị cho người dùng
            "date_label": calendar_tomorrow.strftime("%Y-%m-%d"),
            "data_as_of": pd.Timestamp(prediction_date).strftime("%Y-%m-%d"),
            "models": models_data,
        })
    except Exception as e:
        logger.error(f"predict_tomorrow_compare_api error: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({
            "success": False,
            "error": str(e),
        }, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def daily_predictions_api(request):
    """
    API: Trả về danh sách dự đoán theo ngày đã lưu trong DB (DailyPrediction).

    Query params:
    - limit: số bản ghi (default 30, max 365)
    - since: YYYY-MM-DD — chỉ trả các mốc date >= since (lọc bản demo / dữ liệu rất cũ)

    Response:
    {
      "success": true,
      "data": [
        {"date":"YYYY-MM-DD","rf":1.2,"xgb":2.3,"et":1.8},
        ...
      ]
    }
    """
    try:
        try:
            limit = int(request.GET.get("limit", 30))
        except ValueError:
            limit = 30
        limit = max(1, min(limit, 365))

        qs = DailyPrediction.objects.all()
        since_raw = (request.GET.get("since") or "").strip()
        if since_raw:
            try:
                since_d = _dt.strptime(since_raw, "%Y-%m-%d").date()
                qs = qs.filter(date__gte=since_d)
            except ValueError:
                pass

        # Lấy các mốc dự đoán *gần đây nhất* (theo ngày dự báo), không phải N bản ghi cũ nhất trong DB.
        preds = list(qs.order_by("-date")[:limit])
        preds.sort(key=lambda p: p.date)

        rows = []
        for p in preds:
            rows.append({
                "date": p.date.strftime("%Y-%m-%d"),
                "rf": float(p.rf_pred),
                "xgb": float(p.xgb_pred),
                # ExtraTrees expected mm được lưu trong lr_pred (tương thích schema cũ).
                "et": float(p.lr_pred or 0.0),
            })

        return JsonResponse({"success": True, "data": rows})
    except Exception as e:
        logger.error(f"daily_predictions_api error: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({"success": False, "error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST", "GET"])
def predict_daily_api(request):
    """
    API endpoint: Dự đoán mưa theo ngày
    
    POST: Dự đoán với data từ request
    GET: Test endpoint
    
    Request format:
    {
        "year": 2023,
        "month": 5,
        "day": 15
    }
    
    Response format:
    {
        "success": true,
        "data": {
            "year": 2023,
            "month": 5,
            "day": 15,
            "date": "2023-05-15",
            "has_rain": true,
            "rain_probability": 0.753,
            "predicted_rainfall": 45.60,
            "classifier_model": "XGBoost",
            "regressor_model": "XGBoost"
        },
        "message": "Dự đoán thành công"
    }
    """
    
    try:
        if request.method == 'GET':
            return JsonResponse({
                'success': True,
                'message': 'API dự đoán mưa theo ngày sẵn sàng',
                'usage': 'POST với format: {"year": 2023, "month": 5, "day": 15}'
            })
        
        # POST request
        data = json.loads(request.body)
        year = int(data.get('year'))
        month = int(data.get('month'))
        day = int(data.get('day'))
        
        # Validate input
        if not (1979 <= year <= 2100):
            return JsonResponse({
                'success': False,
                'message': 'Năm phải từ 1979-2100'
            }, status=400)
        
        if not (1 <= month <= 12):
            return JsonResponse({
                'success': False,
                'message': 'Tháng phải từ 1-12'
            }, status=400)
        
        if not (1 <= day <= 31):
            return JsonResponse({
                'success': False,
                'message': 'Ngày phải từ 1-31'
            }, status=400)
        
        # Dự đoán - ưu tiên mô hình 2 giai đoạn (đã train sẵn)
        csv_path = os.path.join(DATA_DIR, 'daily_combined.csv')
        
        if not os.path.exists(csv_path):
            return JsonResponse({
                'success': False,
                'message': 'Dữ liệu huấn luyện không tìm thấy'
            }, status=500)
        
        path_two_stage = os.path.join(MODELS_DIR, 'daily_two_stage_gradient_boosting.pkl')
        if os.path.exists(path_two_stage):
            has_rain, amount_mm, metrics = predict_rainfall_daily_two_stage(
                path_two_stage, year, month, day, csv_path
            )
            result = {
                'year': int(year), 'month': int(month), 'day': int(day),
                'has_rain': bool(has_rain),
                'rain_probability': float(metrics.get('rain_probability', 1.0 if has_rain else 0.0)),
                'predicted_rainfall': float(amount_mm),
                'classifier_model': 'Gradient Boosting',
                'regressor_model': 'Gradient Boosting'
            }
        else:
            try:
                result = predict_with_best_model(
                    csv_path=csv_path,
                    year=year,
                    month=month,
                    day=day,
                    models_dir=MODELS_DIR
                )
            except Exception as e:
                return JsonResponse({
                    'success': False,
                    'message': f'Chưa có mô hình. Chạy: cd DuBao/src && python train_daily_two_stage.py. Lỗi: {str(e)}'
                }, status=500)
        
        # Thêm ngày dự đoán
        result['date'] = f"{year:04d}-{month:02d}-{day:02d}"
        result['rain_probability'] = round(result.get('rain_probability', 0), 4)
        result['predicted_rainfall'] = round(result.get('predicted_rainfall', 0), 2)
        
        return JsonResponse({
            'success': True,
            'data': result,
            'message': 'Dự đoán thành công'
        })
    
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'message': 'Invalid JSON format'
        }, status=400)
    
    except ValueError as e:
        return JsonResponse({
            'success': False,
            'message': f'Giá trị không hợp lệ: {str(e)}'
        }, status=400)
    
    except FileNotFoundError as e:
        return JsonResponse({
            'success': False,
            'message': 'Mô hình chưa được train. Chạy: python main.py --compare'
        }, status=500)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return JsonResponse({
            'success': False,
            'message': f'Lỗi dự đoán: {str(e)}'
        }, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def predict_range_api(request):
    """
    API: Dự đoán mưa cho một khoảng ngày
    
    Query params:
    - year: 2023
    - month: 5
    - start_day: 1
    - num_days: 10 (dự đoán 10 ngày)
    
    Response: Array of predictions
    """
    try:
        try:
            year = int(request.GET.get('year', 2023))
            month = int(request.GET.get('month', 1))
            # support both start_day and day for compatibility
            start_day = int(request.GET.get('start_day', request.GET.get('day', 1)))
            num_days = int(request.GET.get('num_days', 10))
        except ValueError as ve:
            # invalid numeric parameter
            return JsonResponse({
                'success': False,
                'message': f'Thành phần truy vấn không hợp lệ: {ve}'
            }, status=400)
        
        # Validate
        if not (1 <= month <= 12) or not (1 <= start_day <= 31):
            return JsonResponse({
                'success': False,
                'message': 'Tháng hoặc ngày không hợp lệ'
            }, status=400)
        
        if num_days > 31:
            num_days = 31
        
        results = []
        csv_path = os.path.join(DATA_DIR, 'daily_combined.csv')
        
        use_two_stage = os.path.exists(os.path.join(MODELS_DIR, 'daily_two_stage_gradient_boosting.pkl'))
        for day in range(start_day, start_day + num_days):
            if day > 31:
                break
            try:
                if use_two_stage:
                    has_rain, amount_mm, metrics = predict_rainfall_daily_two_stage(
                        os.path.join(MODELS_DIR, 'daily_two_stage_gradient_boosting.pkl'),
                        year, month, day, csv_path
                    )
                    result = {
                        'has_rain': has_rain,
                        'rain_probability': metrics.get('rain_probability', 1.0 if has_rain else 0.0),
                        'predicted_rainfall': amount_mm
                    }
                else:
                    result = predict_with_best_model(
                        csv_path=csv_path, year=year, month=month, day=day, models_dir=MODELS_DIR
                    )
                result['date'] = f"{year:04d}-{month:02d}-{day:02d}"
                results.append(result)
            except Exception:
                continue
        
        return JsonResponse({
            'success': True,
            'data': results,
            'count': len(results),
            'message': f'Dự đoán {len(results)} ngày thành công'
        })
    
    except Exception as e:
        logger.error(f"Range API error: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def model_info_api(request):
    """
    API: Lấy thông tin mô hình đã train
    """
    try:
        comparison_path = os.path.join(MODELS_DIR, 'comparison_results.pkl')
        
        if not os.path.exists(comparison_path):
            return JsonResponse({
                'success': False,
                'models': [],
                'message': 'Chưa train mô hình. Chạy: python main.py --compare'
            })
        
        with open(comparison_path, 'rb') as f:
            comparison = pickle.load(f)
        
        return JsonResponse({
            'success': True,
            'best_classifier': comparison['best_classifier'],
            'best_regressor': comparison['best_regressor'],
            'available_classifiers': list(comparison['classifier_results'].keys()),
            'available_regressors': list(comparison['regressor_results'].keys()),
            'message': 'Thông tin mô hình'
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def model_metrics_api(request):
    """
    API: Lấy metrics của tất cả mô hình
    """
    try:
        # NOTE:
        # Tránh unpickle các file .pkl cũ vì dễ vỡ do lệch phiên bản numpy/sklearn.
        # Thay vào đó, ưu tiên đọc metrics từ model_metrics.json (được train script tạo ra).

        # Load dữ liệu để tính test set info (không cần pickle)
        csv_path = os.path.join(DATA_DIR, 'daily_combined.csv')
        df = pd.read_csv(csv_path)
        split_idx = int(len(df) * 0.8)
        test_size = len(df) - split_idx
        rain_count = int(((df.iloc[split_idx:]['rainfall'] > 0).sum()) if 'rainfall' in df.columns else 0)

        classifier_metrics = {}
        regressor_metrics = {}

        metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    mj = json.load(f) or {}
            except Exception:
                mj = {}

            # keys example: daily_two_stage_gradient_boosting, daily_two_stage_random_forest, daily_two_stage_extra_trees
            for key, m in mj.items():
                if not str(key).startswith("daily_two_stage_"):
                    continue
                name = str(key).replace("daily_two_stage_", "").replace("_", " ").title()
                cls_acc = m.get("cls_accuracy")
                cls_f1 = m.get("cls_f1")
                reg_mae = m.get("reg_mae")
                reg_rmse = m.get("reg_rmse")
                reg_r2 = m.get("reg_r2")

                classifier_metrics[name] = {
                    "accuracy": float(cls_acc) if cls_acc is not None else None,
                    "precision": None,
                    "recall": None,
                    "f1": float(cls_f1) if cls_f1 is not None else None,
                }
                regressor_metrics[name] = {
                    "mae": float(reg_mae) if reg_mae is not None else None,
                    "rmse": float(reg_rmse) if reg_rmse is not None else None,
                    "r2": float(reg_r2) if reg_r2 is not None else None,
                    "mape": None,
                }
        
        return JsonResponse({
            'success': True,
            'data': {
                'classifier': classifier_metrics,
                'regressor': regressor_metrics,
                'test_set_info': {
                    'test_count': int(test_size),
                    'rain_count': int(rain_count)
                }
            },
            'message': 'Model metrics'
        })
    
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def predict_compare_models_api(request):
    """
    API endpoint: Dự đoán với 3 mô hình (Gradient Boosting, Random Forest, XGBoost) và so sánh
    
    Request format:
    {
        "year": 2023,
        "month": 5,
        "day": 15
    }
    
    Response format:
    {
        "success": true,
        "date": "2023-05-15",
        "models": [
            {"model": "GradientBoosting", "has_rain": true, "rain_probability": 0.75, ...},
            ...
        ],
        "consensus": {"has_rain": true, ...}
    }
    """
    import traceback
    try:
        # Parse request
        if request.body:
            data = json.loads(request.body)
        else:
            logger.warning("Empty request body")
            return JsonResponse({
                'success': False,
                'error': 'Request body trống'
            }, status=400)
            
        year = int(data.get('year', 0))
        month = int(data.get('month', 0))
        day = int(data.get('day', 0))
        
        # Validate
        if not (1979 <= year <= 2100):
            logger.warning(f"Invalid year: {year}")
            return JsonResponse({
                'success': False,
                'error': 'Năm phải từ 1979-2100'
            }, status=400)
        if not (1 <= month <= 12):
            logger.warning(f"Invalid month: {month}")
            return JsonResponse({
                'success': False,
                'error': 'Tháng phải từ 1-12'
            }, status=400)
        if not (1 <= day <= 31):
            logger.warning(f"Invalid day: {day}")
            return JsonResponse({
                'success': False,
                'error': 'Ngày phải từ 1-31'
            }, status=400)
        
        csv_path = os.path.join(DATA_DIR, 'daily_combined.csv')
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found at {csv_path}")
            return JsonResponse({
                'success': False,
                'error': f'Dữ liệu huấn luyện không tìm thấy: {csv_path}'
            }, status=500)
        
        # Dự đoán với 3 mô hình 2 giai đoạn: Gradient Boosting, Random Forest, Extra Trees
        model_configs = [
            ('gradient_boosting', 'Gradient Boosting'),
            ('random_forest', 'Random Forest'),
            ('extra_trees', 'Extra Trees'),
        ]
        predictions = {}
        for key, label in model_configs:
            path = os.path.join(MODELS_DIR, f'daily_two_stage_{key}.pkl')
            if not os.path.exists(path):
                predictions[label] = {
                    'has_rain': False, 'rain_probability': 0, 'predicted_rainfall': 0,
                    'mae': 0, 'rmse': 0, 'r2_score': 0, 'metrics': {}
                }
                continue
            try:
                has_rain, amount_mm, metrics = predict_rainfall_daily_two_stage(
                    path, year, month, day, csv_path
                )
                rain_prob = metrics.get('rain_probability', 1.0 if has_rain else 0.0)
                predictions[label] = {
                    'has_rain': has_rain,
                    'rain_probability': rain_prob,
                    'predicted_rainfall': amount_mm,
                    'mae': metrics.get('reg_mae', 0),
                    'rmse': metrics.get('reg_rmse', 0),
                    'r2_score': metrics.get('reg_r2', 0),
                    'metrics': metrics,
                }
            except Exception as e:
                logger.warning(f"Model {label} error: {e}")
                predictions[label] = {
                    'has_rain': False, 'rain_probability': 0, 'predicted_rainfall': 0,
                    'mae': 0, 'rmse': 0, 'r2_score': 0, 'metrics': {}, 'error': str(e)
                }
        
        # Format results for frontend
        models_data = []
        total_rain_prob = 0
        total_rainfall = 0
        rain_count = 0
        
        for model_name, pred in predictions.items():
            models_data.append({
                'model': model_name,
                'has_rain': bool(pred.get('has_rain', False)),
                'rain_probability': round(float(pred.get('rain_probability', 0)), 4),
                'predicted_rainfall': round(float(pred.get('predicted_rainfall', 0)), 2),
                'mae': round(float(pred.get('mae', 0)), 4),
                'rmse': round(float(pred.get('rmse', 0)), 4),
                'r2_score': round(float(pred.get('r2_score', 0)), 4)
            })
            total_rain_prob += pred.get('rain_probability', 0)
            total_rainfall += pred.get('predicted_rainfall', 0)
            if pred.get('has_rain'):
                rain_count += 1
        
        # Calculate consensus
        avg_models = len(models_data) if models_data else 1
        consensus = {
            'has_rain': rain_count > avg_models // 2 if avg_models > 0 else False,
            'avg_rain_probability': round(total_rain_prob / avg_models, 4) if avg_models > 0 else 0,
            'avg_rainfall': round(total_rainfall / avg_models, 2) if avg_models > 0 else 0,
            'agreement_count': rain_count
        }
        
        # Chart data for comparison
        chart_data = {
            'labels': [m['model'] for m in models_data],
            'cls_accuracy': [round(predictions.get(m['model'], {}).get('metrics', {}).get('cls_accuracy', 0) * 100, 1) for m in models_data],
            'cls_f1': [round(predictions.get(m['model'], {}).get('metrics', {}).get('cls_f1', 0) * 100, 1) for m in models_data],
            'reg_r2': [round(predictions.get(m['model'], {}).get('metrics', {}).get('reg_r2', 0) * 100, 1) for m in models_data],
            'reg_mae': [round(predictions.get(m['model'], {}).get('metrics', {}).get('reg_mae', 0), 2) for m in models_data],
        }
        
        logger.info(f"Prediction successful for {year}-{month:02d}-{day:02d}")
        return JsonResponse({
            'success': True,
            'date': f"{year:04d}-{month:02d}-{day:02d}",
            'models': models_data,
            'consensus': consensus,
            'chart_data': chart_data
        })
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({
            'success': False,
            'error': f'JSON format lỗi: {str(e)}'
        }, status=400)
    
    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({
            'success': False,
            'error': f'Giá trị không hợp lệ: {str(e)}'
        }, status=400)
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({
            'success': False,
            'error': f'File không tìm thấy: {str(e)}'
        }, status=500)
    
    except Exception as e:
        logger.error(f"Prediction compare error: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({
            'success': False,
            'error': f'Lỗi dự đoán: {str(e)}'
        }, status=500)
