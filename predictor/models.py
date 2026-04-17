from django.db import models
from django.contrib.auth.models import User

class RainfallPrediction(models.Model):
    """
    Lịch sử dự đoán: theo tháng (day=None) hoặc theo ngày (day có giá trị).
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    year = models.IntegerField()
    month = models.IntegerField()
    day = models.IntegerField(null=True, blank=True, help_text="Ngày trong tháng; null = dự đoán theo tháng")
    predicted_rainfall = models.FloatField()
    historical_avg = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        if self.day is not None:
            return f"Ngày {self.day}/{self.month}/{self.year} - {self.predicted_rainfall:.2f} mm"
        return f"Tháng {self.month}/{self.year} - {self.predicted_rainfall:.2f} mm"


class DailyPrediction(models.Model):
    """
    Dự đoán hàng ngày từ 3 mô hình ML.
    """
    date = models.DateField(unique=True, help_text="Ngày dự đoán (ngày mai)")
    rf_pred = models.FloatField(help_text="Dự đoán từ RandomForest (mm)")
    lr_pred = models.FloatField(help_text="Dự đoán từ LinearRegression (mm)")
    xgb_pred = models.FloatField(help_text="Dự đoán từ XGBoost (mm)")
    best_model = models.CharField(max_length=10, null=True, blank=True, help_text="Mô hình tốt nhất")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-date']

    def __str__(self):
        return f"Dự đoán {self.date} - RF:{self.rf_pred:.3f}, LR:{self.lr_pred:.3f}, XGB:{self.xgb_pred:.3f} mm"


class ActualRainfall(models.Model):
    """
    Lượng mưa thực tế để so sánh với dự đoán.
    """
    date = models.DateField(unique=True, help_text="Ngày có lượng mưa thực tế")
    actual_rainfall = models.FloatField(help_text="Lượng mưa thực tế (mm)")
    created_at = models.DateTimeField(auto_now_add=True)

    # Liên kết với prediction (nếu có)
    prediction = models.OneToOneField(DailyPrediction, null=True, blank=True, on_delete=models.SET_NULL,
                                     related_name='actual')

    # Sai số của từng mô hình
    rf_error = models.FloatField(null=True, blank=True, help_text="Sai số của RandomForest")
    lr_error = models.FloatField(null=True, blank=True, help_text="Sai số của LinearRegression")
    xgb_error = models.FloatField(null=True, blank=True, help_text="Sai số của XGBoost")

    # Cờ retrain
    retrained = models.BooleanField(default=False, help_text="Đã retrain models chưa")

    class Meta:
        ordering = ['-date']

    def __str__(self):
        return f"Actual {self.date}: {self.actual_rainfall:.3f} mm"

    def evaluate_error(self):
        """Tính và lưu lại sai số của từng mô hình."""
        if not self.prediction:
            return None

        self.rf_error = abs(self.prediction.rf_pred - self.actual_rainfall)
        self.lr_error = abs(self.prediction.lr_pred - self.actual_rainfall)
        self.xgb_error = abs(self.prediction.xgb_pred - self.actual_rainfall)
        self.save()

        return {
            'rf_error': self.rf_error,
            'lr_error': self.lr_error,
            'xgb_error': self.xgb_error,
        }

    # Backward-compatible alias (older views call this name)
    def calculate_errors(self):
        return self.evaluate_error()

    def needs_retrain(self, abs_threshold=0.15, relative_threshold=0.4):
        """Xác định retrain theo ngưỡng mới cho actual thấp và actual lớn."""
        if not self.prediction or any(err is None for err in [self.rf_error, self.lr_error, self.xgb_error]):
            return False

        if self.actual_rainfall is None:
            return False

        if self.actual_rainfall < 0.5:
            return any(err > abs_threshold for err in [self.rf_error, self.lr_error, self.xgb_error])

        return any((err / self.actual_rainfall) > relative_threshold
                   for err in [self.rf_error, self.lr_error, self.xgb_error])

    def check_retrain_threshold(self, abs_threshold=0.15, threshold_percent=40):
        """Kiểm tra xem có cần retrain không theo thresholds mới."""
        return self.needs_retrain(abs_threshold=abs_threshold, relative_threshold=threshold_percent / 100.0)

    def get_error_percentages(self):
        """Trả về phần trăm sai số của từng mô hình nếu actual đủ lớn."""
        if not self.prediction or self.actual_rainfall is None or self.actual_rainfall == 0:
            return None

        if self.actual_rainfall < 0.5:
            return None

        return {
            'rf_pct': (self.rf_error / self.actual_rainfall) * 100 if self.rf_error is not None else None,
            'lr_pct': (self.lr_error / self.actual_rainfall) * 100 if self.lr_error is not None else None,
            'xgb_pct': (self.xgb_error / self.actual_rainfall) * 100 if self.xgb_error is not None else None,
        }

    def retrain_summary(self, abs_threshold=0.15, relative_threshold=0.4):
        """Tổng hợp thông tin retrain theo quy tắc mới."""
        if any(err is None for err in [self.rf_error, self.lr_error, self.xgb_error]):
            return {
                'needs_retrain': False,
                'max_pct': None,
                'details': 'Chưa có đủ dữ liệu để đánh giá.'
            }

        needs_retrain = self.needs_retrain(abs_threshold=abs_threshold, relative_threshold=relative_threshold)
        details = []
        if self.actual_rainfall < 0.5:
            details.append(f'Actual nhỏ (<0.5mm), ngưỡng retrain tuyệt đối={abs_threshold:.2f}mm')
            details.append(f'RF={self.rf_error:.3f}mm, XGB={self.xgb_error:.3f}mm, LR={self.lr_error:.3f}mm')
            max_pct = max(self.rf_error, self.lr_error, self.xgb_error)
        else:
            errors_pct = self.get_error_percentages() or {}
            details.append(f'Actual≥0.5mm, ngưỡng retrain tỷ lệ={relative_threshold * 100:.0f}%')
            details.append(
                f"RF={errors_pct.get('rf_pct', 0):.1f}%, XGB={errors_pct.get('xgb_pct', 0):.1f}%, LR={errors_pct.get('lr_pct', 0):.1f}%"
            )
            max_pct = max(errors_pct.values()) if errors_pct else None

        return {
            'needs_retrain': needs_retrain,
            'max_pct': max_pct,
            'details': ' | '.join(details)
        }

    @property
    def rf_error_pct(self):
        if self.actual_rainfall and self.actual_rainfall > 0.5 and self.rf_error is not None:
            try:
                return (self.rf_error / self.actual_rainfall) * 100
            except ZeroDivisionError:
                return None
        return None

    @property
    def xgb_error_pct(self):
        if self.actual_rainfall and self.actual_rainfall > 0.5 and self.xgb_error is not None:
            try:
                return (self.xgb_error / self.actual_rainfall) * 100
            except ZeroDivisionError:
                return None
        return None

    @property
    def lr_error_pct(self):
        if self.actual_rainfall and self.actual_rainfall > 0.5 and self.lr_error is not None:
            try:
                return (self.lr_error / self.actual_rainfall) * 100
            except ZeroDivisionError:
                return None
        return None

