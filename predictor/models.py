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
        return f"Dự đoán {self.date} - RF:{self.rf_pred:.1f}, LR:{self.lr_pred:.1f}, XGB:{self.xgb_pred:.1f} mm"


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
        return f"Actual {self.date}: {self.actual_rainfall:.1f} mm"

    def calculate_errors(self):
        """Tính sai số cho từng mô hình."""
        if not self.prediction:
            return

        self.rf_error = abs(self.prediction.rf_pred - self.actual_rainfall)
        self.lr_error = abs(self.prediction.lr_pred - self.actual_rainfall)
        self.xgb_error = abs(self.prediction.xgb_pred - self.actual_rainfall)
        self.save()

    def check_retrain_threshold(self, threshold_percent=20, abs_threshold=0.5):
        """Kiểm tra xem có cần retrain không."""
        if any(err is None for err in [self.rf_error, self.lr_error, self.xgb_error]):
            return False

        # Nếu actual thấp (dưới 0.5mm), dùng ngưỡng tuyệt đối để tránh % quá lớn
        if self.actual_rainfall is None or self.actual_rainfall <= 0.5:
            return any(err > abs_threshold for err in [self.rf_error, self.lr_error, self.xgb_error])

        # Tính sai số phần trăm (dựa trên actual rainfall)
        rf_error_pct = (self.rf_error / self.actual_rainfall) * 100
        lr_error_pct = (self.lr_error / self.actual_rainfall) * 100
        xgb_error_pct = (self.xgb_error / self.actual_rainfall) * 100

        return any(error_pct > threshold_percent
                   for error_pct in [rf_error_pct, lr_error_pct, xgb_error_pct])

    def get_error_percentages(self):
        """Trả về phần trăm sai số của từng mô hình."""
        if not self.prediction or self.actual_rainfall is None or self.actual_rainfall == 0:
            return None

        return {
            'rf_pct': (self.rf_error / self.actual_rainfall) * 100 if self.rf_error is not None else None,
            'lr_pct': (self.lr_error / self.actual_rainfall) * 100 if self.lr_error is not None else None,
            'xgb_pct': (self.xgb_error / self.actual_rainfall) * 100 if self.xgb_error is not None else None,
        }

    def retrain_summary(self, threshold_percent=20):
        """Tinh tổng quan retrain: có cần retrain không và thông tin chi tiết."""
        errors = self.get_error_percentages()
        if not errors:
            return {
                'needs_retrain': False,
                'max_pct': None,
                'details': 'Chưa có đủ dữ liệu để đánh giá.'
            }

        max_pct = max(errors.values())
        needs_retrain = max_pct > threshold_percent
        details = (
            f"RF: {errors['rf_pct']:.1f}%, XGB: {errors['xgb_pct']:.1f}%, LR: {errors['lr_pct']:.1f}%")

        return {
            'needs_retrain': needs_retrain,
            'max_pct': max_pct,
            'details': details
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

