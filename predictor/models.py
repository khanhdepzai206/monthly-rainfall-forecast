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
