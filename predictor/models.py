from django.db import models
from django.contrib.auth.models import User

class RainfallPrediction(models.Model):
    """
    Model để lưu trữ lịch sử dự đoán
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    year = models.IntegerField()
    month = models.IntegerField()
    predicted_rainfall = models.FloatField()
    historical_avg = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Prediction {self.month}/{self.year} - {self.predicted_rainfall:.2f}mm"
