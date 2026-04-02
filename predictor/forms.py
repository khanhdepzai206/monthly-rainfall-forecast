from django import forms
from .models import ActualRainfall

class ActualRainfallForm(forms.ModelForm):
    """Form để nhập lượng mưa thực tế."""

    class Meta:
        model = ActualRainfall
        fields = ['date', 'actual_rainfall']
        widgets = {
            'date': forms.DateInput(attrs={
                'type': 'date',
                'class': 'form-control',
                'required': True
            }),
            'actual_rainfall': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': '0.1',
                'min': '0',
                'placeholder': '0.0',
                'required': True
            })
        }
        labels = {
            'date': 'Ngày',
            'actual_rainfall': 'Lượng mưa thực tế (mm)'
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set default date to yesterday
        from datetime import datetime, timedelta
        yesterday = datetime.now().date() - timedelta(days=1)
        self.fields['date'].initial = yesterday