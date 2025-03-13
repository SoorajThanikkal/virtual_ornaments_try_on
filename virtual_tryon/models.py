from django.db import models

class OrnamentProduct(models.Model):
    ORNAMENT_TYPES = (
        ('glasses', 'Glasses'),
        ('necklace', 'Necklace'),
        ('earrings', 'Earrings'),
    )
    
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    ornament_type = models.CharField(max_length=20, choices=ORNAMENT_TYPES)
    image = models.ImageField(upload_to='ornaments/')
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.name} ({self.get_ornament_type_display()})"