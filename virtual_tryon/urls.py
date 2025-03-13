from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('products/', views.product_list, name='product_list'),
    path('products/<int:product_id>/', views.product_detail, name='product_detail'),
    path('try-on/<int:product_id>/', views.try_on_page, name='try_on'),
    path('process-try-on/<int:product_id>/', views.process_try_on, name='process_try_on'),
]