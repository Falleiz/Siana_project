from django.urls import path
from . import views
urlpatterns = [
   path('', views.login_view, name='login'),
    path('technician/logout/', views.logout_view, name='logout'),
    path('technician/home/', views.home_view, name='home'),
    path('technician/get-component-3d/<int:component_id>/', views.get_component_3d, name='get_component_3d'),
     path('technician/component-sheet/<int:component_id>/', views.component_technical_sheet, name='component_sheet'),
    path('technician/get-component-3d/<int:component_id>/', views.get_component_3d, name='get_component_3d'),
    path("technician/upload-and-search/", views.upload_and_search, name="upload_and_search"),
]
