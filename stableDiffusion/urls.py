"""Top-level URL dispatcher."""

from django.conf import settings
from django.contrib import admin
from django.urls import include, path
from django.views.static import serve

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('stableDiffusion.urls_core')),
    path('', include('stableDiffusion.urls_generation')),
    path('', include('stableDiffusion.urls_management')),
    path('media/<path:path>', serve, {'document_root': settings.MEDIA_ROOT}),
]
