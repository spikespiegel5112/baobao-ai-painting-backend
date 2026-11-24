from django.urls import path

from . import views
from .controllers import CommonController, MBartController, InpaintController

urlpatterns = [
    path('hello/', views.hello),
    path('testJson/', views.test_json, name='testJson'),
    path('upload/', CommonController.upload, name='upload'),
    path('upload2/', CommonController.upload2, name='upload2'),
    path('addQueue/', CommonController.handle_add_queue, name='addQueue'),
    path('addQueueInpaint/', CommonController.handle_add_queue_inpaint, name='addQueue'),
    path('checkQueueByUserId/', CommonController.handle_check_queue_by_user_id, name='checkQueueByUserId'),
    path('deleteQueue/', CommonController.delete_queue, name='deleteQueue'),
    path('translator/', MBartController.translator, name='translator'),
    path('inpaint/', InpaintController.handle_inpaint, name='inpaint'),
    path('getGenerationStatusSDXLInpaint/', InpaintController.get_generation_status_sdxl_inpaint, name='getGenerationStatusSDXLInpaint'),
]
