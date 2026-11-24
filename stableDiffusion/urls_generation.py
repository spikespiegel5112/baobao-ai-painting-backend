from django.urls import path

from .controllers import FluxController, StableDiffusion3Controller, StableDiffusionXLController

urlpatterns = [
    path('txt2ImgSDXL/', StableDiffusionXLController.handle_txt_2_img, name='txt2Img'),
    path('triggerTxt2ImgSDXLByQueue/', StableDiffusionXLController.trigger_txt_2_img_by_queue, name='triggerTxt2ImgSDXLByQueue'),
    path('getLastImageDataSDXL/', StableDiffusionXLController.get_last_image_data, name='getLastImageDataSDXL'),
    path('getCancellingProgressSDXL/', StableDiffusionXLController.get_cancelling_progress, name='getCancellingProgressSDXL'),
    path('getGenerationStatusSDXL/', StableDiffusionXLController.get_generation_status, name='getGenerationStatusSDXL'),
    path('cancelGenerationSDXL/', StableDiffusionXLController.cancel_generation, name='cancelGenerationSDXL'),
    path('txt2ImgSD3/', StableDiffusion3Controller.handle_txt_2_img, name='txt2Img'),
    path('triggerTxt2ImgSD3ByQueue/', StableDiffusion3Controller.trigger_txt_2_img_by_queue, name='triggerTxt2ImgSD3ByQueue'),
    path('getLastImageDataSD3/', StableDiffusion3Controller.get_last_image_data, name='getLastImageDataSD3'),
    path('getCancellingProgressSD3/', StableDiffusion3Controller.get_cancelling_progress, name='getCancellingProgressSD3'),
    path('getGenerationStatusSD3/', StableDiffusion3Controller.get_generation_status, name='getGenerationStatusSD3'),
    path('cancelGenerationSD3/', StableDiffusion3Controller.cancel_generation, name='cancelGenerationSD3'),
    path('txt2ImgFlux/', FluxController.handle_txt_2_img, name='txt2Img'),
    path('triggerTxt2ImgFluxByQueue/', FluxController.trigger_txt_2_img_by_queue, name='triggerTxt2ImgFluxByQueue'),
    path('getLastImageDataFlux/', FluxController.get_last_image_data, name='getLastImageDataFlux'),
    path('getCancellingProgressFlux/', FluxController.get_cancelling_progress, name='getCancellingProgressFlux'),
    path('getGenerationStatusFlux/', FluxController.get_generation_status, name='getGenerationStatusFlux'),
    path('cancelGenerationFlux/', FluxController.cancel_generation, name='cancelGenerationFlux'),
]
