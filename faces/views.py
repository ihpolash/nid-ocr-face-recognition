from django.shortcuts import render

# Create your views here.
from rest_framework import permissions, serializers, status
from rest_framework.generics import GenericAPIView

# Create your views here.
from rest_framework.response import Response
import client
from rest_framework.throttling import UserRateThrottle, AnonRateThrottle


class FaceEnrollmentSerializer(serializers.Serializer):
    image = serializers.ImageField()
    username = serializers.CharField()

class FaceDetectSerializer(serializers.Serializer):
    image = serializers.ImageField()

class IDOCRSerializer(serializers.Serializer):
    image = serializers.ImageField()



class FaceEnrollmentView(GenericAPIView):
    permission_classes = [permissions.AllowAny]
    serializer_class = FaceEnrollmentSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        image = serializer.validated_data.get('image')
        username = serializer.validated_data.get('username')
        response = client.face_enrollment(image, username)
        return Response(response, status=status.HTTP_200_OK)


class FaceDetectView(GenericAPIView):
    permission_classes = [permissions.AllowAny]
    serializer_class = FaceDetectSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        image = serializer.validated_data.get('image')
        response = client.face_detect(image)
        return Response(response, status=status.HTTP_200_OK)

class IDOCRView(GenericAPIView):
    permission_classes = [permissions.AllowAny]
    serializer_class = IDOCRSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        image = serializer.validated_data.get('image')
        response = client.card_ocr(image)
        return Response(response, status=status.HTTP_200_OK)