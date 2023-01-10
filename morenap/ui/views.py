from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.template import loader
from .forms import *
from django.conf import settings


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

def test(request):
    template = loader.get_template("test.html")
    return HttpResponse(template.render())
 
def hotel_image_view(request):
    if request.method == 'POST':
        form = HotelForm(request.POST, request.FILES)
 
        if form.is_valid():
            form.save()
            return redirect('success')
    else:
        form = HotelForm()
    return render(request, 'hotel_image_form.html', {'form': form})
 
 
def success(request):
    return HttpResponse('successfully uploaded')

def importapp(request):
    return render(request, 'importapp.html')