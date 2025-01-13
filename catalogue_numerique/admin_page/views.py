from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import JsonResponse
from .models import Component
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from django.contrib import messages
from .models import admin_user,Component,Component_details_technique,Component_description_technique_paragraphe,paragraphe_images,Component_model3D,Component_video,Component_document
from technician_page.models import technician_user  # Ajoutez cette importation


def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        try:
            user = admin_user.objects.get(email=email)
            if password == user.password:
                request.session['user_id'] = user.id
                request.session['username'] = user.nom_prenom
                return redirect('admin_home')
            else:
                messages.error(request, "Mot de passe incorrect.")
        except admin_user.DoesNotExist:
            messages.error(request, "Utilisateur introuvable.")

    return render(request, 'admin_page/login.html')




def logout_view(request):
    request.session.flush()  # Supprime toutes les données de session
    messages.success(request, "Vous êtes déconnecté.")
    return redirect('admin_login')  # Redirigez vers la page de connexion


def home_view(request):
    root_components = Component.objects.filter(parent=None)
    users = technician_user.objects.all()
    return render(request, "admin_page/home.html", {
        "components": root_components,
        "users": users
    })



@login_required
@csrf_exempt
@require_http_methods(["POST"])
def add_user(request):
    try:
        data = json.loads(request.body)
        nom_prenom = data.get('nom_prenom')
        email = data.get('email')
        password = data.get('password')
        
        if not all([nom_prenom, email, password]):
            return JsonResponse({'error': 'Tous les champs sont requis'}, status=400)
        
        # Vérifier si l'email existe déjà
        if technician_user.objects.filter(email=email).exists():
            return JsonResponse({'error': 'Cet email existe déjà'}, status=400)
            
        user = technician_user.objects.create(
            nom_prenom=nom_prenom,
            email=email,
            password=password  # Note: Dans un vrai projet, il faudrait hasher le mot de passe
        )
        
        return JsonResponse({
            'id': user.id,
            'nom_prenom': user.nom_prenom,
            'email': user.email
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def edit_user(request, user_id):
    try:
        data = json.loads(request.body)
        user = technician_user.objects.get(id=user_id)
        
        if 'nom_prenom' in data:
            user.nom_prenom = data['nom_prenom']
        if 'email' in data:
            if technician_user.objects.exclude(id=user_id).filter(email=data['email']).exists():
                return JsonResponse({'error': 'Cet email existe déjà'}, status=400)
            user.email = data['email']
        if 'password' in data:
            user.password = data['password']
            
        user.save()
        return JsonResponse({'success': True})
    except technician_user.DoesNotExist:
        return JsonResponse({'error': 'Utilisateur non trouvé'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

@login_required
@csrf_exempt
@require_http_methods(["DELETE"])
def delete_user(request, user_id):
    try:
        user = technician_user.objects.get(id=user_id)
        user.delete()
        return JsonResponse({'success': True})
    except technician_user.DoesNotExist:
        return JsonResponse({'error': 'Utilisateur non trouvé'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

def get_users(request):
    users = technician_user.objects.all()
    return JsonResponse({
        'users': list(users.values('id', 'nom_prenom', 'email'))
    })

# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json


@login_required 
@csrf_exempt
@require_http_methods(["POST"])
def add_component(request):
    print("=== Début add_component ===")
    print("Méthode:", request.method)
    print("Headers:", request.headers)
    print("Body:", request.body)
    
    try:
        if not request.body:
            return JsonResponse({'error': 'Corps de requête vide'}, status=400)
            
        data = json.loads(request.body)
        print("Données JSON décodées:", data)
        
        name = data.get('name')
        if not name:
            return JsonResponse({'error': 'Le nom est requis'}, status=400)
            
        parent_id = data.get('parent_id')
        print(f"Création composant: nom={name}, parent_id={parent_id}")
        
        try:
            if parent_id:
                parent = Component.objects.get(id=parent_id)
                component = Component.objects.create(name=name, parent=parent)
            else:
                component = Component.objects.create(name=name)
                
            response_data = {
                'id': component.id,
                'name': component.name,
                'parent_id': component.parent.id if component.parent else None
            }
            print("Composant créé avec succès:", response_data)
            return JsonResponse(response_data)
            
        except Component.DoesNotExist:
            return JsonResponse({'error': f'Parent avec id={parent_id} non trouvé'}, status=404)
        except Exception as e:
            print("Erreur lors de la création:", str(e))
            return JsonResponse({'error': str(e)}, status=500)
            
    except json.JSONDecodeError as e:
        print("Erreur décodage JSON:", str(e))
        return JsonResponse({'error': f'JSON invalide: {str(e)}'}, status=400)
    except Exception as e:
        print("Erreur générale:", str(e))
        return JsonResponse({'error': str(e)}, status=500)
 
    
    
@csrf_exempt
@require_http_methods(["POST"])
@login_required 
def edit_component(request, component_id):
    try:
        data = json.loads(request.body)
        component = Component.objects.get(id=component_id)
        component.name = data.get('name')
        component.save()
        return JsonResponse({'success': True})
    except Component.DoesNotExist:
        return JsonResponse({'error': 'Component not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

@csrf_exempt
@require_http_methods(["DELETE"])
def delete_component(request, component_id):
    try:
        component = Component.objects.get(id=component_id)
        component.delete()
        return JsonResponse({'success': True})
    except Component.DoesNotExist:
        return JsonResponse({'error': 'Component not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)



from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from .models import (
    Component, Component_details_technique, Component_description_technique_paragraphe,
    paragraphe_images, Component_model3D, Component_video, Component_document
)

# Afficher les détails du composant
@require_http_methods(["GET"])
def component_details_view(request, component_id):
    """Afficher les détails d'un composant."""
    component = get_object_or_404(Component, id=component_id)
    paragraphs = Component_description_technique_paragraphe.objects.filter(component=component).order_by('order')
    details = Component_details_technique.objects.filter(component=component)
    documents = Component_document.objects.filter(component=component)
    videos = Component_video.objects.filter(component=component)
    model3d = Component_model3D.objects.filter(component=component).first()

    return render(request, 'admin_page/component_details.html', {
        'component': component,
        'paragraphs': paragraphs,
        'details': details,
        'documents': documents,
        'videos': videos,
        'model3d': model3d,
    })

# Modifier le nom du composant
@require_http_methods(["POST"])
def edit_component(request, component_id):
    """Modifier le nom du composant."""
    component = get_object_or_404(Component, id=component_id)
    component.name = request.POST.get("name")
    component.save()
    return redirect('component_details', component_id=component.id)

# Gestion des modèles 3D
@require_http_methods(["POST"])
def upload_3d_model(request):
    """Téléverser un modèle 3D au format .glb pour un composant."""
    model_file = request.FILES.get('model3D')
    component_id = request.POST.get('component_id')

    if not model_file.name.endswith(".glb"):
        return JsonResponse({"error": "Seuls les fichiers .glb sont acceptés."}, status=400)

    component = get_object_or_404(Component, id=component_id)
    Component_model3D.objects.filter(component=component).delete()  # Supprimer l'ancien modèle
    Component_model3D.objects.create(component=component, model3D=model_file)  # Ajouter le nouveau
    return redirect('component_details', component_id=component.id)

@require_http_methods(["POST"])
def delete_3d_model(request, model_id):
    """Supprimer un modèle 3D associé à un composant."""
    model = get_object_or_404(Component_model3D, id=model_id)
    component_id = model.component.id
    model.delete()
    return redirect('component_details', component_id=component_id)

# Gestion des paragraphes
@require_http_methods(["POST"])
def add_paragraph(request):
    """Ajouter un paragraphe."""
    component_id = request.POST.get('component_id')
    description = request.POST.get('description')
    component = get_object_or_404(Component, id=component_id)

    Component_description_technique_paragraphe.objects.create(component=component, description=description)
    return redirect('component_details', component_id=component.id)

@require_http_methods(["POST"])
def edit_paragraph(request, paragraph_id):
    """Modifier un paragraphe."""
    paragraph = get_object_or_404(Component_description_technique_paragraphe, id=paragraph_id)
    paragraph.description = request.POST.get('description')
    paragraph.save()
    return redirect('component_details', component_id=paragraph.component.id)

@require_http_methods(["POST"])
def delete_paragraph(request, paragraph_id):
    """Supprimer un paragraphe."""
    paragraph = get_object_or_404(Component_description_technique_paragraphe, id=paragraph_id)
    component_id = paragraph.component.id
    paragraph.delete()
    return redirect('component_details', component_id=component_id)

@require_http_methods(["POST"])
def add_image_to_paragraph(request, paragraph_id):
    """Ajouter une image à un paragraphe."""
    paragraph = get_object_or_404(Component_description_technique_paragraphe, id=paragraph_id)
    image = request.FILES.get('image')

    paragraphe_images.objects.create(paragraphe=paragraph, image=image)
    return redirect('component_details', component_id=paragraph.component.id)

@require_http_methods(["POST"])
def replace_image(request, image_id):
    """Remplacer une image associée à un paragraphe."""
    image_instance = get_object_or_404(paragraphe_images, id=image_id)
    new_image = request.FILES.get('image')

    if not new_image:
        return JsonResponse({'error': 'Aucune image fournie.'}, status=400)

    image_instance.image = new_image
    image_instance.save()
    return redirect('component_details', component_id=image_instance.paragraphe.component.id)

@require_http_methods(["POST"])
def delete_image(request, image_id):
    """Supprimer une image associée à un paragraphe."""
    image_instance = get_object_or_404(paragraphe_images, id=image_id)
    component_id = image_instance.paragraphe.component.id
    image_instance.delete()
    return redirect('component_details', component_id=component_id)

# Gestion des détails techniques
@require_http_methods(["POST"])
def add_technical_detail(request):
    """Ajouter un détail technique."""
    component_id = request.POST.get('component_id')
    caractéristique = request.POST.get('caractéristique')
    valeur = request.POST.get('valeur')
    component = get_object_or_404(Component, id=component_id)

    Component_details_technique.objects.create(
        component=component,
        caractéristique=caractéristique,
        valeur=valeur
    )
    return redirect('component_details', component_id=component.id)

@require_http_methods(["POST"])
def edit_technical_detail(request, detail_id):
    """Modifier un détail technique."""
    detail = get_object_or_404(Component_details_technique, id=detail_id)
    detail.caractéristique = request.POST.get('caractéristique')
    detail.valeur = request.POST.get('valeur')
    detail.save()
    return redirect('component_details', component_id=detail.component.id)

@require_http_methods(["POST"])
def delete_technical_detail(request, detail_id):
    """Supprimer un détail technique."""
    detail = get_object_or_404(Component_details_technique, id=detail_id)
    component_id = detail.component.id
    detail.delete()
    return redirect('component_details', component_id=component_id)

# Gestion des documents
@require_http_methods(["POST"])
def upload_document(request):
    """Ajouter des documents associés."""
    component_id = request.POST.get('component_id')
    component = get_object_or_404(Component, id=component_id)

    for file in request.FILES.getlist('document'):
        Component_document.objects.create(component=component, document=file)

    return redirect('component_details', component_id=component.id)

@require_http_methods(["POST"])
def delete_document(request, document_id):
    """Supprimer un document associé."""
    document = get_object_or_404(Component_document, id=document_id)
    component_id = document.component.id
    document.delete()
    return redirect('component_details', component_id=component_id)

# Gestion des vidéos
@require_http_methods(["POST"])
def upload_video(request):
    """Télécharger une vidéo associée."""
    video_file = request.FILES.get('video')
    component_id = request.POST.get('component_id')
    component = get_object_or_404(Component, id=component_id)

    Component_video.objects.filter(component=component).delete()
    Component_video.objects.create(component=component, video=video_file)
    return redirect('component_details', component_id=component.id)

@require_http_methods(["POST"])
def delete_video(request, video_id):
    """Supprimer une vidéo associée."""
    video = get_object_or_404(Component_video, id=video_id)
    component_id = video.component.id
    video.delete()
    return redirect('component_details', component_id=component_id)


from .models import  ComponentImage

def upload_component_image(request, component_id):
    """Télécharge une ou plusieurs images pour un composant."""
    component = get_object_or_404(Component, id=component_id)

    if request.method == "POST":
        for image in request.FILES.getlist('images'):
            ComponentImage.objects.create(component=component, image=image)
        messages.success(request, "Images ajoutées avec succès.")
    return redirect('component_details', component_id=component.id)

def delete_component_image(request, image_id):
    """Supprime une image associée à un composant."""
    image = get_object_or_404(ComponentImage, id=image_id)
    component_id = image.component.id
    image.delete()
    messages.success(request, "Image supprimée avec succès.")
    return redirect('component_details', component_id=component_id)






