from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.contrib import messages
from .models import technician_user
from admin_page.models import Component


# Connexion
def login_view(request):
    """Gérer la connexion d'un technicien."""
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        try:
            # Vérifier si l'utilisateur existe
            user = technician_user.objects.get(email=email)

            # Vérifier le mot de passe directement
            if password == user.password:
                # Enregistrer les informations de connexion dans la session
                request.session['user_id'] = user.id
                request.session['username'] = user.nom_prenom
                return redirect('home')  # Rediriger après une connexion réussie
            else:
                messages.error(request, "Mot de passe incorrect.")
        except technician_user.DoesNotExist:
            messages.error(request, "Utilisateur introuvable.")

    return render(request, 'technician_page/login.html')


# Déconnexion
def logout_view(request):
    """Gérer la déconnexion."""
    request.session.flush()  # Supprimer toutes les données de session
    messages.success(request, "Vous êtes déconnecté.")
    return redirect('login')  # Rediriger vers la page de connexion


# Page d'accueil dynamique
def home_view(request):
    """Page d'accueil pour les techniciens."""
    # Récupérer tous les composants
    components = Component.objects.all()

    # Identifier le composant principal (sans parent)
    main_component = Component.objects.filter(parent__isnull=True).first()

    return render(request, 'technician_page/home.html', {
        'components': components,
        'main_component': main_component,
    })


from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from admin_page.models import Component

def get_component_3d(request, component_id):
    """Récupérer les informations du modèle 3D d'un composant."""
    component = get_object_or_404(Component, id=component_id)
    model3d = component.appareil_models3d.first()  # Récupérer le premier modèle 3D associé

    return JsonResponse({
        'model3d_url': model3d.model3D.url if model3d else None,  # URL du fichier modèle 3D
        'name': component.name,  # Nom du composant
    })


from django.shortcuts import render, get_object_or_404
from admin_page.models import Component, Component_model3D, Component_description_technique_paragraphe, Component_details_technique, Component_document, Component_video

def component_technical_sheet(request, component_id):
    """Afficher les détails d'un composant pour le technicien."""
    # Récupérer le composant principal
    component = get_object_or_404(Component, id=component_id)

    # Charger les informations associées au composant
    model3d = component.appareil_models3d.first()  # Premier modèle 3D associé
    paragraphs = Component_description_technique_paragraphe.objects.filter(component=component).order_by('order')
    details = Component_details_technique.objects.filter(component=component)
    documents = Component_document.objects.filter(component=component)
    videos = Component_video.objects.filter(component=component)

    # Passer les informations au template
    return render(request, 'technician_page/component_technical_sheet.html', {
        'component': component,
        'model3d': model3d,
        'paragraphs': paragraphs,
        'details': details,
        'documents': documents,
        'videos': videos,
    })


from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from admin_page.models import Component

def search_components(request):
    """Recherche dynamique des composants."""
    query = request.GET.get('q', '')  # Récupérer la requête utilisateur
    results = []

    if query:  # Si une requête est fournie
        components = Component.objects.filter(name__icontains=query)[:10]  # Limite à 10 résultats
        results = [{'id': comp.id, 'name': comp.name} for comp in components]

    return JsonResponse(results, safe=False)




import os
import torch
import clip
from PIL import Image
from django.shortcuts import render, get_object_or_404, redirect
from django.conf import settings
from  admin_page.models import Component, ComponentImage

# Charger le modèle CLIP et le preprocess
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess_clip = clip.load("ViT-B/32", device)

def upload_and_search(request):
    """Téléverse une image temporaire, recherche le composant correspondant et affiche le résultat."""
    if request.method == "POST":
        # Récupérer le fichier téléversé
        uploaded_file = request.FILES.get("image")
        print(f"Fichier téléversé : {uploaded_file}")
        if not uploaded_file:
            print("Aucune image téléversée.")
            return render(request, "technician_page/search.html", {"error": "Aucune image n'a été téléversée."})

        # Sauvegarder l'image temporaire
        temp_dir = os.path.join(settings.MEDIA_ROOT, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_image_path = os.path.join(temp_dir, uploaded_file.name)
        print(f"Chemin de l'image temporaire : {temp_image_path}")
        with open(temp_image_path, "wb") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        # Charger les images des composants
        component_images = ComponentImage.objects.all()
        print(f"Images des composants disponibles : {component_images}")
        component_names = [img.component.name for img in component_images]
        component_image_paths = [img.image.path for img in component_images]

        # Prétraiter l'image téléversée
        print("Prétraitement de l'image téléversée...")
        input_image = preprocess_clip(Image.open(temp_image_path)).unsqueeze(0).to(device)

        # Prétraiter les noms des composants
        print(f"Noms des composants pour la comparaison : {component_names}")
        text_tokens = clip.tokenize(component_names).to(device)

        # Calculer les similarités
        print("Calcul des similarités avec CLIP...")
        with torch.no_grad():
            input_image_features = model_clip.encode_image(input_image)
            text_features = model_clip.encode_text(text_tokens)
            similarities = (input_image_features @ text_features.T).softmax(dim=-1)

        # Trouver la similarité maximale
        max_similarity, max_index = similarities[0].max(dim=0)
        max_similarity = max_similarity.item()
        max_component_name = component_names[max_index]
        print(f"Similitude maximale : {max_similarity} pour le composant {max_component_name}")

        # Supprimer l'image temporaire
        os.remove(temp_image_path)

        # Si la similarité est supérieure à 60 %
        if max_similarity >= 0.6:
            matched_component = get_object_or_404(Component, name=max_component_name)
            print(f"Composant correspondant trouvé : {matched_component.name}")
            return redirect("component_sheet", component_id=matched_component.id)
        else:
            print("Aucun composant ne correspond à plus de 60 % de similarité.")
            return render(request, "technician_page/search.html", {
                "error": "Aucun composant ne correspond à plus de '60%'  de similarité.",
            })

    return render(request, "technician_page/search.html")
