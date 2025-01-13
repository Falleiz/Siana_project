from django.db import models

# Modèle utilisateur (admin_user)
class admin_user(models.Model):
    nom_prenom = models.CharField(max_length=20)
    email = models.EmailField(max_length=254)
    password = models.CharField(max_length=30)

    def __str__(self):
        return f"{self.nom_prenom} ({self.email})"



# Modèle des composants
class Component(models.Model):
    name = models.CharField(max_length=100)  # Nom plus long pour plus de flexibilité
    parent = models.ForeignKey(
        "self",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="enfants"
    )

    def __str__(self):
        if self.parent:
            return f"{self.name} (Enfant de {self.parent.name})"
        return self.name


class ComponentImage(models.Model):
    component = models.ForeignKey(
        Component,
        on_delete=models.CASCADE,
        related_name="component_images"
    )
    image = models.ImageField(upload_to="component_images/")

    def __str__(self):
        return f"Image associée à {self.component.name}"




# Détails techniques d’un composant (Caractéristiques techniques)
class Component_details_technique(models.Model):
    component = models.ForeignKey(
        Component,
        on_delete=models.CASCADE,
        related_name="details_technique"
    )
    caractéristique = models.CharField(max_length=50)  # Plus long pour plus de précision
    valeur = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.caractéristique}: {self.valeur} (Composant: {self.component.name})"


# Paragraphe de description technique
class Component_description_technique_paragraphe(models.Model):
    component = models.ForeignKey(
        Component,
        on_delete=models.CASCADE,
        related_name="description_technique_paragraphes"
    )
    description = models.TextField()
    order = models.PositiveIntegerField(default=0)  # Ordre des paragraphes

    def __str__(self):
        return f"Paragraphe {self.order} - {self.component.name}"


# Images associées aux paragraphes
class paragraphe_images(models.Model):
    paragraphe = models.ForeignKey(
        Component_description_technique_paragraphe,
        on_delete=models.CASCADE,
        related_name="paragraphe_images"
    )
    image = models.ImageField(upload_to="images/")

    def __str__(self):
        return f"Image associée au {self.paragraphe}"


# Modèles 3D associés aux composants
from django.core.validators import FileExtensionValidator
class Component_model3D(models.Model):
    component = models.ForeignKey(
        Component,
        on_delete=models.CASCADE,
        related_name="appareil_models3d"
    )
    model3D = models.FileField(
        upload_to="models3d/",
        validators=[FileExtensionValidator(allowed_extensions=['glb', 'gltf', 'blend', 'fbx', 'obj'])]
    )

    def __str__(self):
        return f"Modèle 3D - {self.component.name}"
# Vidéos associées aux composants
class Component_video(models.Model):
    component = models.ForeignKey(
        Component,
        on_delete=models.CASCADE,
        related_name="appareil_videos"
    )
    video = models.FileField(upload_to="videos/")

    def __str__(self):
        return f"Vidéo associée à {self.component.name}"


# Documents associés aux composants
class Component_document(models.Model):
    component = models.ForeignKey(
        Component,
        on_delete=models.CASCADE,
        related_name="appareil_documents"
    )
    document = models.FileField(upload_to="documents/")

    def __str__(self):
        return f"Document pour {self.component.name}"

