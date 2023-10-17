from django.db import models

from pgvector.django import VectorField


class Embeddings(models.Model):
    raw_text = models.TextField()
    text_vectors = VectorField(dimensions=1536, default=[])

    class Meta:
        db_table = "embeddings"
