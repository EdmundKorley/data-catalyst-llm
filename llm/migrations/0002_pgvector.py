# Generated by Django 4.2.6 on 2023-10-17 15:15

from django.db import migrations
from pgvector.django import VectorExtension


class Migration(migrations.Migration):
    dependencies = [
        ("llm", "0001_initial"),
    ]

    operations = [VectorExtension()]
