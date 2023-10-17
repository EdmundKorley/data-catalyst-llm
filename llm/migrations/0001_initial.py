# Generated by Django 4.2.6 on 2023-10-17 15:15

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Embeddings",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("raw_text", models.TextField()),
            ],
            options={
                "db_table": "embeddings",
            },
        ),
    ]