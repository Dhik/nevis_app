# Generated by Django 5.0.3 on 2024-03-15 01:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('nevisapp', '0002_news'),
    ]

    operations = [
        migrations.AddField(
            model_name='news',
            name='keywords',
            field=models.TextField(default='jokowi'),
            preserve_default=False,
        ),
    ]
