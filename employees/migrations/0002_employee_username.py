# Generated by Django 2.2.3 on 2019-07-24 12:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('employees', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='employee',
            name='username',
            field=models.CharField(max_length=15, null=True, unique=True),
        ),
    ]
