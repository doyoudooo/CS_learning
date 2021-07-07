# Generated by Django 3.2.5 on 2021-07-07 07:55

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='article_kinds',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('kinds', models.CharField(max_length=20)),
            ],
        ),
        migrations.CreateModel(
            name='article',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(default='', max_length=50)),
                ('auther', models.CharField(blank=True, default='', max_length=50, null=True)),
                ('content', models.TextField(default='')),
                ('create_time', models.DateTimeField(auto_now_add=True)),
                ('kind', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='home.article_kinds')),
            ],
        ),
    ]