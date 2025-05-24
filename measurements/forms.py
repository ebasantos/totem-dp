from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import Empresa, UsuarioEmpresa

class EmpresaForm(forms.ModelForm):
    class Meta:
        model = Empresa
        fields = ['nome', 'cnpj', 'email', 'telefone', 'endereco']
        widgets = {
            'endereco': forms.Textarea(attrs={'rows': 3}),
        }

class UsuarioEmpresaForm(UserCreationForm):
    class Meta:
        model = UsuarioEmpresa
        fields = ['username', 'email', 'first_name', 'last_name', 'cargo', 'telefone', 'password1', 'password2'] 