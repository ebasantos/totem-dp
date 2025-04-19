# Sistema de Medição de Distância Pupilar

Sistema desenvolvido para medir a distância entre as pupilas utilizando visão computacional.

## Funcionalidades

- Detecção facial em tempo real
- Medição precisa da distância entre as pupilas
- Interface amigável para o usuário
- Armazenamento de medições
- Visualização do histórico de medições

## Requisitos

- Python 3.13
- Django 5.2
- OpenCV
- PostgreSQL
- Django REST framework

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/ebasantos/totem-dp.git
cd totem-dp
```

2. Crie e ative o ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure o banco de dados PostgreSQL e atualize as configurações em `pupil_distance/settings.py`

5. Execute as migrações:
```bash
python manage.py migrate
```

6. Inicie o servidor:
```bash
python manage.py runserver
```

## Uso

1. Acesse `http://localhost:8000` no navegador
2. Permita o acesso à câmera quando solicitado
3. Posicione seu rosto dentro do guia na tela
4. Aguarde a detecção e medição automática
5. Confirme a medição ou faça ajustes manuais se necessário

## Contribuição

Contribuições são bem-vindas! Por favor, abra uma issue para discutir as mudanças propostas.

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes. 