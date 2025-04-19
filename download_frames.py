import requests
import os

# URLs das imagens de armações
frames = [
    {
        'url': 'https://assets.lenscrafters.com/is/image/LensCrafters/8053672927498__002.png',
        'filename': 'frame1.png'
    },
    {
        'url': 'https://assets.lenscrafters.com/is/image/LensCrafters/8056597243728__002.png',
        'filename': 'frame2.png'
    }
]

# Cria o diretório se não existir
os.makedirs('measurements/static/frames', exist_ok=True)

# Baixa cada imagem
for frame in frames:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.lenscrafters.com/'
        }
        response = requests.get(frame['url'], headers=headers)
        if response.status_code == 200:
            with open(f"measurements/static/frames/{frame['filename']}", 'wb') as f:
                f.write(response.content)
            print(f"Imagem {frame['filename']} baixada com sucesso!")
        else:
            print(f"Erro ao baixar {frame['filename']}: Status code {response.status_code}")
    except Exception as e:
        print(f"Erro ao baixar {frame['filename']}: {str(e)}") 