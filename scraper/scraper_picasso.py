import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
import re

# URL der Webseite
url = 'https://en.wikipedia.org/wiki/List_of_Picasso_artworks_1901%E2%80%931910'

# HTTP-Anfrage an die Seite senden
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Ordner "datenset/Picasso" erstellen, falls er nicht existiert
os.makedirs('datenset/Picasso', exist_ok=True)

# Alle Bildcontainer finden
containers = soup.find_all('figure', class_='mw-default-size')

# Funktion zum Herunterladen und Speichern der Bilder
def download_image(image_url, file_path):
    image_response = requests.get(image_url)
    if image_response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(image_response.content)

# Alle Container durchgehen und Bilder herunterladen
for container in containers:
    # Bild-URL finden
    image_tag = container.find('img')
    if image_tag:
        image_url = urljoin(url, image_tag['src'])
        # Beschriftung aus dem <i>-Tag extrahieren
        caption_tag = container.find('figcaption')
        name_tag = caption_tag.find('i') if caption_tag else None
        name = name_tag.get_text(strip=True) if name_tag else 'Unbenannt'
        print(f"Name: {name}")  # Debug-Info ausgeben
        print(f"Image URL: {image_url}")  # Debug-Info ausgeben
        # Gültigen Dateinamen aus dem Gemäldename erstellen
        valid_name = re.sub(r'[\\/*?:"<>|]', "", name)
        file_path = os.path.join('datenset/Picasso', f"{valid_name}.jpg")
        # Bild herunterladen und speichern
        download_image(image_url, file_path)
        print(f"Downloaded: {name}")

print("Alle Bilder wurden erfolgreich heruntergeladen.")
