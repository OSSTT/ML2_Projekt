import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
import re

# URL der Webseite
url = 'https://nicofranz.art/leonardo-da-vinci/alle-gemaelde'

# HTTP-Anfrage an die Seite senden
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Ordner "datenset/Da Vinci" erstellen, falls er nicht existiert
os.makedirs('datenset/Da Vinci', exist_ok=True)

# Alle Bildcontainer finden
containers = soup.find_all('div', class_='grid-item-c2037')

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
        # Bildname aus dem alt-Attribut extrahieren
        name = image_tag.get('alt', 'Unbenannt')
        print(f"Name: {name}")  # Debug-Info ausgeben
        print(f"Image URL: {image_url}")  # Debug-Info ausgeben
        # Gültigen Dateinamen aus dem Gemäldename erstellen
        valid_name = re.sub(r'[\\/*?:"<>|]', "", name)
        valid_name = re.sub(r'[ -]', "_", valid_name)  # Leerzeichen und Bindestriche durch Unterstriche ersetzen
        file_path = os.path.join('datenset/Da Vinci', f"{valid_name}.jpg")
        # Bild herunterladen und speichern
        download_image(image_url, file_path)
        print(f"Downloaded: {name}")

print("Alle Bilder wurden erfolgreich heruntergeladen.")
