import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
import re

# URL der Wikipedia-Seite
url = 'https://de.wikipedia.org/wiki/Liste_der_Gem%C3%A4lde_von_Caravaggio'

# HTTP-Anfrage an die Seite senden
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Ordner "Data/Caravaggio" erstellen, falls er nicht existiert
os.makedirs('datenset/Caravaggio', exist_ok=True)

# Tabellen auf der Seite finden
tables = soup.find_all('table', {'class': 'wikitable'})

# Funktion zum Herunterladen und Speichern der Bilder
def download_image(image_url, file_path):
    image_response = requests.get(image_url)
    if image_response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(image_response.content)

# Alle Zeilen in den Tabellen durchgehen
for table in tables:
    rows = table.find_all('tr')[1:]  # Überspringe die Kopfzeile
    for row in rows:
        columns = row.find_all('td')
        if len(columns) > 1:
            # Beschriftung aus dem <i>-Tag extrahieren
            name_tag = columns[1].find('i')
            name = name_tag.get_text(strip=True) if name_tag else 'Unbenannt'
            print(f"Name: {name}")  # Debug-Info ausgeben
            # Bild-URL finden
            image_tag = columns[0].find('img')
            if image_tag:
                image_url = urljoin(url, image_tag['src'])
                print(f"Image URL: {image_url}")  # Debug-Info ausgeben
                # Gültigen Dateinamen aus dem Gemäldename erstellen
                valid_name = re.sub(r'[\\/*?:"<>|]', "", name)
                file_path = os.path.join('datenset/Caravaggio', f"{valid_name}.jpg")
                # Bild herunterladen und speichern
                download_image(image_url, file_path)
                print(f"Downloaded: {name}")

print("Alle Bilder wurden erfolgreich heruntergeladen.")
