document.addEventListener('DOMContentLoaded', function() {
    const chatOutput = document.getElementById('chat-output');
    const userInput = document.getElementById('user-input');
    const fileInput = document.getElementById('file-input');
    const submitButton = document.getElementById('submit-button');
    const speechButton = document.getElementById('speech-button');
    const audioPlayer = document.getElementById('audio-player');
    
    let imageProcessed = false;

    submitButton.addEventListener('click', function() {
        const userMessage = userInput.value.trim();
        const file = fileInput.files[0];

        if (userMessage !== '' && (!file || imageProcessed)) {
            appendMessage('user', userMessage);
            sendTextToServer(userMessage);
            userInput.value = '';
            imageProcessed = false;
        } else if (file && !imageProcessed) {
            const formData = new FormData();
            formData.append('file', file);

            fetch('/api/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    appendMessage('bot', data.error);
                } else {
                    const reader = new FileReader();
                    reader.readAsDataURL(file);
                    reader.onload = function() {
                        appendImage(reader.result);
                        sendImageToServer(reader.result);

                        fetch('/api/last_prediction')
                        .then(response => response.json())
                        .then(lastData => {
                            if (lastData.predicted_class) {
                                appendStyledMessage('bot', `The model classified the image as an image of "${lastData.predicted_class}".`);
                            } else {
                                appendStyledMessage('bot', 'Keine vorherigen Vorhersagen gefunden');
                            }
                        })
                        .catch(error => console.error('Error:', error));
                    };
                }
                imageProcessed = true;
                fileInput.value = '';
            })
            .catch(error => console.error('Error:', error));
        }
    });

    speechButton.addEventListener('click', function() {
        fetch('/api/speech', {
            method: 'GET'
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                appendMessage('bot', data.error);
            } else {
                fetch('/api/get_speech', {
                    method: 'GET'
                })
                .then(response => response.blob())
                .then(blob => {
                    const url = URL.createObjectURL(blob);
                    audioPlayer.src = url;
                    audioPlayer.style.display = 'block';
                    audioPlayer.play();
                })
                .catch(error => console.error('Error:', error));
            }
        })
        .catch(error => console.error('Error:', error));
    });

    function appendMessage(sender, message) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender);
        messageElement.innerHTML = `<strong>${sender === 'user' ? 'Du' : 'Chatbot'}:</strong> ${message}`;
        chatOutput.appendChild(messageElement);
        chatOutput.scrollTop = chatOutput.scrollHeight;
    }

    function appendStyledMessage(sender, message) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender);
        messageElement.innerHTML = `<strong>${sender === 'user' ? 'Du' : 'Chatbot'}:</strong> <em style="font-size: smaller;">${message}</em>`;
        chatOutput.appendChild(messageElement);
        chatOutput.scrollTop = chatOutput.scrollHeight;
    }

    function appendImage(imageData) {
        const imageContainer = document.createElement('div');
        const imageElement = document.createElement('img');

        imageElement.src = imageData;

        imageContainer.appendChild(imageElement);
        chatOutput.appendChild(imageContainer);
    }

    function sendTextToServer(text) {
        fetch('/api/text', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        })
        .then(response => response.json())
        .then(data => {
            appendMessage('bot', data.reply);
        })
        .catch(error => console.error('Error:', error));
    }

    function sendImageToServer(imageData) {
        fetch('/api/image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageData })
        })
        .then(response => response.json())
        .then(data => {
            appendMessage('bot', data.reply);
            // Bild wurde erfolgreich gesendet, daher imageProcessed zurücksetzen
            imageProcessed = false;
        })
        .catch(error => console.error('Error:', error));
    }
});

document.getElementById('file-input').addEventListener('change', function() {
    const fileName = this.files[0] ? this.files[0].name : '';
    document.getElementById('file-name').textContent = fileName ? `Hochgeladene Datei: ${fileName}` : '';
    // Datei wurde geändert, daher imageProcessed zurücksetzen
    imageProcessed = false;
});
