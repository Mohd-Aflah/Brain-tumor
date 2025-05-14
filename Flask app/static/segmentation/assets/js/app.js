document.getElementById('predictBtn').addEventListener('click', function() {
    const fileInput = document.getElementById('fileInput');
    const resultDiv = document.getElementById('result');

    if (fileInput.files.length === 0) {
        resultDiv.innerHTML = "Please upload an image first!";
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    resultDiv.innerHTML = "Processing...";

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        resultDiv.innerHTML = `The model predicts: <strong>${data.prediction}</strong>`;
    })
    .catch(err => {
        resultDiv.innerHTML = "Error during prediction";
        console.error(err);
    });
});
