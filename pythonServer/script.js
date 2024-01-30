
let selected = false
let removedPicture = false
let pictureURL = null

function resetModal() {
    const probability = document.getElementById('prediction-probability')
    const final = document.getElementById('gender-prediction')
    const genderPic = document.getElementById('gender-img')

    probability.innerText = ''
    final.innerText = ''
    genderPic.src = 'pending.jpeg'
}

function triggerFileInput() {
    if(removedPicture) {
        removedPicture = false
        return
    }

    if(!selected) {
        document.getElementById("file-input").click();
    }
}

function displaySelectedImage() {
     const fileInput = document.getElementById("file-input");
     const imagePreview = document.getElementById("image-preview");
     const predictedImg = document.getElementById("predicted-image");


    if (fileInput.files && fileInput.files[0]) {
            const reader = new FileReader();

            reader.onload = function (e) {
                // Display the selected image in the image-preview div
                pictureURL = e.target.result
                imagePreview.style.backgroundImage = `url(${e.target.result})`;
                imagePreview.style.backgroundSize = "cover";
                imagePreview.style.backgroundPosition = "center";
                imagePreview.style.cursor = "default"

                predictedImg.src = e.target.result;
            };

            reader.readAsDataURL(fileInput.files[0]);
            selected = true
            const predictButton = document.getElementById("predict-button")
            predictButton.disabled = false

            const removeButton = document.getElementById("remove")
            removeButton.style.display = "inherit"
        }

}

function removePicture() {
    if(selected) {
        const fileInput = document.getElementById("file-input");
        fileInput.value = null
        pictureURL = null

        const imagePreview = document.getElementById("image-preview");
        imagePreview.style.backgroundImage = 'url("plus-icon.svg")'
        imagePreview.style.backgroundSize = '20px 20px'
        imagePreview.style.cursor = 'pointer'

        const removeButton = document.getElementById("remove")
        removeButton.style.display = "none"

        selected = false
        const predictButton = document.getElementById("predict-button")
        predictButton.disabled = true
        removedPicture = true

        const probability = document.getElementById('prediction-probability')
        const final = document.getElementById('gender-prediction')
        const genderPic = document.getElementById('gender-img')
        probability.innerText = ""
        final.innerText = ""
        genderPic.src = "pending.jpeg"
    }
}

function updateModal(response, success){
    const probability = document.getElementById('prediction-probability')
    const final = document.getElementById('gender-prediction')
    const genderPic = document.getElementById('gender-img')

    if(success) {
        const predValue = parseFloat(response) * 100
        const rounded = parseFloat(predValue.toFixed(2));
        if (rounded > 60) {
            final.innerText = "Male"
            probability.innerText = "" + (rounded) + " % Probability"
            genderPic.src = "male.png"
        } else if (rounded < 40) {
            final.innerText = "Female"
            probability.innerText = "" + ((100 - rounded)) + " % Probability"
            genderPic.src = "female.png"
        } else {
            genderPic.src = "pending.jpeg"
            final.innerText = ""
            probability.innerText = "Not Sure"
        }
    } else {
        genderPic.src = "error.png"
        probability.innerText = response
    }
}

async function predict() {
    const data = {picture : pictureURL}
    try {
        const res = await fetch("http://localhost:8082/predict/gender", {
            method: 'POST',
            body: JSON.stringify(data),
            headers: {
                'Content-Type': 'application/json'
            }
        })
        if (res.status === 200) {
            const response = await res.text();
            updateModal(response, true)
        } else {
            updateModal("Could Not Predict\n(Returned with status " + res.status + ")", false)
        }
    } catch(error) {
        setTimeout(() => {
            updateModal("Could Not Predict\n(" + error.toString() + ")", false)
        }, 1000)
    }
}
