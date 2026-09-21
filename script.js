/* ========================================
   EMAIL SPAM DETECTION
   FRONTEND JAVASCRIPT
======================================== */


/* ========================================
   SELECT HTML ELEMENTS
======================================== */

const spamForm = document.getElementById("spam-form");

const emailText = document.getElementById("email-text");

const characterCount = document.getElementById("character-count");

const checkButton = document.getElementById("check-button");

const clearButton = document.getElementById("clear-button");

const resultCard = document.getElementById("result-card");

const resultContent = document.getElementById("result-content");

const resultTitle = document.getElementById("result-title");

const resultMessage = document.getElementById("result-message");

const resultLabel = document.getElementById("result-label");

const loadingContainer = document.getElementById("loading-container");


/* ========================================
   CHARACTER COUNTER
======================================== */

emailText.addEventListener("input", function () {

    const textLength = emailText.value.length;

    characterCount.textContent = `${textLength} characters`;

});


/* ========================================
   CLEAR BUTTON
======================================== */

clearButton.addEventListener("click", function () {

    // Clear textarea

    emailText.value = "";


    // Reset character counter

    characterCount.textContent = "0 characters";


    // Reset result

    resetResult();


    // Focus textarea

    emailText.focus();

});


/* ========================================
   RESET RESULT FUNCTION
======================================== */

function resetResult() {

    resultCard.classList.remove("spam-result");

    resultCard.classList.remove("ham-result");


    resultContent.hidden = false;

    loadingContainer.hidden = true;


    resultTitle.textContent = "Awaiting Analysis";


    resultMessage.textContent =
        "Enter an email message and click Check Email to view the prediction.";


    resultLabel.textContent = "NOT ANALYZED";


    resultLabel.style.background = "";

    resultLabel.style.color = "";


    checkButton.disabled = false;

}


/* ========================================
   DISPLAY LOADING
======================================== */

function showLoading() {

    resultContent.hidden = true;

    loadingContainer.hidden = false;


    checkButton.disabled = true;


    clearButton.disabled = true;

}


/* ========================================
   HIDE LOADING
======================================== */

function hideLoading() {

    resultContent.hidden = false;

    loadingContainer.hidden = true;


    checkButton.disabled = false;

    clearButton.disabled = false;

}


/* ========================================
   DISPLAY ERROR
======================================== */

function showError(message) {

    hideLoading();


    resultCard.classList.remove("spam-result");

    resultCard.classList.remove("ham-result");


    resultTitle.textContent = "Something Went Wrong";


    resultMessage.textContent = message;


    resultLabel.textContent = "ERROR";


    resultLabel.style.background = "#fee2e2";

    resultLabel.style.color = "#dc2626";

}


/* ========================================
   DISPLAY PREDICTION RESULT
======================================== */

function displayResult(prediction) {

    hideLoading();


    resultCard.classList.remove("spam-result");

    resultCard.classList.remove("ham-result");


    const normalizedPrediction =
        String(prediction).trim().toLowerCase();


    if (
        normalizedPrediction === "spam" ||
        normalizedPrediction === "1"
    ) {

        // Spam result

        resultCard.classList.add("spam-result");


        resultTitle.textContent = "Spam Detected";


        resultMessage.textContent =
            "The machine learning model classified this email as spam.";


        resultLabel.textContent = "SPAM";


        resultLabel.style.background = "";

        resultLabel.style.color = "";


    } else if (
        normalizedPrediction === "ham" ||
        normalizedPrediction === "not spam" ||
        normalizedPrediction === "not_spam" ||
        normalizedPrediction === "0"
    ) {

        // Not spam result

        resultCard.classList.add("ham-result");


        resultTitle.textContent = "Not Spam";


        resultMessage.textContent =
            "The machine learning model classified this email as not spam.";


        resultLabel.textContent = "NOT SPAM";


        resultLabel.style.background = "";

        resultLabel.style.color = "";


    } else {

        showError("The backend returned an unexpected prediction.");

    }

}


/* ========================================
   BACKEND API CONNECTION
======================================== */


/*
    IMPORTANT:

    Replace the API URL with the actual
    endpoint provided by Niranjana.

    Example:

    http://127.0.0.1:5000/predict

    The request format must match
    Niranjana's Flask backend.
*/


const API_URL = "http://127.0.0.1:5000/predict";


/* ========================================
   SEND EMAIL TO BACKEND
======================================== */

async function checkEmailWithBackend(email) {

    const response = await fetch(API_URL, {

        method: "POST",

        headers: {

            "Content-Type": "application/json"

        },

        body: JSON.stringify({

            email: email

        })

    });


    if (!response.ok) {

        throw new Error(
            `Backend request failed (${response.status}).`
        );

    }


    const data = await response.json();


    return data;

}


/* ========================================
   FORM SUBMISSION
======================================== */

spamForm.addEventListener("submit", async function (event) {

    event.preventDefault();


    const email = emailText.value.trim();


    // Validate input

    if (email.length === 0) {

        showError("Please enter an email message.");

        return;

    }


    if (email.length < 5) {

        showError("Please enter a longer email message.");

        return;

    }


    // Display loading

    showLoading();


    try {

        /*
            Send email text to backend.

            The expected backend response
            must contain a prediction.

            Example:

            {
                "prediction": "spam"
            }
        */


        const data = await checkEmailWithBackend(email);


        /*
            Adjust this field if Niranjana's
            backend uses another response key.
        */


        const prediction =
            data.prediction ||
            data.result ||
            data.label;


        if (prediction === undefined || prediction === null) {

            throw new Error(
                "Prediction was not found in the backend response."
            );

        }


        // Display actual prediction

        displayResult(prediction);


    } catch (error) {

        console.error("Prediction Error:", error);


        showError(
            "Unable to connect to the backend. Please check whether the Flask server is running."
        );

    }

});


/* ========================================
   INITIAL PAGE SETUP
======================================== */

resetResult();