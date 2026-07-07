function startRecognition() {

    if (!('webkitSpeechRecognition' in window) &&
        !('SpeechRecognition' in window)) {

        alert("Speech Recognition is not supported in this browser.\nUse Google Chrome or Microsoft Edge.");

        return;
    }

    const SpeechRecognition =
        window.SpeechRecognition || window.webkitSpeechRecognition;

    const recognition = new SpeechRecognition();

    recognition.lang = "en-US";

    recognition.continuous = false;

    recognition.interimResults = false;

    recognition.maxAlternatives = 1;

    recognition.start();

    recognition.onstart = function () {

        console.log("Listening...");

    };

    recognition.onresult = function (event) {

        const transcript = event.results[0][0].transcript;

        document.getElementById("question").value = transcript;

        console.log("You said:", transcript);

    };

    recognition.onerror = function (event) {

        console.log(event.error);

        alert("Speech Recognition Error : " + event.error);

    };

    recognition.onend = function () {

        console.log("Recognition Finished");

    };

}