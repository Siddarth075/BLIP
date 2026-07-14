const image = document.getElementById("mainImage");
const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d");

let startX = 0;
let startY = 0;

let currentX = 0;
let currentY = 0;

let drawing = false;

let scaleX = 1;
let scaleY = 1;

image.onload = function () {

    canvas.width = image.clientWidth;
    canvas.height = image.clientHeight;

    canvas.style.width = image.clientWidth + "px";
    canvas.style.height = image.clientHeight + "px";

    scaleX = image.naturalWidth / image.clientWidth;
    scaleY = image.naturalHeight / image.clientHeight;

};

window.addEventListener("resize", function () {

    canvas.width = image.clientWidth;
    canvas.height = image.clientHeight;

    canvas.style.width = image.clientWidth + "px";
    canvas.style.height = image.clientHeight + "px";

    scaleX = image.naturalWidth / image.clientWidth;
    scaleY = image.naturalHeight / image.clientHeight;

});

function getMousePosition(event){

    const rect = canvas.getBoundingClientRect();

    return {

        x: event.clientX - rect.left,

        y: event.clientY - rect.top

    };

}

canvas.addEventListener("mousedown", function(event){

    drawing = true;

    const pos = getMousePosition(event);

    startX = pos.x;
    startY = pos.y;

});

canvas.addEventListener("mousemove", function(event){

    if(!drawing) return;

    const pos = getMousePosition(event);

    currentX = pos.x;
    currentY = pos.y;

    ctx.clearRect(0,0,canvas.width,canvas.height);

    ctx.strokeStyle = "#00ff88";
    ctx.lineWidth = 2;

    ctx.strokeRect(
        startX,
        startY,
        currentX-startX,
        currentY-startY
    );

});

canvas.addEventListener("mouseup", function(){

    drawing = false;

    let x = Math.min(startX,currentX);
    let y = Math.min(startY,currentY);

    let w = Math.abs(currentX-startX);
    let h = Math.abs(currentY-startY);

    document.getElementById("xText").innerHTML = Math.round(x*scaleX);
    document.getElementById("yText").innerHTML = Math.round(y*scaleY);

    document.getElementById("wText").innerHTML = Math.round(w*scaleX);
    document.getElementById("hText").innerHTML = Math.round(h*scaleY);

    document.getElementById("x").value = Math.round(x*scaleX);
    document.getElementById("y").value = Math.round(y*scaleY);

    document.getElementById("w").value = Math.round(w*scaleX);
    document.getElementById("h").value = Math.round(h*scaleY);

});