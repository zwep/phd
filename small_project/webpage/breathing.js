var c = document.getElementById("c");
var ctx = c.getContext("2d");

// Set properties of the canvas..?
ctx.canvas.width  = window.outerWidth;
ctx.canvas.height = window.outerHeight;
ctx.lineWidth = 2;

// Define the area where we are drawing
var rectangleWidth = window.outerWidth;
var rectangleHeight = window.outerHeight;

// Properties of the sine-wave
var sinusWidth = window.outerWidth * 0.99;
var sinusHeight = window.outerHeight/2;
var offsetHeight = 40;  // Arbitrary chosen
var phi = 0;
var frames = 0;

// Properies of the ball
var ballHeight = 0;
var ballColor = 'red';
var ballRadius = 20;

// Properties of time
var period_factor = 10; // This is needed to translate the number to seconds (for my screen at least)

// HTML properties
var frequencyValueElement = document.getElementById('frequency-value');
var periodValueElement = document.getElementById('period-value');

// Set the initial parameter value...
var frequency = Number(document.getElementById("period-value").textContent); // Initial parameter value
var period = Number(document.getElementById("period-value").textContent); // Initial parameter value
// Initialize with the proper factor
period = period * period_factor


function changeValue(delta, parameter_id) {
  if (parameter_id === 'frequency-value'){
    // Change the frequency parameter
    frequency += delta
    // Set the new value in the HTML
    frequencyValueElement.textContent = frequency.toFixed();
  } else {
    // Change the period parameter
    period += delta * period_factor
    // Set the new value in the HTML
    // For visualization, divide by the factor again
    periodValueElement.textContent = period / period_factor;
  }
}


function Draw() {
  frames++
  // Here I can change the speed
  phi = frames / period;
  frequency = Number(document.getElementById("frequency-value").textContent) ;
  ctx.clearRect(0, 0, rectangleWidth, rectangleHeight);
  // Start drawing the sinus wave
  ctx.beginPath();
  for (var x = 0; x < sinusWidth; x++) {
    y = Math.sin(2 * Math.PI * x / frequency + phi) * sinusHeight / 2 + sinusHeight / 2;
    ctx.lineTo(x, y + offsetHeight);
  }
  ctx.stroke();

  // Start drawing the red dot
  ctx.beginPath();
  ballHeight = Math.sin(2 * Math.PI * (sinusWidth / 2) / frequency + phi) * sinusHeight / 2 + sinusHeight / 2 + offsetHeight
  ctx.arc(sinusWidth / 2, ballHeight, ballRadius, 0, Math.PI * 2, false);
  ctx.fillStyle = ballColor;
  ctx.fill();
  ctx.stroke();

  requestId = window.requestAnimationFrame(Draw);
}
requestId = window.requestAnimationFrame(Draw);
