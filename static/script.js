const canvas = document.getElementById('canvas');
const outputCanvas = document.getElementById('outputCanvas');
const ctx = canvas.getContext('2d');
const outputCtx = outputCanvas?.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');

let drawing = false;

ctx.lineWidth = 10;
ctx.lineCap = 'round';
ctx.strokeStyle = 'black';

// Mouse event listeners
canvas.addEventListener('mousedown', (e)=> {
    startDrawing(getTouchPos(e));
});
canvas.addEventListener('mousemove', (e)=> {
    draw(getTouchPos(e));
});
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Touch event listeners
canvas.addEventListener('touchstart', (e) => {
  e.preventDefault();
  startDrawing(getTouchPos(e.touches[0]));
});
canvas.addEventListener('touchmove', (e) => {
  e.preventDefault();
  draw(getTouchPos(e.touches[0]));
});
canvas.addEventListener('touchend', stopDrawing);

function getTouchPos(touch) {
  const rect = canvas.getBoundingClientRect();
  console.log(touch.clientX+","+touch.clientY)
  return {
    x: (touch.clientX - rect.left) * (canvas.width / rect.width),
    y: (touch.clientY - rect.top) * (canvas.height / rect.height)
  };
}

function normalizeEventPosition(e) {
  const rect = canvas.getBoundingClientRect();
  let x, y;

  if ('x' in e) {
    // Touch event, already normalized by getTouchPos
    x = e.x;
    y = e.y;
    console.log(x+","+y)
  }

  // Clamp coordinates to canvas bounds
  x = Math.max(0, Math.min(x, canvas.width));
  y = Math.max(0, Math.min(y, canvas.height));

  return { x, y };
}

function startDrawing(e) {
  drawing = true;
  const { x, y } = normalizeEventPosition(e);
  ctx.beginPath();
  ctx.moveTo(x, y);
}

function draw(e) {
  if (!drawing) return;
  const { x, y } = normalizeEventPosition(e);
  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);
}

function stopDrawing() {
  drawing = false;
  ctx.beginPath();
}

clearBtn.addEventListener('click', () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  outputCtx?.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
});

predictBtn.addEventListener('click', () => {
  const imageData = canvas.toDataURL('image/png');
  fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageData })
  })
    .then(res => res.json())
    .then(data => {
      if (window.innerWidth <= 768) {
        sessionStorage.setItem('predictedImage', data.image);
        window.location.href = '/mobile_output';
      } else if (outputCtx) {
        const img = new Image();
        img.onload = () => outputCtx.drawImage(img, 0, 0, outputCanvas.width, outputCanvas.height);
        img.src = 'data:image/png;base64,' + data.image;
      }
    })
    .catch(err => console.error(err));
});