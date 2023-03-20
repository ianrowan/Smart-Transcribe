const socket = io('http://127.0.0.1:5050');

var prevElements = {};

const hideShow = (id) => {
  const x = document.getElementById(String(id));
  if (x.style.display === "none") {
    x.style.display = "block";
  } else {
    x.style.display = "none";
  }
}

socket.on('transcript', (data) => {
  console.log(`Received transcript: ${data}`);
  document.getElementById('transcript').textContent = data;
});

socket.on('definition', (data) => {
  console.log(`Received transcript: ${data}`);
  
  const defs = JSON.parse(data)
		for (const [key, value] of Object.entries(defs)) {
      if(!(key in prevElements)){
			  document.getElementById('def').innerHTML +=
         `<span class="key" onclick="hideShow('${key}')">${key}</span><span id="${key}" style="display:none">${value}</span><br><br>`;
      }
    }
  prevElements = defs
});
