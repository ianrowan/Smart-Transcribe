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
  //console.log(`Received transcript: ${data}`);
  var words = data.split(' ')
  var content = ''
  var idLookup = {}
  var keys = Object.keys(prevElements).map((key) => {
    const first = key.split(' ')[0]
    idLookup[first] = key;
    return String(first).toLowerCase()
  })
  //Todo handle multi-word keys
  for(var word of words){
    if(keys.includes(word.toLowerCase().replace(/[^\w\s]/gi, ''))){
      content += `<span class="key" onclick="hideShow('${idLookup[word]}')">${word} </span>`
    }else{
      content += word+ " "
    }
  }
  document.getElementById('transcript').innerHTML = content;
});

socket.on('definition', (data) => {
  console.log(`Received defs: ${data}`);
  
  const defs = JSON.parse(data)
		for (const [key, value] of Object.entries(defs)) {
      if(!(key in prevElements)){
			  document.getElementById('def').innerHTML +=
         `<span class="key" onclick="hideShow('${key}')">${key}</span><span id="${key}" style="display:none">${value}</span><br><br>`;
      }
    }
  prevElements = defs
});
