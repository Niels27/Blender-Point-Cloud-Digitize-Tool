// Initialize the WebSocket connection
var socket = io.connect('http://' + document.domain + ':' + location.port);

socket.on('connect', function() {
    console.log('WebSocket connected!');
});

// Function to send data to the server
function sendDataToServer(data) {
    socket.emit('message', data);
    console.log("Data sent to server:", data);
}

// Initialize the map
var map = L.map('map').setView([51.505, -0.09], 13); // Adjust with your desired coordinates

// Add OpenStreetMap tiles
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19
}).addTo(map);

// Loading indicator element
var loadingIndicator = document.createElement('div');
loadingIndicator.innerText = 'Loading shapefile...';
loadingIndicator.style.position = 'absolute';
loadingIndicator.style.top = '10px';
loadingIndicator.style.left = '10px';
document.body.appendChild(loadingIndicator);

// Function to load and display the shapefile
function loadShapefile(url) {
    console.log("Starting to load shapefile:", url);
    
    shp(url).then(function(geojson) {
        console.log("Shapefile loaded:", geojson);
        var layer = L.geoJson(geojson).addTo(map);
        map.fitBounds(layer.getBounds());
        console.log("Shapefile successfully added to the map.");
        loadingIndicator.remove(); // Remove loading indicator after loading
    }).catch(function(error) {
        console.error("Error loading shapefile:", error);
        loadingIndicator.innerText = 'Failed to load shapefile.';
    });
}

// Load and display the shapefile
loadShapefile("http://localhost:5000/shapefiles/shapefile.zip");

// Handle map click events
map.on('click', function(e) {
    var latlng = e.latlng;
    console.log("Clicked at latitude: " + latlng.lat + " and longitude: " + latlng.lng);

    // Prepare the data to be sent
    var data = {
        latitude: latlng.lat,
        longitude: latlng.lng
    };

    // Send the data to the server
    sendDataToServer(data);
});
