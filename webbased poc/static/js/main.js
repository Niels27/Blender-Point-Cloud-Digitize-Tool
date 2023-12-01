// Get the progress element
const progressElement = document.getElementById('progress');

function fetchJSONWithProgress(url) {
    return fetch(url).then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const contentLength = +response.headers.get('Content-Length');
        let receivedLength = 0;
        let chunks = []; // Array to hold received text chunks

        const reader = response.body.getReader();

        function readNextChunk() {
            return reader.read().then(({ done, value }) => {
                if (done) {
                    // All chunks have been read
                    let json = chunks.join('');
                    console.log("JSON start:", json.substring(0, 500));
                    console.log("JSON end:", json.substring(json.length - 500));
                    return JSON.parse(json);
                }

                // Convert chunk to text and accumulate
                let chunkText = new TextDecoder("utf-8").decode(value);
                chunks.push(chunkText);
                receivedLength += value.length;
                progressElement.textContent = `Received ${receivedLength} of ${contentLength}`;

                // Read the next chunk
                return readNextChunk();
            });
        }

        return readNextChunk();
    });
}

function processPointCloud(pointCloudData) {
    console.log("Data processed. Number of points:", pointCloudData.length);
    console.log("Sample data:", pointCloudData.slice(0, 10));
    progressElement.textContent = `Processing ${pointCloudData.length} points...`;

    // Initialize Three.js scene
    init(pointCloudData);
}

fetch('/get-point-cloud')
    .then(response => {
        console.log("Received response, Content-Encoding:", response.headers.get("Content-Encoding"));
        if (response.headers.get("Content-Encoding") === "gzip") {
            // Handle gzipped data
            return response.arrayBuffer().then(buffer => {
                const decompressed = pako.inflate(buffer, { to: 'string' });
                return JSON.parse(decompressed);
            });
        } else {
            // Handle large JSON with progress
            return fetchJSONWithProgress(response.url);
        }
    })
    .then(processPointCloud)
    .catch(error => {
        console.error('Error processing data:', error);
        progressElement.textContent = 'Error processing data.';
    });
function init(pointCloudData) {
    // Scene, camera, and renderer setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    console.log("Initializing scene...");
    progressElement.textContent = "Initializing scene...";

    // Add points to the scene
    const geometry = new THREE.BufferGeometry();
    const vertices = [];
    const colors = [];
    
    pointCloudData.forEach((point, index) => {
        if (point.coords && point.color) {
            // Handle the format with 'coords' and 'color' arrays
            vertices.push(...point.coords);
            colors.push(point.color[0] / 255, point.color[1] / 255, point.color[2] / 255);
        } else if (!isNaN(point.x) && !isNaN(point.y) && !isNaN(point.z)) {
            // Handle the format with 'x', 'y', 'z', and 'color' object
            vertices.push(point.x, point.y, point.z);
            colors.push(point.color.r / 255, point.color.g / 255, point.color.b / 255);
        } else {
            console.log(`Skipping invalid point at index ${index}`);
        }

        if (index % 10000 === 0) {
            console.log(`Processing point ${index} / ${pointCloudData.length}`);
            progressElement.textContent = `Processing point ${index} / ${pointCloudData.length}`;
        }
    });

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({ size: 0.05, vertexColors: true });
    const points = new THREE.Points(geometry, material);
    scene.add(points);

    console.log("Points added to the scene.");
    progressElement.textContent = "Points added to the scene.";

    // Camera position
    camera.position.z = 5;

    console.log("Starting render loop...");
    progressElement.textContent = "Starting render loop...";

    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
    }
    animate();
}
