// Get the progress element
const progressElement = document.getElementById('progress');

fetch('/get-point-cloud')
    .then(response => {
        console.log("Received response, Content-Encoding:", response.headers.get("Content-Encoding"));
        return response.arrayBuffer();
    })
    .then(buffer => {
        try {
            const decompressed = pako.inflate(buffer, { to: 'string' });
            return JSON.parse(decompressed);
        } catch (error) {
            console.error("Decompression error:", error);
            // Try parsing as regular JSON in case the data is not compressed
            return JSON.parse(new TextDecoder("utf-8").decode(buffer));
        }
    })
    .then(pointCloudData => {
        console.log("Data processed. Number of points:", pointCloudData.length);
        progressElement.textContent = `Processing ${pointCloudData.length} points...`;
        
        // Initialize Three.js scene
        init(pointCloudData);
    })
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
        // Check for NaN values
        if (!isNaN(point.x) && !isNaN(point.y) && !isNaN(point.z)) {
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
