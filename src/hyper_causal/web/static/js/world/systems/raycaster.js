import { WebGLRenderer, PCFSoftShadowMap } from 'https://cdn.skypack.dev/three@0.132.2';
import { Raycaster, Vector2 } from 'three';

let mouse;

function createRaycaster(camera) {
    const raycaster = new Raycaster();
    raycaster.params.Line.threshold = 0.1;
    mouse = new Vector2();

    return raycaster;
}

function getIntersectedObjects(event, raycaster, camera, scene) {
    // Calculate mouse position in normalized device coordinates
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    // Raycast from camera to scene
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(scene.children, true);
    return intersects;
}

export { createRaycaster, getIntersectedObjects };
