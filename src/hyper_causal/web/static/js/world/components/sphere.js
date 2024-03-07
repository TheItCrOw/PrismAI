import {
    Mesh,
    MathUtils,
} from 'https://cdn.skypack.dev/three@0.132.2';
import { MeshBasicMaterial, MeshStandardMaterial, SphereGeometry } from 'three';

function createSphere(radius, widthSegment, heightSegment) {
    const geometry = new SphereGeometry(radius, widthSegment, heightSegment);

    const material = new MeshBasicMaterial({
        color: 'gold'
    });;
    const sphere = new Mesh(geometry, material);

    sphere.material.transparent = true;
    sphere.material.opacity = 0.5;

    const radiansPerSecond = MathUtils.degToRad(30);

    sphere.tick = (delta) => {
    };

    return sphere;
}

export { createSphere };

