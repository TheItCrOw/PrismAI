import { BufferGeometry, Line, LineBasicMaterial } from 'three';

function createEdge(fromVec3, toVec3) {

    const geometry = new BufferGeometry().setFromPoints([fromVec3, toVec3]);
    const material = new LineBasicMaterial({ color: 'gold' }); // Red color for example

    const line = new Line(geometry, material);
    line.tick = (delta) => {
    };

    return line;
}

export { createEdge };

