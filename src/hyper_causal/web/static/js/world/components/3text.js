import { Mesh, MeshStandardMaterial, MathUtils, Vector3 } from 'three';
import { TextGeometry } from 'three/addons/geometries/TextGeometry.js';
import { FontLoader } from 'three/addons/loaders/FontLoader.js';

class ThreeText extends Mesh {

    constructor(font, color, text, size = 0.2) {
        super();
        this.material = new MeshStandardMaterial({ color: color });
        this.geometry = new TextGeometry(text, {
            font: font,
            size: size,
            height: size / 5,
        });
    }

    tick = (delta) => {
    };
}

function addTextOntoCube(font, text, cube, size = 0.2) {
    const questionMark = new ThreeText(font, 'black', text, size);
    questionMark.geometry.center();

    // Position the text on each face of the cube
    const positions = [
        { x: -0, y: -0, z: 0.25 }, // Front face
    ];

    positions.forEach(position => {
        const textClone = questionMark.clone();
        textClone.position.set(position.x, position.y, position.z);
        cube.add(textClone);
    });
}

export { ThreeText, addTextOntoCube };

