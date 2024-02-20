import { Mesh, MeshStandardMaterial, MathUtils, Vector3 } from 'three';
import { TextGeometry } from 'three/addons/geometries/TextGeometry.js';
import { FontLoader } from 'three/addons/loaders/FontLoader.js';

class ThreeText extends Mesh {

    constructor(font, color, text, size = 0.5) {
        super();
        this.material = new MeshStandardMaterial({ color: color });
        this.geometry = new TextGeometry(text, {
            font: font,
            size: size,
            height: size / 10,
        });
        this.geometry.center();
    }

    tick = (delta) => {
    };
}

export { ThreeText };

