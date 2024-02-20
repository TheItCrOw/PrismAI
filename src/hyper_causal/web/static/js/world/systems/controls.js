import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

function createControls(camera, canvas) {
    const controls = new OrbitControls(camera, canvas);

    // damping and auto rotation require
    // the controls to be updated each frame

    //controls.autoRotate = true;
    controls.enableDamping = true;

    controls.tick = () => controls.update();

    return controls;
}

export { createControls };