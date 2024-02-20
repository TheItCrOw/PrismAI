import { FontLoader } from 'three/addons/loaders/FontLoader.js';

import { createCamera } from './components/camera.js';
import { createCube } from './components/cube.js';
import { createLights } from './components/lights.js';
import { createScene } from './components/scene.js';
import { ThreeText } from './components/3text.js';

import { createRenderer } from './systems/renderer.js';
import { createControls } from './systems/controls.js';
import { Resizer } from './systems/Resizer.js';
import { Loop } from './systems/Loop.js';

let camera;
let renderer;
let scene;
let loop;

class World {
    constructor(container) {
        camera = createCamera();
        renderer = createRenderer();
        scene = createScene();
        loop = new Loop(camera, scene, renderer);
        container.append(renderer.domElement);

        const fontLoader = new FontLoader();
        fontLoader.load('static/fonts/Open_Sans_Regular.json', (font) => {
            this.onFontLoaded(font);
        });

        const controls = createControls(camera, renderer.domElement);
        const { ambientLight, mainLight } = createLights();

        loop.updatables.push(controls);
        scene.add(ambientLight, mainLight);

        const resizer = new Resizer(container, camera, renderer);
    }

    // Once the font loaded, we can interact with and create text 
    onFontLoaded(font) {
        const text = new ThreeText(font, 'white', 'Test');
        loop.updatables.push(text);
        scene.add(text);
    }

    async init() {

    }

    render() {
        // draw a single frame
        renderer.render(scene, camera);
    }

    start() {
        loop.start();
    }

    stop() {
        loop.stop();
    }
}

export { World };