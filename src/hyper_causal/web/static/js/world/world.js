import { FontLoader } from 'three/addons/loaders/FontLoader.js';

import { createCamera, moveCameraX } from './components/camera.js';
import { createCube } from './components/cube.js';
import { createLights } from './components/lights.js';
import { createScene } from './components/scene.js';
import { ThreeText, addTextOntoCube } from './components/3text.js';

import { createRenderer } from './systems/renderer.js';
import { createControls } from './systems/controls.js';
import { Resizer } from './systems/Resizer.js';
import { Loop } from './systems/Loop.js';
import { Raycaster, Vector2, Vector3 } from 'three';
import { createRaycaster, getIntersectedObjects } from './systems/raycaster.js';
import { getNextTokenBranches } from './systems/api.js';
import { WorldTree } from './components/WorldTree.js';

let camera;
let renderer;
let scene;
let loop;
let inputText;
let maxTokens;
let k;
let controls;
let raycaster;
let worldTree;

let lastHoveredCube = null;

const tokenSpace = 0.2;
const tooltipWidth = 200;
const tooltipHeight = 80;

class World {
    constructor(container) {
        inputText = $('body').data('input');
        maxTokens = $('body').data('maxtokens');
        k = $('body').data('k');
        camera = createCamera();
        renderer = createRenderer();
        scene = createScene();
        loop = new Loop(camera, scene, renderer);
        raycaster = createRaycaster();

        // Add all events here
        renderer.domElement.addEventListener('click', this.handleOnMouseClick);
        renderer.domElement.addEventListener('mousemove', this.handleMouseMove);

        container.append(renderer.domElement);

        controls = createControls(camera, renderer.domElement);
        const { ambientLight, mainLight } = createLights();

        loop.updatables.push(controls);
        scene.add(ambientLight, mainLight);

        const fontLoader = new FontLoader();
        fontLoader.load('static/fonts/Open_Sans_Regular.json', (font) => {
            this.onFontLoaded(font);
        });

        const resizer = new Resizer(container, camera, renderer);
    }

    // Once the font loaded, we can interact with and create text 
    onFontLoaded(font) {
        // Init the world tree.
        worldTree = new WorldTree(scene, loop, font, maxTokens, k);
        console.log(worldTree);
        // First print the starting input text
        this.typewriteNextSequenceIntoScene(inputText, font);
    }

    /**
     * A method that prints a given text into the world with a typewriter delay effect.
     * @param {} text 
     * @param {*} font 
     */
    typewriteNextSequenceIntoScene(text, font) {
        var tokens = text.split(' ');
        var tokenCounter = 0;
        var xOffset = 0;

        function addNextToken() {
            // If we haven't reached the end of the text
            if (tokenCounter < tokens.length) {
                const textMesh = new ThreeText(font, 'white', tokens[tokenCounter]);

                textMesh.geometry.computeBoundingBox();
                textMesh.position.set(xOffset, 0, 0); // Adjust the position with the xoffset
                // Set initial opacity to 0 for fade-in effect
                textMesh.material.transparent = true;
                textMesh.material.opacity = 0;

                loop.updatables.push(textMesh);
                scene.add(textMesh);

                const newTargetPos = new Vector3(xOffset * 1.15, 0, 0);
                controls.target.copy(newTargetPos);
                camera.position.lerp(new Vector3(newTargetPos.x, 0, 10), 0.9);

                tokenCounter++;
                xOffset += textMesh.geometry.boundingBox.max.x + tokenSpace;

                // Fade in the text gradually
                var fadeInInterval = setInterval(function () {
                    textMesh.material.opacity += 0.1; // Adjust the speed of fade-in effect
                    if (textMesh.material.opacity >= 1) {
                        clearInterval(fadeInInterval);
                    }
                }, 150);

                setTimeout(addNextToken, 50); // Do the next token
            } else {
                // Then add the generate button
                const cube = createCube(0.5, 0.5, 0.5);

                cube.geometry.computeBoundingBox();
                cube.position.set(xOffset + tokenSpace, cube.geometry.parameters.height / 6, 0);
                addTextOntoCube(font, '?', cube);

                scene.add(cube);
                loop.updatables.push(cube);
            }
        }

        addNextToken();
    }

    async init() {

    }

    handleOnMouseClick(event) {
        const intersects = getIntersectedObjects(event, raycaster, camera, scene);

        if (intersects.length > 0) {
            const obj = intersects[0].object;
            console.log(obj.position);
            worldTree.plant(inputText, new Vector3(obj.position.x, 0, 0));
        }
    }

    handleMouseMove(event) {
        const intersects = getIntersectedObjects(event, raycaster, camera, scene);

        if (intersects.length > 0) {
            // Pointer cursor
            $('html, body').css('cursor', 'pointer');
            const hoveredObject = intersects[0].object;
            // Let's highlight hovered cubes
            if (hoveredObject.geometry.type === "BoxGeometry") {
                hoveredObject.material.color.set('white');
                hoveredObject.material.opacity = 1;
                lastHoveredCube = hoveredObject;
                // Show a tooltip for the cube
                showTooltip(hoveredObject, '- Generate - ');
            }
        } else {
            $('html, body').css('cursor', 'default');
            if (lastHoveredCube != null)
                lastHoveredCube.material.opacity = 0.5;
            $('#tooltip').fadeOut(125);
        }
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

function showTooltip(hoveredObject, content) {
    // Position the tooltip above the hovered object
    const tooltip = document.getElementById('tooltip');
    const canvasBounds = renderer.domElement.getBoundingClientRect();
    const cubePosition = new Vector3();
    cubePosition.setFromMatrixPosition(hoveredObject.matrixWorld);
    const projectedPosition = cubePosition.project(camera);
    const x = (projectedPosition.x * 0.5 + 0.5) * canvasBounds.width;
    const y = (1 - (projectedPosition.y * 0.5 + 0.5)) * canvasBounds.height;

    // Update tooltip content and position
    tooltip.innerText = content;
    console.log(tooltip.style.height);
    tooltip.style.left = (x - tooltipWidth / 2) + 'px';
    tooltip.style.top = (y - tooltipHeight / 2 - 100) + 'px';
    $(tooltip).fadeIn(200);
}