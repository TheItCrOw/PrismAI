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
import { Branch } from './components/Branch.js';

let camera;
let renderer;
let scene;
let loop;
let inputText;
let maxTokens;
let k;
let beamWidth;
let temp;
let p;
let llm;
let decodingStrategy;
let treeStyle;
let controls;
let raycaster;
let worldTree;

let lastHoveredCube = null;
let lastHoveredLines = [];
let lastHoveredSphere = null;

const tokenSpace = 0.2;
const tooltipWidth = 200;
const tooltipHeight = 80;

class World {
    constructor(container) {
        // Fetch the parameters of the user - this is basically the backend state.
        const $body = $('body');
        inputText = $body.data('input');
        maxTokens = $body.data('maxtokens');
        k = $body.data('k');
        treeStyle = $body.data('treestyle');
        beamWidth = $body.data('beamwidth');
        temp = $body.data('temp');
        p = $body.data('p');
        llm = $body.data('llm')
        decodingStrategy = $body.data('decodingstrategy');

        camera = createCamera();
        renderer = createRenderer();
        scene = createScene();
        loop = new Loop(camera, scene, renderer);
        raycaster = createRaycaster();

        // Add all events here
        renderer.domElement.addEventListener('click', this.handleOnMouseClick);
        renderer.domElement.addEventListener('mousemove', this.handleMouseMove);
        document.getElementById('continue-branch-btn').addEventListener('click', this.handleContinueBranch);
        document.getElementById('stop-generation-btn').addEventListener('click', this.handleStopTreeGrowing);

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
        worldTree = new WorldTree(scene, loop, camera, font, maxTokens, k, treeStyle, decodingStrategy, temp, p, beamWidth, llm);
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

    /**
     * Stops the growing of the tree.
     * @param {*} event 
     */
    handleStopTreeGrowing(event) {
        worldTree.setFinishedGrowing(true);
    }

    /**
     * Handler for the event when the user wants to continue a specifc branch
     * @param {*} event 
     */
    async handleContinueBranch(event) {
        // If the tree is still growing, we cant conitnue a branch exclsuively.
        if (!worldTree.getFinishedGrowing()) return;

        const branchId = $(this).closest('#UI').attr('data-branch');
        // This is the branch it was clicked on. That isnt necesseraly the
        // branch we want to continue as it could have children.
        let clickedBranch = worldTree.getTree()[branchId];
        let branch = clickedBranch;
        while (branch.getChildren().length > 0) {
            let curChild = branch.getChildren().find(b => b.getOrder() == branch.getOrder());
            // Sometimes this specific order isn't available. In that case take any child
            if (curChild === undefined) {
                curChild = branch.getChildren()[0];
            }
            // If this child hasn't been placed into the world yet due to stop, ignore it
            if (curChild.getWorldObject() == null) {
                break;
            }
            branch = curChild;
        }

        // Now, we get the next predicted token and continue for... let's say 6 more steps
        const steps = 6;
        for (let i = 0; i < steps; i++) {
            const nextStep = await getNextTokenBranches(branch.getContext(), k, temp, p, beamWidth, decodingStrategy, llm);
            const nextToken = nextStep.generated_text.trim();

            let nextBranch = new Branch(
                nextStep.steps[0].context + " " + nextToken,
                branch.getDepth() + 1,
                branch.getStartPos()
            );
            nextBranch.setParentBranch(branch);
            nextBranch.setStep(nextToken);
            nextBranch.setProb(nextStep.steps[0].top_k_probs[0]);
            nextBranch.setOrder(branch.getOrder());

            if (branch.getIsOriginalWay()) nextBranch.setIsOriginalWay(true);
            branch.addChild(nextBranch);

            // Now let the branch grow
            nextBranch.placeBranch(branch.getFont(),
                scene,
                loop,
                camera,
                branch.getWorldObjectPos().y);

            // Focus this branch
            createTargetView(nextBranch);

            worldTree.addBranchToTree(nextBranch);

            // So we continue the loop with the correct parent branch
            branch = nextBranch;
        }
    }

    /**
     * Handles the events of clicking onto a three.js object
     * @param {*} event 
     */
    handleOnMouseClick(event) {
        const intersects = getIntersectedObjects(event, raycaster, camera, scene);

        if (intersects.length > 0) {
            const obj = intersects[0].object;
            if (obj.geometry.type == 'BoxGeometry') {
                worldTree.plant(inputText, new Vector3(obj.position.x, 0, 0));
                camera.position.set(15, 0, 60);
                controls.autoRotate = true;
            } else if (obj.type == 'Line') {
                createTargetView(obj.userData.toBranch);
                controls.autoRotate = true;
            } else if (obj.geometry.type == 'SphereGeometry') {
                createTargetView(obj.userData.edge.userData.toBranch);
                controls.autoRotate = true;
            }
        } else {
            $('#UI').removeClass('UI-left');
            controls.autoRotate = false;
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
                showTooltip(hoveredObject, '- Generate -');
            } else if (hoveredObject.type == 'Line') { // And highlight lines
                showTooltipMoving(event, hoveredObject, hoveredObject.userData.prob);
                hoveredObject.material.color.set('gold');
                hoveredObject.material.opacity = 1;
                lastHoveredLines.push(hoveredObject);

                // We only want to highlight the first object, dehighlight the rest.
                for (var i = 1; i < intersects.length; i++) {
                    let cur = intersects[i];
                    if (cur.object.type == 'Line') {
                        dehighlightEdge(cur.object);
                    }
                }
            } else if (hoveredObject.geometry.type == 'SphereGeometry') {
                showTooltip(hoveredObject, hoveredObject.userData.token);
                lastHoveredSphere = hoveredObject;
                recursvileyHighlightEdges(hoveredObject.userData.edge);
            }
        } else {
            $('html, body').css('cursor', 'default');
            if (lastHoveredSphere != null) {
            }
            if (lastHoveredCube != null)
                lastHoveredCube.material.opacity = 0.5;
            for (var i = 0; i < lastHoveredLines.length; i++) {
                dehighlightEdge(lastHoveredLines[i]);
            }
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

function createTargetView(targetBranch) {
    // When we click onto a line, we wanna target and highlight the to branch
    const pos = new Vector3().copy(targetBranch.getWorldObjectPos());
    controls.target.copy(pos);
    pos.z += 20;
    camera.position.copy(pos);
    //controls.autoRotate = true;

    // Show the ui element on the left now
    let fullTextHtml = '';
    let parentBranch = targetBranch;
    while (parentBranch != null) {
        const token = parentBranch.getStep();
        if (token == null) break;
        fullTextHtml = `${token}<span class="prob">(${parentBranch.getProb()}%)</span><br/>` + fullTextHtml;
        parentBranch = parentBranch.getParentBranch();
    }
    $('#UI .full-text').html(inputText + '<br/>' + fullTextHtml);
    $('#UI .title').html(`${targetBranch.getStep()}`);
    $('#UI .header .prob').html(`(${targetBranch.getProb()}%)`);
    $('#UI').addClass('UI-left');
    $('#UI').attr('data-branch', targetBranch.getId());
}

function recursvileyHighlightEdges(hoveredObject) {
    // We want to recursviley highligh the parent lines
    let parentLine = hoveredObject;
    while (parentLine != null) {
        lastHoveredLines.push(parentLine);
        parentLine.material.opacity = 1;
        parentLine.material.color.set('gold');
        parentLine = parentLine.userData.parentEdge;
    }
}

function dehighlightEdge(line) {
    line.material.color.set(line.userData.defaultColor);
    line.material.opacity = line.userData.defaultOpacity;
}

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
    tooltip.style.left = (x - tooltipWidth / 2) + 'px';
    tooltip.style.top = (y - tooltipHeight / 2 - 100) + 'px';
    $(tooltip).fadeIn(200);
}

function showTooltipMoving(event, hoveredObject, content) {
    const canvasBounds = renderer.domElement.getBoundingClientRect();
    const tooltip = document.getElementById('tooltip');
    // Get mouse coordinates relative to the canvas
    const mouseX = event.clientX - canvasBounds.left;
    const mouseY = event.clientY - canvasBounds.top;

    // Update tooltip content and position
    tooltip.innerText = content;
    tooltip.style.left = mouseX - tooltipWidth / 2 + 'px';
    tooltip.style.top = mouseY - tooltipHeight - 60 + 'px';
    $(tooltip).fadeIn(200);
}