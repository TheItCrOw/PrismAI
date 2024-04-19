import { MathUtils, Vector2, Vector3 } from 'three';
import { ThreeText, buildStrokeForThreeText } from './3text.js';
import { createEdge } from './edge.js';
import { createSphere } from './sphere.js';

function uuidv4() {
    return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, c =>
        (+c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> +c / 4).toString(16)
    );
}

class Branch {
    constructor(context, depth, startPos) {
        this.id = uuidv4();
        this.context = context;
        this.depth = depth;
        this.startPos = startPos;
        this.parentBranch = null;
        this.step = null;
        this.children = [];
        this.order = 0;
        this.edge = null;
        this.font = null;
        // The prob of the step token this branch had
        this.prob = -1;
        this.isOriginalWay = false;

        this.worldObject = null;
        this.worldObjectPos = null;
    }

    getId() { return this.id; }
    getContext() { return this.context; }
    getDepth() { return this.depth; }
    getStep() { return this.step; }
    getStartPos() { return this.startPos; }
    getParentBranch() { return this.parentBranch; }
    getChildren() { return this.children; }
    getWorldObject() { return this.worldObject; }
    getWorldObjectPos() { return this.worldObjectPos; }
    getOrder() { return this.order; }
    getEdge() { return this.edge; }
    getIsOriginalWay() { return this.isOriginalWay; }
    getProb() { return this.prob; }
    getFont() { return this.font; }

    setStep(step) { this.step = step; }
    setParentBranch(parentBranch) { this.parentBranch = parentBranch; }
    addChild(child) { this.children.push(child); }
    setOrder(order) { this.order = order; }
    setProb(prob) { this.prob = parseFloat(prob.toFixed(4)); }
    setWorldObjectPos(worldObjectPos) { this.worldObjectPos = worldObjectPos; }
    setIsOriginalWay(isOriginalWay) { this.isOriginalWay = isOriginalWay; }

    async init() {

    }

    /**
     * Takes in the world coordinates and all parameters required and places a branch in the world
     */
    placeBranch(font, scene, loop, camera, y) {
        // ========== First: Place the TextMesh into the world
        const textMesh = new ThreeText(font, 'white', this.step);

        textMesh.position.set(this.startPos.x + (this.depth * 5),
            y,
            1);
        textMesh.material.transparent = true;
        textMesh.material.opacity = 0;
        textMesh.renderOrder = 1;
        textMesh.position.z *= this.order * 2 * Math.PI;
        textMesh.geometry.center();
        textMesh.tick = (delta) => {
            textMesh.lookAt(camera.position);
        };

        loop.updatables.push(textMesh);
        loop.updatables.push(textMesh);
        scene.add(textMesh);

        this.worldObject = textMesh;
        this.worldObjectPos = textMesh.position;
        this.font = font;

        // Fade in the text gradually
        var fadeInInterval = setInterval(function () {
            textMesh.material.opacity += 0.1; // Adjust the speed of fade-in effect
            if (textMesh.material.opacity >= 1) {
                clearInterval(fadeInInterval);
            }
        }, 150);

        // ========== Second: Place the line to the parent branch
        // Add the edge to the parent branch
        this.edge = createEdge(this.parentBranch.getWorldObjectPos(), this.worldObjectPos);
        this.edge.userData.prob = `${this.prob}%`;
        this.edge.userData.parentEdge = this.getParentBranch().getEdge();
        this.edge.userData.defaultColor = 'white';
        this.edge.userData.toBranch = this;
        // According to the probability we give different opacty
        this.edge.material.opacity = 0.05 + this.prob;

        loop.updatables.push(this.edge);
        scene.add(this.edge);

        // ========== Third: Place a sphere around the text to make it look like a node
        // Also add a sphere to each text
        const sphereMesh = createSphere(0.3, 32, 16);
        sphereMesh.renderOrder = 2;
        sphereMesh.position.copy(textMesh.position);
        sphereMesh.material.opacity = this.edge.material.opacity;
        sphereMesh.userData.token = this.step;
        sphereMesh.userData.defaultOpacity = sphereMesh.material.opacity;
        sphereMesh.userData.fullText = this.context;
        sphereMesh.userData.edge = this.edge;

        loop.updatables.push(sphereMesh);
        scene.add(sphereMesh);

        if (this.isOriginalWay) {
            this.edge.material.color.set('limegreen');
            this.edge.userData.defaultColor = 'limegreen';
            this.edge.material.opacity = 1;
        }
        this.edge.userData.defaultOpacity = this.edge.material.opacity;
    }

    /**
     * Visualizes this branch in the world. This can happen async, we don't need to wait for it typically. 
     */
    async grow(font, scene, loop, camera, maxDepthHeight, kBranches, maxTokens) {

        // How much height do we have per branch?
        let y = 0;
        const heightPerBranches = maxDepthHeight / kBranches;
        y = this.parentBranch.getWorldObjectPos().y;
        y = y + (this.order / heightPerBranches * maxTokens * kBranches);

        this.placeBranch(font, scene, loop, camera, y);
    }
}

export { Branch };
