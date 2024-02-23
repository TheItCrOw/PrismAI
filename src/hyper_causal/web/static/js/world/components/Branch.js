import { Vector2, Vector3 } from 'three';
import { ThreeText } from './3text.js';
import { createEdge } from './edge.js';

class Branch {
    constructor(context, depth, startPos) {
        this.context = context;
        this.depth = depth;
        this.startPos = startPos;
        this.parentBranch = null;
        this.step = null;
        this.children = [];
        this.order = 0;
        // The prob of the step token this branch had
        this.prob = -1;

        this.worldObject = null;
        this.worldObjectPos = null;
    }

    getContext() { return this.context; }
    getDepth() { return this.depth; }
    getStep() { return this.step; }
    getStartPos() { return this.startPos; }
    getParentBranch() { return this.parentBranch; }
    getChildren() { return this.children; }
    getWorldObject() { return this.worldObject; }
    getWorldObjectPos() { return this.worldObjectPos; }
    getOrder() { return this.order; }

    setStep(step) { this.step = step; }
    setParentBranch(parentBranch) { this.parentBranch = parentBranch; }
    addChild(child) { this.children.push(child); }
    setOrder(order) { this.order = order; }
    setProb(prob) { this.prob = prob; }
    setWorldObjectPos(worldObjectPos) { this.worldObjectPos = worldObjectPos; }

    async init() {

    }

    /**
     * Visualizes this branch in the world. This can happen async, we don't need to wait for it typically. 
     */
    async grow(font, scene, loop, maxDepthHeight, kBranches, maxTokens) {
        const textMesh = new ThreeText(font, 'white', this.step);

        // How much height do we have per branch?
        let y = 0;
        const heightPerBranches = maxDepthHeight / kBranches;
        y = this.parentBranch.getWorldObjectPos().y;
        y = y + ((this.order - 1) / heightPerBranches * maxTokens * kBranches);

        textMesh.position.set(this.startPos.x + (this.depth * 15),
            y,
            this.startPos.z);
        textMesh.material.transparent = true;
        textMesh.material.opacity = 0;

        loop.updatables.push(textMesh);
        scene.add(textMesh);
        this.worldObject = textMesh;
        this.worldObjectPos = textMesh.position;

        // Fade in the text gradually
        var fadeInInterval = setInterval(function () {
            textMesh.material.opacity += 0.1; // Adjust the speed of fade-in effect
            if (textMesh.material.opacity >= 1) {
                clearInterval(fadeInInterval);
            }
        }, 150);

        // Add the edge to the parent branch
        const edge = createEdge(this.parentBranch.getWorldObjectPos(), this.worldObjectPos);
        loop.updatables.push(edge);
        scene.add(edge);
    }
}

export { Branch };
