import { Mesh, Vector2, Vector3 } from 'three';
import { Branch } from './Branch.js';
import { getNextTokenBranches } from '../systems/api.js';

// The idea is to build a breadth first tree 

class WorldTree {
    constructor(scene, loop, camera, font, maxTokens, kBranches,
        treeStyle, decodingStrategy, temp, p, beamWidth, llm, input,
        uiBranchHoveredCallback, uiBranchUnHoveredCallback, uiBranchClickedCallback) {
        this.queue = [];
        this.tree = {};

        this.scene = scene;
        this.loop = loop;
        this.font = font;
        this.camera = camera;
        this.maxTokens = maxTokens;
        this.input = input;
        this.decodingStrategy = decodingStrategy;
        this.temp = temp;
        this.p = p;
        this.llm = llm;
        this.beamWidth = beamWidth;
        this.kBranches = kBranches;
        this.treeStyle = treeStyle;
        this.uiBranchHoveredCallback = uiBranchHoveredCallback;
        this.uiBranchUnHoveredCallback = uiBranchUnHoveredCallback;
        this.uiBranchClickedCallback = uiBranchClickedCallback;
        this.lastClicked2dBranch = null;
        this.finishedGrowing = false;

        // The amount of branches we have at the final depth
        this.maxDepthBranches = kBranches ** maxTokens;
        // This would be enough to stack k branches on top of each other. We can use
        // this at the final layer.
        this.minHeightOfKBranches = (1 / kBranches) * 1;
        // This is how high the tree must be at least to fit in all branches in the final depth
        this.treeMinHeight = this.maxDepthBranches * this.minHeightOfKBranches;
        // Since we now know how high the last layer will be, we can recrusveily calculate the layers before
        this.maxHeightPerDepth = []
        this.maxHeightPerDepth[maxTokens] = this.treeMinHeight;
        for (var i = maxTokens - 1; i >= 0; i--) {
            this.maxHeightPerDepth[i] = this.maxHeightPerDepth[i + 1] / kBranches;
        }
    }

    getTree() {
        return this.tree;
    }

    addBranchToTree(branch) {
        this.tree[branch.getId()] = branch;
    }

    getFinishedGrowing() {
        return this.finishedGrowing;
    }

    setFinishedGrowing(finishedGrowing) {
        this.finishedGrowing = finishedGrowing;
    }

    async init() {

    }

    /**
     * Adds the given branch to the 2D tree UI
     * @param {} branch 
     */
    addTo2DWorldTree(branch) {
        let $layersContainer = $('#ui-tree-container .layers');
        const curLayer = branch.getDepth();

        // Check if this layer already exists in the UI. Else, we create a container for it
        let exists = false;
        $layersContainer.find('.layer').each(function () {
            if ($(this).data('layer') == curLayer) {
                exists = true;
                return;
            }
        });

        if (!exists) {
            $layersContainer.append(
                `<div class="layer" data-layer=${curLayer}>
                    <div class="flexed align-items-center justify-content-between">
                        <label class="mb-0"><i class="fas fa-layer-group"></i> ${curLayer}</label>
                        <button class="toggle-btn btn" onclick="$(this).closest('.layer').find('.layer-content').toggle()">
                            <i class="fas fa-chevron-down"></i>
                        </button>
                    </div>
                    
                    <div class="layer-content">
                    </div>
                </div>`
            );
        }

        // The layer container now definetly exists and we can fill in the branches
        let $layer = $layersContainer.find(`.layer[data-layer=${curLayer}] .layer-content`);

        const prob = branch.getProb();
        let probColor = 'lightgray';
        if (prob < 0.15) probColor = 'tomato';
        else if (prob > 0.5) probColor = 'limegreen';

        // We want to show the branches full text without the input.
        const text = branch.getContext().replace(this.input, '');
        const originalWayClass = branch.getIsOriginalWay() ? 'underlined font-weight-bold' : '';

        const $branchElement = $(`
            <div class="layer-branch ${originalWayClass}" data-id="${branch.getId()}">
                <div class="w-100 flexed align-items-center justify-content-between">
                    <p class="mb-0 mr-2">
                        ${text}
                    </p>
                    <p class="mb-0 w-auto" style="color:${probColor}">
                        ${prob} <i class="fas fa-code-branch"></i>
                    </p>
                </div>
                <div style="background-color:${probColor}" class="dot"></div>
            </div>
        `);
        $layer.append($branchElement);

        // Attach any events here. Here we handle the hovering over and out of the branch.
        $branchElement.hover(
            () => this.uiBranchHoveredCallback(branch),
            () => this.uiBranchUnHoveredCallback(branch)
        );

        $branchElement.click(() => {
            // When clicked, we hihglight the current clicked branch and give a callback
            const $container = $('#ui-tree-container .layers .layer-content');
            if (this.lastClicked2dBranch != null)
                $container.find(`.layer-branch[data-id="${this.lastClicked2dBranch.getId()}"]`).removeClass('selected-layer-branch');
            let $clickedUIBranch = $container.find(`.layer-branch[data-id="${branch.getId()}"]`);
            $clickedUIBranch.addClass('selected-layer-branch');
            this.lastClicked2dBranch = branch;
            this.uiBranchClickedCallback(branch);
        });
    }

    /**
     * Inits the WorldTree and starts the growing process.
     * @param {*} inputText 
     */
    async plant(inputText, startPos, finishedCallback) {
        var root = new Branch(inputText, 0, startPos);
        root.setWorldObjectPos(new Vector3(startPos.x, 0, 0));
        root.setIsOriginalWay(true);

        this.queue.push(root);

        while (this.queue.length > 0) {

            // This means the tree growing was interrupted
            if (this.finishedGrowing) {
                // There might be some branches which are in the datastructure
                // but which haven't been placed yet. We can't use these, so delete them.
                this.tree = Object.fromEntries(
                    Object.entries(this.tree).filter(([key, value]) => value.getWorldObject() != null)
                );
                break;
            }

            let branch = null;
            if (this.treeStyle == 'breadth-first')
                branch = this.queue.shift();
            else if (this.treeStyle == 'depth-first')
                branch = this.queue.pop();

            // We dont want the root to grow, that already exists.
            if (branch.getDepth() != 0) {
                // This makes the branch grow in 3D
                branch.grow(this.font,
                    this.scene,
                    this.loop,
                    this.camera,
                    this.maxHeightPerDepth[branch.getDepth()],
                    this.kBranches,
                    this.maxTokens);
                // And update the 2d tree UI with the new branch
                this.addTo2DWorldTree(branch);
            }

            const nextBranches = await getNextTokenBranches(branch.getContext(), this.kBranches,
                this.temp, this.p, this.beamWidth, this.decodingStrategy, this.llm);
            // console.log(nextBranches);
            const newStep = nextBranches.steps[0];

            // We need the probability and their corresponding tokens sorted 
            let tokensWithProb = [];
            for (var k = 0; k < newStep.top_k_tokens.length; k++) {
                tokensWithProb.push({
                    token: newStep.top_k_tokens[k],
                    prob: newStep.top_k_probs[k]
                });
            }

            // Sort the k branches according to the tree style
            if (this.treeStyle == 'breadth-first')
                tokensWithProb = tokensWithProb.sort((a, b) => b.prob - a.prob);
            else if (this.treeStyle == 'depth-first')
                tokensWithProb = tokensWithProb.sort((a, b) => a.prob - b.prob);

            for (var i = 0; i < tokensWithProb.length; i++) {
                const next = tokensWithProb[i];
                // We let the model end when it wants to.
                if (next.token == '<|endoftext|>' || next.token === 'EOS') continue;

                // For each new fetched branch, add a branch to the queue and repeat
                const nextBranch = new Branch(
                    newStep.context + " " + next.token, // New context
                    branch.getDepth() + 1, // Add a depth
                    branch.getStartPos()
                );
                nextBranch.setParentBranch(branch);
                nextBranch.setStep(next.token);
                nextBranch.setProb(next.prob);
                nextBranch.setOrder(i);

                // Mark if this branch is the original way, meaning this would be the default output of the llm.
                if (branch.getIsOriginalWay()) {
                    if ((this.treeStyle == 'breadth-first' && i == 0)
                        || (this.treeStyle == 'depth-first' && i == tokensWithProb.length - 1))
                        nextBranch.setIsOriginalWay(true);
                }
                // Store the next branch as a child in the parent
                branch.addChild(nextBranch);

                // If we reached the max desired output, break
                if (nextBranch.getDepth() == this.maxTokens) break;
                // else put it into the queue.
                this.queue.push(nextBranch);
            }
            this.addBranchToTree(branch);
        }

        this.setFinishedGrowing(true);
        console.log(this.tree);
    }
}

export { WorldTree };
