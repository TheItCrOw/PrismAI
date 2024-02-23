import { Mesh, Vector2, Vector3 } from 'three';
import { Branch } from './Branch.js';
import { getNextTokenBranches } from '../systems/api.js';

// The idea is to build a breadth first tree 

class WorldTree {
    constructor(scene, loop, font, maxTokens, kBranches) {
        this.queue = [];
        this.tree = [];

        this.scene = scene;
        this.loop = loop;
        this.font = font;
        this.maxTokens = maxTokens;
        this.kBranches = kBranches;

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

    async init() {

    }

    /**
     * Inits the WorldTree and starts the growing process.
     * @param {*} inputText 
     */
    async plant(inputText, startPos) {
        var root = new Branch(inputText, 0, startPos);
        root.setWorldObjectPos(new Vector3(startPos.x, 0, 0));

        this.queue.push(root);

        while (this.queue.length > 0) {
            const branch = this.queue.shift();
            // We dont want the root to grow, that already exists.
            if (branch.getDepth() != 0)
                branch.grow(this.font,
                    this.scene,
                    this.loop,
                    this.maxHeightPerDepth[branch.getDepth()],
                    this.kBranches,
                    this.maxTokens);

            const nextBranches = await getNextTokenBranches(branch.getContext());
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
            tokensWithProb.sort((a, b) => a.prob > b.prob);

            for (var i = 0; i < tokensWithProb.length; i++) {
                const next = tokensWithProb[i];
                // For each new fetched branch, add a branch to the queue and repeat
                const nextBranch = new Branch(
                    newStep.context + " " + next.token, // New context
                    branch.getDepth() + 1, // Add a depth
                    branch.getStartPos());
                nextBranch.setParentBranch(branch);
                nextBranch.setStep(next.token);
                nextBranch.setProb(next.prob);
                nextBranch.setOrder(i);

                // Store the next branch as a child in the parent
                branch.addChild(nextBranch);

                // If we reached the max desired output, break
                if (nextBranch.getDepth() == this.maxTokens) break;
                // else put it into the queue.
                this.queue.push(nextBranch);
            }
            // this.tree.push(branch);
        }
    }
}

export { WorldTree };
