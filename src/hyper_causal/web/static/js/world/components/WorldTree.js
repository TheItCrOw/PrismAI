import { Mesh, Vector2, Vector3 } from 'three';
import { Branch } from './Branch.js';
import { getNextTokenBranches } from '../systems/api.js';

// The idea is to build a breadth first tree 

class WorldTree {
    constructor(scene, loop, camera, font, maxTokens, kBranches,
        treeStyle, decodingStrategy, temp, p, beamWidth, llm) {
        this.queue = [];
        this.tree = {};

        this.scene = scene;
        this.loop = loop;
        this.font = font;
        this.camera = camera;
        this.maxTokens = maxTokens;
        this.decodingStrategy = decodingStrategy;
        this.temp = temp;
        this.p = p;
        this.llm = llm;
        this.beamWidth = beamWidth;
        this.kBranches = kBranches;
        this.treeStyle = treeStyle;
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
            if (branch.getDepth() != 0)
                branch.grow(this.font,
                    this.scene,
                    this.loop,
                    this.camera,
                    this.maxHeightPerDepth[branch.getDepth()],
                    this.kBranches,
                    this.maxTokens);

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
                if (next.token == '<|endoftext|>') continue;

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
