<div align="center">
  <h1><b>PrismAI</b><br/><h3>Dissecting AI into the Spectrum of Transparency</h3></h1>
  <hr />
</div>

> PrismAI is a framework that aims to make AI more transparent and the outputs of language models more comprehensible. PrismAI is currently built around the 🤗 API which makes it easy to wrap around existing and new projects. The project is a collection of components, each with different functionalities but bundled under the bonnet of PrismAI's primary goal.

## HyperCausal

One component of PrismAI is **HyperCausal: A spacial visualization of Causal Language Models decision making through a Hypertext environment.**

![Loop2-ezgif com-optimize](https://github.com/TheItCrOw/PrismAI/assets/49918134/9701cc27-2e13-4529-92d6-a0a48e801533)

The last step of each Causal or Generative Language Model is to choose the next token for the sequence, which it does by calculating a probability over each token in the vocab and then picking the one with the highest probability. HyperCausal visualizes this process by not only looking at the token with the highest probability, but also the k-nearest neighbors and then branching out more possible sequences the model could or would have generated.

![Animation-ezgif com-optimize](https://github.com/TheItCrOw/PrismAI/assets/49918134/e2b40928-8bfa-4ae6-9390-22629df0c9b0)

HyperCausal is visualized with Three.js and uses PrismAI's core library to generate the probabilities of the LLMs.

**Coming Soon how to use HyperCausal**

## Luminar AI Detection

**Coming soon**.
