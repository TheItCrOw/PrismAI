<div align="center">
  <h1><b>PrismAI</b><br/><h3>Dissecting AI into the Spectrum of Transparency</h3></h1>
  <hr />
</div>

> PrismAI is a framework that aims to make AI more transparent and the outputs of language models more comprehensible. PrismAI is currently built around the 🤗 API which makes it easy to wrap around existing and new projects. The project is a collection of components, each with different functionalities but bundled under the bonnet of PrismAI's primary goal.
**The repo is split into multiple components, each of which can be used independently or together.**

<hr />

<div align="center">
  <h1>HyperCausal</h1>
</div>

<div align="center">
  <a href="https://dl.acm.org/doi/abs/10.1145/3648188.3677049"> <img src="https://img.shields.io/static/v1?label=Hypertext%202024&message=Paper&color=green&style=for-the-badge&logo=acm" alt="Paper: - ACM"></a>
  <a href="http://hypercasual.texttechnologylab.org/"><img src="https://img.shields.io/static/v1?label=&message=Demo&color=orange&style=for-the-badge&logo=abstract" alt="Demo"></a>
  <a href="https://www.texttechnologylab.org"><img src="https://img.shields.io/static/v1?label=&message=Text+Technology+Lab&color=blue&style=for-the-badge&logo=buffer" alt="Text Technology Lab"></a>
  <br/>
</div>
<br/>
<p align="center">
  One component of PrismAI is <b>HyperCausal: Visualizing Causal Inference in 3D Hypertext</b>, presented and published at the 35th ACM Conference on Hypertext and Social Media, 2024.
</p>
<div align="center">
  <video src="https://github.com/user-attachments/assets/a5649f92-13ed-4564-9c0d-80e0e9451bf4"/>
</div>

## About

![Loop2-ezgif com-optimize](https://github.com/TheItCrOw/PrismAI/assets/49918134/9701cc27-2e13-4529-92d6-a0a48e801533)

The last step of each Causal or Generative Language Model is to choose the next token for the sequence, which it does by calculating a probability over each token in the vocab and then picking the one with the highest probability. HyperCausal visualizes this process by not only looking at the token with the highest probability, but also the k-nearest neighbors and then branching out more possible sequences the model could or would have generated.

![Animation-ezgif com-optimize](https://github.com/TheItCrOw/PrismAI/assets/49918134/e2b40928-8bfa-4ae6-9390-22629df0c9b0)

HyperCausal is visualized with Three.js and uses PrismAI's core library to generate the probabilities and inference from the LLMs. This is done to:

- see how confident the llm was when printing out a given token.
- recognize what sequences could have been generated and what their chances were. Truth be told, LLMs are less sure about their predictions than you might think and especially given named-entities, the correct answer was hidden on an alternative path just 2% probability out of reach. 

### Usage

#### Python environment

Clone this repository

```
git clone https://github.com/TheItCrOw/PrismAI.git
```

Create an environment and install requirements

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Navigate to the HyperCausal package

```
cd PrismAI/src/hyper_causal
```

Start the webbrowser with the given command line args

```
python main.py
```

*Please also refer to the `-h` command to get more detailled info about the parameters.* 

Open the host in the browser and wait a moment for three.js to load. Depending on the targeted LLM and your setup this may take a moment. 

#### Docker

#### Pull and run

From Hub:

```
docker run -d --name hyper-causal -p 5678:5678 docker.texttechnologylab.org/prismai/hyper_causal/hyper-causal:0.0.1
```

#### Or: build image from repo

Navigate to the root of the repo, then execute:

```
docker build -f src/hyper_causal/Dockerfile -t hyper-causal .
```

This should create the image. From there, run it:

```
docker run --name hyper-causal -p 5678:5678 hyper-causal
```

### Supported LLMs

HyperCausal and PrismAI in general supports all Causal Language Models that 🤗 has to offer. This includes the fine-tuned versions of these architectures. Please refer to their [documentation](https://huggingface.co/docs/transformers/tasks/language_modeling) for a full list of LLMs.

<hr />

<div align="center">
  <h1>Luminar: AI Detection</h1>
</div>

**Coming soon**.
