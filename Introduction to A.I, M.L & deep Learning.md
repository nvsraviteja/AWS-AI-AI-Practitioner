# What is AI?
It is a computer or a machine that has the ability to perform task that require human intelligence like reasoning , problem solving, decision making etc. 

It is also a simulation of human intelagince in machine that are programed to think, reason, and make decisions. 

AI enables machines to perform task that requires human intelligence such as:

1. **Learning**
2. **Reasoning**
3. **Problem-solving**
4. **Perception** (ability to understand something by sensing it like vision, sound, recognition etc.)


AI is broadly categorized into:
- **Narrow AI**: AI systems designed for specific tasks (e.g., virtual assistants, recommendation systems).
- **General AI**: Hypothetical AI that can perform any intellectual task a human can do.
- **Superintelligent AI**: A future concept where AI surpasses human intelligence.

AI technologies include machine learning, deep learning, computer vision, robotics, and more.

## Applications of AI 
AI applications span various domains: healthcare, finance, transportation, entertainment, and more. 

## Examples of AI:
- Chatbots
- Autonomous vehicles
- Personalized recommendations
- Fraud detection
- Medical diagnosis.

## History of AI: 
* 1950s - Birth of AI research
    *  Alan Turing's "Turing Test."
    * John McCarthy
* 1960s - Early AI milestones
   * Introduction of ELIZA chatbot.
* 1970s - Expert systems emerged.
  * MYCIN medical expert system to detect bacterial infections.
* 1980s - Neural networks gained popularity.
* 1990s - Machine learning algorithms advanced and Data mining techniques developed.
* 1997 - IBM's Deep Blue beats chess champion Garry Kasparov.
* 2010s - Deep learning revolution.
    * AlphaGo defeats Go world champions.
* 2020s - Virtual assistant, autonomous vehicles, health care diagnostics etc.
* Present Day - AI integrated into daily life.

# What is Neural Network?
It is inspired by human brain because of its functionality and structure. it is made up of layers connected neurons which process information. It learns from data and improves over time.
Each neuron combines the weighted input and applies an activation function for non-linearity and produces output.

- Non Linearity: by this the modal can learn more complex patterns or data and represents the relationship between inputs and outputs. Eg: Curves, intersections, etc.
- Linearity: the modal can learn only simple patterns like straight lines

# Structure of Neural Network
1. Input layer: takes raw features from the dataset.
2. Hidden layer: this process one or more layers of input data. These layers learns how to represent the complex data by combining the inputs in various ways. 
3. Output layer: it process the final result based on the hidden layer.

## Mathematical representation of neural network
* Output = (weights * input + bias) * activation
  * weights x inputs - each input has a weight and multiplied with their respective weights.
  * bias - A bias value is added to adjust the position of line.
  * Activation - it goes through an activation process (function), which decides the final output and adds non-linearity to the model.

Example:
  * Inputs x1 = 2, x2 = 3
  * Weights w1 = 0.5, w2 = 0.8
  * Bias b = 0.3
  * Output y = (w1*x1 + w2*x2 + b)
        * y = (0.5*2 + 0.8*3 + 0.3)
        * y = (1+2.4+0.3)
        * y = 3.7

# What is Machine Learning?
Machine learning is a subset of artificial intelligence that focuses on developing algorithms and models that enable computers to learn from data without being explicitly programmed. ML allows machines to improve their performance on a given task by analyzing large amounts of data.

- Instead of writing rules we provide the data and it will figure out the patterns or relationships on its own.
- as more data is given to the model, it becomes better at predicting outcomes and performs well on new unseen data.
- Instead of programming manually, the algorithm learns the rules from the data.
- It has different types of tasks depending on the problem you are solving. eg: classification, regression, clustering etc.
## Types of Machine Learning:
- Supervised learning: The algorithm is trained using labeled data, where both input and desired output are provided. It learns to predict the correct output for new instances.
- Unsupervised learning: The algorithm works with unlabeled data, discovering patterns and structures within the data itself.
- Reinforcement learning: An agent interacts with an environment, receiving rewards or penalties based on actions taken. It learns optimal behavior through trial and error.

AI is not equal to ML 
- AI, it can be any model or system that mimics human intelligence.
- ML is a type of AI that uses statistical methods to learn from data and make predictions or decisions.

# Core ML Terminologies (Very imp for exam)
* Dataset
    * The collection of input data used to train a model.
* Training
    * Process where model learns patterns using training data.
* Inference
    * Using the trained model to make predictions.
* Model
    * The learned pattern or rule extracted from training data.
    * The “brain” created after training.
* Feature
    * Input variables used for prediction.
* Label
    * Correct answers in supervised learning.

# What is Deep Learning?
Deep learning is a subset of machine learning focused on artificial neural networks with multiple layers. The structure is inspired by the human brain. It is designed so that it can learn from large amount of data. It excels at processing complex data like images, audio, and text. By training these networks on vast datasets, deep learning achieves remarkable results in areas like image recognition, natural language understanding, and speech synthesis.

- Artificial neural network uses nodes and connections to process information. 
- It can process complex patterns and relationships in data.
- It is used for tasks like image recognition, natural language processing, speech recognition etc.
- It is more advanced then traditional machine learning models.
- It can understand very complicated patterns in data such as recognizing face or translating languages.
- It has multiple layers of learning and processing units called neurons. As it has multiple layers it is called deep learning.
- It has multiple layers because each layer extract different levels of features from the input data.

# What is Generative AI?
Generative AI is a subset of Deep learning which refers to AI systems capable of creating original content, such as text, images, music, or videos. They generate new material rather than just classifying existing data. This technology leverages powerful algorithms and large datasets to produce creative outputs.

- It generates new content instead of just classifying existing data.
- It creates original content like text, images, music, or videos.
- It uses powerful algorithms and large datasets to produce creative outputs.
- It is used for tasks like generating art, writing stories, composing music, designing products etc.
- It is used in industries like advertising, marketing, gaming, education etc.
- It is used to create realistic images, animations, and simulations.
 Generative AI uses models to generate data which it is trained on.
Major types of Gen AI Models:
- Foundation Models
- Large Language Models (LLMs)
- Diffusion Models
- GANs (Generative Adversarial Networks)
- VAEs (Variational Autoencoders)
- Autoregressive Models (Non-LLM)
- Flow-based Models
- Retrieval-Augmented Models (RAG Models)
- Multimodal Models
- Vision-Language Models (VLMs)
- Reinforcement Learning–Based Generative Models

# What is foundation Model?
It is a large general purpose AI model trained on a huge amount of data that can be adapted for many different tasks. They serve as a starting point for specialized models tailored to specific needs.
By this we can build or tune our AI accordingly instead of building from scratch.
It can be reused across multiple projects and tasks, saving development time and resources.
This models are trained on a wide variety of input data.

# What is Large Language Model(LLM)?
Large language models are a type of generative AI model specifically designed for understanding and generating human-like text. 
They are trained on very large amount of text data.
LLM learns the patterns and structure of language by reading lot of text, articles
They excel at tasks like language translation, summarization, question answering, and even creative writing.
LLMs have achieved remarkable success due to their ability to understand context and generate coherent responses.
LLMs can perform tasks like:
- Text generation
- Translation
- Summarization
- Question answering
- Creative writing and content

# what is Transformers? 
A transformer is an advanced architecture used in deep learning models.
They are particularly useful for Natural Language Processing(NLP).
It process a sentence as a whole and assign importance to each word in the sentence.
Older models that process one word at a time transformers take the entire sentence at once together and decide how important each word is.
this enables them to understand the context better.
It uses attention mechanism so that it can assign more importance to specific words in the sentence.

## What it can do 
- Understand text
- Generates human like text
- FAQ
- Summarizing 
- translation
- creating conversational agents

# What is Embeddings?
Embeddings are numerical representations of words or phrases in a high-dimensional space. They capture semantic meaning and allow similarity comparisons.
Text -> Numbers -> model can understand it.

# AWS AI Services:
- Amazon Bedrock
- Amazon SageMaker
- Amazon Rekognition
- Amazon Textract
- Amazon Comprehend
- Amazon Polly
- Amazon Lex
- Amazon Kendra
- Amazon Translate
- Amazon Transcribe
- Amazon Forecast


