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
- Amazon Augmented AI
- Amazon Comprehend Medical & Transcribe Medical
- Amazon's Hardware for AI
- Amazon Transcribe
- Amazon Forecast

# Amazon Bedrock:
- It is a service provided by AWS to build Gen AI applications.
- It is fully managed by AWS, we don't need to manage anything.
- It can control the data used to train the model
- It follows Pay-Per-use pricing.
- Unified APIs
    * It can access and use different foundation models through a single consistent API interface.
    * Single way to access/use multiple Gen AI models in Amazon Bedrock.
- Can use a wide range of foundation models.

## RAG (Retrieval-Augmented Generation)
- It is a technique used to combine an LLm with external data sources.
- when LLMs don't know private data RAG feeds relevant documents or knowledge to the model at run time.

## Bedrock - Base foundation models:
- Amazon Titan
- Anthropic Claude
- Cohere Command
- Meta Llama
- Stability AI (image models)
- AI21 Labs
- Mistral AI (depending on region)
How to choose a foundation model:
- Model types, performance, requirements, capabilities, constraints, compliance.
- Level of customization, model size, inference option, license agreements, context window, latency
- Multimodal
- Use cases
- Performance
- Cost
- Availability
- Compliance
- Support
### Tokens
tokens are chunk of text that are processed by the model. Each token is assigned a unique number.
The length of the prompt determines the number of tokens.
Tokens are used to measure the complexity of the prompt and determine the cost of running the model.
eg: hello world -> it has 2 tokens

### EULA (End User License Agreement):
- It is a legal agreement between you and the provider of the software.
- It outlines the terms and conditions under which you may use the software.
- It includes restrictions on copying, modifying, distributing, or reverse engineering the software.
- It specifies warranties, liability limitations, indemnification obligations, termination rights, and dispute resolution procedures.

## Amazon Bedrock - Fine Tuning a Model
- Copy a pre-trained foundation model and train it with our own data.
- By this the modal will react and respond to our requests which is unique 
- Fine tuning will charge and change weights of base model.
- Foundation Modal + S3 = Fine tuned model.
### Instruction based Fine tuning
- Improves the performance of FM on domain specific tasks.
- Further trained on particular field or area ok knowledge 
- It uses prompt-response pair to train the Foundation Model.
- Fine-tuning on private data set

### Continued pre-training
- Pre-training on public data sets
- It will provide unlabeled data to continue the training of an Foundation model eg: any document, article, book etc.
- It is also called as domain adaption fine tuning.
- We can feed more and more data to the model for better accuracy.

### Single Turn Messaging
- Used for chatbots, customer support, and personalized interactions.
- It maintains conversation history and responds appropriately.
- It is a part of instruction based fine tuning.
- It is a system which contains the context of conversation
- Messages: Array of message objects
- Role: User/ assistant
- Content: Message content
- Context: Conversation history

### Multi turn messaging
- It is used for Chatbot and Virtual Assistants
- Supports long conversations and context management.
- Maintains memory of previous messages.
- It is a part of instruction based fine tuning.
- It is a system which contains the context of conversation
- Messages: Array of message objects
- Role: User/ assistant
- Content: Message content
- Context: Conversation history

### Transfer Learning
- In this when a model is trained on specific domain it can be reused to different domain and it will work perfectly fine.
- Fine tune is kind of transfer learning.
- It is a method of transferring knowledge from one model to another.

### Amazon Bedrock - Evaluating a model
- By Automatic Evaluation we can easily find the desired foundation model.
- Evaluate a model for quality control and then train the model accordingly.
- Built-in task types:
    * Text summarization
    * Question and answer
    * text classification
    * Open ended text generation

#### How it works: 
    Benchmark Questions -> Model to evaluate -> Generate Answers -> Compare Answers -> Get Score
- Grading or Model scores are calculated using various statical methods:
- Manual/Human evaluation:
    * Human evaluators assess the quality of generated responses.
    * Provide feedback on correctness, relevance, clarity, and coherence.
    * Identify areas for improvement and refine the model further.
- Benchmark Datasets
    * It is helpful to measure accuracy, speed, scalability, and efficiency.
    * Some datasets allow you to quickly detect any kind of bias and potential discrimination against a group of people 

- Accuracy
### Metrics to evaluate a model: (Not sure about it need to learn)
- Precision
- Recall
- F1 score
- BLEU(blue Language Evaluation Utility) score
- ROUGE score
- Perplexity

### Amazon Bedrock - RAG & Knowledge base
- It allows a foundation model to refer a data source outside of its training data.
    * Eg: Training data source is S3 but the required data is not in S3 then it will refer to other data source outside of its training data i.e: Vector database.
- When AI fails to retrieve data from its training data then it falls back to another data source like a vector database to retrieve the required data.
- It is used where real-time data is needed and the data is updated frequently.
### Vector Database
- It stores the previously asked quires and allows to quickly search for similar ones.
- How it works: Input text -> Embedding model -> Vector database -> similar queries -> Answer.

### Embedding model
- It is a type of AI model that converts data like text, images, video into a list of numbers that capture the meaning of data.
- So that similar text or idea gets similar numbers.

### Vector Database types:
* Amazon OpenSearch Service
    * It is a search & analytics database used for full-text search, Real time analytics, log & event monitoring, Vector similarity search.
    * It provides real-time similarity queries
        * Run fast, symmetric search using vector embeddings.
    * Store millions of vector embeddings
    * Scalable index management
        * can easily handle large volumes of data from multiple sources
    * Fast nearest neighbor search
        * Efficiently finds nearest neighbors in high dimensional spaces.
    * Secure and scalable deployment
        * Deployed securely and scalably in production environments.
- Amazon QLDB
    * It is a ledger database used for storing transactional records, financial transactions, audit trails, IoT events, supply chain data, regulatory compliance data.
    * It provides ACID properties (Atomicity, Consistency, Isolation, Durability)
    * It supports SQL queries
    * It provides real-time similarity queries
        * Run fast, symmetric search using vector embeddings.
    * Store millions of vector embeddings
- Amazon Document DB
- Amazon Aurora
- Amazon RDS for PostgreSQL
- Amazon Neptune

### RAG Data Sources:
- Amazon S3
- Confluence
- MS Sharepoint
- Salesforce
- Web Pages
- Custom API

### RAG Use Cases:
- Customer Service Chatbot
    * Knowledge base - Products, Troubleshooting guides.
    * RAG - Chatbot to answer clients
- Legal Research and Analysis 
    * Knowledge Base - Laws, Regulations, Case Studies.
    * RAG - Lawyers to research and analyze laws. Chatbot to ans law related queries.
- Product Recommendations
    * Knowledge Base - Product descriptions, reviews, specifications.
    * RAG - Customers to recommend products.
- Healthcare Diagnosis
    * Knowledge Base - Medical journals, patient records, clinical guidelines.
    * RAG - Doctors to diagnose patients.

## Gen AI Concepts:


### Tokenization:
Converting text into a sequence of tokens.
- Word based tokenization: text is split into individual words.
- Character based tokenization: text is split into characters.
- Subword tokenization: text is split into subwords or parts of words. eg: inconsistent -> in, consistent.

### Context Window:
- Input + Output tokens a model can handle.
- It means the number of tokens on LLM can consider when generating text
- The larger the window then we get more info
- Large context window require more memory and processing power.

### Embeddings:
- It creates vector over text, img, Audio etc.
    - 1st text is converted to tokens and then embedding creates vector over tokens.
    - Then embedding model creates vector over tokens.
- Embeddings are vectors that represent the meaning of text, images, audio etc.
- Embeddings are used to compare similarities between texts/images/audio etc.
- Embeddings are used to find similar texts/images/audio etc.

## Amazon Bedrock Guardrails:
- It controls the interaction between user and Foundation model.
- Used to filter Harmful content
- Enhanced privacy 
- It can create multiple guardrails to monitor user inputs that can violate the guardrail.
- It can be used to prevent harmful content from being generated by the model.
- It can be used to ensure that the model stays within certain parameters or boundaries.
- It can be used to protect sensitive information from being exposed.
- It can be used to enforce ethical standards and regulations.

## Amazon Bedrock Agents:
- Agents are integrated with other systems, services, databases and API to exchange data or initiate actions
- Uses RAG to retrieve information when required.
#### How it works
    Task -> Agent -> Bedrock Model -> Steps performed -> Result -> Agent -> Task + Result -> Bedrock model -> Final response -> Agent -> Final response to user.

## Amazon Bedrock & Cloudwatch:
In cloudwatch we have some metrics, logs & alarms.
- It helps us to monitor the usage of the model.
* identify if there are any errors in the model.
    * any security breaches in the model.
    * any performance issues in the model.
    * any compliance issues in the model.
- Model invocation logging is an account setting that enables logging of all LLM requests made for the models that are hosted on AWS Bedrock. 
- When ever or every invocations are stored in S3 or cloudwatch logs
- It includes text, images and embeddings.
- By this we can analyze the usage of the model and optimize it.
- Cloudwatch metrics are published from bedrock to cloudwatch.

## Amazon Bedrock Pricing:
### On Demand Pricing:
 * We only pay for what we have requested.
 * There is no upfront commitment or contract.
 * We only pay for what we use.
 * It is flexible and easy to scale up/down depending on demand.

### Text Models
 * It is charged for both input and output generated tokens. 
 * For example, if you ask a question and receive a response of 50 tokens, you would be charged for those 50 tokens plus the additional tokens used to generate the response.
 * Each token is assigned a unique number.
 * The length of the prompt determines the number of tokens.
 * Tokens are used to measure the complexity of the prompt and determine the cost of running the model.

 ### Embedding Models
 * It is charged for converting the paragraphs into tokens.

 ### Image Models
 * It is used to generating images from prompt and charged per image generated.

 ### Batch mode
 * It sends multiple requests at once and then stored in S3 bucket.
 * It is cheaper than invoking model individually.
 * Gets discounts upto 50%

 ### Provisioned Throughput
 * Purchase model units for certain time period.
 * Throughput- Max number of Input/Output tokens processed per second.
 * It is charged for throughput unit consumed.
 * Useful for predictable or high-volume workloads.

### Model Improvement Techniques Cost Order (Low to High)
1. Prompt Engineering
    * No additional computation or Fine-tuning required.
2. RAG 
    * Uses external Knowledge bases.
    * No Foundation model training required.
3. Fine-Tuning / Instruction based
    * Requires model training.
    * Foundation model is fine-tuned with specific instructions.
    * More expensive than prompt engineering and RAG.
4. Domain Adaption Fine-Tuning
    * Model is trained on specific dataset
    * Requires a large Computation.
    * Most expensive among all techniques.

## Prompt Engineering:
- Prompt is an instruction given to the model to get the desired output.
- To archive the expected output from AI then the instruction given should be proper and clear.
- Developing designing and optimizing prompts to enhance the output of AI.
- Improved prompting technique consists of:
    * Instructions - a task for the model
    * Context - external info to guide the model
    * Input data - what response we want
    * Output indicator - what we expect from the model like type and format.

### Negative Prompting:
* It is a technique to instruct the model on what not to include in the output.
- It helps to 
    * Avoid unwanted context
    * Maintain focus.
    * Enhance clarity.

### Prompt Performance Optimization;
1. System Prompts - We can instruct model that how it should reply or behave.
    * Example: "You are a helpful assistant."
2. Temperature (0 to 1) - Controls the creativity of the Model.
    * Low Temp (0.2) - It gives more accurate, predictable and repetitive answers.
        * Good for FAQ's.
        * By this the model will not use it's creativity.
    * High Temp (1.0) - The model becomes more creative and surprising.
        * Take risks and generate new ideas or even make up things.
        * Good for story telling and marketing
    * Low - Serious, Safe, Repetitive (FAQs, Code)
    * Medium - balanced, less creative (General, chat)
    * High - Fun, creative, unpredictable (Stories, adds)
3. Top P (0 to 1) - It controls how many word choices the AI considers when generating a response - based on their probability of being correct.
    * eg: Pick a word from the list of words which fits the next word . the size of list depends on the value of P.
    * Low Top P (0.3) - The model chooses from the top 30%
    * this makes the ans more consistent.
    * Used to get accurate and more clarity
    * High Top P (0.9) - The model chooses from the top 90% of options.
    * this makes the ans more creative.
    * Used to get more creative and diverse answers.
    * Model picks from a large variety of possible words
    * This creates more diverse and creative response
    * Good for creative work.
4. Top K - it controls how many top possible next words the AI can pick when generating a response 
    * Pick the next word from the list
    Low K (10)
    - Keep response clean and focused
    - It looks at top 10 most likely words
    High k (500)
    - It can choose a wider set of possible words 
    - make response more creative and unpredictable 
5. Length - It controls the maximum number of tokens the AI model can generate in a reply
    - Helps to avoid very long response. 
6. Stop sequences - By this we tell the model to stop generating response after seeing those special word or symbols.
7. Prompt Latency - We can say how fast the model responds.
    - its impacted by few parameters
        * the model size
        * model type
        * No. of tokens in the i/p or o/p

8. Frequency Penalty (0 to 1) - It penalizes frequent words in the output.
    * Low FP (0.2) - The model generates more varied outputs.
    * Used to avoid repetition and encourage diversity in responses.
    * High FP (1.0) - The model avoids repeating itself.
    * Used to maintain consistency and reduce redundancy in responses.

### Prompt Engineering Techniques
#### Zero-Shot Prompting
* It is a method where a LLM completes a task based on a written instruction, without being given any specific examples of the desired output
* It will be a task for model because we don't provide examples or any explicit training for that specific task
* Model's general knowledge will work on the task completely
* the larger and tuned model, the good results we get

#### Few Shots Prompting
* It is a technique that provides an AI model with a few examples of a task in the prompt to guide its output.
* We provide few examples to the model to perform the task
* if we provide only one example then it is called one shot or single shot

#### Chain of thought Prompting
* Dividing a task into steps so that we can get a proper answer for the query
* while solving a complex problem then we can use these type of prompting
* by sentence like 'think step by step' helps AI to provide a good or best output
* can be combined with zero and few-shots prompting

### Prompt Templates
This templates are used to make task easy by providing pre-made questions or instruction 
* What it do 
    - Take the user's text and turn it into a well structured prompt for the AI model
    - Organize and send the AI's response back
### Prompt Template Injections 
By this users can hijack our prompt and can access the prohibited or harmful content
* Protection
    - By adding explicit instructions to ignore any potential content
## AWS AI Services Overview
#### AWS Bedrock (Foundation Models (LLMs & GenAI))
It is like using AI models without giving any training
Purpose:

✔ Access LLMs (Claude, Titan, Llama, Mistral, etc.)

✔ Image models (Stable Diffusion)

✔ Embedding models

✔ Agents

✔ Guardrails

#### AWS Sagemaker (Custom Machine Learning or Custom AI)
Used to train a ML with own data
Purpose:
custom ML models.

✔ Build

✔ Train

✔ Deploy

Used when:
- You want your own ML model
- You need MLOps pipelines
- You need AutoML (Autopilot)

#### Amazon Rekognition (Vision)
Purpose:

✔ Image analysis

✔ Video analysis


✔ Label detection

✔ Face detection

✔ Emotion detection

✔ Moderation

Used for:
- Security cameras
- Content moderation
- Facial analysis
- Object detection

#### Amazon Textract (Document AI)
Purpose:

✔ Extract text from documents

✔ Extract tables

✔ Extract forms

✔ Read scanned PDFs

Used for:
- Invoices
- Bills
- ID cards
- Contracts

#### Amazon Comprehend (Natural Language Processing)
Purpose:

✔ Sentiment analysis

✔ Entity recognition

✔ Key phrase extraction

✔ Topics

✔ PII detection

✔ Custom classification (high-level)



Used for:
- Customer reviews
- Support tickets
- Text analytics

#### Amazon Polly (Speech AI)
Purpose:

✔ Text-to-speech

Use-case:
- Announcements
- Voice bots
- Accessibility apps

#### Amazon Transcribe (Speech AI)

Purpose:

✔ Speech-to-text

✔ Call analytics

✔ Real-time or batch transcription


Use-case:
- Customer support call analysis
- Meeting transcription

#### Amazon Lex (Conversational AI)
Purpose:

✔ Chatbot builder

✔ Voice bots

✔ Conversational agents

Used for:
- Customer service chatbots
- IVR systems
- Booking assistants

#### Amazon Kendra (Search)
Purpose:

✔ Enterprise search engine

✔ Natural-language queries

✔ “Google-like search” for enterprise documents
Use-case:
- Company knowledge base
- Internal document search

| Scenario                    | AWS Service       |
| --------------------------- | ----------------- |
| Use LLMs / GenAI            | **Bedrock**       |
| Fine-tune models            | **SageMaker**     |
| Image recognition           | **Rekognition**   |
| Extract text from documents | **Textract**      |
| Sentiment analysis          | **Comprehend**    |
| Speech → text               | **Transcribe**    |
| Text → speech               | **Polly**         |
| Build chatbot               | **Lex**           |
| Enterprise search           | **Kendra**        |
| Custom ML training          | **SageMaker**     |
| Document QA with your data  | **Bedrock + RAG** |

### AI Responsibilities
While building a model it is very important to make sure the AI is safe, fair, and trustworthy.

1. Bias
2. Fairness
3. Explainability
4. Privacy & Security

#### Bias
The AI may generate content which is unfair, skewed for certain groups of people

✔ Why it happens:
- Model trained on poor or unbalanced data
- Human bias in labeling
- Sampling errors
- Underrepresented categories

✔ Effects:
- Unfair predictions
- Discrimination
- Poor user experience

✔ AWS Tools to reduce bias:
- SageMaker Clarify → detects & explains bias in datasets and models
- Responsible AI Guardrails (Bedrock)

#### Fairness
The AI will give fair and behaves equally for all groups of people

✔ Goal:
- AI should not favor or discriminate against anyone based on:
    - gender
    - race
    - age
    - location
    - or any sensitive attribute

✔ How to ensure fairness:
- Balanced training data
- Bias detection tools (like Clarify)
- Regular monitoring of model outputs
- Human oversight (HITL)
- Fairness ≠ absence of bias completely, but minimizing it using best practices.

#### Explainability 
The AI will ensures you that you can understand why the AI made a decision

✔ Why it matters:
- Builds trust
- Helps debugging models
- Required for many industries (finance, healthcare, insurance)

✔ Examples:
- Why did the model reject a loan?
- Why did the model detect fraud?

✔ AWS Services:
- SageMaker Clarify → feature importance
- Bedrock Guardrails → helps explain why it blocked content

✔ Explainability helps:
- Users trust AI decisions
- Developers fix errorsAuditors verify compliance

#### Privacy Security
The AI should protect the user data from misuse and unauthorized access.


✔ Privacy (Protect personal data):
- Do not store sensitive info unnecessarily
- Anonymize data
- Avoid training on private user inputs
- Follow data retention policies
- AWS Bedrock, for example:
    - Does NOT use your prompts to train models
    - No data stored unless you enable logging

✔ Security (Protect systems & models):
- Encryption (at rest and in transit)
- IAM access control
- VPC endpoints for private access
- Secure APIs
- Audit logging (CloudTrail)

✔ Important for exam:
- AI responsibility = ensuring models are safe, private, non-harmful, and trustworthy.

#### Human-in-the-Loop (HITL)
This required human to confirm the final output of the AI
Used when:
Critical decisions (loans, medical, legal)
High-risk content

#### Model Drift
This happens when the data or information changes frequently which leads to model inaccuracy or the model accuracy gets dropped 
- Example: In 2020 you trained a model to predict the price of houses in a city as time changes the price of houses keep increasing but the model still predicts the same old rate.

# Amazon Sagemaker
It is a fully managed AWS service that provides tools and infrastructure for building, training, and deploying machine learning models.
It simplifies the workflow of ML by enabling data scientists and developers to create, train, and deploy models quickly and efficiently.
In a normal machine learning project, you follow these main steps:
- Collect, clean, and understand the data.
- Create and choose the right features.
- Train the model and adjust its settings.
- Deploy the model and keep monitoring it.
Each step normally needs different tools and experts, but Amazon SageMaker combines all of these stages into one unified platform.

#### Benefits of Amazon Sagemaker
- End-to-end ML service: 
    * Covers the entire ML lifecycle — data prep, training, deployment, and monitoring — all in one place.
- Integrated IDE: 
    * SageMaker Studio lets you build, train, and deploy models from a single interface.
- Easy to use: 
    * Tools like Autopilot allow even non-coders to create high-quality models quickly.
- Scalable: 
    * Automatically scales resources for training and inference without manual setup.
- Cost-effective: 
    * Features like Spot Training and multi-model endpoints help reduce costs.
- Supports all major ML frameworks: 
    * Works with TensorFlow, PyTorch, XGBoost, and more.
- Automated tuning: 
    * Automatically finds the best hyperparameters for top model performance.
- Simple deployment: 
    * Deploy models as real-time APIs or batch jobs with minimal effort.
- Continuous monitoring: 
    * Model Monitor checks for drift, anomalies, and performance issues.
- Model registry: 
    * Handles versioning, approvals, and deployment history.
- Strong integrations: 
    * Works seamlessly with AWS services like S3, Glue, Lambda, and CloudWatch.
- Secure and compliant: 
    * Offers VPC support, encryption, and IAM control to protect your data.

#### Use the Sagemaker when:
- You need to train your own model
- You want full control
- You need AutoML, MLOps, pipelines
- Bedrock’s pre-trained models are not enough 

### Features of Sagemaker:
1. SageMaker Studio (VS Code of ML)

SageMaker Studio is an all-in-one Machine Learning IDE, just like how VS Code is an all-in-one editor for developers.
It provides a single web-based interface where you can do everything needed for an ML project without switching tools.
What you can do in SageMaker Studio
- Build notebooks:
    * Create and run Jupyter notebooks for data exploration, model building, and experiments.
    * Train models:
    * Trigger training jobs directly from the interface using AWS-managed infrastructure.
    * Debug models:
    * Use built-in debugging tools to inspect training issues (overfitting, gradients, system errors).
    * Deploy models:
    * Deploy models to real-time or batch endpoints in a few clicks.
- In short:
    * SageMaker Studio = One place for the full ML lifecycle: build → train → debug → deploy.

2. SageMaker Notebooks

These are Jupyter notebooks hosted on AWS.
- Key points:
    * Used for experimentation, data analysis, and prototyping ML models.
    * Fully managed → AWS handles compute, storage, kernels, lifecycle.
    * Easy to share and collaborate with teams.
- Purpose:
    - Run ML code interactively during development.

3. SageMaker Training Jobs

Training Jobs let you train ML models on AWS-managed infrastructure at scale.

- Why use Training Jobs?
    - Your local machine may not handle large datasets or heavy models.
    - Training jobs let you use powerful CPUs/GPUs (like p3, g5 instances).
    - You only pay for the time the job runs.
* What they do:
    - Run your code in a containerized environment.
    - Automatically scale compute and storage.
    - Store model artifacts (trained model files) in S3 after completion.
- Use case:
    - When you need large-scale, efficient, and fast ML training.

4. SageMaker Inference

Once a model is trained, it needs to be used to make predictions.
- SageMaker Inference provides two main ways to deploy models:
- Types of Inference
- Real-time inference:
    - Low-latency, live responses.
    - Used for apps like fraud detection, chatbots, product recommendations.
- Batch inference:
    - Run predictions on large files or datasets at once.
    - Used for monthly reports, scoring millions of records, etc.
- Purpose:
    - Production-ready model deployment on AWS.

5. SageMaker Autopilot

Autopilot is an AutoML service.
- What it does automatically:
    - When you upload data (CSV, S3 dataset), Autopilot will:
    - Clean the data
    - Select algorithms
    - Train multiple models
    - Tune hyperparameters
    - Pick the best model
    - Optionally deploy it
- Why use it?
    - No ML expertise required.
    - Best for beginners, business users, or when you want a fast baseline model.
- Use case:
    - Build ML models automatically with minimal coding or ML knowledge.

6. Data Wrangler

Data Wrangler is a tool for visual data preparation.

- Features:
    - Clean, filter, join, and transform data without writing code.
    - Supports 300+ data transformations.
    - Export the cleaned data directly to training jobs or pipelines.
- Why it's useful:
    - Data preparation takes 60–80% of time in ML.
    - Data Wrangler makes it extremely fast and beginner-friendly.
- Purpose:
    - The easiest way to prepare and clean ML datasets visually.

7. SageMaker Feature Store

A centralized place to store, manage, and reuse ML features (i.e., model inputs).
- Why feature store matters:
    - Features remain consistent across training and inference.
    - Teams can reuse features instead of computing them again.
    - Helps maintain versioning, lineage, and accuracy in ML pipelines.
- Use case:
    - Used in ML pipelines to ensure data consistency and improve collaboration.

8. SageMaker Pipelines

SageMaker Pipelines is the ML workflow automation service.
- It automates end-to-end ML processes like:
    - Data preparation
    - Feature engineering
    - Model training
    - Model evaluation
    - Deployment
    - Monitoring
    - Why use Pipelines?
    - Ensures repeatability and automation.
    - Reduces manual errors.
    - Enables real MLops on AWS.
- Purpose:
    - Automate ML workflows from start to finish (MLOps for SageMaker).


# Amazon Rekognition
It is an AWS AI Service used to recognize and analyze images and videos.

It can detect:
- Objects
- Faces
- Emotions
- Labels (what’s in the image)
- Text
- Celebrities
- Unsafe content

#### What it can do?
1. Object and Scene detection
- Detects objects like:
    - Car
    - Person
    - Dog
    - Tree
    - Road
    - Keyword: “Label detection”

2. Face Detection 
- Find faces in images
    - Bounding boxes
    - Facial landmarks

3. Face Recognition
- Match faces to a collection. Used for:
    - Security systems
    - Attendance apps

4. Emotion Detection
- Detects emotions:
- Happy
- Sad
- Angry
- Calm
- Fear

5. Text Detection - OCR(Optical Character Recognition)
- Extract text from images:
    - Signboards
    - License plates
    - Book pages

6. Unsafe Content Detection (Moderation)
- Used to Detect
    - Violent images
    - Nudity
    - Alcohol
    - Drugs
    - Hate Symbols

7. Celebrity Recognition
- Recognizes celebrities using built-in-database

#### What it can't do
- Translate text
- Read tables from documents
- Summarize anything
- Analyze Audio
- Generate images
- Chat with user
- Use LLMs

#### Use cases
- Security surveillance
- Retail analytics
- Moderating images on social media
- Finding unsafe content
- Counting people
- Face-based authentication
- Recognizing celebrities
- Detecting vehicles, pets, objects

# Amazon Comprehend
It is an AWS Natural language processor service that analyzes text to extract meaning, sentiment, keywords, entities, Personally Identifiable Information.

#### What it can do
1. Sentiment Analysis
- Determines if text is:
    - Positive
    - Negative
    - Neutral
    - Mixed

2. Entity Recognition
- Extract key items from text, like:
    - Person
    - Location
    - Organization
    - Date
    - Amount
    - Product names

3. Key Phrase Extraction
- It pulls out important phrases.

4. Personally Identifiable Information
- Phone number
- Email
- Aadhar number
- Address
- Card details
- Name

5. Language Detection
- It will detect the language by the given text

6. Topic Modeling
- Groups documents into topics.

- Example:
    - Topic 1: "refund", 'Payment', 'order'
    - Topic 2: 'delivery',shipping','delay'

7. Custom Classification
- We can train custom text classifiers.
Example:
- Classify support tickets as:
    - billing
    - technical
    - Account issues

# Amazon Textract
It is an AWS AI service used to extract text from forms, tables, handwritten documents, and documents.
- Textract understands:
    - PDFs
    - Scanned documents
    - Forms
    - Contracts
    - Tables
    - Receipts
    - Handwritten forms

#### Textract Capabilities
1. Detect text (Optical Character Recognition)

- Extract normal printed text from scanned documents.

2. Detect Forms (Key-Value Pairs)

- Extract fields like:
    - Name: Ravi Kumar

3. Detect Tables

- It reads Tables, Rows, Columns.

4. Extract Handwriting

- Textract can extract handwritten text in forms.

# Amazon Translate (Text Translation)
Translate = Text → text (different language)
- What it does:
    - Converts text from one language to another
    - Fast, accurate, real-time
    - Supports many languages
- Examples:
    - Convert English → Hindi
    - Convert English → Telugu
    - Convert French → English

- Key words in exam:
    - “Translate a document”
    - “Convert language”
    - “Localization”
    - “Multi-language website”

- What it CANNOT do:
    - Speech
    - OCR
    - Summaries
    - Sentiment

# Amazon Polly (Text-to-Speech)
* Polly = Text → natural human-like voice
- Used for:
    - Voice assistants
    - IVR systems
    - Reading news
    - Accessibility tools
    - Podcast narration


- Key words in exam:
    - “Convert text to natural speech”
    - “Generate audio”
    - “Voice output”


- What Polly CANNOT do:
    - Speech to text
    - Translation
    - Sentiment analysis

# Amazon Transcribe (Speech-to-Text)
* Transcribe = Speech → text
- Used for:
    - Meeting notes
    - Call center recordings
    - Live captioning
    - Medical transcription
    - Voice commands


- Key words in exam:
    - “Convert audio to text”
    - “Transcribe meeting call”
    - “Speech recognition”
    - “Call center recordings”


- What Transcribe CANNOT do:
    -  Text-to-speech
    -  Image-to-text
    -  Translation

# Quick Recall
| Need                        | Service    |
| --------------------------- | ---------- |
| Translate languages         | Translate  |
| Generate speech             | Polly      |
| Convert audio to text       | Transcribe |
| Detect language             | Comprehend |
| Analyze meaning             | Comprehend |
| Summarize text              | Bedrock    |
| Extract text from documents | Textract   |




# Amazon Bedrock:
Amazon bedrock is a service that provides access to large language models(LLM). It offers pre-trained LLMs and custom models built on top of these foundations.
AWS Bedrock is a serverless platform that lets you use pre-trained Gen AI foundation model
It is fully managed service, we don't have to worry about it anymore.
- Amazon Bedrock is like a app store for AI models. We just pick the model, customize it, and deploy our own AI app and AWS handles everything behind the scenes.

## Why Bedrock Exists:
- Before Bedrock:
    - You needed GPUs
    - Huge training costs
    - Complex model hosting
    - Hard to switch between models
    - Security risks
- With Bedrock:
    - No infrastructure
    - Pay per request
    - Enterprise security
    - Models from many vendors
    - Easy model switching

## What Bedrock Provides:
1. Foundation Models - These are huge pre-trained AI models that perform:
    * Text generation
    * Chat 
    * Code generation
    * Image generation
    * Embeddings
    * Multi modal tasks
 - Providers inside Bedrock:
    - Amazon Titan
    - Anthropic Claude
    - Cohere Command
    - Meta Llama
    - Stability AI (image models)
    - AI21 Labs
    - Mistral AI (depending on region)'

2. Customization Methods - Bedrock lets you tailor a model to your business needs by:
    - Fine-tuning - Train the model on your own data or requirements
    - RAG (Retrieval-Augmented Generation) - we can connect our own data or private documents, it reads them and gives accurate answers.
        * this is handled by Bedrock knowledge bases.
    - Agents - Create AI bots or assistants that can interact with users and answer questions.
        * Call APIs, Query database, Make decisions, Trigger actions.
    - Prompt Engineering - Customize prompts to get desired responses.
3. Serverless inference (inference is an idea or conclusion that's drawn from evidence and reasoning)
    - This is the biggest advantage:
        * No EC2
        * No GPUs
        * No containers
        * No scaling issues
        * AWS handles everything
    - You only pay for:
        * Input tokens
        * Output tokens
        * Model customizations
4. Enterprise Level Security:
* IAM Access Control
    * No API key exposure - This is why companies trust Bedrock more than external AI APIs (OpenAI, Gemini, etc.).
* VPC Endpoints
    * VPC endpoints allows a private connection between your VPC and AWS services, without needing an internet gateway
* Encryption in transit and rest
    * In Transit - Whenever your application talks to Bedrock:
        * Data is encrypted using HTTPS/TLS
        * Prevents man in the middle attacks
        * Ensures safe communication
    * At Rest - Any data that Bedrock temporarily stores is encrypted using AWS KMS when data is stored in Bedrock:
        * Protects against unauthorized access
        * Ensures secure storage
* No Data retention
    * Bedrock does not store, train or learn from your data.
    * Our prompts are not used to train, improve, fine-tune models or build datasets.
    * No Long-term storage

## (need to rewrite)Integration with the AWS ecosystem
AWS Bedrock goes beyond providing powerful models it integrates with other AWS services to support end-to-end AI workflows. Some integrations include:
- Amazon SageMaker: Enables fine-tuning of foundation models to meet specific requirements.
- AWS Lambda: Facilitates event-driven AI applications, such as triggering a model to fine-tune new data or review inference results.
- Amazon CloudWatch: Provides monitoring and logging capabilities for analyzing model performance and gathering user feedback.
- Amazon S3: Serves as a storage solution for datasets, enabling performance tracking and cost analysis.

# Amazon SageMaker:

