# Seminal AI Papers, ASO 2015-2025
Collection of Critical AI Papers and research.
__Order is of papers read/finished, not indicative of their timeline nor relative importance.__

Pivotal papers including Transformers Architecture (Attention is All You Need), Early ChatGPT papers, Chain of Thought, etc.

0. **"Scaling Laws for Neural Language Models" (2020) by Jared Kaplan et al.**
   - *Summary:* This paper investigates how the performance of neural language models improves as model size, dataset size, and compute resources increase. The authors establish empirical scaling laws, showing that loss decreases predictably with increased resources following a power-law relationship. (The writers of this paper continued work at OpenAI and later founding Anthropic upon establishing the NL scaling laws, recommending need for GPUs, data, and training.)
   - *Link:* ["Scaling Laws for Neural Language Models" (2020)](https://arxiv.org/abs/2001.08361)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/Scaling_Laws-2001.08361v1.pdf)

1. **"Deep Residual Learning for Image Recognition" (2015) by Kaiming He et al.**
   - *Summary:* Introduced ResNet, a deep convolutional neural network architecture utilizing residual connections, enabling the training of extremely deep networks and achieving state-of-the-art results in image recognition tasks.
   - *Link:* [Deep Residual Learning for Image Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/DeepResidualLearning_1512.03385v1.pdf)

2. **"Attention Is All You Need" (2017) by Ashish Vaswani et al.**
   - *Summary:* Proposed the Transformer architecture, relying entirely on self-attention mechanisms, which has become foundational in natural language processing and beyond.
   - *Link:* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/Attention_1706.03762v7.pdf)

3. **"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018) by Jacob Devlin et al.**
   - *Summary:* Introduced BERT, a model pre-trained on vast text corpora to understand language context bidirectionally, setting new benchmarks in multiple NLP tasks.
   - *Link:* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/BERT_1810.04805v2.pdf)

4. **"Scaling Vision Transformers to 22 Billion Parameters" (2024) by Xianzhi Du et al.**
   - *Summary:* Explored the scalability of Vision Transformers (ViTs) to unprecedented sizes, demonstrating their potential in achieving superior performance across various vision tasks.
   - *Link:* [Scaling Vision Transformers to 22 Billion Parameters](https://arxiv.org/abs/2302.05442)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/ScalingVision_2302.05442v1.pdf)

5. **"AlphaMissense: AI System for Predicting the Pathogenicity of Genetic Variants" (2024) by Pushmeet Kohli et al.**
   - *Summary:* Introduced AlphaMissense, an AI model capable of predicting the pathogenicity of missense mutations, aiding in the interpretation of genetic variations and their implications for diseases.
   - *Link:* [Accurate proteome-wide missense variant effect prediction with AlphaMissense](https://www.science.org/doi/10.1126/science.adg7492)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/Alpha-Disease%2Bscience.adg7492.pdf)

6. **"Mamba: A Memory-Augmented Model for Efficient AI" (2024) by Albert Gu and Tri Dao**
   - *Summary:* Presented Mamba, a model design enhancing AI efficiency by compressing data points into summaries, providing the AI with a form of working memory, and improving performance across various domains.
   - *Link:* [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/Mamba_2312.00752v2.pdf)

7. **"Liquid Neural Networks: A New Approach to AI Model Flexibility" (2020) by Ramin Hasani et al.**
   - *Summary:* Introduced liquid neural networks, where neurons adapt over time through linked equations, resulting in models that are more flexible and capable of learning post-training, with applications in various fields.
   - *Link:* [Liquid Time-constant Networks](https://arxiv.org/abs/2006.04439)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/Liquid_2006.04439v4.pdf)

8. **GPT-3 Release via OpenAI**
**"Language Models are Few-Shot Learners" (2020) by Tom B. Brown et al.**  
   - *Summary:* Introduced GPT-3, a 175-billion parameter autoregressive language model capable of performing diverse natural language processing tasks without explicit fine-tuning, demonstrating strong performance in translation, question-answering, and more through few-shot learning.  
   - *Link:* [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/FewShot-GPT_2005.14165v4.pdf)

9. **"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022) by Jason Wei et al.**  
   - *Summary:* Demonstrated that prompting large language models with a chain of thought—a series of intermediate reasoning steps—significantly enhances their ability to perform complex reasoning tasks, leading to improved performance in arithmetic, commonsense, and symbolic reasoning.  
   - *Link:* [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/CoT_Responses_2201.11903v6.pdf)

10. **"Self-Consistency Improves Chain of Thought Reasoning in Language Models" (2022) by Xuezhi Wang et al.**  
   - *Summary:* Proposed a new decoding strategy called self-consistency, which samples multiple reasoning paths and selects the most consistent answer, further enhancing the performance of chain-of-thought prompting in large language models across various reasoning benchmarks.  
   - *Link:* [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/Self-Consistency_CoT_2203.11171v4.pdf)

11. **"The Lottery Ticket Hypothesis" (2019) by Jonathan Frankle et al"**  
   - *Summary:* The paper presents the "lottery ticket hypothesis," which suggests that within large neural networks exist smaller, sparse subnetworks that, when initialized correctly, can achieve comparable performance to the full network.
   - *Link:* [The Lottery Ticket Hypothesis (2019)](https://arxiv.org/abs/1803.03635)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/LotteryTicketHypothesis_1803.03635v5.pdf)

12. **"On Calibration of Modern Neural Networks" (2017) by Chuan Guo et al**  
   - *Summary:*  The study examines the reliability of neural networks' confidence estimates, revealing that while modern networks achieve high accuracy, they tend to be poorly calibrated, leading to overconfidence in incorrect predictions.
   - *Link:* [On Calibration of Modern Neural Networks (2017)](https://arxiv.org/abs/1706.04599)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/Calibration-Modern-NN_1706.04599v2.pdf)

13. **NeuralGCM: AI-Assisted Weather and Climate Forecasting (2024) by Google Research**
   - *Summary:* Introduced NeuralGCM, a hybrid model combining machine learning with traditional atmospheric forecasting tools, enabling accurate long-term weather and climate predictions, including tracking extreme events like cyclones.
   - *Link:* [Nature: Neural general circulation models for weather and climate](https://www.nature.com/articles/s41586-024-07744-y)
   - *Link:* [FT on Weather AI](https://www.ft.com/content/78d1314b-2879-40cc-bb87-ffad72c8a0f4)   
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/Neural-GCMs_s41586-024-07744-y.pdf)

14. **AlphaDev: AI-Discovered Algorithms for Enhanced Computing (2023) by DeepMind**  
   - *Summary:* Presented AlphaDev, an AI system that discovered faster algorithms for fundamental tasks like sorting and hashing, leading to significant performance improvements in computing efficiency.
   - *Link:* [Nature: Faster sorting algorithms discovered using deep reinforcement learning](https://www.nature.com/articles/s41586-023-06004-9)
   - *Link:* [AlphaDev on Wiki]https://en.wikipedia.org/wiki/AlphaDev)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/AlphaDev_s41586-023-06004-9.pdf)

15.**"Highly accurate protein structure prediction with AlphaFold" (2021) by John Jumper et al.**
   - * Underpinning the latest version of AlphaFold is a novel machine learning approach that incorporates physical and biological knowledge about protein structure, leveraging multi-sequence alignments, into the design of the deep learning algorithm.*
   - Link:* [Highly accurate protein structure prediction with AlphaFold](https://www.nature.com/articles/s41586-021-03819-2)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/AlphaFold_s41586-021-03819-2.pdf)

16. **"AlphaFold 2 and the Future of AI-Assisted Protein Structure Prediction" (2023) by John Jumper et al.**
   - *Summary:* Detailed AlphaFold 2, an AI system capable of predicting protein structures with remarkable accuracy, revolutionizing computational biology and bioinformatics.
   - *Link:* [AlphaFold 2 and the Future of AI-Assisted Protein Structure Prediction](https://www.nature.com/articles/s41392-023-01381-z)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/AlphaFold2_s41392-023-01381-z.pdf)

17.**AlphaFold 3: Extending AI Predictions to DNA and RNA Structures (2024) by John Jumper et al.**  
   - *Summary:* Expanded upon previous versions like AlphaFold 2, to predict structures of other molecules, such as DNA and RNA, further accelerating research in structural biology and protein folding. Together, these results show that high-accuracy modelling across biomolecular space is possible within a single unified deep-learning framework.
   - *Link:* [Nature: Accurate structure prediction of biomolecular interactions with AlphaFold 3 (2024)](https://www.nature.com/articles/s41586-024-07487-w)
   - *Link:* [Time: John Jumper - Time 100](https://time.com/7012710/john-jumper-2)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/AlphaFold3_s41586-024-07487-w.pdf)

18. **"The performance of AlphaMissense to identify genes influencing disease" (2024) by J. Brent Richards et al.**
   - *Summary:* Introduced AlphaMissense, an AI model capable of predicting the pathogenicity of missense mutations, aiding in the interpretation of genetic variations and their implications for diseases.
   - *Link:* [The performance of AlphaMissense to identify genes influencing disease](https://www.sciencedirect.com/science/article/pii/S2666247724000848)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/AlphaMissense-Genes_S2666247724000848.pdf)

19. **"Understanding Chain-of-Thought Prompting" (2022) by Boshi Wang et al**  
   - *Summary:* This paper explores how chain-of-thought (CoT) prompting improves reasoning capabilities in large language models by breaking down complex problems into intermediate steps, helping models achieve higher performance in reasoning tasks.
   - *Link:* [Understanding Chain-of-Thought Prompting (2022)](https://arxiv.org/abs/2212.10001)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/Understanding-CoT_2212.10001v2.pdf)

20. **"Class Imbalance and Deep Learning" (2017) by Mateusz Buda**  
   - *Summary:* This paper analyzes the impact of class imbalance on deep neural networks, highlighting how imbalance can significantly degrade performance and proposing techniques to mitigate its effects.
   - *Link:* [Class Imbalance and Deep Learning (2017)](https://arxiv.org/abs/1710.05381)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/Class-Imbalance_NN_1710.05381v2.pdf)

21. **"Dopamine: A Reinforcement Learning Framework" (2020) by Will Daney et al**  
   - *Summary:* Dopamine is an RL research framework designed to support rapid prototyping and experimentation with value-based deep reinforcement learning algorithms, offering a flexible and lightweight solution for research purposes.
   - *Link:* [Dopamine: A Reinforcement Learning Framework (2020)](https://pubmed.ncbi.nlm.nih.gov/31942076/)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/Dopamine_RL_Dabney-2020-supp.pdf)

22. **"Training Compute-Optimal Large Language Models" (2022) by Jordan Hoffman et al**  
   - *Summary:* DeepMind investigates how model performance scales with compute, data, and parameter size, offering key insights into optimizing AI training efficiency and predicting future capabilities.
   - *Link:* [DeepMind: Training Compute-Optimal Large Language Models (2022)](https://arxiv.org/abs/2203.15556)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/DeepMind-TrainCompute_2203.15556v1.pdf)

23. **"DeepSeek Model Overview - R1" (2024) by Daya Guo et al**  
   - *Summary:* This report outlines the development and performance of DeepSeek, a large-scale AI model, detailing its architecture, training strategies, and potential applications across various domains.
   - *Link:* [DeepSeek Model Overview (2024)](https://arxiv.org/html/2501.12948v1)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/DeepSeek_R1.pdf)

24. **"----**  
   - *Summary:*  
   - *Link:* [Name](https://link.com)
   - PDF in Repo: [PDF---](https://github.com/link)

NeurIPS, Top Papers 2020-2024

**NeurIPS 2024 Best Paper Awards** [2024](https://blog.neurips.cc/2024/12/10/announcing-the-neurips-2024-best-paper-awards/)

1. **"Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"**
   - *Authors:* Keyu Tian et al.
   - *Summary:* This paper introduces a novel visual autoregressive (VAR) model that iteratively predicts images at progressively higher resolutions. The VAR model demonstrates superior performance in image generation, outperforming existing autoregressive models in efficiency and achieving results competitive with diffusion-based methods.
   - *Link:* [NeurIPS 2024 Best Paper Awards](https://openreview.net/forum?id=gojL67CfS8)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/Visual_Autoregressive_Mo-19076_.pdf)

2. **"Stochastic Taylor Derivative Estimator: Efficient Amortization for Arbitrary Differential Operators"**
   - *Authors:* Zekun Shi, Zheyuan Hu, Min Lin, Kenji Kawaguchi
   - *Summary:* This work proposes a tractable approach to train neural networks using supervision that incorporates higher-order derivatives. The method provides significant speed-up and memory reduction over traditional approaches, enabling the solution of high-dimensional partial differential equations efficiently.
   - *Link:* [NeurIPS 2024 Awards](https://openreview.net/forum?id=J2wI2rCG2u)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/Stochastic_Taylor_Deriva_14856.pdf)
  
*- Datasets and Benchmarks Track:*
3. **"The PRISM Alignment Dataset: What Participatory, Representative and Individualised Human Feedback Reveals About the Subjective and Multicultural Alignment of Large Language Models"**
   - *Authors:* Hannah Rose Kirk et al.
   - *Summary:* This paper introduces the PRISM dataset, mapping the sociodemographics and stated preferences of 1,500 diverse participants to their contextual preferences and fine-grained feedback in interactions with large language models. The study provides insights into the subjective and multicultural alignment of these models.
   - *Link:* [NeurIPS 2024 Awards](https://openreview.net/forum?id=DFr5hteojx#discussion)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/The_PRISM_Alignment_Datas_1310.pdf)
 
*- Runner Ups, Main Track:*
4.**"Not All Tokens Are What You Need for Pretraining"**
   - *Authors:* Zhenghao Lin et al.
   - *Summary:* This paper presents a simple method to filter pre-training data when training large language models (LLM)
   - *Link:* [NeurIPS 2024, Runner Up](https://openreview.net/forum?id=0NMzBwqaAJ)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/Not_All_Tokens_Are_What_Y_5208.pdf)

5. **"Guiding a Diffusion Model with a Bad Version of Itself"**
   - *Authors:* Tero Karras et al.
   - *Summary:*  This paper proposes an alternative (Autoguidance) to classifier free guidance (CFG) in the context of text-2-image (T2I) models. CFG is a guidance technique (a correction in diffusion trajectories) that is extensively used by practitioners to obtain better prompt alignment and higher-quality images, that Autoguidance could improve. This change leads to notable improvements in diversity and image quality.
   - *Link:* [NeurIPS 2024 Awards](https://openreview.net/forum?id=bg6fVPVs3s)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/Guiding_a_Diffusion_Mode_10530.pdf)

**NeurIPS 2023 Best Paper Awards** [2023](https://blog.neurips.cc/2023/12/11/announcing-the-neurips-2023-paper-awards/)

**NeurIPS 2022 Best Paper Awards** [2022](https://blog.neurips.cc/2022/11/21/announcing-the-neurips-2022-awards/)

**NeurIPS 2021 Best Paper Awards** [2021](https://nips.cc/virtual/2021/awards_detail)

**NeurIPS 2020 Best Paper Awards** [2020](https://nips.cc/virtual/2020/awards_detail)

[Awards selection: 2020](https://neuripsconf.medium.com/announcing-the-neurips-2020-award-recipients-73e4d3101537) 

1. **"No-Regret Learning Dynamics for Extensive-Form Correlated Equilibrium" (2020) by Andrea Celli et al**
   - Summary:
   - *Link:* [No-Regret Learning Dynamics for Extensive-Form Correlated Equilibrium" (2020](https://arxiv.org/abs/2004.00603)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/NoRegret_2004.00603v5.pdf)

2. **"Improved guarantees and a multiple-descent curve for Column Subset Selection and the Nyström method" (2020) by Michael Dereziński et al.**
   - Summary:
   - *Link:* ["Improved guarantees and a multiple-descent curve for Column Subset Selection and the Nyström method" (2020)](https://arxiv.org/abs/2002.09073)
   - PDF in Repo: [PDF](https://github.com/kaybeeinc/critical-ai-papers_25/blob/main/ImprGaur_2002.09073v3.pdf)

3. **"Language Models are Few-Shot Learners" by Tom B. Brown et al** (Represented in #8 above)

?????
????
 / **"Rethinking 'Batch' in Batch Normalization"**
   - *Authors:* Simone Wu, Zhirong Wu, Yann LeCun
   - *Summary:* This paper revisits the concept of batch normalization, proposing alternatives that address its limitations, particularly in scenarios with small batch sizes. The study offers theoretical insights and practical solutions to improve training stability and performance.
   - *Link:* [NeurIPS 2020 Awards](https://neuripsconf.medium.com/announcing-the-neurips-2020-award-recipients-73e4d3101537)
   - PDF in Repo: [PDF---]()

/ **"Neural Tangents: Fast and Easy Infinite Neural Networks in Python"**
   - *Authors:* Roman Novak, Lechao Xiao, Yasaman Bahri, Jaehoon Lee, Greg Yang, Jascha Sohl-Dickstein
   - *Summary:* This work introduces Neural Tangents, a library designed to facilitate research and experimentation with infinite-width neural networks, providing tools to compute exact predictions of infinitely wide networks in practical scenarios.
   - *Link:* [NeurIPS 2020 Awards](https://neuripsconf.medium.com/announcing-the-neurips-2020-award-recipients-73e4d3101537)
   - PDF in Repo: [PDF---]()
  

