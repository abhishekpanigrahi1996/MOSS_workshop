import { useState } from "react";

const Card = ({ children }) => (
  <div className="border rounded-2xl shadow p-4 bg-white">{children}</div>
);

const Input = ({ className = "", ...props }) => (
  <input
    {...props}
    className={`border p-2 rounded w-full ${className}`.trim()}
  />
);

const papers = [
    {
        "title": "Understanding Attention Glitches with Threshold Relative Attention",
        "id": 96,
        "authors": [
            "Mattia Opper",
            "Roland Fernandez",
            "Paul Smolensky",
            "Jianfeng Gao"
        ],
        "keywords": [
            "length generalisation",
            "attention glitches",
            "flip-flops",
            "algorithmic reasoning"
        ],
        "abstract": "Transformers struggle with generalisation, displaying poor performance even on basic yet fundamental tasks, such as flip-flop language modeling. We test whether these limitations can be explained through two key failures of self-attention. The first is the inability to fully remove irrelevant information. The second concerns position, even when a key is completely irrelevant learned positional biases may unintentionally up-weight it - dangerous when distances fall out of distribution. To probe this we propose TRA, a novel attention mechanism with which we demonstrate that these issues underlie generalisation failures on the flip-flop task.",
        "topic": "Attention Mechanisms & Transformer Analysis",
        "url": "https://openreview.net/forum?id=yhNOZsCPUi"
    },
    {
        "title": "Learning Gaussian Mixture Models via Transformer Measure Flows",
        "id": 89,
        "authors": [
            "Aleksandr Zimin",
            "Anastasiia Kutakh",
            "Yury Polyanskiy",
            "Philippe Rigollet"
        ],
        "keywords": [
            "Transformers",
            "GMM",
            "Measure-to-measure flow map"
        ],
        "abstract": "We introduce a transformer architecture for approximating Gaussian Mixture Models (GMMs) through a measure-to-measure flow interpretation. Rather than estimating explicit cluster parameters, our model predicts the underlying cluster probability distribution by minimizing Wasserstein distance to the true measure. A key innovation is the flow speed hyperparameter, which adjusts clustering intensity by varying transformer step size and indirectly controlling model depth based on the desired output complexity. Experimental results show performance comparable to or exceeding classical algorithms like K-means, while the synthetic setup provides a lightweight, interpretable sandbox for investigating transformer flow foundations without computational overhead of language-based benchmarks.",
        "topic": "Attention Mechanisms & Transformer Analysis",
        "url": "https://openreview.net/forum?id=MGyqEvYn1T"
    },
    {
        "title": "Continuous Chain of Thought Enables Parallel Exploration and Reasoning",
        "id": 65,
        "authors": [
            "Halil Alperen Gozeten",
            "Muhammed Emrullah Ildiz",
            "Xuechen Zhang",
            "Hrayr Harutyunyan",
            "Ankit Singh Rawat",
            "Samet Oymak"
        ],
        "keywords": [
            "chain-of-thought",
            "latent space reasoning",
            "parallel exploration",
            "transformers",
            "policy optimization",
            "multi token sampling"
        ],
        "abstract": "We propose CoT2, a framework using continuously-valued tokens that enables language models to track multiple reasoning paths in parallel and provide a novel CoT2 supervision strategy where we match the softmax outputs to the empirical token distributions of a set of target traces. Theoretically, we show that CoT2 offers sample-complexity benefits and construct a one-layer transformer that solves the subset-sum problem with sufficient embedding capacity. We also introduce continuous sampling methods, showing that reinforcement learning with CoT2 notably improves logical reasoning performance compared to discrete and continuous baselines.",
        "topic": "Attention Mechanisms & Transformer Analysis",
        "url": "https://openreview.net/forum?id=1ORJaYuMJc"
    },
    {
        "title": "On the Emergence of Position Bias in Transformers",
        "id": 62,
        "authors": [
            "Xinyi Wu",
            "Yifei Wang",
            "Stefanie Jegelka",
            "Ali Jadbabaie"
        ],
        "keywords": [
            "attention mechanism",
            "transformers",
            "position bias",
            "positional encoding",
            "deep learning theory"
        ],
        "abstract": "Recent studies have revealed various manifestations of position bias in transformer architectures, from the \u201clost-in- the-middle\u201d phenomenon to attention sinks, yet a comprehensive theoretical understanding of how attention masks and positional encodings shape these biases remains elusive. This paper presents a graph-theoretic framework for analyzing position bias in multi-layer attention. Modeling attention masks as directed graphs, we quantify how tokens interact with contextual information based on their sequential positions. We uncover two key insights: First, causal masking inherently biases attention toward earlier positions, as tokens in deeper layers attend to increasingly more contextualized representations of earlier tokens. Second, we characterize the competing effects of the causal mask and relative positional encodings, such as the decay mask and rotary positional encoding (RoPE): while both mechanisms introduce distance-based decay within individual attention maps, their aggregate effect across multiple attention layers\u2014coupled with the causal mask\u2014leads to a trade-off between the long-term decay effects and the cumulative importance of early sequence positions. Through controlled numerical experiments, we not only validate our theoretical findings but also reproduce position biases observed in real-world LLMs. Our framework offers a principled foundation for understanding positional biases in transformers, shedding light on the complex interplay of attention mechanism components and guiding more informed architectural design.",
        "topic": "Attention Mechanisms & Transformer Analysis",
        "url": "https://openreview.net/forum?id=3Dq32m6m2M"
    },
    {
        "title": "Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought",
        "id": 52,
        "authors": [
            "Hanlin Zhu",
            "Shibo Hao",
            "Zhiting Hu",
            "Jiantao Jiao",
            "Stuart Russell",
            "Yuandong Tian"
        ],
        "keywords": [
            "reasoning",
            "chain of continuous thought",
            "superposition",
            "transformer"
        ],
        "abstract": "In this paper, we prove that a two-layer transformer with $D$ steps of continuous chain-of-thoughts (CoTs) can solve the directed graph reachability problem, where $D$ is the diameter of the graph, while the best known result of constant-depth transformers with discrete CoTs requires $O(n^2)$ decoding steps where $n$ is the number of vertices ($D<n$).  In our construction, each continuous thought vector is a superposition state that encodes multiple search frontiers simultaneously (i.e., parallel breadth-first search (BFS)), while discrete CoTs must choose a single path sampled from the superposition state, which leads to sequential search that requires many more steps and may be trapped into local solutions. We also performed extensive experiments to verify that our theoretical construction aligns well with the empirical solution obtained via training dynamics, and observed that encoding of multiple search frontiers as a superposition state automatically emerges in training continuous CoTs, without explicit supervision to guide the model to explore multiple paths simultaneously.",
        "topic": "Attention Mechanisms & Transformer Analysis",
        "url": "https://openreview.net/forum?id=1cD9iO5Isv"
    },
    {
        "title": "What Happens During the Loss Plateau? Understanding Abrupt Learning in Transformers",
        "id": 27,
        "authors": [
            "Pulkit Gopalani",
            "Wei Hu"
        ],
        "keywords": [
            "Abrupt learning",
            "attention map",
            "transformer training dynamics",
            "interpretability",
            "science of language models"
        ],
        "abstract": "Training Transformers on algorithmic tasks frequently demonstrates an intriguing *abrupt learning* phenomenon: an extended performance plateau followed by a sudden, sharp improvement. This work investigates the underlying mechanisms for such dynamics, primarily in shallow Transformers. We reveal that during the plateau, the model often develops an interpretable *partial solution* while simultaneously exhibiting a strong *repetition bias* in their outputs. This output degeneracy is accompanied by *internal representation collapse*, where hidden states across different tokens become nearly parallel. We further identify the slow learning of optimal attention maps as a key bottleneck. Hidden progress in attention configuration during the plateau precedes the eventual rapid convergence, and directly intervening on attention significantly alters plateau duration and the severity of repetition bias and representational collapse. We validate that these phenomena\u2014repetition bias and representation collapse\u2014are not artifacts of toy setups but also manifest in the early pre-training stage of LLMs like Pythia and OLMo.",
        "topic": "Attention Mechanisms & Transformer Analysis",
        "url": "https://openreview.net/forum?id=9ZnblbFpgc"
    },
    {
        "title": "Quantitative Bounds for Length Generalization in Transformers",
        "id": 17,
        "authors": [
            "Zachary Izzo",
            "Eshaan Nichani",
            "Jason D. Lee"
        ],
        "keywords": [
            "transformers",
            "LLM theory",
            "length generalization"
        ],
        "abstract": "We provide quantitative bounds on the length of sequences required to be observed during training for a transformer to length generalize, e.g., to continue to perform well on sequences unseen during training. Our results improve on Huang et al. (2024), who show that there is a finite training length beyond which length generalization is guaranteed, but for which they do not provide quantitative bounds.",
        "topic": "Attention Mechanisms & Transformer Analysis",
        "url": "https://openreview.net/forum?id=4a61vS0Dky"
    },
    {
        "title": "Efficient B-Tree Insertions Using Proximal Policy Optimization and Hierarchical Attention Models",
        "id": 24,
        "authors": [
            "Alexander Kastius",
            "Nick Lechtenb\u00f6rger",
            "Felix Schulz",
            "Johann Schulze Tast",
            "Rainer Schlosser",
            "Ralf Herbrich"
        ],
        "keywords": [
            "Reinforcement Learning",
            "Optimization",
            "Databases",
            "Attention",
            "B-Trees"
        ],
        "abstract": "B-trees are a fundamental component of any large database management system. They can grow to noticeable sizes, but their handling is non-trivial. We present a novel approach to use attention-based models with weight sharing across the hierarchical structure of the tree to parse such large trees fast and without the need for excessive training on large clusters. We present a use case in which the model is used in conjunction with PPO to manage write operations on such a tree.",
        "topic": "Efficiency, Calibration & Robustness",
        "url": "https://openreview.net/forum?id=MumGcsHUA6"
    },
    {
        "title": "Review, Remask, Refine: Process-Guided Block Diffusion for Text Generation",
        "id": 93,
        "authors": [
            "Nikita Mounier",
            "Parsa Idehpour"
        ],
        "keywords": [
            "ext Generation",
            "Masked Diffusion Models",
            "Block Diffusion",
            "Process Reward Models",
            "PRM",
            "Iterative Refinement",
            "Error Correction",
            "Inference-Time Guidance",
            "Self-Correction",
            "Mathematical Reasoning",
            "Computational Efficiency",
            "Windowed Evaluation",
            "LLaDA",
            "Qwen2.5-Math-PRM"
        ],
        "abstract": "A key challenge for iterative text generation is enabling models to efficiently identify and correct their own errors. We propose Review, Remask, Refine (R3), a relatively simple yet elegant framework that requires no additional model training and can be applied to any pre-trained masked text diffusion model (e.g., LLaDA or BD3-LM). In R3, a Process Reward Model (PRM) is utilized for the $\\textbf{Review}$ of intermediate generated blocks. The framework then translates these PRM scores into a $\\textbf{Remask}$ strategy: the lower a block's PRM score, indicating potential mistakes, the greater the proportion of tokens within that block are remasked. Finally, the model is compelled to $\\textbf{Refine}$ these targeted segments, focusing its efforts more intensively on specific sub-optimal parts of past generations, leading to improved final output.",
        "topic": "Efficiency, Calibration & Robustness",
        "url": "https://openreview.net/forum?id=v2H3nOJepW"
    },
    {
        "title": "Approximate Message Passing on General Factor Graphs using Shallow Neural Networks",
        "id": 82,
        "authors": [
            "Leonhard Hennicke",
            "Jan Lemcke",
            "Rainer Schlosser",
            "Ralf Herbrich"
        ],
        "keywords": [
            "Message Passing",
            "Probabilistic Machine Learning",
            "Sampling",
            "Factor Graphs",
            "Shallow Neural Networks"
        ],
        "abstract": "Factor graphs offer an efficient framework for probabilistic inference through message passing, with the added benefit of uncertainty quantification, which is crucial in safety-critical applications. However, their applicability is limited by the need to analytically solve update equations for factors, which are problem-specific and may involve intractable integrals. We propose to approximate the message update equations of individual factors with shallow neural networks, which we train on data generated by sampling from the respective factor equations, to capture complex factor relationships while maintaining computational tractability.",
        "topic": "Efficiency, Calibration & Robustness",
        "url": "https://openreview.net/forum?id=JFh2uWR9Rt"
    },
    {
        "title": "Personalizing AI Interventions in Multiple Health Behavioral Change Settings",
        "id": 79,
        "authors": [
            "Samantha Marks",
            "Michelle Chang",
            "Eura Nofshin",
            "Weiwei Pan",
            "Finale Doshi-Velez"
        ],
        "keywords": [
            "reinforcement learning",
            "RL",
            "mobile health",
            "mHealth",
            "multiple health behavioral change",
            "MHBC",
            "personalization"
        ],
        "abstract": "We introduce a novel reinforcement learning (RL) framework for personalizing AI interventions in multiple health behavior change (MHBC) settings. Our key contribution is a simple, interpretable model that captures empirically observed human behaviors.  Using this model, we provide insight into how the AI will intervene, including when it has varying degrees of knowledge about the human model.",
        "topic": "Efficiency, Calibration & Robustness",
        "url": "https://openreview.net/forum?id=GP4VzuxY4n"
    },
    {
        "title": "Foundation Models on a Budget: Approximating Blocks in Large Vision Models",
        "id": 59,
        "authors": [
            "Irene Cannistraci",
            "Simone Antonelli",
            "Emanuele Palumbo",
            "Thomas M. Sutter",
            "Emanuele Rodol\u00e0",
            "Bastian Rieck",
            "Julia E Vogt"
        ],
        "keywords": [
            "latent representations",
            "representation learning",
            "neural network similarities",
            "classification",
            "foundation models",
            "large models"
        ],
        "abstract": "Foundation Models have shown impressive performance in various tasks and domains, yet they require massive computational resources, raising concerns about accessibility and sustainability. In this paper, we propose Transformer Blocks Approximation (TBA), a novel method that leverages intra-network similarities to identify and approximate transformer blocks in large vision models using only a small amount of training data. TBA replaces these blocks using lightweight, closed-form transformations, without any additional training steps. The proposed method reduces the number of parameters while having minimal impact on the downstream task.",
        "topic": "Efficiency, Calibration & Robustness",
        "url": "https://openreview.net/forum?id=XI9tNjMZhd"
    },
    {
        "title": "Performance Plateaus in Inference-Time Scaling for Text-to-Image Diffusion Without External Models",
        "id": 54,
        "authors": [
            "Changhyun Choi",
            "Sungha Kim",
            "H. Jin Kim"
        ],
        "keywords": [
            "Text-to-Image Diffusion Models",
            "Inference-Time Scaling",
            "Initial Noise Optimization",
            "VRAM-Limited GPUs"
        ],
        "abstract": "Recently, it has been shown that investing computing resources in searching for good initial noise for a text-to-image diffusion model helps improve performance. However, previous studies required external models to evaluate the resulting images, which is impossible on GPUs with small VRAM. For these reasons, we apply Best-of-N inference-time scaling to algorithms that optimize the initial noise of a diffusion model without external models across multiple datasets and backbones. We demonstrate that inference-time scaling for text-to-image diffusion models in this setting quickly reaches a performance plateau, and a relatively small number of optimization steps suffices to achieve the maximum achievable performance with each algorithm.",
        "topic": "Efficiency, Calibration & Robustness",
        "url": "https://openreview.net/forum?id=blyUISwF6P"
    },
    {
        "title": "CaliPSo: Calibrated Predictive Models with Sharpness as Loss Function",
        "id": 51,
        "authors": [
            "Alexandre Capone",
            "Kamron Zaidi",
            "Tianyu Xu",
            "Brian Yang",
            "Geoff Pleiss",
            "Jeff Schneider"
        ],
        "keywords": [
            "Conformal prediction",
            "calibration",
            "quantile regression"
        ],
        "abstract": "Conformal prediction methods have become increasingly common for accurately capturing uncertainty with machine learning models. However, conformal prediction typically recalibrates an existing model, making it heavily reliant on the quality of the uncalibrated model. Moreover, they either enforce marginal calibration strictly, yielding potentially coarse predictive intervals, or attempt to strike a balance between interval coarseness and calibration. Motivated by these shortcomings, we present CaliPSo a neural network model that is marginally calibrated out-of-the-box and stays so throughout training. This property is achieved by adding a model-dependent constant to the model prediction that shifts it in a way that ensures calibration. During training, we then leverage this to focus exclusively on sharpness - the property of returning tight predictive intervals - rendering the model more useful at test time.  We show thorough experimental results, where our method exhibits superior performance compared to several state-of-the-art approaches.",
        "topic": "Efficiency, Calibration & Robustness",
        "url": "https://openreview.net/forum?id=aj7JZBV5t8"
    },
    {
        "title": "AdaptMI: Adaptive Skill-based In-context Math Instructions for Small Language Models",
        "id": 50,
        "authors": [
            "Yinghui He",
            "Abhishek Panigrahi",
            "Yong Lin",
            "Sanjeev Arora"
        ],
        "keywords": [
            "Small language models",
            "large language models",
            "in-context learning",
            "natural language processing",
            "test-time adaption"
        ],
        "abstract": "In-context learning (ICL) enhances language model performance by providing relevant contextual information. Recent works (Didolkar et al., 2024a;b) show that ICL performance can be improved by leveraging a frontier large language model\u2019s (LLM) ability to predict required skills to solve a problem, popularly referred to as an LLM\u2019s metacognition, and using the recommended skills to construct necessary in-context examples. While this improves performance in larger models, smaller language models (SLMs) see minimal benefit, revealing a performance gap.  We show that skill-based prompting can hurt SLM performance on easy questions by introducing unnecessary information, akin to cognitive overload. To mitigate this, we introduce AdaptMI, an Adaptive strategy for selecting skill-based Math Instructions. Guided by cognitive load theory, AdaptMI introduces skill-based examples only when the model performs poorly. We further propose AdaptMI+ , which provides targeted examples for specific missing skills. In 5-shot evaluations on popular math benchmarks and five SLMs (1B\u20137B; Qwen, Llama), AdaptMI+ improves accuracy by up to 6% compared to naive skill-based methods.",
        "topic": "Efficiency, Calibration & Robustness",
        "url": "https://openreview.net/forum?id=0nuohwdAvM"
    },
    {
        "title": "TinyServe: Query-Aware Cache Selection for Efficient LLM Inference",
        "id": 43,
        "authors": [
            "Dong Liu",
            "Yanxuan Yu"
        ],
        "keywords": [
            "Efficient serving",
            "Token selection",
            "Cache management",
            "LLMs system behavior",
            "small LLMs"
        ],
        "abstract": "Serving large language models (LLMs) efficiently remains challenging due to the high memory and latency overhead of key-value (KV) cache access during autoregressive decoding. We present \\textbf{TinyServe}, a lightweight and extensible runtime system for deploying tiny LLMs (e.g., TinyLLaMA, GPT2-345M) with support for structured KV sparsity, plugin-based token selection, and hardware-efficient attention kernels. Unlike prior simulation frameworks, TinyServe executes real-time decoding with configurable sparsity strategies and fine-grained instrumentation.  To reduce decoding cost, we introduce a \\textit{query-aware page selection} mechanism that leverages bounding-box metadata to estimate attention relevance between the query and KV cache blocks. This enables selective KV loading with minimal overhead and no model modifications. Our fused CUDA kernel integrates page scoring, sparse memory access, and masked attention in a single pass.  Experiments show that TinyServe achieves up to \\textbf{3.4\u00d7} speedup and over \\textbf{2\u00d7} memory savings with negligible accuracy drop. Additional analysis of cache reuse, page hit rate, and multi-GPU scaling confirms its practicality as a system-level testbed for LLM inference research on resource-constrained hardware.",
        "topic": "Efficiency, Calibration & Robustness",
        "url": "https://openreview.net/forum?id=sOdtl4jLci"
    },
    {
        "title": "Dynamic Low-Rank Training with Spectral Regularization: Achieving Robustness in Compressed Representations",
        "id": 22,
        "authors": [
            "Steffen Schotth\u00f6fer",
            "H. Lexie Yang",
            "Stefan Schnake"
        ],
        "keywords": [
            "Dynamical Low-Rank Aproximation",
            "Low-Rank",
            "Adversarial Robustness",
            "Compression"
        ],
        "abstract": "Deployment of neural networks on resource-constrained devices demands models that are both compact and robust to adversarial inputs. However, compression and adversarial robustness often conflict. In this work, we introduce a dynamical low-rank training scheme enhanced with a novel spectral regularizer that controls the condition number of the low-rank core in each layer. This approach mitigates the sensitivity of compressed models to adversarial perturbations without sacrificing clean accuracy. The method is model- and data-agnostic, computationally efficient, and supports rank adaptivity to automatically compress the network at hand. Extensive experiments across standard architectures, datasets, and adversarial attacks show the regularized networks can achieve over 94\\% compression while recovering or improving adversarial accuracy relative to uncompressed baselines.",
        "topic": "Efficiency, Calibration & Robustness",
        "url": "https://openreview.net/forum?id=yZY0w0Nr7E"
    },
    {
        "title": "Effective Reinforcement Learning for Reasoning in Language Models",
        "id": 21,
        "authors": [
            "Lianghuan Huang",
            "Shuo Li",
            "Sagnik Anupam",
            "Insup Lee",
            "Osbert Bastani"
        ],
        "keywords": [
            "large language models",
            "efficient training",
            "reinforcement learning",
            "reasoning"
        ],
        "abstract": "Reinforcement learning (RL) has emerged as a promising strategy for improving the reasoning capabilities of language models (LMs) in domains such as mathematics and coding. However, most modern RL algorithms were designed to target robotics applications, which differ significantly from LM reasoning. We analyze RL algorithm design decisions for LM reasoning, for both accuracy and computational efficiency, focusing on relatively small models due to computational constraints. Our findings are: (i) on-policy RL significantly outperforms supervised fine-tuning (SFT), (ii) PPO-based off-policy updates increase accuracy instead of reduce variance, and (iii) removing KL divergence can lead to concise generations and higher accuracy. Furthermore, we find that a key bottleneck to computational efficiency is that the optimal batch sizes for inference and backpropagation are different. We propose a novel algorithm, DASH, that performs $\\textit{preemptive sampling}$ (i.e., sample a large batch and accumulate gradient updates in small increments), and $\\textit{gradient filtering}$ (i.e., drop samples with small advantage estimates). We show that DASH reduces training time by 83% compared to a standard implementation of GRPO without sacrificing accuracy. Our findings provide valuable insights on designing effective RL algorithms for LM reasoning.",
        "topic": "Efficiency, Calibration & Robustness",
        "url": "https://openreview.net/forum?id=BwXTNDA9ZV"
    },
    {
        "title": "LiteByte: Efficient and Fast-Adapting MLPs for Online Byte-Level Prediction",
        "id": 19,
        "authors": [
            "Yu Mao",
            "Yuyan Lin",
            "Xue Liu",
            "Chun Jason Xue"
        ],
        "keywords": [
            "Byte-Level Modeling",
            "Online Learning",
            "MLP Architecture",
            "Attention-Free",
            "Soft Expert Routing",
            "Autoregressive Prediction"
        ],
        "abstract": "Transformer-based architectures have become the de facto standard for sequence modeling, largely due to their scalability and ability to capture long-range dependencies. However, their high computational cost, reliance on long contexts, and limited adaptability under online updates make them less suitable for small-scale or streaming scenarios. In this paper, we revisit MLP-based models for byte-level next-token prediction under fully online training. We propose a simple yet effective architecture, LiteByte, which is composed of alternating feedforward layers and soft-shared expert projections, without attention or recurrence. Each sample is dynamically routed through a learned mixture of compact shared MLPs, enabling adaptive token-wise transformations with minimal overhead. Despite its simplicity, our model achieves significantly faster convergence and lower perplexity than Transformer, RNN, and vanilla MLP baselines on Enwik8, Text8, and a curated Dickens corpus. It also demonstrates superior runtime efficiency in terms of inference latency and throughput. We further argue that the soft expert mechanism introduces a reusable and modular structure that may serve as a lightweight adapter or differentiable controller in broader applications such as LoRA-style fine-tuning or modular agents.",
        "topic": "Efficiency, Calibration & Robustness",
        "url": "https://openreview.net/forum?id=hQeFXhHcw6"
    },
    {
        "title": "Mind the Gap: Removing the Discretization Gap in Differentiable Logic Gate Networks",
        "id": 18,
        "authors": [
            "Shakir Yousefi",
            "Andreas Plesner",
            "Till Aczel",
            "Roger Wattenhofer"
        ],
        "keywords": [
            "Logic gate networks",
            "Gumbel noise",
            "Faster training",
            "Smoother minima"
        ],
        "abstract": "Modern neural networks exhibit state-of-the-art performance on many benchmarks, but their high computational requirements and energy usage have researchers exploring more efficient solutions for real-world deployment.     Logic gate networks (LGNs) learns a large network of logic gates for efficient image classification. However, learning a network that can solve a simple problem like CIFAR-10 can take days to weeks to train. Even then, almost half of the network remains unused, causing a \\emph{discretization gap}. This discretization gap hinders real-world deployment of LGNs, as the performance drop between training and inference negatively impacts accuracy.     We inject Gumbel noise with a straight-through estimator during training to significantly speed up training, improve neuron utilization, and decrease the discretization gap.      We theoretically show that this results from implicit Hessian regularization, which improves the convergence properties of LGNs. We train networks $4.5 \\times$ faster in wall-clock time, reduce the discretization gap by 98\\%, and reduce the number of unused gates by 100\\%.",
        "topic": "Efficiency, Calibration & Robustness",
        "url": "https://openreview.net/forum?id=eDxggWdZyg"
    },
    {
        "title": "ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training",
        "id": 2,
        "authors": [
            "Feijiang Han",
            "Xiaodong Yu",
            "Jianheng Tang",
            "Qingyun Zeng",
            "Licheng Guo",
            "Lyle Ungar"
        ],
        "keywords": [
            "Large Language Models",
            "Training-Free Methods",
            "Attention",
            "Interpretability"
        ],
        "abstract": "Training-free methods for enhancing large language models (LLMs) have attracted growing interest recently, with token-level attention tuning emerging as an interpretable and promising direction. However, existing methods typically rely on auxiliary mechanisms to identify important or irrelevant task-specific tokens, introducing potential bias and limiting applicability. In this work, we uncover a surprising and elegant alternative: the semantically empty initial token (e.g., <BOS> in Llama) serves as a powerful and underexplored control point for optimizing model behavior. Through theoretical analysis, we show that tuning the initial token\u2019s attention sharpens or flattens the attention distribution over subsequent tokens, and its role as an attention sink amplifies this effect. Empirically, we find that: (1) tuning its attention improves LLM performance across tasks more effectively than tuning other task-specific tokens; (2) the effect follows a consistent trend across layers, with earlier layers having greater impact, but varies across attention heads, with different heads showing distinct preferences in how they attend to this token. Based on these findings, we propose ZeroTuning, a training-free approach that improves LLM performance by applying head-specific attention adjustments to this special token. Despite tuning only one token, ZeroTuning achieves higher average performance on text classification, multiple-choice QA, and multi-turn conversation tasks across models such as LLama, Qwen, and DeepSeek. For example, ZeroTuning improves Llama-3.1-8B by 11.71% on classification tasks, 2.64% on QA tasks, and raises its multi-turn score from 7.804 to 7.966. The method is also robust to limited resources, few-shot settings, long contexts, quantization, decoding strategies, and prompt variations. Our work sheds light on a previously overlooked control point in LLMs, offering new insights into both inference-time tuning and model interpretability.",
        "topic": "Efficiency, Calibration & Robustness",
        "url": "https://openreview.net/forum?id=THSbsRWy9v"
    },
    {
        "title": "Parity Requires Unified Input Dependence and Negative Eigenvalues in SSMs",
        "id": 92,
        "authors": [
            "Behnoush Khavari",
            "Jayesh Khullar",
            "Mehran Shakerinava",
            "Jerry Huang",
            "Siamak Ravanbakhsh",
            "Sarath Chandar"
        ],
        "keywords": [
            "SSMs",
            "Linear Recurrent Neural Networks",
            "State-tracking",
            "expressivity"
        ],
        "abstract": "Recent work has shown that LRNN models such as S4D, Mamba, and DeltaNet lack state-tracking capability due to either time-invariant transition matrices or restricted eigenvalue ranges. To address this, input-dependent transition matrices, particularly those that are complex or non-triangular, have been proposed to enhance SSM performance on such tasks. While existing theorems demonstrate that both input-independent and non-negative SSMs are incapable of solving simple state-tracking tasks like parity, regardless of depth, they do not explore whether combining these two types in a multilayer SSM could help. We investigate this question for efficient SSMs with diagonal transition matrices and show that such combinations still fail to solve parity. This implies that a recurrence layer must be both input-dependent and include negative eigenvalues. Our experiments support this conclusion by analyzing an SSM model that combines S4D and Mamba layers.",
        "topic": "Optimization Dynamics",
        "url": "https://openreview.net/forum?id=ZL0J1T4xAu"
    },
    {
        "title": "From SGD to Spectra: A Theory of Neural Network Weight Dynamics",
        "id": 86,
        "authors": [
            "Brian Richard Olsen",
            "Sam Fatehmanesh",
            "Frank Xiao",
            "Adarsh Kumarappan",
            "Anirudh Gajula"
        ],
        "keywords": [
            "stochastic differential equations",
            "SGD dynamics",
            "singular\u2011value spectra",
            "Dyson Brownian motion",
            "heavy\u2011tailed distributions"
        ],
        "abstract": "Deep neural networks have revolutionized machine learning, yet their training dynamics remain theoretically unclear\u2014we develop a continuous-time, matrix-valued stochastic differential equation (SDE) framework that rigorously connects the microscopic dynamics of SGD to the macroscopic evolution of singular-value spectra in weight matrices. We derive exact SDEs showing that squared singular values follow Dyson Brownian motion with eigenvalue repulsion, and characterize stationary distributions as gamma-type densities with power-law tails, providing the first theoretical explanation for the heavy-tailed \"bulk+tail\" spectral structure observed empirically in trained networks. Through controlled experiments on transformer and MLP architectures, we validate our theoretical predictions and demonstrate quantitative agreement between SDE-based forecasts and observed spectral evolution, providing a rigorous foundation for understanding why deep learning works.",
        "topic": "Optimization Dynamics",
        "url": "https://openreview.net/forum?id=PAcc7wfxZd"
    },
    {
        "title": "Gradient descent in presence of extreme flatness and steepness",
        "id": 85,
        "authors": [
            "Dravyansh Sharma"
        ],
        "keywords": [
            "Gradient descent",
            "Newton's method",
            "learning rate",
            "non-convex optimization",
            "non-smooth optimization"
        ],
        "abstract": "Typical theoretical analysis of convergence of gradient descent requires assumptions like convexity and smoothness that do not hold in practice. Towards understanding the challenges and potential solutions for learning in the presence of non-convex and non-smooth functions, we study the convergence of gradient descent for a simple sigmoid based function family. The functions in this family simultaneously exhibit extreme flatness and extreme sharpness, making it particularly challenging to choose a step size. We show that both small and large step sizes fail; in fact, convergence is a highly volatile function of initialization and learning rate. We observe similar challenges with a known regularized version of Newton's method. We propose a novel Newton-damped gradient descent that performs well for the non-convex, non-smooth family under study, in the sense that most settings of the learning rate lead to convergence. Our small scale experiments indicate interesting directions for both future empirical and theoretical research.",
        "topic": "Optimization Dynamics",
        "url": "https://openreview.net/forum?id=XQmAG9ZLnd"
    },
    {
        "title": "Geometry of Rank Constraints in Shallow Polynomial Neural Networks",
        "id": 75,
        "authors": [
            "Param Mody",
            "Maksym Zubkov"
        ],
        "keywords": [
            "machine learning theory",
            "optimisation",
            "algebraic geometry"
        ],
        "abstract": "We study shallow quadratic and cubic polynomial neural networks of width 2. In this setting, the ambient space is the space of symmetric polynomials, which is finite-dimensional. We consider four target functions that correspond to rank-2 and rank-3 symmetric matrices, and rank-2 and rank-3 symmetric tensors. We compare the learning dynamics when the target function lies within versus outside the function space (neuromanifold), and we analyze the patterns of critical points in both the parameter space and the corresponding functional space.",
        "topic": "Optimization Dynamics",
        "url": "https://openreview.net/forum?id=43cfYyiNiY"
    },
    {
        "title": "Decomposed Learning: An Avenue for Mitigating Grokking",
        "id": 70,
        "authors": [
            "Gabryel Mason-Williams",
            "Israel Mason-Williams"
        ],
        "keywords": [
            "grokking",
            "optimisation",
            "linear algebra",
            "SVD",
            "compression"
        ],
        "abstract": "Grokking is a delayed transition from memorisation to generalisation in neural networks. It challenges perspectives on efficient learning, particularly in structured tasks and small-data regimes. We explore grokking in modular arithmetic from the perspective of a training pathology. We use Singular Value Decomposition (SVD) to modify the weight matrices of neural networks by changing the representation of the weight matrix, $W$, into the product of three matrices, $U$, $\\Sigma$ and $V^T$. Through empirical evaluations on the modular addition task, we show that this representation significantly reduces the effect of grokking and, in some cases, eliminates it.",
        "topic": "Optimization Dynamics",
        "url": "https://openreview.net/forum?id=LVuzwpMovE"
    },
    {
        "title": "Why Loss Re-weighting Works If You Stop Early: Training Dynamics of Unconstrained Features",
        "id": 66,
        "authors": [
            "Yize Zhao",
            "Christos Thrampoulidis"
        ],
        "keywords": [
            "Loss reweighting",
            "Early stopping",
            "Class imbalance",
            "Learning dynamics",
            "Unconstrained features model (UFM)"
        ],
        "abstract": "The application of loss reweighting in modern deep learning presents a nuanced picture. While it fails to alter the terminal learning phase in overparameterized deep neural networks (DNNs) trained on high-dimensional datasets, empirical evidence consistently shows it offers significant benefits early in training. To transparently demonstrate and analyze this phenomenon, we introduce a small-scale model (SSM). This model is specifically designed to abstract the inherent complexities of both the DNN architecture and the input data, while maintaining key information about the structure of imbalance within its spectral components.  On the one hand, the SSM reveals how vanilla empirical risk minimization preferentially learns to distinguish majority classes over minorities early in training, consequently delaying minority learning. In stark contrast, reweighting restores balanced learning dynamics, enabling the simultaneous learning of features associated with both majorities and minorities.",
        "topic": "Optimization Dynamics",
        "url": "https://openreview.net/forum?id=tfIaWUfdnU"
    },
    {
        "title": "Emergence, pretraining loss and associative recall: a toy model",
        "id": 56,
        "authors": [
            "Sultan Daniels",
            "Dylan Davis",
            "Dhruv Gautam",
            "Wentinn Liao",
            "Gireeja Ranade",
            "Anant Sahai"
        ],
        "keywords": [
            "emergence",
            "time-series",
            "toy models",
            "interpretability"
        ],
        "abstract": "To study emergence in LLM-style neural networks, we introduce a new family of toy problems that combine features of linear-regression style continuous in-context learning (ICL) with discrete associative recall --- specifically symbolically labeled interleaved observations from randomly drawn deterministic linear dynamical systems. We pretrain transformer models on sample traces from this toy, and explore the idea that the emergence of an ability is largely a function of the pretraining loss. During training, this toy model exhibits the emergence of at least three different abilities, and we use simple out-of-distribution experiments to show how some of these abilities seem to completely ignore what feels to a human as being very salient context.",
        "topic": "Optimization Dynamics",
        "url": "https://openreview.net/forum?id=MU4NsMVrFw"
    },
    {
        "title": "Neural Stochastic Differential Equations on Compact State-Spaces",
        "id": 45,
        "authors": [
            "Yue-Jane Liu",
            "Malinda Lu",
            "Matthew K. Nock",
            "Yaniv Yacoby"
        ],
        "keywords": [
            "Neural SDEs",
            "Inductive Bias",
            "Viability Theory",
            "Generative Models",
            "Dynamical Systems",
            "Time Series"
        ],
        "abstract": "Many modern probabilistic models rely on SDEs, but their adoption is hampered by instability, poor inductive bias outside bounded domains, and reliance on restrictive dynamics or training tricks. While recent work constrains SDEs to compact spaces using reflected dynamics, these approaches lack continuous dynamics and efficient high-order solvers, limiting interpretability and applicability. We propose a novel class of neural SDEs on compact spaces with continuous dynamics, amenable to higher-order solvers and with favorable inductive bias.",
        "topic": "Optimization Dynamics",
        "url": "https://openreview.net/forum?id=6BGJrnRxYW"
    },
    {
        "title": "Universal Dynamics of Warmup Stable Decay: understanding WSD beyond Transformers",
        "id": 44,
        "authors": [
            "Annalisa Belloni",
            "Lorenzo Noci",
            "Antonio Orvieto"
        ],
        "keywords": [
            "LR schedule",
            "Deep Learning",
            "Optimization",
            "Warmup Stable Decay",
            "CNN"
        ],
        "abstract": "The Warmup Stable Decay (WSD) learning rate scheduler has recently become popular, largely due to its good performance and flexibility when training large language models. It remains an open question whether the remarkable performance of WSD - using a decaying learning rate for only a fraction of training compared to cosine decay - is a phenomenon specific to transformer-based language models that can potentially offer new theoretical insights into their training dynamics. Inspired by the usage of learning rate schedulers as a new lens into understanding landscape geometry (e.g., river valley, connected minima, progressive sharpening), in this work we compare the WSD path of the Adam optimizer on a Pythia-like language model to that of a small CNN trained to classify CIFAR10 images. We observe most training signals, optimizer path features, and sharpness dynamics to be qualitatively similar in such architectures. This consistency points to shared geometric characteristics of the loss landscapes of old and new nonconvex problems, and hints to future research questions around the geometry of high dimensional optimization problems.",
        "topic": "Optimization Dynamics",
        "url": "https://openreview.net/forum?id=2HNQqMBvC2"
    },
    {
        "title": "Cross-Validation Error Dynamics in Smaller Datasets",
        "id": 30,
        "authors": [
            "Bethany austhof",
            "Lev Reyzin"
        ],
        "keywords": [
            "cross-validation",
            "negative correlation",
            "hypergeometric sampling"
        ],
        "abstract": "Cross-validation (CV) is the de facto standard for estimating a model\u2019s generalization performance, but in smaller datasets it exhibits an underappreciated quirk: across folds, training and test errors are strongly negatively correlated, while training and holdout errors show a moderate anticorrelation and test versus holdout errors are essentially uncorrelated. Herein, we document these phenomena empirically\u2014on both real and synthetic datasets under AdaBoost\u2014and introduce a simple generative model that explains them. By viewing each CV split as hypergeometric sampling from a finite population and incorporating an overfitting parameter \u03b4 that shifts expected errors on train, test, and holdout sets, we derive closed-form expressions for the covariances among observed error rates. Our analysis shows that sampling-induced anticorrelation dominates in small datasets, while overfitting contributes an additional negative term, thus accounting for the observed error dynamics. We discuss the limitations of our approach and suggest directions for more refined models and extensions to regression settings.",
        "topic": "Optimization Dynamics",
        "url": "https://openreview.net/forum?id=aBHCgj4FdW"
    },
    {
        "title": "Emergence of Hebbian Dynamics in Regularized Non-Local Learners",
        "id": 16,
        "authors": [
            "David Aaron Koplow",
            "Tomaso Poggio",
            "Liu Ziyin"
        ],
        "keywords": [
            "Neuroscience",
            "Hebbian Learning",
            "Gradient Descent"
        ],
        "abstract": "Stochastic gradient descent (SGD) is often viewed as biologically implausible, while local Hebbian rules dominate theories of synaptic plasticity in our brain. We prove and empirically demonstrate--on small MLPs and transformers that can be trained on a single GPU--that SGD with weight decay can naturally produce Hebbian-like dynamics near stationarity, whereas injected gradient noise can flip the alignment to be anti-Hebbian. The effect holds for nearly any learning rule, even some random ones, revealing Hebbian behavior as an emergent epiphenomenon of deeper optimization dynamics during training. These results narrow the gap between artificial and biological learning and caution against treating observed Hebbian signatures as evidence against global error-driven mechanisms in our brains.",
        "topic": "Optimization Dynamics",
        "url": "https://openreview.net/forum?id=M8R1HZtPsr"
    },
    {
        "title": "An Empirical Investigation of Initialization Strategies for Kolmogorov\u2013Arnold Networks",
        "id": 8,
        "authors": [
            "Spyros Rigas",
            "Dhruv Verma",
            "Georgios Alexandridis",
            "Yixuan Wang"
        ],
        "keywords": [
            "Kolmogorov\u2013Arnold networks",
            "deep learning",
            "initialization",
            "power law"
        ],
        "abstract": "Kolmogorov\u2013Arnold Networks (KANs) are a recently introduced neural architecture that use trainable activation functions instead of fixed ones, offering greater flexibility and interpretability. Although KANs have shown promising results across various tasks, little attention has been given to how they should be initialized. In this work, we explore alternative initialization strategies, including two variance-preserving methods based on classical ideas and an empirical power-law approach with tunable exponents. Using function fitting as a small-scale testbed, we run a large grid search over architectures and initialization settings. We find that power-law configurations consistently outperform the standard baseline initialization across all architectures. The variance-preserving methods tend to underperform on smaller models but outperform the baseline as networks grow deeper and wider, though they still do not match the performance of power-law initialization. Overall, our results highlight initialization as an important yet underexplored aspect of KANs and point to several directions for future work.",
        "topic": "Optimization Dynamics",
        "url": "https://openreview.net/forum?id=eC285SNCiW"
    },
    {
        "title": "How Much Context Does Natural Language Actually Require? An Analysis Using LLMs as Statistical Oracles",
        "id": 71,
        "authors": [
            "Vala Vakilian",
            "Sadegh Mahdavi",
            "Christos Thrampoulidis"
        ],
        "keywords": [
            "LLMs",
            "Long Context",
            "Sampling",
            "Decoding",
            "Language Structure",
            "Inference"
        ],
        "abstract": "Despite the growing trend towards large-context transformer models, key questions remain about how much context is truly required for accurate language modeling. We explore this by treating large language models as statistical oracles and measuring the smallest prefix needed to replicate full-context next-token predictions. Using samples from diverse natural text sources, we evaluate minimal context length  requirements across various decoding strategies using correctness and support set overlap metrics. Under greedy decoding, we find that over 80\\% of tokens require less than 10\\% of the most recent context to yield identical predictions. For general sampling strategies, we define Recall and Risk metrics to assess context dependence, and find that dynamic strategies offer higher support coverage at low percentiles\u2014while also increasing Risk due to broader supports at shorter contexts.",
        "topic": "Representation Dynamics",
        "url": "https://openreview.net/forum?id=yNhbnum0iQ"
    },
    {
        "title": "Pruning Increases Orderedness in Weight-Tied Recurrent Computation",
        "id": 69,
        "authors": [
            "YIDING SONG"
        ],
        "keywords": [
            "pruning",
            "directionality",
            "hierarchical organisation",
            "perceptron"
        ],
        "abstract": "Inspired by the prevalence of recurrent circuits in biological brains, we investigate the degree to which directionality is a helpful inductive bias for artificial neural networks. Taking directionality as topologically-ordered information flow between neurons, we formalise a perceptron layer with all-to-all connections (mathematically equivalent to a weight-tied recurrent neural network) and demonstrate that directionality, a hallmark of modern feed-forward networks, can be \\emph{induced} rather than hard-wired by applying appropriate pruning techniques. Across different random seeds our pruning schemes successfully induce greater topological ordering in information flow between neurons without compromising performance, suggesting that directionality is \\emph{not} a prerequisite for learning, but may be an advantageous inductive bias discoverable by gradient descent and sparsification.",
        "topic": "Representation Dynamics",
        "url": "https://openreview.net/forum?id=aUdJ9UlNLK"
    },
    {
        "title": "In-Context Occam\u2019s Razor: How Transformers Prefer Simpler Hypotheses on the Fly",
        "id": 67,
        "authors": [
            "Puneesh Deora",
            "Bhavya Vasudeva",
            "Tina Behnia",
            "Christos Thrampoulidis"
        ],
        "keywords": [
            "In-context learning",
            "markov chains",
            "Occam's razor",
            "linear regression",
            "transformers",
            "varying complexity"
        ],
        "abstract": "In-context learning (ICL) enables transformers to adapt to new tasks through contextual examples without parameter updates. While existing research has typically studied ICL in fixed-complexity environments, real-world language models encounter tasks spanning diverse complexity levels. This paper investigates how transformers navigate hierarchical task structures where higher-complexity categories can perfectly represent any pattern generated by simpler ones. We design testbeds based on both Markov chains and linear regression that reveal transformers not only identify the appropriate complexity level for each task but also accurately infer the corresponding parameters\u2014even when the in-context examples are compatible with multiple complexity hypotheses. Notably, when presented with data generated by simpler processes, transformers consistently favor the least complex sufficient explanation. We theoretically explain this behavior through a Bayesian framework, demonstrating that transformers effectively implement an in-context Bayesian Occam's razor by balancing model fit against complexity penalties.",
        "topic": "Representation Dynamics",
        "url": "https://openreview.net/forum?id=KgdkO1KNxP"
    },
    {
        "title": "Understanding How Chess-Playing Language Models Compute Linear Board Representations",
        "id": 49,
        "authors": [
            "Aaron Mei"
        ],
        "keywords": [
            "Mechanistic Interpretability",
            "Language Models",
            "Chess",
            "World Models"
        ],
        "abstract": "The field of mechanistic interpretability seeks to understand the internal workings of neural networks, particularly language models. While previous research has demonstrated that language models trained on games can develop linear board representations, the mechanisms by which these representations arise are unknown. This work investigates the internal workings of a GPT-2 style transformer trained on chess PGNs, and proposes an algorithm for how the model computes the board state.",
        "topic": "Representation Dynamics",
        "url": "https://openreview.net/forum?id=Z9OV9NygER"
    },
    {
        "title": "Generative or Discriminative? Revisiting Text Classification in the Era of Transformers",
        "id": 42,
        "authors": [
            "Siva Rajesh Kasa",
            "Sumegh Roychowdhury",
            "Karan Gupta",
            "Yaswanth Biruduraju",
            "SANTHOSH KUMAR KASA",
            "Ashutosh Kumar",
            "Pattisapu Nikhil Priyatam",
            "Arindam Bhattacharya",
            "Shailendra Agarwal",
            "Vijay huddar"
        ],
        "keywords": [
            "Generative Classifiers",
            "Discrete Diffusion Models",
            "Autoregressive models",
            "Encoder Models",
            "Masked Language Models"
        ],
        "abstract": "In text classification, the classical comparison between discriminative and generative classifiers gains renewed relevance in the transformer era, where computational constraints often limit thorough experimentation. Through systematic small-scale experiments on text classification tasks, we investigate how the fundamental ``two regimes\" phenomenon\u2014where generative classifiers excel with limited data but show higher asymptotic error\u2014manifests across modern architectures (Auto-regressive, Masked Language Models, Discrete Diffusion, and Encoders). By training models from scratch on controlled text datasets, we isolate and analyze core architectural behaviors in terms of sample efficiency, calibration, and preservation of ordinal relationships. Our findings provide insights into the inherent trade-offs of different modelling approaches for text classification, demonstrating how small-scale experimentation can inform both theoretical understanding and practical architectural choices.",
        "topic": "Representation Dynamics",
        "url": "https://openreview.net/forum?id=S92PUZIsDs"
    },
    {
        "title": "Exploring Diverse Solutions for Underdetermined Problems",
        "id": 33,
        "authors": [
            "Eric Volkmann",
            "Andreas Radler",
            "Johannes Brandstetter",
            "Arturs Berzins"
        ],
        "keywords": [
            "theory-informed learning",
            "data-free",
            "diversity",
            "mode collapse"
        ],
        "abstract": "This work explores the utility of a recently proposed diversity loss in training generative, theory-informed models on underdetermined problems with multiple solutions. Unlike data-driven methods, theory-informed learning often operates in data-free settings, optimizing neural networks to satisfy objectives and constraints. We demonstrate how this diversity loss encourages the generation of diverse solutions across various example problems, effectively avoiding mode collapse and enabling exploration of the solution space.",
        "topic": "Representation Dynamics",
        "url": "https://openreview.net/forum?id=HEqtcVOicH"
    },
    {
        "title": "Transformers May Learn to Classify In-Context by Context-Adaptive Kernel Gradient Descent",
        "id": 32,
        "authors": [
            "Sara Dragutinovi\u0107",
            "Andrew M Saxe",
            "Aaditya K Singh"
        ],
        "keywords": [
            "transformers",
            "in-context learning",
            "gradient descent",
            "mechanistic interpretability"
        ],
        "abstract": "The remarkable ability of transformers to learn new concepts solely by reading examples within the input prompt, termed in-context learning (ICL), is a crucial aspect of intelligent behavior. Here, we focus on understanding the learning algorithm transformers use to learn from context. Existing theoretical work, often based on simplifying assumptions, has primarily focused on linear self-attention and continuous regression tasks, finding transformers can learn in-context by gradient descent. Given that transformers are typically trained on discrete and complex tasks, we bridge the gap from this existing work to the setting of *classification*, with *non-linear* (importantly, *softmax*) activation. We find that transformers still learn to do gradient descent in-context, though on functionals in the kernel feature space and with a context-adaptive learning rate in the case of softmax transformer.",
        "topic": "Representation Dynamics",
        "url": "https://openreview.net/forum?id=ngs41s5vvX"
    },
    {
        "title": "Koopman Autoencoders Learn Neural Representation Dynamics",
        "id": 20,
        "authors": [
            "Nishant Suresh Aswani",
            "Saif Jabari"
        ],
        "keywords": [
            "dynamics",
            "neural representations",
            "koopman theory"
        ],
        "abstract": "This paper explores a simple question: can we model the internal transformations of a neural network using dynamical systems theory? We introduce Koopman autoencoders to capture how neural representations evolve through network layers, treating these representations as states in a dynamical system. Our approach learns a surrogate model that predicts how neural representations transform from input to output, with two key advantages. First, by way of lifting the original states via an autoencoder, it operates in a linear space, making editing the dynamics straightforward. Second, it preserves the topologies of the original representations by regularizing the autoencoding objective. We demonstrate that these surrogate models naturally replicate the progressive topological simplification observed in neural networks. As a practical application, we show how our approach enables targeted class unlearning in the Yin-Yang and MNIST classification tasks.",
        "topic": "Representation Dynamics",
        "url": "https://openreview.net/forum?id=iqqrXH2tMk"
    },
    {
        "title": "Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry",
        "id": 81,
        "authors": [
            "Sai Sumedh R. Hindupur",
            "Ekdeep Singh Lubana",
            "Thomas Fel",
            "Demba E. Ba"
        ],
        "keywords": [
            "Sparse Autoencoders",
            "Dictionary Learning",
            "Interpretability"
        ],
        "abstract": "Sparse Autoencoders (SAEs) are widely used to interpret neural networks by identifying meaningful concepts from their representations.  We show that each SAE imposes structural assumptions about how concepts are encoded in model representations, which in turn shapes what it can and cannot detect. We train SAEs on synthetic data with specific structure to show that SAEs fail to recover concepts when their assumptions are ignored, and we design a new SAE---called SpaDE---that enables the discovery of previously hidden concepts (those with heterogenous intrinsic dimensionality and nonlinear separation boundaries) and reinforces our theoretical insights.",
        "topic": "Representation Dynamics",
        "url": "https://openreview.net/forum?id=AKaoBzhIIF"
    },
    {
        "title": "Evaluating Sparse Autoencoders: From Shallow Design to Matching Pursuit",
        "id": 76,
        "authors": [
            "Val\u00e9rie Costa",
            "Thomas Fel",
            "Ekdeep Singh Lubana",
            "Bahareh Tolooshams",
            "Demba E. Ba"
        ],
        "keywords": [
            "Interpretability",
            "Representation Learning",
            "Dictionary Learning",
            "Sparse Autoencoders"
        ],
        "abstract": "Sparse autoencoders (SAEs) have recently become central tools for interpretability, leveraging dictionary learning principles to extract sparse, interpretable features from neural representations whose underlying structure is typically unknown. This paper evaluates SAEs in a controlled setting using MNIST, which reveals that current shallow architectures implicitly rely on a quasi-orthogonality assumption that limits the ability to extract correlated features. To move beyond this, we compare them with an iterative SAE that unrolls Matching Pursuit (MP-SAE), enabling the residual-guided extraction of correlated features that arise in hierarchical settings such as handwritten digit generation while guaranteeing monotonic improvement of the reconstruction as more atoms are selected.",
        "topic": "Representation Dynamics",
        "url": "https://openreview.net/forum?id=SLGftRJVUN"
    },
    {
        "title": "Stats or Facts: Decomposing Generalization in Language Models with Small-Scale Models",
        "id": 68,
        "authors": [
            "Tina Behnia",
            "Puneesh Deora",
            "Christos Thrampoulidis"
        ],
        "keywords": [
            "factual recall",
            "diversity",
            "markov chains",
            "language models",
            "transformers",
            "training dynamics"
        ],
        "abstract": "Large language models learn both statistical patterns  that make text fluent and factual associations between specific tokens that represent knowledge information. The complexity of natural language interweaving linguistic patterns and factual content challenges a systematic study of this capability. To address this, we introduce a Small-Scale Data Model (SSDM) designed to disentangle these components. The SSDM consists of a statistical stream of generic tokens, endowed with designated positional information, which composes with a separate factual stream of source-target token pairs representing knowledge. Partitioning the generating distribution of the statistical stream into sub-distributions, which we term templates, allows us to: (i) Independently vary the format of the templates (i.e., contextual structure) and the frequency with which facts appear within each template during training (i.e., contextual diversity); (ii) Measure both in-distribution and out-of-distribution generalization; and (iii) Distinguish between statistical, structural, and factual aspects of language model generalization. We demonstrate the flexibility of the SSDM by reporting example findings concerning: (a) the potentially catastrophic impact of low contextual diversity on either factual recall, statistical generalization, or both, contingent on the contextual structure; (b) observed stage-wise learning dynamics; and (c) hallucination.",
        "topic": "Synthetic Benchmarks & Inductive Bias",
        "url": "https://openreview.net/forum?id=2cp7X3Z7Hc"
    },
    {
        "title": "SynDaCaTE: A Synthetic Dataset For Evaluating Part-Whole Hierarchical Inference",
        "id": 88,
        "authors": [
            "Jake Levi",
            "Mark van der Wilk"
        ],
        "keywords": [
            "Synthetic dataset",
            "part-whole hierarchy",
            "inductive bias",
            "data efficiency",
            "capsule models"
        ],
        "abstract": "Learning to infer object representations, and in particular part-whole hierarchies, has been the focus of extensive research in computer vision, in pursuit of improving data efficiency, systematic generalisation, and robustness. Models which are \\emph{designed} to infer part-whole hierarchies, often referred to as capsule networks, are typically trained end-to-end on supervised tasks such as object classification, in which case it is difficult to evaluate whether such a model \\emph{actually} learns to infer part-whole hierarchies, as claimed. To address this difficulty, we present a SYNthetic DAtaset for CApsule Testing and Evaluation, abbreviated as SynDaCaTE, and establish its utility by (1) demonstrating the precise bottleneck in a prominent existing capsule model, and (2) demonstrating that permutation-equivariant self-attention is highly effective for parts-to-wholes inference, which motivates future directions for designing effective inductive biases for computer vision.",
        "topic": "Synthetic Benchmarks & Inductive Bias",
        "url": "https://openreview.net/forum?id=SpppKsudMo"
    },
    {
        "title": "Restoring Task-Relevant Information in Synthetic Data: A Small-Scale V-Information View",
        "id": 87,
        "authors": [
            "Sid Bharthulwar"
        ],
        "keywords": [
            "Synthetic Data",
            "V-Information",
            "Information Restoration",
            "Small-Scale Experiments",
            "Model Capacity",
            "CNNs",
            "LLMs",
            "Inductive Biases",
            "Alignment"
        ],
        "abstract": "This paper investigates synthetic data generation as a mechanism for restoring or reformatting task-relevant information that is obscured or unusable for a specific, computationally bounded learner. We conduct a small-scale, controlled experiment on CIFAR-10, involving pixel permutation to corrupt data, a Convolutional Autoencoder (Conv-AE) synthesizer for information restoration, and a downstream CNN learner. Framed through V-Information, which quantifies information accessible to such a learner, empirical results demonstrate that while permutation drastically reduces usable V-Information, the synthesizer partially restores it, leading to significant performance recovery. We further explore how model capacities interact with this process, finding learner capacity beneficial only when usable information is present. This highlights computation\u2019s role in making latent information accessible, a principle highly relevant to current synthetic data practices in capabilities and alignment of foundation models.",
        "topic": "Synthetic Benchmarks & Inductive Bias",
        "url": "https://openreview.net/forum?id=FtNJa2n8wk"
    },
    {
        "title": "Improving Pathfinding with Anchoring Tokens",
        "id": 80,
        "authors": [
            "Huaqing Zhang",
            "Bingbin Liu",
            "Juno Kim",
            "Andrej Risteski"
        ],
        "keywords": [
            "path-finding",
            "planning",
            "next-token prediction"
        ],
        "abstract": "Planning is a critical aspect of multi-step reasoning, yet it remains challenging for large language models (LLMs). In this work, we use pathfinding in graphs as a sandbox for understanding and improving the planning abilities of LLMs. Our results show that while conventional autoregressive training generalizes poorly, an anchoring strategy, whereby a model first predicts a small subset of intermediate nodes along the path, significantly improves the path finding performance. We confirm these gains on two families of graphs with markedly different structures and provide preliminary heuristics for selecting effective anchor nodes, offering guidance for more realistic settings.",
        "topic": "Synthetic Benchmarks & Inductive Bias",
        "url": "https://openreview.net/forum?id=AQbLem2j2B"
    },
    {
        "title": "Towards Understanding Self-Pretraining for Sequence Classification",
        "id": 78,
        "authors": [
            "Omar Coser",
            "Antonio Orvieto"
        ],
        "keywords": [
            "Attention",
            "Pretraining",
            "Sequence Models"
        ],
        "abstract": "It was recently shown by Amos et al. (2023) that to boost test accuracy of transformer models on sequence classification, it can be highly effective to first pretrain with a masked token prediction objective on exactly the same data (self-pretraining, SPT). While the focus of Amos et al. (2023) is to show that transformers \u2013 and not only state-space models (SSMs, like S4) \u2013 can perform well on the Long-Range Arena (LRA, a collection of challenging synthetic sequence classification tasks), their finding is intriguing from a more fundamental perspective. Indeed, even though it can be easily claimed that the observed gains come from the benefits of data-driven initialization and pretraining inductive biases, it is unclear which precise mechanism unlocks performance and why standard supervised learning can fail. To better understand this intriguing phenomenon, we replicate and ablate the results of Amos et al. (2023). We show that substantial gains can be observed even at an extremely small scale, using a self-pretraining pipeline that requires little extra compute. We further identify in the attention mechanism weights the source of SPT improved performance. We hope our insights lead to future investigations around SPT, and that our work exposes this unusual yet promising technique for data-scarce learning to a broader audience.",
        "topic": "Synthetic Benchmarks & Inductive Bias",
        "url": "https://openreview.net/forum?id=w4usPIgago"
    },
    {
        "title": "Discovering Hidden Algebraic Structures via Transformers with Rank-Aware Beam GRPO",
        "id": 72,
        "authors": [
            "Jaeha Lee",
            "Gio Huh",
            "Ning Su",
            "Tony Yue YU"
        ],
        "keywords": [
            "functional decomposition",
            "polynomial decomposition",
            "beam search",
            "reinforcement learning",
            "symbolic reasoning",
            "transformer model"
        ],
        "abstract": "We study the capabilities of small-scale transformer models in symbolic reasoning, focusing on the NP-hard algebraic task of multivariate polynomial decomposition, with widespread applications in science and engineering. Our approach includes a fine-grained synthetic data generation pipeline, supervised pretraining, beam search, evaluations for scaling behavior and generalizability, and a novel rank-aware reinforcement learning method called Beam Grouped Relative Policy Optimization (BGRPO), which improves accuracy while reducing inference compute by up to 75%. Additionally, our model demonstrates competitive performance in polynomial simplification, outperforming Mathematica in various cases.",
        "topic": "Synthetic Benchmarks & Inductive Bias",
        "url": "https://openreview.net/forum?id=lO9q5itiqK"
    },
    {
        "title": "Permutations as a testbed for studying the effect of input representations on learning",
        "id": 58,
        "authors": [
            "Sarah McGuire Scullen",
            "Davis Brown",
            "Robert Jasper",
            "Henry Kvinge",
            "Helen Jenne"
        ],
        "keywords": [
            "data representation",
            "permutations",
            "deep learning",
            "data",
            "model architecture"
        ],
        "abstract": "Quality data is crucial for deep learning. However, relative to progress in model training and data curation, there is a lesser focus on understanding the effects of how data is encoded and passed to the neural network\u2014 the \u201cdata representation.\u201d This is especially true for non-textual domains, where there are often challenges in distinguishing between the difficulty of the learning task versus difficulties arising merely from the format of the input data. We propose using permutations, which have multiple natural mathematical representations, as a systematic way to study how task difficulty and learning outcomes are influenced by the choice of input data representation. In our setting, we find that the model performance on a data representation can change significantly with the number of examples and architecture type; however, with enough examples most tasks are learned regardless of data representation.",
        "topic": "Synthetic Benchmarks & Inductive Bias",
        "url": "https://openreview.net/forum?id=VbqocgftJF"
    },
    {
        "title": "Extrapolation by Association: Length Generalization Transfer in Transformers",
        "id": 53,
        "authors": [
            "Ziyang Cai",
            "Nayoung Lee",
            "Avi Schwarzschild",
            "Samet Oymak",
            "Dimitris Papailiopoulos"
        ],
        "keywords": [
            "Transformers",
            "Language Models",
            "Length Generalization",
            "Composition"
        ],
        "abstract": "Transformer language models have demonstrated impressive generalization capabilities in natural language domains, yet we lack a fine-grained understanding of how such generalization arises. In this paper, we investigate length generalization\u2014the ability to extrapolate from shorter to longer inputs\u2014through the lens of \\textit{task transfer}. We find that length generalization can be \\textit{transferred} across related tasks. That is, training a model with a longer and related auxiliary task can lead the model to generalize to unseen and longer inputs from some other target task. We demonstrate this length generalization transfer across a diverse suite of algorithmic tasks, including arithmetic operations, string transformations, and maze navigation. Our results show that transformer models can inherit generalization capabilities from similar tasks when trained jointly. Moreover, we observe similar transfer effects in pretrained language models, suggesting that pretraining equips models with reusable computational scaffolding that facilitates extrapolation in downstream settings. Finally, we provide initial mechanistic evidence that length generalization transfer correlates with the re-use of the same attention heads between the tasks. Together, our findings deepen our understanding of how transformers generalize to out-of-distribution inputs and highlight the compositional reuse of inductive structure across tasks.",
        "topic": "Synthetic Benchmarks & Inductive Bias",
        "url": "https://openreview.net/forum?id=DG244g0EUW"
    },
    {
        "title": "Do Larger Language Models Imply Better Generalization? A Pretraining Scaling Law for Implicit Reasoning",
        "id": 48,
        "authors": [
            "Xinyi Wang",
            "Shawn Tan",
            "Mingyu Jin",
            "William Yang Wang",
            "Rameswar Panda",
            "Yikang Shen"
        ],
        "keywords": [
            "Language Models",
            "Scaling Law",
            "Reasoning",
            "Knowledge Graph"
        ],
        "abstract": "Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks requiring complex reasoning. However, the effects of scaling on their reasoning abilities remain insufficiently understood. In this paper, we introduce a synthetic multihop reasoning environment designed to closely replicate the structure and distribution of real-world large-scale knowledge graphs. Our reasoning task involves completing missing edges in the graph, which requires advanced multi-hop reasoning and mimics real-world reasoning scenarios. To evaluate this, we pretrain language models (LMs) from scratch solely on triples from the incomplete graph and assess their ability to infer the missing edges. Interestingly, we observe that overparameterization can impair reasoning performance due to excessive memorization. We investigate different factors that affect this U-shaped loss curve, including graph structure, model size, and training steps. To predict the optimal model size for a specific knowledge graph, we find an empirical scaling that linearly maps the knowledge graph search entropy to the optimal model size. This work provides new insights into the relationship between scaling and reasoning in LLMs, shedding light on possible ways to optimize their performance for reasoning tasks.",
        "topic": "Synthetic Benchmarks & Inductive Bias",
        "url": "https://openreview.net/forum?id=JNTaZD7Iam"
    },
    {
        "title": "Measuring Memorization and Generalization in Forecasting Models via Structured Perturbations of Chaotic Systems",
        "id": 34,
        "authors": [
            "Max Kanwal",
            "Caryn Tran"
        ],
        "keywords": [
            "dynamical systems",
            "OOD genearalization"
        ],
        "abstract": "We introduce a benchmarking method for evaluating generalization and memorization in time series forecasting models of chaotic dynamical systems. By generating two complementary types of test sets\u2014by perturbating training trajectories to minimally/maximally diverge over a fixed time horizon\u2014we quantify each model's sensitivity to distribution shift. Our results reveal consistent trade-offs between training accuracy and OOD generalization across neural architectures, offering a lightweight diagnostic tool for model evaluation in the small-data regime.",
        "topic": "Synthetic Benchmarks & Inductive Bias",
        "url": "https://openreview.net/forum?id=DHo0eqNkvm"
    },
    {
        "title": "Encoding Domain Insights into Multi-modal Fusion: Improved Performance at the Cost of Robustness",
        "id": 29,
        "authors": [
            "Jackson Sam Michaels",
            "Sidong Zhang",
            "Madalina Fiterau"
        ],
        "keywords": [
            "Multi-modal Fusion",
            "Robustness",
            "Inductive Priors",
            "Synthetic Tasks",
            "Small-Scale Experiments",
            "Interpretability"
        ],
        "abstract": "Using small-scale experiments with real and synthetic tasks, we compare multi-modal fusion methods, including a proposed `Product Fusion', to demonstrate how encoding task-specific priors affects performance. Our results highlight a crucial trade-off: aligning fusion design with priors boosts clean-data accuracy with limited data but significantly diminishes robustness to noisy inputs.",
        "topic": "Synthetic Benchmarks & Inductive Bias",
        "url": "https://openreview.net/forum?id=VkH4WH1cOO"
    },
    {
        "title": "Dataset Distillation for Memorized Data: Soft Labels can Leak Held-Out Teacher Knowledge",
        "id": 26,
        "authors": [
            "Freya Behrens",
            "Lenka Zdeborova"
        ],
        "keywords": [
            "knowledge distillation",
            "dataset distillation",
            "memorization",
            "learning theory",
            "model transfer",
            "shortcut learning"
        ],
        "abstract": "Dataset and knowledge distillation transfer capabilities between models.  Their efficiency is often linked to structure in the data.  However, next to general skills, modern neural networks encode specific facts, but if and how such memorized information is transferred remains less understood. To analyze the transfer of memorized information in isolation, we consider finite random i.i.d. datasets where generalization is a priori impossible and a successful teacher fit implies pure memorization. Yet, we show that students can learn non-trivial accuracy on held out memorized teacher data they never directly observed - in some cases up to perfect accuracy.  This notebook showcases this phenomenon in three different contexts, and sets up the framework required for a deeper empirical and theoretical analysis.",
        "topic": "Synthetic Benchmarks & Inductive Bias",
        "url": "https://openreview.net/forum?id=RXiLRoUEpf"
    },
    {
        "title": "Transformers Pretrained on Procedural Data Contain Modular Structures for Algorithmic Reasoning",
        "id": 6,
        "authors": [
            "Zachary Shinnick",
            "Liangze Jiang",
            "Hemanth Saratchandran",
            "Anton van den Hengel",
            "Damien Teney"
        ],
        "keywords": [
            "Inductive Biases",
            "Procedural Data",
            "Algorithmic Reasoning",
            "Pre-training",
            "Transformers"
        ],
        "abstract": "$\\textbf{Context.}$ Pretraining on large, semantically rich datasets is key for developing language models. Surprisingly, recent studies have shown that even synthetic data, generated procedurally through simple semantic-free algorithms, can yield some of the same benefits as natural language pretraining. It is unclear $\\textit{what}$ specific capabilities such simple synthetic data instils in a model, $\\textit{where}$ these capabilities reside in the architecture, and $\\textit{how}$ they manifest within its weights.  $\\textbf{Findings.}$ In this short paper, we identify several beneficial forms of procedural data, together with specific algorithmic reasoning skills that improve in small transformers. Our core finding is that different procedural rules instil $\\textit{distinct but complementary inductive structures}$ in the model. With extensive ablations and partial-transfer experiments, we discover that these structures reside in different parts of the model. Attention layers often carry the most transferable information, but some pretraining rules impart useful structure to MLP blocks instead. Most interestingly, the structures induced by multiple rules can be composed to jointly reinforce multiple capabilities.   $\\textbf{Implications.}$ These results suggest an exciting possibility of disentangling the acquisition of knowledge from reasoning in language models, with the goal of improving their robustness and data efficiency.",
        "topic": "Synthetic Benchmarks & Inductive Bias",
        "url": "https://openreview.net/forum?id=mMfVdDyd4h"
    },
    {
        "title": "The Necessity for Intervention Fidelity: Unintended Side Effects When Steering LLMs",
        "id": 64,
        "authors": [
            "Jonas B Raedler",
            "Weiyue Li",
            "Alyssa Mia Taliotis",
            "Manasvi Goyal",
            "Siddharth Swaroop",
            "Weiwei Pan"
        ],
        "keywords": [
            "Steering",
            "Representation Engineering",
            "LLM",
            "AI",
            "Social Bias"
        ],
        "abstract": "Steering (inference-time modification of activations) offers a lightweight alternative to fine-tuning for aligning large language models (LLMs). While effective on targeted behaviors, we do not yet understand its effects on unrelated model behaviors. Here, we present a systematic comparison of steering across pretrained and fine-tuned models in the context of social bias. We find that in pretrained models, steering suppresses the intended (stereotypical) behavior, as expected. However, in fine-tuned models, steering primarily suppresses unrelated outputs, and this is both unexpected and undesired. This misalignment reveals aggregate metrics masks side-effects, highlighting the need for a focus on intervention fidelity (the degree to which an intervention impacts models as intended.) We hypothesize that this is due to fine-tuning increasing anisotropy of the latent space, entangling unrelated behaviors and thereby reducing steering precision.",
        "topic": "Trust, Fairness & Explanation",
        "url": "https://openreview.net/forum?id=6gLxi32SSu"
    },
    {
        "title": "Optimizing Explanations: Nuances Matter When Evaluation Metrics Become Loss Functions",
        "id": 63,
        "authors": [
            "Jonas B Raedler",
            "Hiwot Belay Tadesse",
            "Weiwei Pan",
            "Finale Doshi-Velez"
        ],
        "keywords": [
            "Explanations",
            "Properties",
            "Optimization",
            "Feature Attribtution Explanations"
        ],
        "abstract": "Recent work has introduced a framework that allows users to directly optimize explanations for desired properties and their trade-offs. While powerful in principle, this method repurposes evaluation metrics as loss functions \u2013 an approach whose implications are not yet well understood. In this paper, we study how different robustness metrics influence the outcome of explanation optimization, holding faithfulness constant. We do this in the transductive setting, in which all points are available in advance. Contrary to our expectations, we observe that the choice of robustness metric can lead to highly divergent explanations, particularly in higher-dimensional settings. We trace this behavior to the use of metrics that evaluate the explanation set as a whole, rather than imposing constraints on individual points, and to how these \u201cglobal\u201d metrics interact with other optimization objectives. These interactions can allow the optimizer to produce locally inconsistent, unintuitive, and even undesirable explanations, despite satisfying the desired trade-offs. Our findings highlight the need for metrics whose mathematical structure more closely aligns with their intended use in optimization, and we advocate for future work that rigorously investigates metrics that incorporate a pointwise evaluation and their influence on the optimization landscape.",
        "topic": "Trust, Fairness & Explanation",
        "url": "https://openreview.net/forum?id=HUq8YbDpFt"
    },
    {
        "title": "Generalizing Trust: Weak-to-Strong Trustworthiness in Language Models",
        "id": 41,
        "authors": [
            "Lillian Sun",
            "Martin Pawelczyk",
            "Zhenting Qi",
            "Aounon Kumar",
            "Himabindu Lakkaraju"
        ],
        "keywords": [
            "Weak-to-strong generalization",
            "Trustworthiness",
            "Fairness",
            "Robustness",
            "Adversarial Robustness",
            "OOD Robustness",
            "Privacy"
        ],
        "abstract": "As large language models continue to advance, ensuring their trustworthiness is critical. However, inaccessible real-world ground truth labels pose a significant challenge in high-stakes domains. Recent studies have highlighted weak-to-strong generalization, where a strong model trained only on a weak model's labels surpasses the weak model in task performance. Yet, whether critical trustworthiness properties such as robustness, fairness, and privacy can generalize similarly remains an open question. This is the first work to study this question by examining if a stronger model can enhance trustworthiness when fine-tuned on a weaker model\u2019s labels, a paradigm we term weak-to-strong trustworthiness. To address this, we introduce two fundamental fine-tuning strategies that leverage trustworthiness regularization during the fine-tuning of the weak and weak-to-strong models. Our experimental evaluation on real-world datasets reveals that while some trustworthiness properties, such as fairness, adversarial, and OOD robustness, show significant improvement in trustworthiness generalization when both models were regularized, others like privacy do not exhibit signs of weak-to-strong trustworthiness. Our results highlight the potential of weak-to-strong trustworthiness as a practical pathway for enhancing the trustworthiness of increasingly capable AI systems, even under imperfect real-world conditions.",
        "topic": "Trust, Fairness & Explanation",
        "url": "https://openreview.net/forum?id=yrW1uzfiMP"
    },
    {
        "title": "Evaluating Generalization and Representation Stability in Small LMs via Prompting, Fine-Tuning and Out-of-Distribution Prompts",
        "id": 35,
        "authors": [
            "Rahul Raja",
            "Arpita Vats"
        ],
        "keywords": [
            "small language models",
            "few-shot prompting",
            "supervised fine-tuning",
            "model adaptation",
            "representation learning",
            "generalization",
            "low-resource NLP",
            "out-of-distribution robustness",
            "prompt engineering",
            "parameter efficiency",
            "transformer models",
            "task-specific tuning",
            "in-context learning",
            "experimental analysis",
            "model scaling"
        ],
        "abstract": "We investigate the generalization capabilities of small language models under two popular adaptation paradigms: few-shot prompting and supervised fine-tuning. While prompting is often favored for its parameter efficiency and flexibility, it remains unclear how robust this approach is in low-resource settings and under distributional shifts. This paper presents a comparative study of prompting and fine-tuning across task formats, prompt styles, and model scales, with a focus on their behavior in both in-distribution and out-of-distribution (OOD) settings.  Beyond accuracy, we analyze the internal representations learned by each approach to assess the stability and abstraction of task-specific features. Our findings highlight critical differences in how small models internalize and generalize knowledge under different adaptation strategies. This work offers practical guidance for model selection in low-data regimes and contributes empirical insight into the ongoing debate over prompting versus fine-tuning.",
        "topic": "Trust, Fairness & Explanation",
        "url": "https://openreview.net/forum?id=kTliXXhj5f"
    },
    {
        "title": "Is Visual Prompting the Right Setup for Knowledge Transfer in new Foundation Models?",
        "id": 12,
        "authors": [
            "Niclas Hergenr\u00f6ther",
            "Antonio Orvieto"
        ],
        "keywords": [
            "Visual Prompting",
            "VP",
            "Transfer Learning",
            "Adversarial Reprogramming"
        ],
        "abstract": "Visual Prompting (VP) has emerged as a promising technique for efficient knowledge transfer. As new foundation model families (like Mamba) get introduced and VP pipelines such as AutoVP reach greater maturity, we find a growing need for a systematic evaluation of current approaches. In this work, we assess the performance of the latest models, comparing them to earlier architectures and alternative fine-tuning methods, to better understand the progress, challenges and opportunities in the field of efficient fine-tuning under resource limitations. Towards this goal, this paper provides a concise empirical overview of the interactions among foundation model families (Attention-, Convolution-, and Mamba-based) and transfer paradigms: VP, Linear Probing (LP), and Full Finetuning (FFT). Our work builds up on previous findings by broadening the selection of evaluated models, tuning hyperparameters, and techniques. In the interest of delivering practical guidelines for the user, we also explore application of prevalent regularization techniques to boost performance in the context of VP.",
        "topic": "Trust, Fairness & Explanation",
        "url": "https://openreview.net/forum?id=iajFbllZKt"
    }
];

const topicSummaries = {
'Efficiency, Calibration & Robustness':'Methods that make models cheaper to serve or more reliable at test time. Topics include parameter/compute reduction (e.g., TinyServe, LiteByte), low‑rank & spectral compression with robustness, inference‑time scaling for diffusion, calibrated prediction intervals, and application‑level efficiency tricks.',
'Synthetic Benchmarks & Inductive Bias':'New toy datasets, algorithmic tasks, or analytical setups for probing what models really learn. Emphasises synthetic diagnostics such as SynDaCaTE, Stats‑vs‑Facts playgrounds, anchoring‑token pathfinding, polynomial decomposition tasks, permutation representations, and chaotic‑system perturbations.',
'Optimization Dynamics':'Theoretical & empirical analyses of how parameters evolve and why training sometimes stalls or ‘grocks’. Covers parity limits in SSMs, SGD‑to‑spectra SDE theory, gradient descent under flat/steep regions, SVD‑based grokking mitigation, early‑stopping re‑weighting, and error anticorrelation in cross‑validation.',
'Representation Dynamics':'Studies how internal representations form, simplify, or can be manipulated. Includes context‑length minimality, pruning‑induced directionality, Bayesian Occam’s razor in context, linear board probes for chess LMs, Koopman autoencoders, and generative vs. discriminative classifier behaviour.',
'Attention Mechanisms & Transformer Analysis':'Deep dives into attention behaviour, position bias, and chain‑of‑thought reasoning. Features Threshold Relative Attention for glitch avoidance, continuous CoT reasoning proofs, position‑bias emergence, measure‑flow transformers for GMMs, abrupt‑learning plateau analysis, and quantitative length‑generalization bounds.',
'Trust, Fairness & Explanation':'Ensuring models behave as intended: intervention fidelity when steering LLMs, weak‑to‑strong transfer of trustworthiness, pitfalls when turning explanation metrics into losses, robustness trade‑offs in prompt vs. fine‑tune adaptation, and critical evaluation of visual prompting.',
};

export default function PaperBrowser() {
    const [query, setQuery] = useState("");
    const [openAbstract, setOpenAbstract] = useState(null);
    const [activeKeyword, setActiveKeyword] = useState(null);
  
    const topics = [...new Set(papers.map(p => p.topic))];
    const keywords = [...new Set(papers.flatMap(p => p.keywords))].sort();
  
    const filtered = papers.filter(paper => {
      const matchesQuery = paper.title.toLowerCase().includes(query.toLowerCase()) ||
        paper.authors.some(a => a.toLowerCase().includes(query.toLowerCase())) ||
        paper.keywords.some(k => k.toLowerCase().includes(query.toLowerCase()));
      const matchesKeyword = activeKeyword ? paper.keywords.includes(activeKeyword) : true;
      return matchesQuery && matchesKeyword;
    });
  
    const resetFilters = () => {
      setQuery("");
      setActiveKeyword(null);
    };
  
    return (
      <div className="p-6 max-w-5xl mx-auto w-full">
        <h1 className="text-2xl font-bold mb-4">MOSS 2025 Accepted Papers</h1>
  
        <p className="text-gray-700 text-sm mb-4">
          This webpage presents all accepted papers at the MOSS 2025 Workshop. The papers are grouped into topical clusters for easier navigation. You can:
          <ul className="list-disc list-inside mt-2 space-y-1">
            <li>Click a topic name to jump to the relevant group of papers.</li>
            <li>Click a keyword to view all papers associated with it.</li>
            <li>Use the search bar to filter papers by title, author, or keyword.</li>
            <li>Click "Show Abstract" to expand the abstract for any paper.</li>
            <li>Click "Reset" to clear filters and return to the full list.</li>
          </ul>
        </p>
  
        <div className="flex items-center gap-4 mb-6">
          <Input
            placeholder="Search by title, author, keyword..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button
            onClick={resetFilters}
            className="bg-gray-200 hover:bg-gray-300 text-sm px-4 py-2 rounded"
          >
            Reset
          </button>
        </div>
  
        <div className="mb-6">
          <h2 className="text-lg font-semibold mb-2">Topics</h2>
          <ul className="space-y-2 ml-2">
            {topics.map(topic => (
              <li key={topic} className="flex items-start gap-2">
                <span className="mt-1 w-2 h-2 rounded-full bg-black"></span>
                <div>
                  <a href={`#${topic.replace(/\s+/g, "-")}`} className="text-blue-600 font-medium hover:underline">
                    {topic}
                  </a>
                  <div className="text-sm text-gray-600">{topicSummaries[topic]}</div>
                </div>
              </li>
            ))}
          </ul>
        </div>
  
        <div className="mb-8">
          <h2 className="text-lg font-semibold mb-2">Keywords</h2>
          <div className="flex flex-wrap gap-3 text-sm text-gray-700 mb-2">
            {keywords.map(keyword => (
              <button
                key={keyword}
                onClick={() => setActiveKeyword(current => current === keyword ? null : keyword)}
                className={`border px-2 py-1 rounded ${activeKeyword === keyword ? "bg-blue-200" : "bg-gray-100"}`}
              >
                {keyword}
              </button>
            ))}
          </div>
          <button
            onClick={resetFilters}
            className="bg-gray-200 hover:bg-gray-300 text-sm px-3 py-1 rounded"
          >
            Reset Filters
          </button>
        </div>
  
        {topics.map(topic => {
          const topicPapers = filtered.filter(p => p.topic === topic);
          if (topicPapers.length === 0) return null;
  
          return (
            <div key={topic} id={topic.replace(/\s+/g, "-")} className="mb-8">
              <h2 className="text-xl font-semibold mb-3">{topic}</h2>
              <p className="text-sm text-gray-600 mb-4">{topicSummaries[topic]}</p>
              <ul className="list-disc list-inside ml-6 space-y-4">
                {topicPapers.map(paper => (
                  <li key={paper.id} id={`paper-${paper.id}`}>
                    <Card>
                      <div className="flex flex-col gap-2">
                        <div className="flex justify-between items-start">
                          <div className="w-full">
                            <a
                              href={paper.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-lg font-semibold text-blue-600 hover:underline"
                            >
                              {paper.title}
                            </a>
                            <p className="text-sm text-gray-700">{paper.authors.join(", ")}</p>
                            <p className="text-sm text-gray-600 italic">{paper.tldr}</p>
                            <p className="text-xs text-gray-500 flex flex-wrap gap-1">
                              Keywords: {paper.keywords.map((k, i) => (
                                <span key={i}>{k}{i < paper.keywords.length - 1 ? "," : ""}</span>
                              ))}
                            </p>
                            <a
                              href={`https://github.com/abhishekpanigrahi1996/MOSS/tree/main/submissions/submission-${paper.id}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-sm text-blue-500 hover:underline"
                            >
                              Code link
                            </a>
                          </div>
                          <div className="ml-4">
                            <button
                              onClick={() => setOpenAbstract(openAbstract === paper.id ? null : paper.id)}
                              className="text-sm text-blue-500 hover:underline whitespace-nowrap"
                            >
                              {openAbstract === paper.id ? "Hide Abstract" : "Show Abstract"}
                            </button>
                          </div>
                        </div>
                        {openAbstract === paper.id && (
                          <p className="text-sm text-gray-800 border-t pt-2">{paper.abstract}</p>
                        )}
                      </div>
                    </Card>
                  </li>
                ))}
              </ul>
            </div>
          );
        })}
      </div>
    );
  }