# GenAI Usage for this project

**Tool used**: Perplexity research mode (combination of different unspecified models)

### Initial prompt 

I am a 2nd year master's student taking a course on complex systems simulations. 
My master's is in AI, so my programming knowledge is good but my math/physics background is less experienced (but still proficient enough to get through an AI master's degree).
My project team and I are building a model that simulates the growth of corals
based on the following two papers: Llabrés et al. 2024 & Halsey 2000 (uploaded documents).
We are building a DLA with the following extensions:
- Seed grows from the "ground"
- No "downward" growth i.e. we modify Moore's neighbourhood from 8 to 5 valid neighborhood directions on a square lattice: left, top left, top, top right, right
- Growth parameters (we want to translate the Llabrés growth parameters for the DLA)

Your role: 
- Help me understand the intuition behind concepts relevant to this project
- Help me develop clean, reproducible, and readable code by providing pseudocode suggestions
- Act as a learning amplifier, not a solution provider
- Ask follow-up questions that help refine your understanding if something is unclear, vague, or ambiguous
- Explain jargon if used, provide plain language explanations as much as possible without compromising "correctness"
- Be concise. Do not be sycophantic or ingratiating
- Wait for my questions. I don't need anything from you yet.

### Ideation

Ideation (research questions, hypothesis, plan of action): Done as a team. No GenAI usage.

### Understanding 
#### Prompting strategy:
- As I read through the paper, ask for clarification about concepts I don't understand
- Ask for intuition / questions that help me reason
- Iteratively refine answer by identifying mistakes made by the tool

#### Examples:

Llabrés et al. 2024
- **Prompt1**: My understanding of the 5 parameters is as follows: growth mode = growth direction (but I'm confused how the parameter translates to which direction). 
elongation rate = how fast the coral grows? subdivision distance = how big the mesh triangles can get before a new polyp is made, 
interbranching angle = angle of the branches (if any), and I'm confused about what inter-branching length is.
- **Answer1**:  "Plain language" explanation, ~500 words (100/parameter)
- **Result**: This helped me re-read the paper with better context/intuition for the equations, and from here I was able to reason that
about how each behaviour would be expressed on a 2D grid with pixels. From here I decided on the growth_mode & friendliness parameters (without GenAI)

Halsey 2000. 
- **Prompt1**: I'm having trouble understanding the multifractality concept in the DLA paper. What is the growth probability, and how can I interpret the sigma function? I also don't fully understand where q comes from.
- **Answer1**: "Plain language" explanation + follow-up questions, ~500 words.
- **Result**: With better context for the mathematical concepts described in the paper (and after looking up DLA animations on YouTube) I then reread the multifractality section a few more times and actually understood it.


- **Prompt2**: Concisely, help me understand the first scaling relation, and its relevance to multifractality. Ask me questions, and provide examples to help guide my understanding.
- **Answer2**: "Plain language" explanation, ~300 words.
- **Result**: I went back to the paper and tried to connect the intuition to what I was reading. 
I actually realized that explanation/equations provided were VERY wrong/unintuitive, so I went back and shared a screenshot instead of a pdf of the equations.
Turns out it had read the equations incorrectly. The new explanation made so much more sense.


- **Prompt3**: I repeated prompt2 (with the correct equations) for scaling relations 2 & 3, paying close attention to whether my understanding of the paper matches the intuition being conveyed
- **Result**: This really helped me understand the scaling relations, their notation, and understood that part of my experiment will be to plot sigma(q) and observe the behaviour around q=1, at q=3, and as q -> inf.

- **Prompt4**: Explain what capacitance is in plain language & its relevance in this context, and provide some sources for further reading. 
- **Result**: This wasn't really relevant for advancing the project, but I wanted to understand. This helped.

### Coding
#### Prompting strategy
- Provide my own code, pseudocode, or plan of action to help constrain the solution
- Iteratively ask for feedback / updated solutions based on whether something worked or not

#### Task: Parameters
- **Prompt1** I've decided that I want to implement two parameters in my DLA that will behave like the 5 Llabrés parameters.  
Here is my baseline dla: (code block of baseline dla model)
Let's start with the first parameter: I want to define horizontal or vertical growth preference. 
Basically, I want to modify the attachment probability within the valid neighborhood of a cell chosen by a random walker.
New cells cannot attach to sites that already exist in the cluster, and they cannot grow downwards. 
To implement this, I want to interpolate between the following keyframes:
```
horizontal   = np.array([1.0, 0.0, 0.0, 0.0, 1.0])  # growth_mode= -1.0 -> horizontal only (left/right)
diagonal_sides = np.array([1.0, 1.0, 0.0, 1.0, 1.0])  # growth_mode= -0.5 -> sides + top-diagonals equal (no top)
uniform  = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # growth_mode= 0.0 -> all 5 allowed directions are equal (no bottom)
diagonal_top   = np.array([0.0, 1.0, 1.0, 1.0, 0.0])  # growth_mode= 0.5 -> top + top-diagonals equal (no sides)
vertical  = np.array([0.0, 0.0, 1.0, 0.0, 0.0]) 
 ````
This is my solution for interpolating between the frames, but I don't think it's working correctly. Help me understand what I'm doing wrong:
````
probabilities = (1 - growth_mode) * weights_diagonal_sides + growth_mode * weights_uniform
````
The result should be between [1.0, 0.0, 0.0, 0.0, 1.0] and [1.0, 1.0, 0.0, 1.0, 1.0] given a value of -1 to -0.5, but that's not what's happening.
- **Answer**: ~10 lines of pseudocode & explanation about the interpolation factor.
- **Result**:  After chatting a bit longer about what this interpolation factor actually is doing, I finally understood what I wanted.
Then, using this knowledge I was able to determine the correct interpolation factor (kind of like a binomial distribution??) and was able to apply the same logic to defining the friendliness parameter
Asked follow up questions like: "why use min max scaling instead of normalizing by dividing by total sum? what's the difference?" 

- **Prompt**: (I tried asking a separate unrelated chat how it might go about implementing these two parameters to compare 
whether I was on the right track, and it gave me absolute garbage. It wanted to use like angles and stuff, in a square grid lattice!! I told it "that's literally insane" and then it gave me attitude. I killed the chat.)

#### Task: Multifractality
- **Prompt1**: I am trying to compute the growth probability of my coral. Here is my intuition: 
As the coral grows, sum the number of times a site has been hit divided by the total number of walks that have been attempted until that point. 
Here is my code block (insert code). I don't think I'm calculating this correctly because my probabilities are all uniform??? I am expecting there to be 
some sites that have higher probability than others, but that's not what I'm seeing
- **Answer1**: Pseudocode / codeblock fragment, ~50 lines, plain language explanation (~400 words)
- **Result**: Helped me identify that I was taking _entirely_ the wrong approach to calculating the growth probability and I had totally misunderstood it.
Asked follow-up questions like "ahhhh ok so once I have my final cluster, /then/ I calculate pi?"

#### Task: Parallelization for computing 
- **Prompt1**: My compute_growth_probability is taking way too long to run. Each walker is independent of the other, so I can parallelize this process.
I want to us the multiprocessing tool to parallelize the process. 
- **Answer1**: Pseudocode block + code fragments, ~50 lines
- **Result**: Used my knowledge of parallelization from a previous course to decide what to keep. Asked follow-up questions to understand why I needed a new seed for each parallel process for the random number generator
- **Prompt2**: Help, when I run the compute_growth_probability_parallelized function, it's eating up my memory!! Help me understand why!
- **Answer2**: Gave me unuseful garbage. The answer actually came to me as I was trying to fall asleep: I was recording the path for every single random walker which was about 80,000 steps, and I had about 100,000 
random walkers... So I decided to record just the first walk so I could plot an example of what it looks like (I was curious to see how far it was going to understand why the code was taking so long)
and then my memory issues were fixed.

#### Task: Bugs
- Ask for help interpreting error messages

#### Task: Visualization
- **Prompt**: I want to plot two heat maps in one, where each square is split into two. This is my single heatmap code block (pasted code). 
I want the bottom-left triange to be the expected value, and the top right hand triangle to be the experimental value. My data is two 2D numpy arrays. 
I want the respective values to be displayed in each triangle. Add a legend to the bottom explaning which side of the triangle refers to which data source.
Here is an example: https://stackoverflow.com/questions/63530701/python-package-to-plot-two-heatmaps-in-one-split-each-square-into-two-triangles
- **Answer**:  Multi-heatmap codeblock that was mostly wrong (wrong triangle directions, ugly formatting, wrong naming, values placed in the wrong area of the triangle)
Provided a base that allowed me to iterate towards what I actually wanted

- **Prompt**: Minor visualization tips I either couldn't find on StackOverflow or needed help understanding further, like plotting the tangent of a set of datapoints at a particular x value. 

### Documentation
- (In cursor) Gave explicit instructions to add type hints to function definitions. Manually checked for correctness
- Doc strings were added manually

# GenAI Usage for this project

**Tool used**: ChatGPT (GPT-5.2 Thinking)

### Initial prompt

I am a master’s student in Computational Science working on a coral-growth project based on DLA.
I already implemented a DLA model in a previous course assignment, so I am comfortable coding the core loop (seed → walkers → random walk → sticking). I have a math and physics background.
However, last year I simulated in a finite box with boundary conditions, and for this project I want a setup that behaves more like growth in an effectively unbounded plane.
Your role:
- Help me sanity-check algorithmic choices (boundary conditions vs kill radius + reinjection)
- Explain performance tricks in plain language (frontier set, kill radius)
- Help me interpret errors when I paste a traceback
- Keep explanations practical and tied to implementation

### Ideation

Ideation (research questions, hypothesis, plan of action): Done as a team. No GenAI usage.

---

### Understanding
#### Prompting strategy:
- Use GenAI to sanity-check implementation choices I was already considering
- Ask for trade-offs and failure modes (what can bias morphology, what changes runtime)
- Keep answers conceptual (plain language + pseudocode-level detail), then validate by running locally

#### Example: baseline DLA algorithm + why we use a frontier set and a kill radius

**Context**:
I had already implemented a DLA model in a previous course assignment, so I was comfortable coding the core loop (seed → walkers → random walk → sticking). However, last year I simulated in a finite box with boundary conditions, and for this project I wanted a setup that behaves more like growth in an effectively unbounded plane. I used GenAI to sanity-check the algorithmic choices and to clarify trade-offs.

- **Prompt1**: “I implemented DLA before in a finite box with boundary conditions. For this project, we want something closer to unbounded growth. Should we still use boundary conditions, or is a kill radius + reinjection better? Explain the trade-offs.”
- **Output1**: Short conceptual explanation comparing finite-domain simulation (with reflecting/periodic/absorbing boundaries) vs kill radius + reinjection, including likely sources of bias and efficiency implications (~300–500 words).
- **Result1**: I adopted the reasoning that boundary conditions can bias morphology unless the domain is very large (reflecting walls can cause artificial wall-sticking, periodic boundaries introduce wrap-around effects, absorbing boundaries change effective flux). I therefore kept a kill radius + reinjection approach to approximate unbounded growth while keeping computation focused near the aggregate.

- **Prompt2**: “Explain the frontier set trick and why it makes the simulation faster than checking neighbors every step.”
- **Output2**: Plain-language explanation + pseudocode-level description (~150–300 words + short code fragment).
- **Result2**: I adopted the frontier idea: maintain the set of empty sites adjacent to the cluster so sticking becomes a fast membership check, and update the frontier locally after attachment. I used this to justify the optimization choice in documentation.

- **Prompt3**: “What kinds of biases can reflecting / periodic / absorbing boundaries introduce in DLA?”
- **Output3**: Short list of biases and when they matter (~150–250 words).
- **Result3**: Helped me write a plausible justification for why we avoided finite-box boundary conditions for the baseline, and why kill radius + reinjection was the better match for our goal.

#### Observations / limitations
- GenAI was useful for summarizing trade-offs and giving me a clean explanation to reuse in writing.
- I still validated decisions by running simulations locally and checking that behaviour matched expected DLA morphology.

