
# AI-Powered Analysis of ORCID and Research Link Australia Data for Technology Landscaping


## Abstract

ORCID profiles provide a substantial amount of information about research communities. ORCID is considered a key data source for providing valuable and open-access data about researchers and their research works. However, ORCID also has potential to inform Technology Landscaping regarding researchers and their associated research networks, with potential applications including identifying technical expertise, emerging research communities, and investment opportunities. 

This presentation features a case study leveraging over 100,000 Australian ORCID researcher profiles and their research outcome collected from ARDC Research Link platform. We demonstrate how we have collected, connected, and analysed this dataset to observe the time-series evolution of the Australian research communities. The resulting animation will illustrate five years of historical developments within these communities. This will also establish a robust foundation for projecting future technological trends in Australia. 

Aligning with the technical aspect of this presentation, we demonstrate an AI pipeline using LLM model to analyse our data and achieve the visualisation results. This makes the solution presented in our work accessible and interoperable with many research infrastructure solutions in universities. We will discuss the technical challenges and opportunities observed while developing and deploying this solution in Nectar Cloud. We believe the lessons learned from our work can help identify potential new use cases for application of AI pipelines to analyse ORCID and other open scholarly data. 

## Methodology

In this work, we devise a pipeline that automatically create taxonomy that can detect emerging technologies from the first principle. 

### Keywords Extraction 

The keyword extraction pipeline leverages a large language model (LLM) to systematically identify and extract research-relevant keywords from grant information. This automated approach processes grant titles and descriptions to discover technical terms, methodologies, technologies, and emerging research concepts that characterize the innovation landscape.

The extraction system employs an asynchronous pipeline built on the OpenAI API interface, enabling concurrent processing of multiple grants to maximize throughput. The architecture implements semaphore-based concurrency control to manage API rate limits while maintaining efficient parallel processing. Grant records are streamed through the pipeline with checkpoint recovery support, allowing the system to resume processing after interruptions without re-extracting keywords from previously processed grants.

#### Prompt Engineering for Precision

The extraction prompt instructs the LLM to function as an expert research analyst with multi-disciplinary knowledge. The system prompt enforces several critical constraints to ensure keyword quality:

**Text Grounding Requirement**: All extracted keywords must be directly present in or clearly derivable from the grant text. This prevents hallucination and ensures keywords accurately represent the stated research focus rather than inferred concepts.

**Specificity Constraints**: The prompt explicitly prohibits overly broad or generic terms. Instead of accepting general keywords like "engineering" or "artificial intelligence," the system demands precise technical terminology such as "bio-integrated nano-photonics" or "graph neural networks." This specificity requirement ensures keywords can effectively distinguish between research projects.

**Uniqueness Test**: Keywords must pass a uniqueness filter—if a term could reasonably appear in 80% or more of grants, it is rejected. This eliminates administrative terms, funding-related language, and generic methodological phrases that provide no discriminative value for technology landscaping.

**Length Limitation**: Keywords are constrained to 1-4 words to maintain conciseness and prevent extraction of lengthy descriptive phrases. This encourages selection of established technical terminology over verbose explanations.

#### Structured Output and Validation

The LLM returns keywords in a structured JSON format validated against a Pydantic schema (`KeywordsList`). Each keyword includes a name, type classification (General, Methodology, Application, or Technology), and a brief description explaining its relevance to the research. This structured approach ensures consistent output format and enables programmatic validation of extraction quality.

Extracted keywords undergo post-processing to normalize variants and eliminate duplicates. The pipeline uses NLTK's Porter Stemmer to identify morphological variants (e.g., "optimization" and "optimisation") and consolidates them by selecting the variant with the longest description. This normalization step produces a deduplicated keyword set while preserving the most informative descriptions from the extraction process.

The pipeline supports large-scale processing through incremental output to JSONL files and automatic checkpoint recovery. Each successfully extracted grant is immediately written to disk, allowing the system to resume from the last processed grant in case of interruption. This design enables processing of datasets containing tens of thousands of grants without risk of data loss from mid-pipeline failures.



