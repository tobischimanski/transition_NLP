# Using AI to assess corporate climate transition disclosures

This repository contains code and data for the project "Using AI to assess corporate climate transition disclosures". This is the implementation of the Retrieval Augmented Generation (RAG) system for the project.

## Simple Usage

> [!IMPORTANT]
> If you use our code, please make sure to reference by citing the paper (Colesanti Senni et. al., 2024). If you want to use the code for commercial purposes, please get in touch with Markus Leippold (markus.leippold@df.uzh.ch).

For a detailed walk-through, follow [this tutorial](https://medium.com/@schimanski.tobi/ai-for-sustainability-1-a-tool-for-analyzing-company-transition-plans-7d75853f933b?source=friends_link&sk=d7c5aaf0af36d4618d26fdc1c34abf01).

In order to use this, first clone the repository.
```python
!git clone https://github.com/tobischimanski/transition_NLP.git
# move to folder and show content
%cd transition_NLP/
%ls
```
As a second step, we install the requirements.
```python
!pip install -r requirements.txt
```

Now, we are ready to go. Simply follow the usage pattern and you can analyze your report.
```python
# Usage pattern: python transition_analysis.py api_key report model [num indicators]
# Example
!python transition_analysis.py sk-... ./Test_Data/CSR_IP_2022.pdf gpt-3.5-turbo-1106 4
```
Usage pattern explanation:
- api_key: Create and use your own OPENAI API KEY. You can follow [this tutorial](https://www.merge.dev/blog/chatgpt-api-key).
- report: This is the path of the report that you want to be analyzed.
- model: Specify the (openai) model, you want to use. For an overview of the models, [see here](https://platform.openai.com/docs/models/overview).
- num indicators [optional]: To get a feeling for the tool, you can also specify that you only want a subset of the indicators to be created. Put any positive number here.

If you follow these steps, there will be an Excel file in the folder "Excel_Output" that carries the same name as your report. You can see that the one report in "Test_Data" was already analyzed with different specifications.

## License
This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License - see the [LICENSE](LICENSE) file for details.

## Read and Cite the Paper

Find the [published paper here](https://iopscience.iop.org/article/10.1088/2515-7620/ad9e88/meta).

```
@article{10.1088/2515-7620/ad9e88,
	author={Colesanti Senni, Chiara and Schimanski, Tobias and Bingler, Julia and Ni, Jingwei and Leippold, Markus},
	title={Using AI to assess corporate climate transition disclosures},
	journal={Environmental Research Communications},
	url={http://iopscience.iop.org/article/10.1088/2515-7620/ad9e88},
	year={2024}
}
```
