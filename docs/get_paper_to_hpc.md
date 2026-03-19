First run `paper_getting.ipynb` (can do locally, using the local pdfs you already have available). This generates `papers_with_names.csv`, which you need to move to the HPC, and give the path to `$PAPERS_CSV` in `function_extraction_from_paper.sh`. 


Then, open Terminal, to move all downloaded papers to hpc like: 

```scp local_folder_with_papers\* hpc_folder```

e.g. 

```scp .\projects\hands-on-llms\Notebooks\downloaded_papers\* yyin@hex:/cephfs2/yyin/llm_circuit_analysis/pdfs```

The `hpc_folder` is the one for `$PAPERS_DIR` in `function_extraction_from_paper.sh`. 