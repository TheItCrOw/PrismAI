'''
We use the main.ipynb to experiment and build the collection and through this main.py script,
we orchestrate the collection and synthesization in parallel by firstly writing the notebook to 
a script file with as much instances as we like, and then starting each script as its own 
process from here to log to a destined file. That way, we can collect many sources in parallel.
'''
import subprocess
import os

notebook_path = 'main.ipynb'
output_dir = 'processes'
log_dir = 'processes/logs'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

collectors = [
    'collectors.bundestag_collector.BundestagCollector',
    'collectors.house_of_commons_collector.HouseOfCommonsCollector',
    'collectors.student_essays_collector.StudentEssaysCollector',
    'collectors.arxiv_collector.ArxivCollector',
    'collectors.spiegel_collector.SpiegelCollector',
    'collectors.cnn_news_collector.CNNNewsCollector',
    #'collectors.open_legal_data_collector.OpenLegalDataCollector',
    #'collectors.euro_court_cases_collector.EuroCourtCasesCollector',
    #'collectors.religion_collector.ReligionCollector',
    #'collectors.gutenberg_collector.GutenbergCollector',
    #'collectors.blog_corpus_collector.BlogCorpusCollector'
]

if __name__ == '__main__':
    print('============ Orchestrator started. ============')

    # Foreach collector, we create its own collection instance.
    # The stepts are always the same foreach collector
    for collector in collectors:

        out = f'process_{collector.split('.')[len(collector.split('.')) - 1]}'

        # Step 1: Convert the Jupyter notebook to a Python script
        convert_command = [
            'jupyter', 'nbconvert', '--to', 'script',
            notebook_path,
            '--output-dir', output_dir,
            '--output', out
        ]
        print('Converting notebook to script...')
        subprocess.run(convert_command, check=True)
        print('Notebook converted successfully.')

        # Step 2: Run the generated script in the background
        script_path = os.path.join(output_dir, f'{out}.py')
        log_path = os.path.join(log_dir, f'{out}.log')

        print(f'Running {script_path} in the background...')
        with open(log_path, "w") as log_file:
            subprocess.Popen(
                ["python", '-u', script_path, '--collectors', collector],
                stdout=log_file,
                stderr=log_file,
                start_new_session=True  # set the process detached
            )

        print(f"Script is running in the background. Logs are being written to {log_path}.")
        print('\n=========================================================================\n')