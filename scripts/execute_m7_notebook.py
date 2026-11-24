"""Execute every M7 cell in a fresh .venv process, with rich saved outputs.

Uses an in-process IPython shell because sandbox policy blocks Jupyter's local
sockets. No stale notebook namespace is reused. This is not a Jupyter kernel
integration test; execution method is recorded in notebook metadata.
"""
from pathlib import Path
import os
import sys
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
path=ROOT/'source/experiments/06_comparative_experiments.ipynb'
notebook=nbformat.read(path,as_version=4)
shell=InteractiveShell.instance()
for cell in notebook.cells:
    if cell.cell_type!='code':
        continue
    cell.outputs=[]
    with capture_output(stdout=True,stderr=True,display=True) as captured:
        result=shell.run_cell(cell.source,store_history=True)
    cell.execution_count=result.execution_count
    if captured.stdout: cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
    if captured.stderr: cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
    for output in captured.outputs:
        cell.outputs.append(nbformat.v4.new_output('display_data',data=output.data,metadata=output.metadata))
    error=result.error_before_exec or result.error_in_exec
    if error:
        cell.outputs.append(nbformat.v4.new_output('error',ename=type(error).__name__,evalue=str(error),traceback=[]))
        nbformat.write(notebook,path)
        raise RuntimeError(f'Cell {cell.execution_count} failed') from error
notebook.metadata['m7_execution']={'method':'fresh-process IPython InteractiveShell, sequential all code cells',
                                 'reason':'Sandbox denies socket creation (PermissionError errno1)',
                                 'python_version':sys.version.split()[0],
                                 'environment':'.venv', 'jupyter_kernel_transport_verified':False}
nbformat.write(notebook,path)
print(f'Executed {sum(c.cell_type=="code" for c in notebook.cells)} cells successfully with saved rich outputs.')
