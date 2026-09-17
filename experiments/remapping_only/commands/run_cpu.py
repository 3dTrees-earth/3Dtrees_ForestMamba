"""CPU-only analysis in an existing local environment with explicit resource caps."""
import os,resource,sys
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:10])
resource.setrlimit(resource.RLIMIT_AS,(50000000000,50000000000))
os.environ.update(OMP_NUM_THREADS='10',OPENBLAS_NUM_THREADS='10',MKL_NUM_THREADS='10',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.path.dirname(os.path.abspath(__file__)))
python=os.environ.get('FM_CPU_PYTHON',sys.executable)
os.execv(python,[python,*sys.argv[1:]])
