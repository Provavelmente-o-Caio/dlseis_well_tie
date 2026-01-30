"""
Wrapper para executar script Python com ambiente isolado das DLLs do Petrel
"""
import os
import sys
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Uso: python run_isolated.py <script.py> [args...]")
        sys.exit(1)
    
    script_path = sys.argv[1]
    script_args = sys.argv[2:]
    
    # Cria um ambiente completamente novo
    env = {}
    
    # Copia apenas variáveis essenciais do sistema
    essential_vars = [
        'SystemRoot', 'TEMP', 'TMP', 'APPDATA', 'LOCALAPPDATA',
        'USERPROFILE', 'USERNAME', 'COMPUTERNAME', 'HOMEDRIVE', 
        'HOMEPATH', 'COMSPEC', 'PROGRAMFILES', 'PROGRAMFILES(X86)'
    ]
    
    for var in essential_vars:
        value = os.environ.get(var)
        if value:
            env[var] = value
    
    # Copia variáveis do Conda
    conda_vars = [
        'CONDA_PREFIX', 'CONDA_DEFAULT_ENV', 'CONDA_PYTHON_EXE',
        'CONDA_SHLVL', 'CONDA_PROMPT_MODIFIER'
    ]
    
    for var in conda_vars:
        value = os.environ.get(var)
        if value:
            env[var] = value
    
    # Constrói PATH com DLLs do Conda PRIMEIRO
    conda_prefix = os.environ.get('CONDA_PREFIX', r'D:\anaconda3\envs\wtie')
    
    path_parts = [
        os.path.join(conda_prefix, 'Library', 'bin'),
        os.path.join(conda_prefix, 'Library', 'usr', 'bin'),
        os.path.join(conda_prefix, 'Library', 'mingw-w64', 'bin'),
        os.path.join(conda_prefix, 'Scripts'),
        os.path.join(conda_prefix, 'bin'),
        os.path.join(conda_prefix),
    ]
    
    # Adiciona apenas System32 (SEM outros caminhos do Windows que possam ter DLLs do Petrel)
    system_root = env.get('SystemRoot', r'C:\Windows')
    path_parts.extend([
        os.path.join(system_root, 'System32'),
        system_root,
        os.path.join(system_root, 'System32', 'Wbem')
    ])
    
    env['PATH'] = ';'.join(path_parts)
    
    # Força Conda a usar modificação de busca de DLL
    env['CONDA_DLL_SEARCH_MODIFICATION_ENABLE'] = '1'
    
    # Remove variáveis Python que podem causar conflito
    for var in ['PYTHONPATH', 'PYTHONHOME']:
        env.pop(var, None)
    
    print("=" * 60)
    print("EXECUTANDO EM AMBIENTE ISOLADO")
    print("=" * 60)
    print(f"Script: {script_path}")
    print(f"Python: {sys.executable}")
    print(f"PATH (primeiros 300 chars): {env['PATH'][:300]}")
    print("=" * 60)
    print()
    
    # Executa o script em um subprocesso completamente isolado
    try:
        result = subprocess.run(
            [sys.executable, script_path] + script_args,
            env=env,
            cwd=os.path.dirname(script_path) or '.'
        )
        sys.exit(result.returncode)
    except Exception as e:
        print(f"ERRO ao executar script: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()