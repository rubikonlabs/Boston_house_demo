import subprocess

if __name__ == "__main__":

    subprocess.call(['mlflow', 'server', '--backend-store-uri', './mlruns', '--default-artifact-root', './mlruns', '--host', '0.0.0.0', '--port', '5000'], shell=False)




# mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0 --port 5000
