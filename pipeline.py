#!/usr/bin/env python3
"""
pipeline.py
End-to-end ML pipeline orchestrator

Runs the complete MLOps workflow:
1. Data preparation
2. Model training
3. Model evaluation
4. Drift monitoring  
5. API deployment
6. Dashboard generation
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_step(step_num, total, message):
    """Print formatted step header"""
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}Step {step_num}/{total}: {message}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")

def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"{Colors.OKCYAN}▶ {description}...{Colors.ENDC}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"{Colors.OKGREEN}✅ {description} completed{Colors.ENDC}")
        if result.stdout:
            print(result.stdout)
        return True
    else:
        print(f"{Colors.FAIL}❌ {description} failed{Colors.ENDC}")
        print(result.stderr)
        return False

def main():
    """Run the complete ML pipeline"""
    
    print(f"{Colors.HEADER}")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║         MLOps Pipeline - Real Estate Price Engine             ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")
    
    start_time = time.time()
    pipeline_results = {}
    
    # Configuration
    BASE_DIR = Path(__file__).parent
    VENV_PYTHON = BASE_DIR / ".venv" / "bin" / "python"
    
    # Check if venv exists
    if not VENV_PYTHON.exists():
        print(f"{Colors.FAIL}❌ Virtual environment not found at {VENV_PYTHON}{Colors.ENDC}")
        print(f"{Colors.WARNING}Run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt{Colors.ENDC}")
        sys.exit(1)
    
    TOTAL_STEPS = 7
    current_step = 0
    
    # Step 1: Data Preparation
    current_step += 1
    print_step(current_step, TOTAL_STEPS, "Data Preparation")
    success = run_command(
        f"{VENV_PYTHON} src/data_prep.py",
        "Processing real estate data"
    )
    pipeline_results['data_prep'] = {'success': success}
    
    if not success:
        print(f"{Colors.WARNING}⚠️ Using existing processed data{Colors.ENDC}")
    
    # Step 2: Model Training
    current_step += 1
    print_step(current_step, TOTAL_STEPS, "Model Training")
    success = run_command(
        f"{VENV_PYTHON} src/model_training.py",
        "Training Ridge and XGBoost models"
    )
    pipeline_results['training'] = {'success': success}
    
    if not success:
        print(f"{Colors.FAIL}❌ Training failed, aborting pipeline{Colors.ENDC}")
        sys.exit(1)
    
    # Load training metrics
    try:
        with open('models/training_summary.json', 'r') as f:
            training_summary = json.load(f)
            best_model = training_summary['best_model']
            print(f"{Colors.OKGREEN}✨ Best Model: {best_model['tag']}{Colors.ENDC}")
            print(f"   MAE: ¥{best_model['test_metrics']['MAE']/1e6:.2f}M")
            print(f"   RMSE: ¥{best_model['test_metrics']['RMSE']/1e6:.2f}M")
            print(f"   MAPE: {best_model['test_metrics']['MAPE']:.2f}%")
            pipeline_results['best_model'] = best_model
    except Exception as e:
        print(f"{Colors.WARNING}⚠️ Could not load training summary: {e}{Colors.ENDC}")
    
    # Step 3: Model Evaluation
    current_step += 1
    print_step(current_step, TOTAL_STEPS, "Model Evaluation")
    
    MAE_THRESHOLD = 20_000_000  # ¥20M
    MAPE_THRESHOLD = 50  # 50%
    
    if 'best_model' in pipeline_results:
        mae = best_model['test_metrics']['MAE']
        mape = best_model['test_metrics']['MAPE']
        
        if mae < MAE_THRESHOLD and mape < MAPE_THRESHOLD:
            print(f"{Colors.OKGREEN}✅ Model meets quality thresholds{Colors.ENDC}")
            pipeline_results['evaluation'] = {'passed': True}
        else:
            print(f"{Colors.WARNING}⚠️ Model exceeds quality thresholds{Colors.ENDC}")
            pipeline_results['evaluation'] = {'passed': False}
    
    # Step 4: Drift Monitoring
    current_step += 1
    print_step(current_step, TOTAL_STEPS, "Drift Monitoring")
    success = run_command(
        f"{VENV_PYTHON} src/monitor_drift.py",
        "Checking for data drift"
    )
    pipeline_results['drift_monitoring'] = {'success': success}
    
    # Step 5: Generate Reports
    current_step += 1
    print_step(current_step, TOTAL_STEPS, "Report Generation")
    success = run_command(
        f"{VENV_PYTHON} src/report_generate.py",
        "Creating HTML dashboard"
    )
    pipeline_results['reports'] = {'success': success}
    
    if success:
        print(f"{Colors.OKGREEN}📊 Report: reports/price_engine_report.html{Colors.ENDC}")
    
    # Step 6: Start API Server
    current_step += 1
    print_step(current_step, TOTAL_STEPS, "API Deployment")
    
    # Check if API is already running
    api_check = subprocess.run("lsof -i :8000", shell=True, capture_output=True)
    if api_check.returncode == 0:
        print(f"{Colors.WARNING}⚠️ API already running on port 8000{Colors.ENDC}")
    else:
        print(f"{Colors.OKCYAN}▶ Starting API server...{Colors.ENDC}")
        subprocess.Popen(
            f"{VENV_PYTHON} -m uvicorn src.api_server_grafana:app --host 0.0.0.0 --port 8000 > /tmp/ml_api.log 2>&1 &",
            shell=True
        )
        time.sleep(3)
        
        # Test API
        api_test = subprocess.run("curl -s http://localhost:8000/health", shell=True, capture_output=True)
        if api_test.returncode == 0:
            print(f"{Colors.OKGREEN}✅ API server started on port 8000{Colors.ENDC}")
            print(f"{Colors.OKGREEN}   Health: {api_test.stdout.decode()[:50]}...{Colors.ENDC}")
            print(f"{Colors.OKGREEN}   Metrics: http://localhost:8000/metrics/{Colors.ENDC}")
            pipeline_results['api'] = {'success': True, 'port': 8000}
        else:
            print(f"{Colors.FAIL}❌ API failed to start{Colors.ENDC}")
            pipeline_results['api'] = {'success': False}
    
    # Step 7: Start Streamlit Dashboard
    current_step += 1
    print_step(current_step, TOTAL_STEPS, "Dashboard Deployment")
    
    # Check if Streamlit is already running
    streamlit_check = subprocess.run("lsof -i :8501", shell=True, capture_output=True)
    if streamlit_check.returncode == 0:
        print(f"{Colors.WARNING}⚠️ Streamlit already running on port 8501{Colors.ENDC}")
    else:
        print(f"{Colors.OKCYAN}▶ Starting Streamlit dashboard...{Colors.ENDC}")
        subprocess.Popen(
            f"{VENV_PYTHON} -m streamlit run src/dashboard_pro.py --server.headless true --server.port 8501 > /tmp/streamlit.log 2>&1 &",
            shell=True
        )
        time.sleep(5)
        print(f"{Colors.OKGREEN}✅ Streamlit dashboard started{Colors.ENDC}")
        print(f"{Colors.OKGREEN}   URL: http://localhost:8501{Colors.ENDC}")
        pipeline_results['dashboard'] = {'success': True, 'port': 8501}
    
    # Pipeline Summary
    elapsed = time.time() - start_time
    
    print(f"\n{Colors.HEADER}")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║                    PIPELINE COMPLETE                           ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")
    
    print(f"{Colors.BOLD}⏱️  Total Time: {elapsed:.1f}s{Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}📊 Results:{Colors.ENDC}")
    print(f"   {'Data Prep:':<20} {'✅' if pipeline_results.get('data_prep', {}).get('success') else '❌'}")
    print(f"   {'Training:':<20} {'✅' if pipeline_results.get('training', {}).get('success') else '❌'}")
    print(f"   {'Evaluation:':<20} {'✅' if pipeline_results.get('evaluation', {}).get('passed') else '⚠️'}")
    print(f"   {'Drift Monitor:':<20} {'✅' if pipeline_results.get('drift_monitoring', {}).get('success') else '❌'}")
    print(f"   {'Reports:':<20} {'✅' if pipeline_results.get('reports', {}).get('success') else '❌'}")
    print(f"   {'API:':<20} {'✅' if pipeline_results.get('api', {}).get('success') else '❌'}")
    print(f"   {'Dashboard:':<20} {'✅' if pipeline_results.get('dashboard', {}).get('success') else '❌'}")
    
    print(f"\n{Colors.BOLD}🌐 Access Points:{Colors.ENDC}")
    print(f"   API:         http://localhost:8000/docs")
    print(f"   Metrics:     http://localhost:8000/metrics/")
    print(f"   Dashboard:   http://localhost:8501")
    print(f"   Report:      reports/price_engine_report.html")
    
    if 'best_model' in pipeline_results:
        print(f"\n{Colors.BOLD}🎯 Model Performance:{Colors.ENDC}")
        print(f"   Model:       {pipeline_results['best_model']['tag']}")
        print(f"   MAE:         ¥{pipeline_results['best_model']['test_metrics']['MAE']/1e6:.2f}M")
        print(f"   MAPE:        {pipeline_results['best_model']['test_metrics']['MAPE']:.2f}%")
    
    print(f"\n{Colors.OKGREEN}{'='*70}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}🎉 MLOps Pipeline Successfully Completed!{Colors.ENDC}")
    print(f"{Colors.OKGREEN}{'='*70}{Colors.ENDC}\n")
    
    # Save pipeline metadata
    pipeline_metadata = {
        'run_at': datetime.utcnow().isoformat(),
        'elapsed_seconds': elapsed,
        'results': pipeline_results,
        'success': all([
            pipeline_results.get('training', {}).get('success'),
            pipeline_results.get('api', {}).get('success'),
        ])
    }
    
    with open('pipeline_run.json', 'w') as f:
        json.dump(pipeline_metadata, f, indent=2)
    
    print(f"{Colors.OKCYAN}📝 Pipeline metadata saved to pipeline_run.json{Colors.ENDC}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}⚠️ Pipeline interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}❌ Pipeline failed: {e}{Colors.ENDC}")
        sys.exit(1)

