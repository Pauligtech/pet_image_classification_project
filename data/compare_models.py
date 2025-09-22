import subprocess
import sys
import argparse
from datetime import datetime
import os

def run_model_and_capture_results(model_name, image_dir, dogfile):
    """Run check_images.py with the specified model and capture output"""
    print(f"\nRunning {model_name.upper()} model...")

    try:
        cmd = [
            sys.executable, "check_images.py",
            "--dir", image_dir,
            "--arch", model_name,
            "--dogfile", dogfile
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"Error running {model_name}: {result.stderr}")
            return None
        return result.stdout
    except subprocess.TimeoutExpired:
        print(f"Timeout running {model_name}")
        return None
    except Exception as e:
        print(f"Error running {model_name}: {e}")
        return None


def extract_statistics(output_text):
    """Extract key statistics, image counts, and runtime from check_images.py output"""
    lines = output_text.split('\n')
    stats = {}
    in_results_section = False
    
    # Extract basic counts and runtime
    for line in lines:
        if "Results Summary for CNN Model Architecture" in line:
            in_results_section = True
            continue
        elif in_results_section and "N Images" in line and ":" in line:
            parts = line.split(":")
            if len(parts) == 2:
                try:
                    stats['n_images'] = int(parts[1].strip())
                except:
                    pass
        elif in_results_section and "N Dog Images" in line and ":" in line:
            parts = line.split(":")
            if len(parts) == 2:
                try:
                    stats['n_dogs'] = int(parts[1].strip())
                except:
                    pass
        elif in_results_section and "N Not-Dog Images" in line and ":" in line:
            parts = line.split(":")
            if len(parts) == 2:
                try:
                    stats['n_notdogs'] = int(parts[1].strip())
                except:
                    pass
        elif in_results_section and ":" in line and "%" in line:
            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip().replace(" ", "_").lower()
                value = parts[1].strip().replace("%", "")
                try:
                    stats[key] = float(value)
                except:
                    stats[key] = value
        elif "Total Elapsed Runtime" in line:
            # Extract runtime - format is "** Total Elapsed Runtime: h:m:s"
            # Find the runtime part after the last ': '
            runtime_start = line.rfind(': ') + 2
            if runtime_start > 1:
                runtime = line[runtime_start:].strip()
                stats['runtime'] = runtime
            break

    return stats


def create_visual_results_table(all_results):
    """Create results table with matplotlib visualization"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
    except ImportError:
        print("Matplotlib not available. Skipping visual table creation.")
        return None
    
    # Extract common statistics (assuming they're the same across models)
    sample_stats = next(iter(all_results.values()))
    if not sample_stats:
        print("No valid results to display")
        return None
    
    n_images = sample_stats.get('n_images', 0)
    n_dogs = sample_stats.get('n_dogs', 0) 
    n_notdogs = sample_stats.get('n_notdogs', 0)
    
    # Create figure with increased width for better text fitting
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('white')  # Ensure white background
    
    # Title
    ax.text(5, 9.5, 'Results Table', fontsize=26, fontweight='bold', ha='center')
    
    # Summary statistics table (top section) - increased width
    summary_data = [
        ['# Total Images', str(n_images)],
        ['# Dog Images', str(n_dogs)],
        ['# Not-a-Dog Images', str(n_notdogs)]
    ]
    
    # Draw summary table with increased width
    summary_table = ax.table(
        cellText=summary_data,
        cellLoc='left',
        loc='center',
        bbox=[0.15, 0.73, 0.4, 0.18]  # Increased width
    )
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(14)
    summary_table.scale(1, 1.6)
    
    # Style summary table
    for i in range(len(summary_data)):
        summary_table[(i, 0)].set_text_props(weight='bold')
        summary_table[(i, 0)].set_facecolor('#f0f0f0')
        summary_table[(i, 1)].set_text_props(weight='bold')
        summary_table[(i, 0)].set_edgecolor('black')
        summary_table[(i, 1)].set_edgecolor('black')
    
    # Main results table (bottom section) with Runtime column
    models = ['ResNet', 'AlexNet', 'VGG']  # Order to match reference
    headers = ['Model Architecture', '% Not-a-Dog\nCorrect', '% Dogs\nCorrect', 
               '% Breeds\nCorrect', '% Match\nLabels', 'Total Elapsed\nRuntime']
    
    # Prepare data for main table
    table_data = []
    
    for model in models:
        model_key = model.lower()
        if model_key in all_results and all_results[model_key]:
            stats = all_results[model_key]
            notdog_pct = f"{stats.get('pct_correct_notdogs', 0):.1f}%"
            dog_pct = f"{stats.get('pct_correct_dogs', 0):.1f}%"
            breed_pct = f"{stats.get('pct_correct_breed', 0):.1f}%"
            match_pct = f"{stats.get('pct_match', 0):.1f}%"
            runtime = stats.get('runtime', '0:0:0')
            
            table_data.append([model, notdog_pct, dog_pct, breed_pct, match_pct, runtime])
        else:
            table_data.append([model, 'ERROR', 'ERROR', 'ERROR', 'ERROR', 'ERROR'])
    
    # Create main results table with increased width
    main_table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        bbox=[0.05, 0.28, 0.9, 0.35]  # Increased height and adjusted position
    )
    
    main_table.auto_set_font_size(False)
    main_table.set_fontsize(12)
    main_table.scale(1, 2.0)  # Increased scale for better text fitting
    
    # Style main table
    # Header row
    for j in range(len(headers)):
        main_table[(0, j)].set_text_props(weight='bold', fontsize=11)
        main_table[(0, j)].set_facecolor('#e6e6e6')
        main_table[(0, j)].set_edgecolor('black')
    
    # Data rows with alternating colors and special formatting
    colors = ['white', '#f9f9f9', '#f0f0f0']
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = main_table[(i, j)]
            cell.set_edgecolor('black')  # Add borders
            
            if j == 0:  # Model name column
                cell.set_text_props(weight='bold', fontsize=12)
                cell.set_facecolor('#e6e6e6')
            else:
                cell.set_facecolor(colors[i-1])
                
                # Highlight 100% values in blue
                cell_text = cell.get_text().get_text()
                if '100.0%' in cell_text:
                    cell.set_text_props(color='blue', weight='bold', fontsize=11)
                elif any(x in cell_text for x in ['93.3%', '90.0%', '80.0%']):
                    cell.set_text_props(color='blue', weight='bold', fontsize=11)
                else:
                    cell.set_text_props(fontsize=11)
    
    # Add "Project Results" footer
    ax.text(5, 0.1, 'Project Results', fontsize=18, fontweight='bold', ha='center')
    
    # Save the image with white background
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_table_uploaded-images.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300, facecolor='white', 
                edgecolor='none', pad_inches=0.2)
    plt.close()
    
    print(f"\nVisual results table saved as: {filename}")
    return filename


def create_console_results_table(all_results):
    """Create a formatted console table"""
    print("\n" + "="*80)
    print("RESULTS COMPARISON TABLE")
    print("="*80)

    header = ['Model', '% Dogs', '% Breeds', '% Not-Dogs', '% Match']
    print(f"{header[0]:<10} {header[1]:<12} {header[2]:<12} {header[3]:<12} {header[4]:<12}")
    print("-" * 60)

    for model, stats in all_results.items():
        if stats:
            dogs = stats.get('pct_correct_dogs', 'N/A')
            breeds = stats.get('pct_correct_breed', 'N/A')
            notdogs = stats.get('pct_correct_notdogs', 'N/A')
            match = stats.get('pct_match', 'N/A')

            dogs_str = f"{dogs:.1f}%" if isinstance(dogs, float) else str(dogs)
            breeds_str = f"{breeds:.1f}%" if isinstance(breeds, float) else str(breeds)
            notdogs_str = f"{notdogs:.1f}%" if isinstance(notdogs, float) else str(notdogs)
            match_str = f"{match:.1f}%" if isinstance(match, float) else str(match)

            print(f"{model.upper():<10} {dogs_str:<12} {breeds_str:<12} {notdogs_str:<12} {match_str:<12}")
        else:
            print(f"{model.upper():<10} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12}")


def analyze_results(all_results):
    """Print analysis and recommend best model"""
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    def safe_get(stats, key):
        return stats.get(key, 0) if stats else 0

    best_dogs = max(all_results.items(), key=lambda x: safe_get(x[1], 'pct_correct_dogs'))
    best_breeds = max(all_results.items(), key=lambda x: safe_get(x[1], 'pct_correct_breed'))
    best_notdogs = max(all_results.items(), key=lambda x: safe_get(x[1], 'pct_correct_notdogs'))
    best_overall = max(all_results.items(), key=lambda x: safe_get(x[1], 'pct_match'))

    print(f"Best at identifying dogs: {best_dogs[0].upper()} ({safe_get(best_dogs[1], 'pct_correct_dogs'):.1f}%)")
    print(f"Best at breed classification: {best_breeds[0].upper()} ({safe_get(best_breeds[1], 'pct_correct_breed'):.1f}%)")
    print(f"Best at identifying non-dogs: {best_notdogs[0].upper()} ({safe_get(best_notdogs[1], 'pct_correct_notdogs'):.1f}%)")
    print(f"Best overall match rate: {best_overall[0].upper()} ({safe_get(best_overall[1], 'pct_match'):.1f}%)")
    print(f"\nRECOMMENDED MODEL: {best_overall[0].upper()}")


def main():
    parser = argparse.ArgumentParser(description="Compare models with results table")
    parser.add_argument("--dir", default="uploaded_images/", help="Path to image directory")
    parser.add_argument("--dogfile", default="dognames.txt", help="Dog names file")
    args = parser.parse_args()

    # Validate required files exist
    if not os.path.exists("check_images.py"):
        print("ERROR: check_images.py not found in current directory")
        return
    
    if not os.path.exists(args.dir):
        print(f"ERROR: Image directory '{args.dir}' not found")
        return
        
    if not os.path.exists(args.dogfile):
        print(f"ERROR: Dog names file '{args.dogfile}' not found")
        return

    models = ['vgg', 'alexnet', 'resnet']
    all_results = {}

    print("Uploaded Image Classification - Model Comparison")
    print("="*50)

    for model in models:
        output = run_model_and_capture_results(model, args.dir, args.dogfile)
        if output:
            stats = extract_statistics(output)
            all_results[model] = stats
            print(f"✓ {model.upper()} completed")
        else:
            all_results[model] = None
            print(f"✗ {model.upper()} failed")

    # Create both console and visual tables
    create_console_results_table(all_results)
    analyze_results(all_results)
    create_visual_results_table(all_results)

    print("\nComparison complete!")


if __name__ == "__main__":
    main()


# Run this in the terminal using 
# python compare_models.py --dir uploaded_images/ --dogfile dognames.txt