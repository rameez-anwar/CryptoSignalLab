import generator
import signal_pipeline

def main():
    print("Running generator...")
    # generator.py runs its logic on import, but to be explicit, reload if needed
    # If you want to refactor generator.py and signal_pipeline.py to have main() functions, you can do so and call them here
    print("Generator complete. Running signal pipeline...")
    print("Signal pipeline complete.")

if __name__ == '__main__':
    main() 